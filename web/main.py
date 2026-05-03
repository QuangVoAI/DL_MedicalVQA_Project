import asyncio
import collections
import gc
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor
from src.models.medical_vqa_model import MedicalVQAModelA
from src.models.multimodal_vqa import MultimodalVQA
from src.utils.answer_rewriter import MedicalAnswerRewriter
from src.utils.helpers import majority_answer
from src.utils.text_utils import postprocess_answer
from src.utils.translator import MedicalTranslator
from src.utils.visualization import MedicalImageTransform


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "medical_vqa.yaml"

VARIANT_ORDER = ["A1", "A2", "B1", "B2", "DPO", "PPO"]

VARIANT_META = {
    "A1": {
        "family": "A",
        "title": "A1",
        "subtitle": "LSTM baseline",
        "description": "DenseNet-121 + PhoBERT + LSTM",
    },
    "A2": {
        "family": "A",
        "title": "A2",
        "subtitle": "Transformer decoder",
        "description": "DenseNet-121 + PhoBERT + Transformer",
    },
    "B1": {
        "family": "B",
        "title": "B1",
        "subtitle": "Zero-shot",
        "description": "LLaVA-Med base",
    },
    "B2": {
        "family": "B",
        "title": "B2",
        "subtitle": "Fine-tuned",
        "description": "LLaVA-Med + LoRA",
    },
    "DPO": {
        "family": "B",
        "title": "DPO",
        "subtitle": "Alignment",
        "description": "B2 + Direct Preference Optimization",
    },
    "PPO": {
        "family": "B",
        "title": "PPO",
        "subtitle": "RL refinement",
        "description": "B2 + Proximal Policy Optimization",
    },
}

SUGGESTION_DATA_PATH = ROOT_DIR / "data" / "merged_vqa_vi_cleaned.json"
SUGGESTION_LIMIT = int(os.getenv("WEB_SUGGESTION_LIMIT", "8"))


def _read_config() -> dict[str, Any]:
    try:
        import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to read config at {CONFIG_PATH}: {exc}") from exc


CFG = _read_config()

app = FastAPI(title="Medical VQA Compare API", version="2.0.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class VQAServerState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self.image_size = int(CFG.get("data", {}).get("image_size", 224))
        self.answer_max_words = int(CFG.get("data", {}).get("answer_max_words", 10))
        self.max_question_len = int(CFG.get("data", {}).get("max_question_len", 64))
        self.max_answer_len = int(CFG.get("data", {}).get("max_answer_len", 20))
        self.model_a_cfg = CFG.get("model_a", {})
        self.model_b_cfg = CFG.get("model_b", {})
        self.eval_cfg = CFG.get("eval", {})
        self.models_dir = ROOT_DIR / "checkpoints"
        self.qa_tokenizer = None
        self.translator = MedicalTranslator(device="cpu")
        self.answer_rewriter = MedicalAnswerRewriter()
        self.image_transform = MedicalImageTransform(size=self.image_size)
        self.cache_lock = asyncio.Lock()
        self.b_lock = asyncio.Lock()
        self.a_models: dict[str, dict[str, Any]] = {}
        self.llava_bundle: dict[str, Any] | None = None
        self.question_suggestions: list[dict[str, Any]] = []
        self.preload_models = os.getenv("WEB_PRELOAD_MODELS", "1" if self.device.type == "cuda" else "0") == "1"

    @property
    def phobert_model(self) -> str:
        return self.model_a_cfg.get("phobert_model", "vinai/phobert-base")

    @property
    def llava_model_id(self) -> str:
        return self.model_b_cfg.get("model_name", "chaoyinshe/llava-med-v1.5-mistral-7b-hf")


state = VQAServerState()
load_lock = asyncio.Lock()


def _artifact_exists(path: Path) -> bool:
    return path.exists()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _normalize_text_key(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _suggestion_category(item: dict[str, Any], question: str) -> str:
    content_type = str(item.get("content_type", "")).strip()
    if content_type:
        return content_type

    q = question.lower()
    if any(token in q for token in ["bất thường", "abnormal", "normal", "có vẻ"]):
        return "Abnormality"
    if any(token in q for token in ["phương thức", "modality", "chụp", "scan", "x-ray", "ct", "mri"]):
        return "Modality"
    if any(token in q for token in ["mặt phẳng", "plane", "lát cắt"]):
        return "Plane"
    if any(token in q for token in ["bao nhiêu", "how many", "số lượng"]):
        return "Quantity"
    if any(token in q for token in ["màu", "color"]):
        return "Color"
    if any(token in q for token in ["ở đâu", "vị trí", "where"]):
        return "Position"
    if any(token in q for token in ["chứa", "contain", "có "]):
        return "Organ"
    return "General"


def _load_question_suggestions(limit: int = SUGGESTION_LIMIT) -> list[dict[str, Any]]:
    if not SUGGESTION_DATA_PATH.exists():
        return []

    try:
        with SUGGESTION_DATA_PATH.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as exc:
        print(f"[WARNING] Failed to read suggestion dataset: {exc}")
        return []

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in dataset:
        if not _as_bool(item.get("question_vi_valid", True)):
            continue
        if _as_bool(item.get("low_quality", False)):
            continue
        question = str(item.get("question_vi") or "").strip()
        if not question:
            continue
        groups[_normalize_text_key(question)].append(item)

    candidates: list[dict[str, Any]] = []
    for items in groups.values():
        if len(items) < 8:
            continue

        question = str(items[0].get("question_vi") or "").strip()
        if not question:
            continue

        answer_texts = []
        content_types = []
        answer_types = []
        modalities = []
        for item in items:
            answer = str(item.get("answer_vi") or item.get("answer") or "").strip()
            if answer:
                answer_texts.append(_normalize_text_key(answer))
            content_types.append(str(item.get("content_type", "")).strip())
            answer_types.append(str(item.get("answer_type", "")).strip().upper())
            modalities.append(str(item.get("modality", "")).strip())

        if not answer_texts:
            continue

        answer_counter = collections.Counter(answer_texts)
        top_answer, top_count = answer_counter.most_common(1)[0]
        total = len(answer_texts)
        confidence = top_count / total
        answer_type = collections.Counter(answer_types).most_common(1)[0][0] if answer_types else ""
        content_type = collections.Counter([c for c in content_types if c]).most_common(1)[0][0] if any(content_types) else ""
        modality = collections.Counter([m for m in modalities if m]).most_common(1)[0][0] if any(modalities) else ""

        if answer_type == "CLOSED":
            if confidence < 0.85:
                continue
        elif confidence < 0.92:
            continue
        if answer_type != "CLOSED" and len(top_answer.split()) > 3:
            continue
        if len(question) > 140:
            continue

        category = _suggestion_category(items[0], question)
        category_bonus = {
            "Abnormality": 5.0,
            "Modality": 4.5,
            "Plane": 4.25,
            "Organ": 4.0,
            "Position": 3.5,
            "Quantity": 3.25,
            "Color": 3.0,
            "General": 2.0,
        }.get(category, 2.0)
        score = confidence * 100.0 + min(total, 80) * 0.15 + category_bonus - len(question) * 0.02

        candidates.append(
            {
                "question": question,
                "question_key": _normalize_text_key(question),
                "answer": top_answer,
                "answer_type": answer_type or "OPEN",
                "content_type": content_type or category,
                "modality": modality,
                "confidence": round(confidence, 3),
                "sample_count": total,
                "score": round(score, 3),
            }
        )

    if not candidates:
        return []

    priority_order = ["Abnormality", "Modality", "Plane", "Organ", "Position", "Quantity", "Color", "General"]
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    per_category_limit = 2
    category_counts: dict[str, int] = collections.defaultdict(int)

    for category in priority_order:
        category_candidates = sorted(
            (c for c in candidates if c["content_type"].lower() == category.lower()),
            key=lambda item: (item["score"], item["confidence"], item["sample_count"]),
            reverse=True,
        )
        for candidate in category_candidates:
            if candidate["question_key"] in used_keys:
                continue
            if category_counts[category] >= per_category_limit:
                break
            selected.append(candidate)
            used_keys.add(candidate["question_key"])
            category_counts[category] += 1
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for candidate in sorted(candidates, key=lambda item: (item["score"], item["confidence"], item["sample_count"]), reverse=True):
            if candidate["question_key"] in used_keys:
                continue
            selected.append(candidate)
            used_keys.add(candidate["question_key"])
            if len(selected) >= limit:
                break

    return selected[:limit]


def _select_best_b2_checkpoint(checkpoint_root: Path) -> Optional[Path]:
    if not checkpoint_root.exists():
        return None

    best_dir: Optional[Path] = None
    best_metric: Optional[float] = None

    for ckpt_dir in sorted(checkpoint_root.glob("checkpoint-*")):
        state_file = ckpt_dir / "trainer_state.json"
        if not state_file.exists():
            continue
        try:
            trainer_state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        metric = trainer_state.get("best_metric")
        if isinstance(metric, str):
            try:
                metric = float(metric)
            except ValueError:
                metric = None

        if metric is None:
            eval_losses = [
                rec.get("eval_loss")
                for rec in trainer_state.get("log_history", [])
                if isinstance(rec, dict) and rec.get("eval_loss") is not None
            ]
            metric = min(eval_losses) if eval_losses else None

        if metric is None:
            continue

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_dir = ckpt_dir

    if best_dir is not None:
        return best_dir

    checkpoints = sorted(checkpoint_root.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


def _resolve_variant_artifact(variant: str) -> dict[str, Any]:
    if variant in {"A1", "A2"}:
        ckpt_path = ROOT_DIR / "checkpoints" / f"medical_vqa_{variant}_best.pth"
        if not ckpt_path.exists():
            resume_path = ROOT_DIR / "checkpoints" / f"medical_vqa_{variant}_resume.pth"
            ckpt_path = resume_path if resume_path.exists() else ckpt_path
        return {"type": "direction_a", "path": ckpt_path}

    if variant == "B1":
        return {"type": "llava_base", "path": state.llava_model_id}

    if variant == "B2":
        ckpt_dir = _select_best_b2_checkpoint(ROOT_DIR / "checkpoints" / "B2")
        return {"type": "llava_adapter", "path": ckpt_dir}

    if variant == "DPO":
        final_adapter = ROOT_DIR / "checkpoints" / "DPO" / "final_adapter"
        fallback = ROOT_DIR / "checkpoints" / "DPO" / "checkpoint-25"
        return {"type": "llava_adapter", "path": final_adapter if final_adapter.exists() else fallback}

    if variant == "PPO":
        final_adapter = ROOT_DIR / "checkpoints" / "PPO" / "final_adapter"
        return {"type": "llava_adapter", "path": final_adapter}

    raise ValueError(f"Unknown variant: {variant}")


def _llava_adapter_specs() -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for variant in ("B2", "DPO", "PPO"):
        artifact = _resolve_variant_artifact(variant)["path"]
        if isinstance(artifact, Path) and artifact.exists():
            specs.append((variant, artifact))
    return specs


def _ensure_qa_tokenizer():
    if state.qa_tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(state.phobert_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        state.qa_tokenizer = tokenizer
    return state.qa_tokenizer


def _looks_vietnamese(text: str) -> bool:
    vi_marks = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    lowered = text.lower()
    if any(ch in vi_marks for ch in lowered):
        return True
    vi_keywords = {
        "không",
        "có",
        "bệnh",
        "phổi",
        "tim",
        "sọ",
        "xương",
        "ảnh",
        "hỏi",
        "đâu",
        "gì",
        "như thế nào",
    }
    return any(keyword in lowered for keyword in vi_keywords)


def _looks_closed_question(question: str) -> bool:
    normalized = question.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    closed_prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "has ",
        "have ",
        "had ",
        "có ",
        "có phải",
        "liệu ",
    )
    closed_keywords = {
        "yes",
        "no",
        "không",
        "có",
        "normal",
        "abnormal",
        "present",
        "absent",
        "sốt",
    }
    open_prefixes = ("what ", "where ", "when ", "who ", "which ", "how ", "why ")
    if normalized.startswith(open_prefixes):
        return False
    if normalized.startswith(closed_prefixes):
        return True
    return any(word in normalized.split() for word in closed_keywords)


def _normalize_closed_answer(question_vi: str, question_en: str, pred_vi: str, pred_en: str = "") -> str:
    question_text = f"{question_vi} {question_en}".lower()
    combined = " ".join(part for part in [pred_vi, pred_en] if part).lower().strip()
    combined_norm = re.sub(r"\s+", " ", combined)

    is_normality_question = any(pattern in question_text for pattern in ["bình thường", "normal", "abnormal"])

    if is_normality_question:
        if any(pattern in combined_norm for pattern in ["không bình thường", "not normal"]):
            return "không"
        if any(pattern in combined_norm.split() for pattern in ["có", "yes"]):
            return "có"
        if any(pattern in combined_norm for pattern in ["bình thường", "normal", "unremarkable", "no significant abnormalities"]):
            return "có"
        if any(pattern in combined_norm for pattern in ["bất thường", "abnormal", "fracture", "lesion", "mass", "effusion", "pneumothorax"]):
            return "không"
    else:
        if any(pattern in combined_norm for pattern in ["không", "no", "absent", "negative", "none"]):
            return "không"
        if any(pattern in combined_norm for pattern in ["có", "yes", "present", "detected", "positive"]):
            return "có"

    if any(pattern in combined_norm for pattern in ["bình thường", "normal", "unremarkable", "no significant abnormalities"]):
        return "có"
    if any(pattern in combined_norm for pattern in ["bất thường", "abnormal", "fracture", "lesion", "mass", "effusion", "pneumothorax"]):
        return "không"
    return pred_vi or pred_en or ""


def _build_bad_words_ids(processor, variant: str) -> list[list[int]] | None:
    if variant not in {"B1", "B2", "DPO", "PPO"}:
        return None
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    banned_phrases = [
        "yes",
        "no",
        "the answer is",
        "the image is",
        "this image is",
        "the image shows",
        "the scan shows",
        "there is",
        "there are",
        "it appears",
        "the finding is",
    ]
    bad_words_ids = []
    for phrase in banned_phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)
    return bad_words_ids or None


def _build_b1_prompt(question_en: str, max_words: int) -> str:
    instruction = f"Answer in Vietnamese, concise, at most {max_words} words."
    return f"USER: <image>\n{question_en}\n{instruction} ASSISTANT:"


def _rewrite_final_answer(question: str, raw_answer: str, language: str = "vi") -> str:
    """
    Chỉ rewrite phần output hiển thị cuối cùng.
    Raw prediction vẫn được giữ nguyên trong payload để debug.
    """
    candidate = state.answer_rewriter.rewrite(question=question, answer=raw_answer, language=language)
    candidate = postprocess_answer(candidate, max_words=state.answer_max_words)
    if candidate:
        return candidate
    return postprocess_answer(raw_answer, max_words=state.answer_max_words)


def _extract_key_medical_term(raw_en: str, max_words: int) -> str:
    text = re.sub(r"\s+", " ", (raw_en or "").strip())
    if not text:
        return ""
    return " ".join(text.split()[:max_words])


def _en_to_vi_direct(en_text: str) -> Optional[str]:
    text = (en_text or "").strip().lower()
    mapping = {
        "yes": "có",
        "no": "không",
        "normal": "bình thường",
        "abnormal": "bất thường",
        "present": "có",
        "absent": "không",
    }
    return mapping.get(text)


def _prepare_question_text(question: str, variant: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "", ""

    if variant == "B1":
        question_en = question if not _looks_vietnamese(question) else state.translator.translate_vi2en(question)
        return question, question_en

    question_vi = question if _looks_vietnamese(question) else state.translator.translate_en2vi(question)
    return question_vi, question


async def _ensure_direction_a_model(variant: str):
    if variant not in {"A1", "A2"}:
        raise ValueError(f"Unsupported direction A variant: {variant}")

    cached = state.a_models.get(variant)
    if cached is not None:
        return cached

    async with state.cache_lock:
        cached = state.a_models.get(variant)
        if cached is not None:
            return cached

        tokenizer = _ensure_qa_tokenizer()
        ckpt_path = _resolve_variant_artifact(variant)["path"]
        if not isinstance(ckpt_path, Path) or not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint cho {variant}: {ckpt_path}")

        decoder_type = "lstm" if variant == "A1" else "transformer"
        model = MedicalVQAModelA(
            decoder_type=decoder_type,
            vocab_size=len(tokenizer),
            hidden_size=int(state.model_a_cfg.get("hidden_size", 768)),
            phobert_model=state.phobert_model,
        ).to(state.device)

        payload = torch.load(ckpt_path, map_location=state.device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        bundle = {
            "variant": variant,
            "family": "A",
            "model": model,
            "tokenizer": tokenizer,
            "checkpoint": str(ckpt_path),
        }
        state.a_models[variant] = bundle
        return bundle


def _build_llava_base_and_processor():
    if not torch.cuda.is_available():
        raise RuntimeError("Các model LLaVA (B1/B2/DPO/PPO) cần CUDA để chạy trong web này.")

    wrapper = MultimodalVQA(
        model_id=state.llava_model_id,
        lora_r=int(state.model_b_cfg.get("lora_r", 16)),
        lora_alpha=int(state.model_b_cfg.get("lora_alpha", 32)),
        lora_dropout=float(state.model_b_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=state.model_b_cfg.get("lora_target_modules"),
    )
    processor = LlavaProcessor.from_pretrained(wrapper.model_id)
    processor.tokenizer.padding_side = "left"
    base_model = LlavaForConditionalGeneration.from_pretrained(
        wrapper.model_id,
        quantization_config=wrapper.bnb_config,
        device_map="auto",
    )
    base_model.config.use_cache = False
    return wrapper, processor, base_model


async def _ensure_llava_bundle():
    cached = state.llava_bundle
    if cached is not None:
        return cached

    async with state.cache_lock:
        cached = state.llava_bundle
        if cached is not None:
            return cached

        wrapper, processor, base_model = _build_llava_base_and_processor()
        adapter_specs = _llava_adapter_specs()
        adapter_name_map = {variant: variant for variant, _ in adapter_specs}

        if adapter_specs:
            first_variant, first_path = adapter_specs[0]
            model = PeftModel.from_pretrained(
                base_model,
                str(first_path),
                adapter_name=first_variant,
                is_trainable=False,
            )
            for variant, path in adapter_specs[1:]:
                model.load_adapter(str(path), adapter_name=variant, is_trainable=False)
            model.set_adapter(first_variant)
        else:
            model = base_model

        model.eval()
        bundle = {
            "family": "B",
            "model": model,
            "processor": processor,
            "wrapper": wrapper,
            "checkpoint": adapter_specs[0][1].as_posix() if adapter_specs else state.llava_model_id,
            "adapter_name_map": adapter_name_map,
            "peft": bool(adapter_specs),
        }
        state.llava_bundle = bundle
        return bundle


def _predict_direction_a(bundle: dict[str, Any], question_vi: str, image: Image.Image) -> dict[str, Any]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_tensor = state.image_transform(image.convert("L")).unsqueeze(0).to(state.device)

    inputs = tokenizer(
        question_vi,
        padding="max_length",
        truncation=True,
        max_length=state.max_question_len,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(state.device)
    attention_mask = inputs["attention_mask"].to(state.device)
    is_closed = _looks_closed_question(question_vi)

    with torch.inference_mode():
        logits_closed, pred_ids = model.inference(
            image_tensor,
            input_ids,
            attention_mask,
            beam_width=int(state.eval_cfg.get("beam_width_a", 5)),
            max_len=state.max_answer_len,
        )

    if is_closed:
        prediction_raw = "có" if logits_closed.argmax(dim=1).item() == 1 else "không"
        prediction = _rewrite_final_answer(question_vi, prediction_raw, language="vi")
    else:
        prediction_raw = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        prediction = _rewrite_final_answer(question_vi, prediction_raw, language="vi")

    return {
        "prediction": prediction,
        "prediction_raw": prediction_raw,
        "status": "ok",
    }


async def _predict_direction_b(
    bundle: dict[str, Any],
    question_vi: str,
    question_en: str,
    image: Image.Image,
    variant: str,
) -> dict[str, Any]:
    model = bundle["model"]
    processor = bundle["processor"]
    wrapper = bundle["wrapper"]
    is_closed = _looks_closed_question(question_vi if variant != "B1" else question_en)
    question_for_variant = question_en if variant == "B1" else question_vi
    adapter_name = bundle.get("adapter_name_map", {}).get(variant)

    if variant == "B1":
        prompt = _build_b1_prompt(question_for_variant, state.answer_max_words)
        num_beams = int(state.eval_cfg.get("beam_width_b_open", 5))
        max_new_tokens = int(state.eval_cfg.get("max_new_tokens_b_open", state.answer_max_words + 6))
    else:
        prompt = wrapper.build_instruction_prompt(question_for_variant, language="vi", include_answer=False)
        num_beams = int(state.eval_cfg.get("beam_width_b_closed", 1)) if is_closed else int(
            state.eval_cfg.get("beam_width_b_open", 5)
        )
        max_new_tokens = int(state.eval_cfg.get("max_new_tokens_b_closed", 4)) if is_closed else int(
            state.eval_cfg.get("max_new_tokens_b_open", state.answer_max_words + 6)
        )

    bad_words_ids = _build_bad_words_ids(processor, variant)
    inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True)
    inputs = inputs.to(state.device)
    if "pixel_values" in inputs and torch.cuda.is_available():
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    async with state.b_lock:
        if adapter_name and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        if variant == "B1" and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=num_beams,
                        early_stopping=num_beams > 1,
                        bad_words_ids=bad_words_ids,
                    )
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    early_stopping=num_beams > 1,
                    bad_words_ids=bad_words_ids,
                )

    input_token_len = inputs.input_ids.shape[1]
    pred_raw = processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

    if variant == "B1":
        pred_en = _extract_key_medical_term(pred_raw, 50)
        if is_closed:
            prediction = _normalize_closed_answer(question_vi, question_en, pred_en, pred_en)
        else:
            prediction = _en_to_vi_direct(pred_en)
            if prediction is None:
                prediction = state.translator.translate_en2vi(pred_en)
            prediction = postprocess_answer(prediction, max_words=state.answer_max_words)
    else:
        if is_closed:
            prediction = _normalize_closed_answer(question_vi, question_en, pred_raw)
        else:
            prediction = postprocess_answer(pred_raw, max_words=state.answer_max_words)

    prediction = _rewrite_final_answer(question_vi or question_en, prediction, language="vi")

    return {
        "prediction": prediction,
        "prediction_raw": pred_raw,
        "status": "ok",
    }


async def predict_variant(variant: str, question: str, image: Image.Image) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if variant in {"A1", "A2"}:
            bundle = await _ensure_direction_a_model(variant)
        else:
            artifact = _resolve_variant_artifact(variant)["path"]
            if variant != "B1" and (not isinstance(artifact, Path) or not artifact.exists()):
                raise FileNotFoundError(f"Không tìm thấy artifact cho {variant}: {artifact}")
            bundle = await _ensure_llava_bundle()
        question_vi, question_en = _prepare_question_text(question, variant)
        if variant == "B1":
            if not question_en:
                question_en = question
            result = await _predict_direction_b(bundle, question_vi, question_en, image, variant)
        elif bundle["family"] == "A":
            result = _predict_direction_a(bundle, question_vi, image)
        else:
            result = await _predict_direction_b(bundle, question_vi, question_en, image, variant)

        result.update(
            {
                "variant": variant,
                "checkpoint": (
                    bundle.get("checkpoint", "")
                    if variant in {"A1", "A2"}
                    else str(_resolve_variant_artifact(variant)["path"])
                    if variant != "B1"
                    else state.llava_model_id
                ),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )
        return result
    except Exception as exc:
        return {
            "variant": variant,
            "prediction": "",
            "prediction_raw": "",
            "status": f"error: {exc}",
            "checkpoint": "",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


def _parse_model_selection(raw_model_name: Optional[str], raw_model_names: Optional[str]) -> list[str]:
    if raw_model_names:
        try:
            parsed = json.loads(raw_model_names)
        except Exception:
            parsed = [part.strip() for part in raw_model_names.split(",") if part.strip()]
        if isinstance(parsed, str):
            parsed = [parsed]
        selected = [name for name in parsed if name in VARIANT_ORDER]
        if selected:
            return selected

    if raw_model_name and raw_model_name in VARIANT_ORDER:
        return [raw_model_name]

    return VARIANT_ORDER[:]


def _variant_availability() -> dict[str, dict[str, Any]]:
    b2_checkpoint = _select_best_b2_checkpoint(ROOT_DIR / "checkpoints" / "B2")
    cuda_ready = torch.cuda.is_available()
    return {
        "A1": {"available": (_artifact_exists(ROOT_DIR / "checkpoints" / "medical_vqa_A1_best.pth")), "artifact": "checkpoints/medical_vqa_A1_best.pth"},
        "A2": {"available": (_artifact_exists(ROOT_DIR / "checkpoints" / "medical_vqa_A2_best.pth")), "artifact": "checkpoints/medical_vqa_A2_best.pth"},
        "B1": {"available": cuda_ready, "artifact": state.llava_model_id},
        "B2": {"available": cuda_ready and b2_checkpoint is not None, "artifact": str(b2_checkpoint) if b2_checkpoint else ""},
        "DPO": {"available": cuda_ready and (_artifact_exists(ROOT_DIR / "checkpoints" / "DPO" / "final_adapter") or _artifact_exists(ROOT_DIR / "checkpoints" / "DPO" / "checkpoint-25")), "artifact": "checkpoints/DPO/final_adapter"},
        "PPO": {"available": cuda_ready and _artifact_exists(ROOT_DIR / "checkpoints" / "PPO" / "final_adapter"), "artifact": "checkpoints/PPO/final_adapter"},
    }


@app.on_event("startup")
async def startup_event() -> None:
    _ensure_qa_tokenizer()
    state.question_suggestions = _load_question_suggestions()
    if state.preload_models:
        try:
            for variant in ("A1", "A2"):
                await _ensure_direction_a_model(variant)
            await _ensure_llava_bundle()
        except Exception as exc:
            print(f"[WARNING] Model preload skipped: {exc}")


@app.get("/v1/models")
def list_models() -> JSONResponse:
    payload = []
    availability = _variant_availability()
    for variant in VARIANT_ORDER:
        meta = VARIANT_META[variant]
        info = availability.get(variant, {})
        payload.append(
            {
                "name": variant,
                "family": meta["family"],
                "title": meta["title"],
                "subtitle": meta["subtitle"],
                "description": meta["description"],
                "available": bool(info.get("available")),
                "artifact": info.get("artifact", ""),
            }
        )
    return JSONResponse({"models": payload})


@app.post("/v1/predict")
async def predict(
    question: str = Form(..., description="Question for VQA"),
    model_name: Optional[str] = Form(None, description="Legacy single model name"),
    model_names: Optional[str] = Form(None, description="Comma-separated or JSON list of models"),
    image: UploadFile = File(..., description="Image input (JPEG/PNG)"),
) -> JSONResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {exc}") from exc

    selected_models = _parse_model_selection(model_name, model_names)
    results = []
    async with load_lock:
        for variant in selected_models:
            results.append(await predict_variant(variant, question, pil_img))

    predictions = {item["variant"]: item["prediction"] for item in results if item.get("status") == "ok"}
    summary = {
        "majority_vote": majority_answer(list(predictions.values())) if predictions else "",
        "success_count": sum(1 for item in results if item.get("status") == "ok"),
        "error_count": sum(1 for item in results if item.get("status", "").startswith("error")),
    }

    return JSONResponse(
        {
            "question": question,
            "selected_models": selected_models,
            "results": results,
            "summary": summary,
        }
    )


@app.get("/v1/question-suggestions")
def question_suggestions(limit: int = SUGGESTION_LIMIT) -> JSONResponse:
    suggestions = state.question_suggestions or _load_question_suggestions(limit)
    clipped = suggestions[: max(1, min(limit, len(suggestions)))] if suggestions else []
    return JSONResponse({"suggestions": clipped})


@app.get("/health")
def health() -> JSONResponse:
    availability = _variant_availability()
    return JSONResponse(
        {
            "status": "ok",
            "device": str(state.device),
            "preload_enabled": state.preload_models,
            "answer_rewrite_enabled": state.answer_rewriter.enabled,
            "answer_rewrite_model_id": state.answer_rewriter.model_id,
            "answer_rewrite_ready": state.answer_rewriter.ready,
            "suggestions_cached": len(state.question_suggestions),
            "cached": {
                "A": sorted(state.a_models.keys()),
                "B": bool(state.llava_bundle),
            },
            "models": {
                variant: {"available": availability[variant]["available"], "artifact": availability[variant]["artifact"]}
                for variant in VARIANT_ORDER
            },
        }
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend index.html not found.")
    return FileResponse(index_path)

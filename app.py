import asyncio
import gc
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from peft import PeftModel
from PIL import Image
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor

from src.engine.medical_eval import (
    _build_b1_prompt,
    _build_bad_words_ids,
    _en_to_vi_direct,
    _extract_key_medical_term,
    _normalize_closed_answer,
)
from src.models.medical_vqa_model import MedicalVQAModelA
from src.models.multimodal_vqa import MultimodalVQA
from src.utils.answer_rewriter import MedicalAnswerRewriter
from src.utils.text_utils import normalize_answer, postprocess_answer
from src.utils.translator import MedicalTranslator
from src.utils.visualization import MedicalImageTransform


os.environ.setdefault("ANSWER_REWRITE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
os.environ.setdefault("ANSWER_REWRITE_USE_4BIT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "configs" / "medical_vqa.yaml"
VARIANT_ORDER = ["A1", "A2", "B1", "B2", "DPO", "PPO", "SOUP"]
MODEL_DISPLAY_NAMES = {
    "A1": "A1 LSTM",
    "A2": "A2 Transformer",
    "B1": "B1 Zero-shot",
    "B2": "B2 Fine-tuned",
    "DPO": "DPO Alignment",
    "PPO": "PPO RL refinement",
    "SOUP": "SOUP Model Soup",
}
HF_MODEL_REPOS = {
    "A1": "SpringWang08/medical-vqa-a1",
    "A2": "SpringWang08/medical-vqa-a2",
    "B1": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    "B2": "SpringWang08/medical-vqa-b2",
    "DPO": "SpringWang08/medical-vqa-dpo",
    "PPO": "SpringWang08/medical-vqa-ppo",
    "SOUP": "SpringWang08/medical-vqa-soup",
}

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANSWER_MAX_WORDS = int(CFG["data"].get("answer_max_words", 10))
IMAGE_SIZE = int(CFG["data"].get("image_size", 224))
MAX_QUESTION_LEN = int(CFG["data"].get("max_question_len", 64))
MAX_ANSWER_LEN = int(CFG["data"].get("max_answer_len", 20))
MODEL_A_CFG = CFG.get("model_a", {})
MODEL_B_CFG = CFG.get("model_b", {})
EVAL_CFG = CFG.get("eval", {})
PHOBERT_MODEL = MODEL_A_CFG.get("phobert_model", "vinai/phobert-base")
LLAVA_MODEL_ID = MODEL_B_CFG.get("model_name", HF_MODEL_REPOS["B1"])

qa_tokenizer = None
image_transform = MedicalImageTransform(size=IMAGE_SIZE)
translator = MedicalTranslator(device=DEVICE.type)
rewriter = MedicalAnswerRewriter()
loaded_a_models: dict[str, dict[str, Any]] = {}
llava_bundle: dict[str, Any] | None = None
b_lock = asyncio.Lock()


def _ensure_qa_tokenizer():
    global qa_tokenizer
    if qa_tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        qa_tokenizer = tokenizer
    return qa_tokenizer


def _looks_closed_question(question: str) -> bool:
    normalized = normalize_answer(question)
    closed_prefixes = (
        "có ",
        "không ",
        "phải ",
        "đây có",
        "hình ảnh có",
        "ảnh có",
        "is ",
        "are ",
        "does ",
        "do ",
        "can ",
        "has ",
    )
    open_prefixes = ("what ", "where ", "when ", "who ", "which ", "how ", "why ")
    if normalized.startswith(open_prefixes):
        return False
    if normalized.startswith(closed_prefixes):
        return True
    return any(word in normalized.split() for word in {"có", "không", "normal", "abnormal"})


def _prepare_question_text(question: str) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "", ""
    # B1 benefits from English when users provide English; otherwise it still works
    # with the concise Vietnamese instruction used in the notebook.
    return question, question


def _download_direction_a_checkpoint(variant: str) -> str:
    filename = f"medical_vqa_{variant}_best.pth"
    local_path = ROOT_DIR / "checkpoints" / filename
    if local_path.exists():
        return str(local_path)
    return hf_hub_download(repo_id=HF_MODEL_REPOS[variant], filename=filename)


def _ensure_direction_a_model(variant: str) -> dict[str, Any]:
    if variant in loaded_a_models:
        return loaded_a_models[variant]

    tokenizer = _ensure_qa_tokenizer()
    ckpt_path = _download_direction_a_checkpoint(variant)
    decoder_type = "lstm" if variant == "A1" else "transformer"
    model = MedicalVQAModelA(
        decoder_type=decoder_type,
        vocab_size=len(tokenizer),
        hidden_size=int(MODEL_A_CFG.get("hidden_size", 768)),
        phobert_model=PHOBERT_MODEL,
    ).to(DEVICE)

    payload = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    bundle = {
        "variant": variant,
        "family": "A",
        "model": model,
        "tokenizer": tokenizer,
        "checkpoint": HF_MODEL_REPOS[variant],
    }
    loaded_a_models[variant] = bundle
    return bundle


def _build_llava_base_and_processor():
    if not torch.cuda.is_available():
        raise RuntimeError("B1/B2/DPO/PPO cần GPU CUDA trên Hugging Face Space.")

    wrapper = MultimodalVQA(
        model_id=LLAVA_MODEL_ID,
        lora_r=int(MODEL_B_CFG.get("lora_r", 16)),
        lora_alpha=int(MODEL_B_CFG.get("lora_alpha", 32)),
        lora_dropout=float(MODEL_B_CFG.get("lora_dropout", 0.05)),
        lora_target_modules=MODEL_B_CFG.get("lora_target_modules"),
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


def _ensure_llava_bundle() -> dict[str, Any]:
    global llava_bundle
    if llava_bundle is not None:
        return llava_bundle

    wrapper, processor, base_model = _build_llava_base_and_processor()
    adapter_variants = ["B2", "DPO", "PPO", "SOUP"]
    first_variant = adapter_variants[0]
    model = PeftModel.from_pretrained(
        base_model,
        HF_MODEL_REPOS[first_variant],
        adapter_name=first_variant,
        is_trainable=False,
    )
    for variant in adapter_variants[1:]:
        model.load_adapter(HF_MODEL_REPOS[variant], adapter_name=variant, is_trainable=False)

    model.eval()
    llava_bundle = {
        "family": "B",
        "model": model,
        "processor": processor,
        "wrapper": wrapper,
        "checkpoint": LLAVA_MODEL_ID,
        "adapter_name_map": {variant: variant for variant in adapter_variants},
    }
    return llava_bundle


def _predict_direction_a(bundle: dict[str, Any], question_vi: str, image: Image.Image) -> dict[str, str]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_tensor = image_transform(image.convert("L")).unsqueeze(0).to(DEVICE)
    inputs = tokenizer(
        question_vi,
        padding="max_length",
        truncation=True,
        max_length=MAX_QUESTION_LEN,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    is_closed = _looks_closed_question(question_vi)

    with torch.inference_mode():
        logits_closed, pred_ids = model.inference(
            image_tensor,
            input_ids,
            attention_mask,
            beam_width=int(EVAL_CFG.get("beam_width_a", 5)),
            max_len=MAX_ANSWER_LEN,
        )

    if is_closed:
        prediction_raw = "có" if logits_closed.argmax(dim=1).item() == 1 else "không"
        prediction = prediction_raw
    else:
        prediction_raw = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        prediction = postprocess_answer(prediction_raw, max_words=ANSWER_MAX_WORDS)
    return {"prediction": prediction, "prediction_raw": prediction_raw}


async def _predict_direction_b(
    bundle: dict[str, Any],
    question_vi: str,
    question_en: str,
    image: Image.Image,
    variant: str,
) -> dict[str, str]:
    model = bundle["model"]
    processor = bundle["processor"]
    wrapper = bundle["wrapper"]
    is_closed = _looks_closed_question(question_vi if variant != "B1" else question_en)
    question_for_variant = question_en if variant == "B1" else question_vi
    adapter_name = bundle.get("adapter_name_map", {}).get(variant)

    if variant == "B1":
        prompt = _build_b1_prompt(question_for_variant, ANSWER_MAX_WORDS)
        num_beams = int(EVAL_CFG.get("beam_width_b_open", 5))
        max_new_tokens = int(EVAL_CFG.get("max_new_tokens_b_open", ANSWER_MAX_WORDS + 6))
    else:
        prompt = wrapper.build_instruction_prompt(question_for_variant, language="vi", include_answer=False)
        num_beams = int(EVAL_CFG.get("beam_width_b_closed", 1)) if is_closed else int(EVAL_CFG.get("beam_width_b_open", 5))
        max_new_tokens = (
            int(EVAL_CFG.get("max_new_tokens_b_closed", 4))
            if is_closed
            else int(EVAL_CFG.get("max_new_tokens_b_open", ANSWER_MAX_WORDS + 6))
        )

    bad_words_ids = _build_bad_words_ids(processor, variant)
    inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True).to(DEVICE)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    async with b_lock:
        if adapter_name and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        if variant == "B1" and hasattr(model, "disable_adapter"):
            context = model.disable_adapter()
        else:
            context = torch.inference_mode()

        with context:
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
                prediction = translator.translate_en2vi(pred_en)
            prediction = postprocess_answer(prediction, max_words=ANSWER_MAX_WORDS)
    else:
        prediction = _normalize_closed_answer(question_vi, question_en, pred_raw) if is_closed else pred_raw
        prediction = postprocess_answer(prediction, max_words=ANSWER_MAX_WORDS)

    return {"prediction": prediction, "prediction_raw": pred_raw}


async def _predict_variant(variant: str, question: str, image: Image.Image) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        question_vi, question_en = _prepare_question_text(question)
        if variant in {"A1", "A2"}:
            bundle = _ensure_direction_a_model(variant)
            out = _predict_direction_a(bundle, question_vi, image)
        else:
            bundle = _ensure_llava_bundle()
            out = await _predict_direction_b(bundle, question_vi, question_en, image, variant)

        answer_for_rewrite = out["prediction"] or out["prediction_raw"]
        rewritten = rewriter.rewrite(
            question=question_vi,
            answer=answer_for_rewrite,
            language="vi",
            source_model=variant,
        )
        return {
            "model": variant,
            "Model": MODEL_DISPLAY_NAMES.get(variant, variant),
            "prediction": rewritten,
            "Prediction": rewritten,
            "prediction_before_rewrite": out["prediction"],
            "raw": out["prediction_raw"],
            "answer_used_for_rewrite": answer_for_rewrite,
            "checkpoint": HF_MODEL_REPOS.get(variant, ""),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model": variant,
            "Model": MODEL_DISPLAY_NAMES.get(variant, variant),
            "prediction": "",
            "Prediction": "",
            "prediction_before_rewrite": "",
            "raw": "",
            "answer_used_for_rewrite": "",
            "checkpoint": HF_MODEL_REPOS.get(variant, ""),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "status": f"error: {exc}",
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict_all(image: Image.Image, question: str, selected_models: list[str]) -> pd.DataFrame:
    if image is None:
        raise gr.Error("Vui lòng upload ảnh y khoa.")
    if not question or not question.strip():
        raise gr.Error("Vui lòng nhập câu hỏi.")
    variants = selected_models or VARIANT_ORDER

    async def _run():
        rows = []
        for variant in variants:
            rows.append(await _predict_variant(variant, question, image))
        return rows

    rows = asyncio.run(_run())
    return pd.DataFrame(rows)[["Model", "Prediction"]]


CSS = """
.gradio-container { max-width: 1180px !important; }
#run-btn { height: 44px; }
"""

with gr.Blocks(css=CSS, title="Medical VQA Compare") as demo:
    gr.Markdown("# Medical VQA Compare")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Ảnh y khoa", type="pil", image_mode="RGB", sources=["upload", "clipboard"])
            question_input = gr.Textbox(
                label="Câu hỏi",
                value="Hình ảnh này có bất thường không?",
                lines=2,
            )
            model_input = gr.CheckboxGroup(
                label="Model",
                choices=VARIANT_ORDER,
                value=VARIANT_ORDER,
            )
            run_button = gr.Button("Chạy dự đoán", variant="primary", elem_id="run-btn")
        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                label="Kết quả",
                headers=[
                    "Model",
                    "Prediction",
                ],
                wrap=True,
            )

    run_button.click(
        fn=predict_all,
        inputs=[image_input, question_input, model_input],
        outputs=output_table,
        show_progress="full",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)

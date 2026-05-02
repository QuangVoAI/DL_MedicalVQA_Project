import argparse
import html
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor

from src.data.medical_dataset import MedicalVQADataset
from src.models.medical_vqa_model import MedicalVQAModelA
from src.models.multimodal_vqa import MultimodalVQA
from src.utils.text_utils import normalize_answer, postprocess_answer
from src.utils.translator import MedicalTranslator
from src.utils.visualization import MedicalImageTransform as MedicalTransform


def vqa_collate_fn(batch):
    elem = batch[0]
    collated = {}
    for key in elem.keys():
        if key in ["image", "input_ids", "attention_mask", "label_closed", "target_ids", "chosen_ids", "rejected_ids"]:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated


def normalize_for_metric(text: str) -> str:
    return str(text).strip().lower()


def _normalize_closed_answer(question_vi: str, question_en: str, pred_vi: str, pred_en: str = "") -> str:
    question_vi_norm = normalize_answer(question_vi)
    question_en_norm = normalize_answer(question_en)
    pred_vi_norm = normalize_answer(pred_vi)
    pred_en_norm = normalize_answer(pred_en)
    combined = " ".join(part for part in [pred_vi_norm, pred_en_norm] if part).strip()

    is_normality_question = any(
        pattern in " ".join([question_vi_norm, question_en_norm])
        for pattern in ["bình thường", "normal", "abnormal", "bat thuong"]
    )

    if is_normality_question:
        if any(pattern in combined for pattern in ["không bình thường", "not normal"]):
            return "không"
        if any(pattern in combined.split() for pattern in ["có", "yes"]):
            return "có"
        if any(pattern in combined for pattern in [
            "bình thường", "normal", "no significant abnormalities", "no abnormality",
            "unremarkable", "appears to be normal", "without significant abnormalities",
            "không phát hiện bất thường",
        ]):
            return "có"
        if any(pattern in combined for pattern in [
            "bất thường", "abnormal", "abnormality detected", "fracture", "lesion",
            "mass", "effusion", "pneumothorax",
        ]):
            return "không"
    else:
        if any(pattern in combined for pattern in ["không", "no", "absent", "not seen", "negative", "none"]):
            return "không"
        if any(pattern in combined for pattern in ["có", "yes", "present", "detected", "positive"]):
            return "có"

    return pred_vi_norm or pred_en_norm


_B1_FEW_SHOT = (
    "Q: Is there cardiomegaly? A: yes\n"
    "Q: What organ is shown? A: lung\n"
    "Q: Is the aorta normal? A: no\n"
    "Q: What abnormality is present? A: pleural effusion\n"
)


def _build_b1_prompt(question_en: str, max_words: int) -> str:
    return (
        f"USER: <image>\n"
        f"Answer each question with medical terminology only, "
        f"no more than {max_words} words, no full sentences.\n"
        f"{_B1_FEW_SHOT}"
        f"Q: {question_en} A: ASSISTANT:"
    )


_EN_VI_DIRECT = {
    "yes": "có", "no": "không", "present": "có", "absent": "không",
    "normal": "bình thường", "abnormal": "bất thường", "true": "có", "false": "không",
    "positive": "có", "negative": "không", "lung": "phổi", "lungs": "phổi",
    "heart": "tim", "liver": "gan", "spleen": "lách", "kidney": "thận", "brain": "não",
    "bladder": "bàng quang", "chest": "ngực", "abdomen": "bụng", "pelvis": "xương chậu",
    "spine": "cột sống", "rib": "xương sườn", "ribs": "xương sườn", "trachea": "khí quản",
    "aorta": "động mạch chủ", "diaphragm": "cơ hoành", "mediastinum": "trung thất",
    "chest x-ray": "x-quang ngực", "x-ray": "x-quang", "xray": "x-quang", "mri": "mri",
    "ct": "ct", "ultrasound": "siêu âm", "ct scan": "ct", "mri scan": "mri",
    "axial": "mặt phẳng ngang", "coronal": "mặt phẳng vành", "sagittal": "mặt phẳng dọc",
    "transverse": "mặt phẳng ngang", "cardiomegaly": "tim to", "pneumonia": "viêm phổi",
    "pleural effusion": "tràn dịch màng phổi", "pneumothorax": "tràn khí màng phổi",
    "fracture": "gãy xương", "edema": "phù nề", "pulmonary edema": "phù phổi",
    "consolidation": "đông đặc", "atelectasis": "xẹp phổi", "opacity": "mờ đục",
    "mass": "khối u", "nodule": "nốt", "lesion": "tổn thương", "tumor": "khối u",
    "effusion": "tràn dịch", "infiltrate": "thâm nhiễm", "fibrosis": "xơ hóa",
    "calcification": "vôi hóa", "carcinoma": "ung thư", "metastasis": "di căn",
    "bilateral": "hai bên", "unilateral": "một bên", "left": "trái", "right": "phải",
    "upper": "trên", "lower": "dưới", "upper left": "phía trên bên trái", "upper right": "phía trên bên phải",
    "lower left": "phía dưới bên trái", "lower right": "phía dưới bên phải",
}


def _extract_key_medical_term(raw_en: str, max_words: int) -> str:
    import re
    text = raw_en.strip().lower()
    prefixes = [
        r"^the (image|scan|x-ray|xray|mri|ct|picture|photo|radiograph) (shows?|depicts?|demonstrates?|reveals?|indicates?|presents?)\s+",
        r"^based on the (image|scan|x-ray|mri|ct)\s*,?\s*",
        r"^in (this|the) (image|scan|x-ray|mri|ct)\s*,?\s*",
        r"^i (can see|observe|notice|see)\s+",
        r"^there (is|are)\s+(a |an |some )?",
        r"^(it |this )(shows?|is|appears?|looks?)\s+(like\s+)?",
        r"^the (patient|subject)\s+(has|shows?|presents?)\s+",
        r"^(a|an|the)\s+",
    ]
    for pat in prefixes:
        text = re.sub(pat, "", text)
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:max_words]) if words else raw_en.strip()


def _en_to_vi_direct(en_text: str):
    return _EN_VI_DIRECT.get(en_text.strip().lower())


def predict_direction_a(model, dataloader, device, tokenizer, beam_width=1, max_len=32, max_words=10):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting A"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_closed"]
            logits_closed, pred_ids = model.inference(images, input_ids, attention_mask, beam_width=beam_width, max_len=max_len)
            preds_text_raw = [postprocess_answer(t, max_words=max_words) for t in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
            preds_text = list(preds_text_raw)
            closed_map = {0: "không", 1: "có"}
            closed_preds_idx = torch.argmax(logits_closed, dim=-1)
            for i in range(len(preds_text)):
                if labels[i].item() != -1:
                    preds_text[i] = closed_map[closed_preds_idx[i].item()]
                preds_text[i] = postprocess_answer(preds_text[i], max_words=max_words)

            for i in range(len(preds_text)):
                rows.append({
                    "ground_truth": normalize_for_metric(postprocess_answer(batch["raw_answer"][i], max_words=max_words)),
                    "ground_truth_en": normalize_for_metric(batch.get("raw_answer_en", [""])[i] if "raw_answer_en" in batch else ""),
                    "predicted": normalize_for_metric(preds_text[i]),
                    "predicted_raw": normalize_for_metric(preds_text_raw[i]),
                    "predicted_display": normalize_for_metric(preds_text_raw[i]),
                    "predicted_en": "",
                })
    return rows


def predict_direction_b(model, dataloader, device, processor, variant="B1", beam_width=1, beam_width_closed=1, beam_width_open=1, max_new_tokens_closed=4, max_new_tokens_open=16, generation_batch_size=1, max_words=10):
    model.eval()
    rows = []
    translator = MedicalTranslator(device=device.type)
    wrapper = MultimodalVQA()

    def _run_generation(raw_images, prompts, sample_indices, num_beams, max_new_tokens):
        if not sample_indices:
            return []
        decoded_outputs = []
        chunk_size = generation_batch_size if num_beams > 1 else max(generation_batch_size, 2)
        for start in range(0, len(sample_indices), chunk_size):
            chunk_indices = sample_indices[start:start + chunk_size]
            text_subset = [prompts[i] for i in chunk_indices]
            image_subset = [raw_images[i] for i in chunk_indices]
            inputs = processor(text=text_subset, images=image_subset, return_tensors="pt", padding=True).to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=num_beams > 1,
            )
            input_token_len = inputs.input_ids.shape[1]
            decoded_outputs.extend(processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True))
            del inputs, output_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return decoded_outputs

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {variant}"):
            raw_images = batch["raw_image"]
            questions_vi = batch.get("raw_questions", [])
            questions_en = batch.get("raw_questions_en", [])
            refs_vi_raw = batch.get("raw_answer", [])
            refs_en_raw = batch.get("raw_answer_en", [])
            labels = batch["label_closed"]

            if variant == "B1":
                if not questions_en or any(not str(q).strip() for q in questions_en):
                    questions_en = translator.translate_vi2en(questions_vi)
                prompts = [_build_b1_prompt(q, max_words) for q in questions_en]
            else:
                prompts = [wrapper.build_instruction_prompt(q, language="vi", include_answer=False) for q in questions_vi]

            preds_raw = [""] * len(prompts)
            closed_idx = [i for i, lbl in enumerate(labels.tolist()) if lbl != -1]
            open_idx = [i for i, lbl in enumerate(labels.tolist()) if lbl == -1]

            if variant == "B1":
                preds_raw = _run_generation(raw_images, prompts, list(range(len(prompts))), beam_width_open, max_new_tokens_open)
            else:
                for idx, pred in zip(closed_idx, _run_generation(raw_images, prompts, closed_idx, beam_width_closed, max_new_tokens_closed)):
                    preds_raw[idx] = pred
                for idx, pred in zip(open_idx, _run_generation(raw_images, prompts, open_idx, beam_width_open, max_new_tokens_open)):
                    preds_raw[idx] = pred

            preds_vi = []
            preds_vi_display = []
            preds_en_clean = []

            if variant == "B1":
                preds_en_clean = [_extract_key_medical_term(p, 50) for p in preds_raw]
                needs_translate_idx = []
                needs_translate_txt = []
                for i, pred_en in enumerate(preds_en_clean):
                    if labels[i].item() != -1:
                        preds_vi.append(_normalize_closed_answer(questions_vi[i], questions_en[i], pred_en, pred_en))
                    else:
                        vi_direct = _en_to_vi_direct(pred_en)
                        if vi_direct is not None:
                            preds_vi.append(postprocess_answer(vi_direct, max_words=max_words))
                        else:
                            preds_vi.append(None)
                            needs_translate_idx.append(i)
                            needs_translate_txt.append(pred_en)
                if needs_translate_txt:
                    translated = translator.translate_en2vi(needs_translate_txt)
                    if isinstance(translated, str):
                        translated = [translated]
                    for idx, vi in zip(needs_translate_idx, translated):
                        preds_vi[idx] = postprocess_answer(vi, max_words=max_words)
                preds_vi_display = list(preds_vi)
            else:
                preds_vi_display = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_raw]
                for i, pred_vi in enumerate(preds_raw):
                    if labels[i].item() != -1:
                        preds_vi.append(_normalize_closed_answer(questions_vi[i], questions_en[i] if i < len(questions_en) else "", pred_vi))
                    else:
                        preds_vi.append(pred_vi)
                preds_en_clean = [""] * len(preds_raw)

            preds_vi = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_vi]
            preds_vi_display = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_vi_display]
            preds_vi_raw = list(preds_vi_display)
            refs_vi = [postprocess_answer(r, max_words=max_words) for r in refs_vi_raw]
            refs_en = [postprocess_answer(r, max_words=max_words) if r else "" for r in refs_en_raw]

            for i in range(len(preds_vi)):
                rows.append({
                    "ground_truth": normalize_for_metric(refs_vi[i]),
                    "ground_truth_en": normalize_for_metric(refs_en[i]),
                    "predicted": normalize_for_metric(preds_vi[i]),
                    "predicted_raw": normalize_for_metric(preds_vi_raw[i]),
                    "predicted_display": normalize_for_metric(preds_vi_display[i]),
                    "predicted_en": normalize_for_metric(preds_en_clean[i] if i < len(preds_en_clean) else ""),
                })

    return rows


def select_best_adapter_checkpoint(checkpoint_root: str):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục checkpoint: {checkpoint_root}")

    checkpoint_dirs = sorted(
        p for p in checkpoint_root.glob("checkpoint-*")
        if (p / "adapter_config.json").exists()
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"Không có adapter checkpoint trong {checkpoint_root}")

    for state_file in sorted(checkpoint_root.glob("checkpoint-*/trainer_state.json"), reverse=True):
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        best_path = state.get("best_model_checkpoint")
        if best_path:
            best_dir = Path(best_path.replace("./", ""))
            if not best_dir.is_absolute():
                best_dir = Path.cwd() / best_dir
            if (best_dir / "adapter_config.json").exists():
                return best_dir.resolve()

    return checkpoint_dirs[-1].resolve()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset_and_loader(config, split: str, tokenizer):
    hf_repo = config["data"].get("hf_dataset")
    if not hf_repo:
        raise ValueError("Script này hiện yêu cầu dataset từ Hugging Face Hub.")

    dataset_dict = load_dataset(hf_repo)
    if split not in dataset_dict:
        raise ValueError(f"Dataset không có split '{split}'. Các split hiện có: {list(dataset_dict.keys())}")

    answer_max_words = int(config["data"].get("answer_max_words", 10))
    transform = MedicalTransform(size=config["data"]["image_size"])
    dataset = MedicalVQADataset(
        hf_dataset=dataset_dict[split],
        tokenizer=tokenizer,
        transform=transform,
        max_seq_len=config["data"]["max_question_len"],
        max_ans_len=config["data"]["max_answer_len"],
        answer_max_words=answer_max_words,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("eval_batch_size", 8)),
        shuffle=False,
        collate_fn=vqa_collate_fn,
    )
    return dataset_dict[split], loader


def load_direction_a_model(variant: str, config, tokenizer, device):
    ckpt_path = Path(f"checkpoints/medical_vqa_{variant}_best.pth")
    if not ckpt_path.exists():
        resume_path = Path(f"checkpoints/medical_vqa_{variant}_resume.pth")
        ckpt_path = resume_path if resume_path.exists() else None
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint cho {variant}")

    decoder_type = "lstm" if variant == "A1" else "transformer"
    model = MedicalVQAModelA(
        decoder_type=decoder_type,
        vocab_size=len(tokenizer),
        hidden_size=config["model_a"].get("hidden_size", 768),
        phobert_model=config["model_a"].get("phobert_model", "vinai/phobert-base"),
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, str(ckpt_path)


def build_llava_base_and_processor(config):
    wrapper = MultimodalVQA(
        model_id=config["model_b"]["model_name"],
        lora_r=int(config["model_b"].get("lora_r", 16)),
        lora_alpha=int(config["model_b"].get("lora_alpha", 32)),
        lora_dropout=float(config["model_b"].get("lora_dropout", 0.05)),
        lora_target_modules=config["model_b"].get("lora_target_modules"),
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


def load_direction_b_model(variant: str, config):
    wrapper, processor, base_model = build_llava_base_and_processor(config)

    if variant == "B1":
        model = base_model
        checkpoint = config["model_b"]["model_name"]
    elif variant == "B2":
        ckpt_dir = select_best_adapter_checkpoint(config["train"].get("b2_output_dir", "./checkpoints/B2"))
        model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=False)
        checkpoint = str(ckpt_dir)
    else:
        raise ValueError(f"Variant không hỗ trợ trong script này: {variant}")

    model.eval()
    return model, processor, checkpoint


def convert_prediction_rows(hf_split, prediction_rows, variant: str, checkpoint: str):
    rows = []

    for idx, item in enumerate(hf_split):
        pred_row = prediction_rows[idx] if idx < len(prediction_rows) else {}
        rows.append({
            "idx": idx,
            "variant": variant,
            "checkpoint": checkpoint,
            "id": item.get("id"),
            "source": item.get("source"),
            "image_name": item.get("image_name"),
            "answer_type": item.get("answer_type"),
            "question": item.get("question"),
            "question_vi": item.get("question_vi"),
            "ground_truth": pred_row.get("ground_truth", ""),
            "ground_truth_en": pred_row.get("ground_truth_en", ""),
            "predicted": pred_row.get("predicted", ""),
            "predicted_raw": pred_row.get("predicted_raw", ""),
            "predicted_display": pred_row.get("predicted_display", ""),
            "predicted_en": pred_row.get("predicted_en", ""),
        })
    return rows


def build_side_by_side(hf_split, prediction_map):
    variants = list(prediction_map.keys())
    combined = []
    for idx, item in enumerate(hf_split):
        row = {
            "idx": idx,
            "id": item.get("id"),
            "source": item.get("source"),
            "image_name": item.get("image_name"),
            "answer_type": item.get("answer_type"),
            "question": item.get("question"),
            "question_vi": item.get("question_vi"),
            "ground_truth": item.get("answer_vi"),
            "ground_truth_full_vi": item.get("answer_full_vi"),
        }
        for variant in variants:
            preds = prediction_map[variant]
            row[f"{variant}_predicted"] = preds[idx]["predicted"] if idx < len(preds) else ""
            row[f"{variant}_predicted_raw"] = preds[idx]["predicted_raw"] if idx < len(preds) else ""
        combined.append(row)
    return combined


def export_preview_images(hf_split, output_dir: Path, split: str, image_size: int = 256):
    image_dir = output_dir / f"{split}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_refs = []

    for idx, item in enumerate(hf_split):
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        preview = image.copy()
        preview.thumbnail((image_size, image_size))
        image_name = Path(str(item.get("image_name") or f"{idx}.jpg")).name
        save_name = f"{idx:04d}_{image_name}"
        save_path = image_dir / save_name
        preview.save(save_path, format="JPEG", quality=90)
        image_refs.append(save_path.relative_to(output_dir).as_posix())

    return image_refs


def render_compare_html(compare_rows, variants, output_dir: Path, split: str):
    html_path = output_dir / f"compare_{split}_{'_'.join(variants)}.html"
    cards = []

    for row in compare_rows:
        img_src = html.escape(row.get("image_preview", ""))
        question_vi = html.escape(str(row.get("question_vi", "")))
        question_en = html.escape(str(row.get("question", "")))
        answer_type = html.escape(str(row.get("answer_type", "")))
        ground_truth = html.escape(str(row.get("ground_truth", "")))
        image_name = html.escape(str(row.get("image_name", "")))
        preds_html = []
        for variant in variants:
            pred = html.escape(str(row.get(f"{variant}_predicted", "")))
            raw = html.escape(str(row.get(f"{variant}_predicted_raw", "")))
            preds_html.append(
                f"""
                <div class="pred">
                  <div class="pred-title">{variant}</div>
                  <div><strong>Pred:</strong> {pred}</div>
                  <div class="muted"><strong>Raw:</strong> {raw}</div>
                </div>
                """
            )

        cards.append(
            f"""
            <article class="card">
              <div class="media">
                <img src="{img_src}" alt="{image_name}" loading="lazy" />
                <div class="meta">
                  <div><strong>Idx:</strong> {row.get("idx", "")}</div>
                  <div><strong>Image:</strong> {image_name}</div>
                  <div><strong>Type:</strong> {answer_type}</div>
                </div>
              </div>
              <div class="content">
                <div><strong>Q (VI):</strong> {question_vi}</div>
                <div class="muted"><strong>Q (EN):</strong> {question_en}</div>
                <div class="gt"><strong>GT:</strong> {ground_truth}</div>
                <div class="pred-grid">
                  {''.join(preds_html)}
                </div>
              </div>
            </article>
            """
        )

    page = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Compare Predictions - {split}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1d1b16;
      --muted: #6e675c;
      --line: #d8cfbf;
      --accent: #8f3d2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #efe7d7 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 24px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 30px rgba(40, 28, 12, 0.06);
    }}
    .media img {{
      width: 100%;
      border-radius: 12px;
      display: block;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .meta {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .content {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      line-height: 1.5;
    }}
    .muted {{
      color: var(--muted);
    }}
    .gt {{
      padding: 10px 12px;
      background: #f6efe4;
      border-left: 4px solid var(--accent);
      border-radius: 8px;
    }}
    .pred-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 8px;
    }}
    .pred {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }}
    .pred-title {{
      font-weight: 700;
      margin-bottom: 6px;
      color: var(--accent);
    }}
    @media (max-width: 820px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
      .pred-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>So sánh prediction {html.escape(split)}</h1>
    <div class="sub">Models: {html.escape(', '.join(variants))}</div>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Xuất prediction của A1/A2/B1/B2 để so sánh.")
    parser.add_argument("--config", default="configs/medical_vqa.yaml")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--variants", nargs="+", default=["A1", "A2", "B1", "B2"])
    parser.add_argument("--output-dir", default="results/predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_a"]["phobert_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    hf_split, dataloader = build_dataset_and_loader(config, args.split, tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_refs = export_preview_images(hf_split, output_dir, args.split)

    summary = {}
    prediction_map = {}

    for variant in args.variants:
        print(f"[INFO] Đang chạy prediction cho {variant} trên split '{args.split}'...")
        if variant in {"A1", "A2"}:
            model, checkpoint = load_direction_a_model(variant, config, tokenizer, device)
            prediction_rows = predict_direction_a(
                model,
                dataloader,
                device,
                tokenizer,
                beam_width=int(config["eval"].get("beam_width_a", 5)),
                max_len=int(config["data"].get("max_answer_len", 20)),
                max_words=int(config["data"].get("answer_max_words", 10)),
            )
        else:
            model, processor, checkpoint = load_direction_b_model(variant, config)
            prediction_rows = predict_direction_b(
                model,
                dataloader,
                device,
                processor,
                beam_width=int(config["eval"].get("beam_width_b", 5)),
                beam_width_closed=int(config["eval"].get("beam_width_b_closed", 1)),
                beam_width_open=int(config["eval"].get("beam_width_b_open", config["eval"].get("beam_width_b", 5))),
                max_new_tokens_closed=int(config["eval"].get("max_new_tokens_b_closed", 4)),
                max_new_tokens_open=int(config["eval"].get("max_new_tokens_b_open", int(config["data"].get("answer_max_words", 10)) + 6)),
                generation_batch_size=int(config["eval"].get("generation_batch_size_b", 1)),
                max_words=int(config["data"].get("answer_max_words", 10)),
                variant=variant,
            )

        rows = convert_prediction_rows(hf_split, prediction_rows, variant, checkpoint)
        prediction_map[variant] = rows
        out_path = output_dir / f"{variant}_{args.split}_predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        summary[variant] = {
            "checkpoint": checkpoint,
            "num_predictions": len(rows),
        }
        print(f"[SUCCESS] Đã lưu {out_path}")

        del model
        if variant in {"B1", "B2"}:
            del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compare_rows = build_side_by_side(hf_split, prediction_map)
    for idx, row in enumerate(compare_rows):
        row["image_preview"] = image_refs[idx] if idx < len(image_refs) else ""
    compare_path = output_dir / f"compare_{args.split}_{'_'.join(args.variants)}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_rows, f, ensure_ascii=False, indent=2)

    summary_path = output_dir / f"summary_{args.split}_{'_'.join(args.variants)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    html_path = render_compare_html(compare_rows, args.variants, output_dir, args.split)

    print(f"[SUCCESS] Đã lưu file so sánh tại {compare_path}")
    print(f"[SUCCESS] Đã lưu summary tại {summary_path}")
    print(f"[SUCCESS] Đã lưu HTML hiển thị ảnh tại {html_path}")


if __name__ == "__main__":
    main()

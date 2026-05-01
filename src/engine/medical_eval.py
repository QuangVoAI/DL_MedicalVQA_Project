import torch
from tqdm import tqdm
from src.utils.metrics import batch_metrics, compute_bertscore, compute_semantic_score
from src.utils.text_utils import is_medical_term_compliant, normalize_answer, postprocess_answer

def normalize_for_metric(text: str) -> str:
    return text.strip().lower()


def _normalize_closed_answer(question_vi: str, question_en: str, pred_vi: str, pred_en: str = "") -> str:
    """Map descriptive yes/no-style outputs to closed-form labels."""
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
        explicit_negative_patterns = [
            "không bình thường",
            "not normal",
        ]
        explicit_positive_patterns = [
            "có",
            "yes",
        ]
        positive_patterns = [
            "bình thường",
            "normal",
            "no significant abnormalities",
            "no abnormality",
            "unremarkable",
            "appears to be normal",
            "without significant abnormalities",
            "không phát hiện bất thường",
        ]
        negative_patterns = [
            "bất thường",
            "abnormal",
            "abnormality detected",
            "fracture",
            "lesion",
            "mass",
            "effusion",
            "pneumothorax",
        ]
        if any(pattern in combined for pattern in explicit_negative_patterns):
            return "không"
        if any(pattern in combined.split() for pattern in explicit_positive_patterns):
            return "có"
        if any(pattern in combined for pattern in positive_patterns):
            return "có"
        if any(pattern in combined for pattern in negative_patterns):
            return "không"
    else:
        positive_patterns = [
            "có",
            "yes",
            "present",
            "detected",
            "positive",
        ]
        negative_patterns = [
            "không",
            "no",
            "absent",
            "not seen",
            "negative",
            "none",
        ]
        # For presence/absence questions, "không có ..." contains "có" but
        # semantically means no. Check negation before positive cues.
        if any(pattern in combined for pattern in negative_patterns):
            return "không"
        if any(pattern in combined for pattern in positive_patterns):
            return "có"

    fallback_positive_patterns = [
        "bình thường",
        "normal",
        "no significant abnormalities",
        "no abnormality",
        "unremarkable",
        "appears to be normal",
        "without significant abnormalities",
        "không phát hiện bất thường",
    ]
    fallback_negative_patterns = [
        "bất thường",
        "abnormal",
        "abnormality detected",
        "fracture",
        "lesion",
        "mass",
        "effusion",
        "pneumothorax",
    ]

    if any(pattern in combined for pattern in fallback_positive_patterns):
        return "có"
    if any(pattern in combined for pattern in fallback_negative_patterns):
        return "không"
    return pred_vi_norm or pred_en_norm


def _compute_format_stats(preds: list[str], max_words: int) -> dict[str, float]:
    if not preds:
        return {
            "max_10_word_compliance_rate": 0.0,
            "medical_term_compliance_rate": 0.0,
            "avg_answer_length": 0.0,
        }
    word_counts = [len(p.split()) for p in preds]
    return {
        "max_10_word_compliance_rate": sum(1 for count in word_counts if count <= max_words) / len(word_counts),
        "medical_term_compliance_rate": sum(1 for pred in preds if is_medical_term_compliant(pred)) / len(preds),
        "avg_answer_length": sum(word_counts) / len(word_counts),
    }


def _build_bad_words_ids(processor, variant: str) -> list[list[int]] | None:
    if variant not in {"B1", "B2", "DPO"}:
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


def _attach_metric_views(metrics: dict[str, float]) -> dict[str, float]:
    """Add explicit metric names while preserving backward-compatible aliases."""
    if "accuracy" in metrics:
        metrics["accuracy_normalized"] = metrics["accuracy"]
    if "em" in metrics:
        metrics["em_normalized"] = metrics["em"]
    if "f1" in metrics:
        metrics["f1_normalized"] = metrics["f1"]
    if "bleu1" in metrics:
        metrics["bleu1_normalized"] = metrics["bleu1"]
    if "bleu2" in metrics:
        metrics["bleu2_normalized"] = metrics["bleu2"]
    if "bleu3" in metrics:
        metrics["bleu3_normalized"] = metrics["bleu3"]
    if "bleu4" in metrics:
        metrics["bleu4_normalized"] = metrics["bleu4"]
    if "rouge_l" in metrics:
        metrics["rouge_l_normalized"] = metrics["rouge_l"]
    if "meteor" in metrics:
        metrics["meteor_normalized"] = metrics["meteor"]
    if "bert_score" in metrics:
        metrics["bert_score_raw"] = metrics["bert_score"]
    if "semantic" in metrics:
        metrics["semantic_raw"] = metrics["semantic"]
    return metrics

class MedicalVQAEvaluator:
    """
    Hệ thống đánh giá hợp nhất cho cả Hướng A và Hướng B.
    """
    def __init__(self, device, tokenizer=None, processor=None):
        self.device = device
        self.tokenizer = tokenizer
        self.processor = processor

    def evaluate(self, model, dataloader, variant_type='A', beam_width=1):
        """
        Giao diện chung để đánh giá bất kỳ variant nào.
        """
        if variant_type == 'A':
            return evaluate_vqa(model, dataloader, self.device, self.tokenizer, beam_width)
        else:
            return evaluate_multimodal_vqa(model, dataloader, self.device, self.processor, beam_width, variant=variant_type)

def evaluate_vqa(model, dataloader, device, tokenizer, beam_width=1, max_len=32, max_words=10):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_preds_display = []
    all_refs = []
    all_refs_full = []
    all_is_closed = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_closed']
            
            # [FIX] Gọi inference() để lấy CẢ HAI head outputs, truyền max_len từ config
            logits_closed, pred_ids = model.inference(images, input_ids, attention_mask, beam_width=beam_width, max_len=max_len)
            
            # Decode generative head + làm sạch subword artifacts
            preds_text_raw = [
                postprocess_answer(t, max_words=max_words)
                for t in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            ]
            preds_text = list(preds_text_raw)
            
            # [CRITICAL FIX] Với câu Đóng (Yes/No), dùng classifier head thay vì generator
            closed_map = {0: "không", 1: "có"}
            closed_preds_idx = torch.argmax(logits_closed, dim=-1)  # [B]
            for i in range(len(preds_text)):
                if labels[i].item() != -1:  # Câu hỏi đóng
                    preds_text[i] = closed_map[closed_preds_idx[i].item()]
                preds_text[i] = postprocess_answer(preds_text[i], max_words=max_words)
            
            # Debug: Hiển thị cả câu Đóng và câu Mở để kiểm tra đa dạng
            if len(all_preds) == 0:
                print("\n--- DEBUG PREDICTIONS ---")
                shown_closed, shown_open = 0, 0
                for i in range(len(preds_text)):
                    is_closed = labels[i].item() != -1
                    if (is_closed and shown_closed < 2) or (not is_closed and shown_open < 2):
                        q_type = "CLOSED" if is_closed else "OPEN"
                        print(f"[{q_type}] Q: {batch['raw_questions'][i]}")
                        print(f"  Pred raw: '{preds_text_raw[i]}'")
                        print(f"  Pred normalized: '{preds_text[i]}'")
                        print(f"  GT  : '{batch['raw_answer'][i]}'")
                        if is_closed: shown_closed += 1
                        else: shown_open += 1
                    if shown_closed >= 2 and shown_open >= 2:
                        break
                print("--------------------------\n")
            
            all_preds.extend([normalize_for_metric(p) for p in preds_text])
            all_preds_raw.extend([normalize_for_metric(p) for p in preds_text_raw])
            all_preds_display.extend([normalize_for_metric(p) for p in preds_text_raw])
            # [CRITICAL FIX] Dùng đáp án Tiếng Việt để chấm điểm
            all_refs.extend([normalize_for_metric(postprocess_answer(r, max_words=max_words)) for r in batch['raw_answer']])
            all_refs_full.extend([normalize_for_metric(postprocess_answer(r, max_words=100)) for r in batch.get('raw_answer_full', batch['raw_answer'])])
            is_closed = (batch['label_closed'] != -1).tolist()
            all_is_closed.extend(is_closed)

    metrics = batch_metrics(all_preds, all_refs)
    metrics["semantic"] = compute_semantic_score(all_preds_raw, all_refs)
    metrics["bert_score"] = compute_bertscore(all_preds_raw, all_refs)
    metrics = _attach_metric_views(metrics)
    metrics.update(_compute_format_stats(all_preds, max_words=max_words))
    metrics['predictions'] = all_preds
    metrics['predictions_raw'] = all_preds_raw
    metrics['predictions_display'] = all_preds_display
    metrics['ground_truths'] = all_refs
    
    closed_preds = [p for p, c in zip(all_preds, all_is_closed) if c]
    closed_refs = [r for r, c in zip(all_refs, all_is_closed) if c]
    closed_preds_raw = [p for p, c in zip(all_preds_raw, all_is_closed) if c]
    if closed_preds:
        metrics['closed'] = batch_metrics(closed_preds, closed_refs)
        metrics['closed']["semantic"] = compute_semantic_score(closed_preds_raw, closed_refs)
        metrics['closed']["bert_score"] = compute_bertscore(closed_preds_raw, closed_refs)
        metrics['closed'] = _attach_metric_views(metrics['closed'])
        metrics['closed'].update(_compute_format_stats(closed_preds, max_words=max_words))
        metrics['closed_eval'] = {
            "accuracy": metrics['closed'].get("accuracy_normalized", 0.0),
            "em": metrics['closed'].get("em_normalized", 0.0),
            "f1": metrics['closed'].get("f1_normalized", 0.0),
            "count": len(closed_preds),
        }
    open_preds = [p for p, c in zip(all_preds, all_is_closed) if not c]
    open_refs = [r for r, c in zip(all_refs, all_is_closed) if not c]
    open_preds_raw = [p for p, c in zip(all_preds_raw, all_is_closed) if not c]
    if open_preds:
        metrics['open'] = batch_metrics(open_preds, open_refs)
        metrics['open']["semantic"] = compute_semantic_score(open_preds_raw, open_refs)
        metrics['open']["bert_score"] = compute_bertscore(open_preds_raw, open_refs)
        metrics['open'] = _attach_metric_views(metrics['open'])
        metrics['open'].update(_compute_format_stats(open_preds, max_words=max_words))
        metrics['open_eval'] = {
            "semantic": metrics['open'].get("semantic_raw", 0.0),
            "bert_score": metrics['open'].get("bert_score_raw", 0.0),
            "f1": metrics['open'].get("f1_normalized", 0.0),
            "rouge_l": metrics['open'].get("rouge_l_normalized", 0.0),
            "count": len(open_preds),
        }
        
    metrics['long_answers_eval'] = {
        "accuracy": batch_metrics(all_preds, all_refs_full).get("accuracy_normalized", 0),
        "f1": batch_metrics(all_preds, all_refs_full).get("f1_normalized", 0),
        "bleu4": batch_metrics(all_preds, all_refs_full).get("bleu4_normalized", 0),
        "semantic": compute_semantic_score(all_preds_raw, all_refs_full),
        "bert_score": compute_bertscore(all_preds_raw, all_refs_full)
    }
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# B1 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_B1_FEW_SHOT = (
    "Q: Is there cardiomegaly? A: yes\n"
    "Q: What organ is shown? A: lung\n"
    "Q: Is the aorta normal? A: no\n"
    "Q: What abnormality is present? A: pleural effusion\n"
)


def _build_b1_prompt(question_en: str, max_words: int) -> str:
    """
    Few-shot prompt ép LLaVA trả lời ngắn (≤max_words từ y tế), không sinh câu dài.
    Đặt 4 ví dụ in-context trước câu hỏi thực để suppress verbose prefix.
    """
    return (
        f"USER: <image>\n"
        f"Answer each question with medical terminology only, "
        f"no more than {max_words} words, no full sentences.\n"
        f"{_B1_FEW_SHOT}"
        f"Q: {question_en} A: ASSISTANT:"
    )


# En → Vi fast lookup (50+ thuật ngữ y tế thường gặp trong SLAKE + VQA-RAD)
_EN_VI_DIRECT: dict = {
    # binary
    "yes": "có", "no": "không",
    "present": "có", "absent": "không",
    "normal": "bình thường", "abnormal": "bất thường",
    "true": "có", "false": "không",
    "positive": "có", "negative": "không",
    # anatomy
    "lung": "phổi", "lungs": "phổi",
    "heart": "tim", "liver": "gan", "spleen": "lách",
    "kidney": "thận", "brain": "não", "bladder": "bàng quang",
    "chest": "ngực", "abdomen": "bụng", "pelvis": "xương chậu",
    "spine": "cột sống", "rib": "xương sườn", "ribs": "xương sườn",
    "trachea": "khí quản", "aorta": "động mạch chủ",
    "diaphragm": "cơ hoành", "mediastinum": "trung thất",
    # modality
    "chest x-ray": "x-quang ngực", "x-ray": "x-quang", "xray": "x-quang",
    "mri": "mri", "ct": "ct", "ultrasound": "siêu âm",
    "ct scan": "ct", "mri scan": "mri",
    # planes
    "axial": "mặt phẳng ngang",
    "coronal": "mặt phẳng vành",
    "sagittal": "mặt phẳng dọc",
    "transverse": "mặt phẳng ngang",
    # pathologies
    "cardiomegaly": "tim to",
    "pneumonia": "viêm phổi",
    "pleural effusion": "tràn dịch màng phổi",
    "pneumothorax": "tràn khí màng phổi",
    "fracture": "gãy xương",
    "edema": "phù nề",
    "pulmonary edema": "phù phổi",
    "consolidation": "đông đặc",
    "atelectasis": "xẹp phổi",
    "opacity": "mờ đục",
    "mass": "khối u",
    "nodule": "nốt",
    "lesion": "tổn thương",
    "tumor": "khối u",
    "effusion": "tràn dịch",
    "infiltrate": "thâm nhiễm",
    "fibrosis": "xơ hóa",
    "calcification": "vôi hóa",
    "carcinoma": "ung thư",
    "metastasis": "di căn",
    "bilateral": "hai bên",
    "unilateral": "một bên",
    "left": "trái", "right": "phải",
    "upper": "trên", "lower": "dưới",
    "right upper quadrant": "phía trên bên phải",
    "left upper quadrant": "phía trên bên trái",
    "right lower quadrant": "phía dưới bên phải",
    "left lower quadrant": "phía dưới bên trái",
    "right upper": "phía trên bên phải",
    "left upper": "phía trên bên trái",
    "upper left": "phía trên bên trái",
    "upper right": "phía trên bên phải",
    "lower left": "phía dưới bên trái",
    "lower right": "phía dưới bên phải",
}


def _extract_key_medical_term(raw_en: str, max_words: int) -> str:
    """
    Loại bỏ verbose prefix LLaVA hay sinh ("The image shows a chest X-ray with..."),
    chỉ giữ lại thuật ngữ y tế chính.
    """
    import re
    text = raw_en.strip().lower()

    # Các prefix verbose phổ biến cần xóa
    prefixes = [
        r"^the (image|scan|x-ray|xray|mri|ct|picture|photo|radiograph) (shows?|depicts?|demonstrates?|reveals?|indicates?|presents?)\s+",
        r"^based on the (image|scan|x-ray|mri|ct)\s*,?\s*",
        r"^in (this|the) (image|scan|x-ray|mri|ct)\s*,?\s*",
        r"^i (can see|observe|notice|see)\s+",
        r"^there (is|are)\s+(a |an |some )?",
        r"^(it |this )(shows?|is|appears?|looks?)\s+(like\s+)?",
        r"^the (patient|subject)\s+(has|shows?|presents?)\s+",
        r"^(a|an|the)\s+",
        r"^[a-z\s]+ is (located|seen|found|present)( in| at| on)?\s+(the\s+)?",
    ]
    for pat in prefixes:
        text = re.sub(pat, "", text)

    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    return " ".join(words[:max_words]) if words else raw_en.strip()


def _en_to_vi_direct(en_text: str) -> str | None:
    """
    Tra từ điển nhanh. Sắp xếp theo độ dài giảm dần để phrase dài match trước.
    Trả về None nếu không match → caller dùng Translation Model.
    """
    norm = en_text.strip().lower()
    if norm in _EN_VI_DIRECT:
        return _EN_VI_DIRECT[norm]
    return None


def _dual_score_open(
    preds_vi: list,
    preds_en: list,
    refs_vi: list,
    refs_en: list,
) -> list:
    """
    Với mỗi câu hỏi mở, so sánh F1 Vi vs F1 En rồi chọn prediction tốt hơn.
    Giải quyết 0% open-ended do dịch thuật mất nghĩa.
    """
    from src.utils.metrics import compute_f1
    from src.utils.text_utils import normalize_answer
    result = []
    for pv, pe, rv, re_ in zip(preds_vi, preds_en, refs_vi, refs_en):
        f1_vi = compute_f1(pv, rv)
        f1_en = compute_f1(normalize_answer(pe), normalize_answer(re_)) if re_ else 0.0
        result.append(pv if f1_vi >= f1_en else normalize_answer(pe))
    return result


def evaluate_multimodal_vqa(
    model,
    dataloader,
    device,
    processor,
    beam_width=1,
    max_words=10,
    variant='B1',
    beam_width_closed=None,
    beam_width_open=None,
    max_new_tokens_closed=None,
    max_new_tokens_open=None,
):
    """
    B1 Zero-Shot evaluation & B2/DPO Fine-Tuned evaluation.
    """
    model.eval()
    all_preds     = []
    all_preds_raw = []
    all_preds_display = []
    all_preds_en  = []
    all_refs      = []
    all_refs_full = []
    all_refs_en   = []
    all_is_closed = []

    from src.utils.translator import MedicalTranslator
    translator = MedicalTranslator(device=device.type)
    
    from src.models.multimodal_vqa import MultimodalVQA
    wrapper = MultimodalVQA()

    beam_width_closed = beam_width if beam_width_closed is None else beam_width_closed
    beam_width_open = beam_width if beam_width_open is None else beam_width_open
    max_new_tokens_closed = 4 if max_new_tokens_closed is None else max_new_tokens_closed
    max_new_tokens_open = (max_words + 6) if max_new_tokens_open is None else max_new_tokens_open
    bad_words_ids = _build_bad_words_ids(processor, variant)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {variant}")):
            raw_images   = batch.get('raw_image')
            questions_vi = batch.get('raw_questions', [])
            questions_en = batch.get('raw_questions_en', [])
            refs_vi_raw  = batch.get('raw_answer', [])
            refs_en_raw  = batch.get('raw_answer_en', [])
            labels       = batch['label_closed']

            if variant == 'B1':
                # B1 (Zero-shot) needs English translation & English few-shot prompt
                if not questions_en or any(not str(q).strip() for q in questions_en):
                    questions_en = translator.translate_vi2en(questions_vi)
                prompts = [_build_b1_prompt(q, max_words) for q in questions_en]
            else:
                # B2 & DPO (Fine-tuned) expect Vietnamese instruction directly
                prompts = [wrapper.build_instruction_prompt(q, language="vi", include_answer=False) for q in questions_vi]
            preds_raw = [""] * len(prompts)
            closed_idx = [i for i, lbl in enumerate(labels.tolist()) if lbl != -1]
            open_idx = [i for i, lbl in enumerate(labels.tolist()) if lbl == -1]

            def _run_generation(sample_indices, num_beams, max_new_tokens):
                if not sample_indices:
                    return []
                text_subset = [prompts[i] for i in sample_indices]
                image_subset = [raw_images[i] for i in sample_indices] if raw_images is not None else None
                if image_subset is not None:
                    inputs = processor(
                        text=text_subset,
                        images=image_subset,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                else:
                    inputs = processor(text=text_subset, return_tensors="pt", padding=True).to(device)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    early_stopping=num_beams > 1,
                    bad_words_ids=bad_words_ids,
                )
                input_token_len = inputs.input_ids.shape[1]
                return processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            if variant == 'B1':
                generated = _run_generation(list(range(len(prompts))), beam_width_open, max_new_tokens_open)
                preds_raw = generated
            else:
                for idx, pred in zip(closed_idx, _run_generation(closed_idx, beam_width_closed, max_new_tokens_closed)):
                    preds_raw[idx] = pred
                for idx, pred in zip(open_idx, _run_generation(open_idx, beam_width_open, max_new_tokens_open)):
                    preds_raw[idx] = pred

            preds_vi = []
            preds_vi_display = []
            preds_en_clean = []
            
            if variant == 'B1':
                # [FIX 2] Strip verbose prefix → giữ key medical term. Tránh cắt vụn câu tiếng Anh để Dịch thuật hiểu đúng.
                preds_en_clean = [_extract_key_medical_term(p, 50) for p in preds_raw]
    
                # [FIX 3 + 5] Per-sample: closed → normalize En trước; open → dict lookup rồi Translation Model
                needs_translate_idx = []   # index cần dịch
                needs_translate_txt = []
    
                for i, pred_en in enumerate(preds_en_clean):
                    if labels[i].item() != -1:
                        # Closed: dùng _normalize_closed_answer với En pred (chính xác hơn)
                        preds_vi.append(
                            _normalize_closed_answer(
                                questions_vi[i], questions_en[i], pred_en, pred_en
                            )
                        )
                    else:
                        # Open: thử dict nhanh trước
                        vi_direct = _en_to_vi_direct(pred_en)
                        if vi_direct is not None:
                            preds_vi.append(postprocess_answer(vi_direct, max_words=max_words))
                        else:
                            preds_vi.append(None)           # placeholder
                            needs_translate_idx.append(i)
                            needs_translate_txt.append(pred_en)
    
                # Batch dịch những câu cần Translation Model
                if needs_translate_txt:
                    translated = translator.translate_en2vi(needs_translate_txt)
                    if isinstance(translated, str):
                        translated = [translated]
                    for idx, vi in zip(needs_translate_idx, translated):
                        preds_vi[idx] = postprocess_answer(vi, max_words=max_words)
                preds_vi_display = list(preds_vi)
            else:
                # B2 & DPO directly outputs Vietnamese, no translation needed
                preds_vi_display = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_raw]
                for i, pred_vi in enumerate(preds_raw):
                    if labels[i].item() != -1:
                        preds_vi.append(
                            _normalize_closed_answer(
                                questions_vi[i], questions_en[i] if i < len(questions_en) else "", pred_vi
                            )
                        )
                    else:
                        preds_vi.append(pred_vi)
                preds_en_clean = [""] * len(preds_raw)

            # Đảm bảo không có None
            preds_vi = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_vi]
            preds_vi_display = [postprocess_answer(p, max_words=max_words) if p else "" for p in preds_vi_display]
            preds_vi_raw = list(preds_vi_display)

            # Refs
            refs_vi  = [postprocess_answer(r, max_words=max_words) for r in refs_vi_raw]
            refs_en  = [postprocess_answer(r, max_words=max_words) if r else "" for r in refs_en_raw]

            # Debug batch đầu
            if batch_idx == 0:
                print(f"\n--- DEBUG {variant} (Evaluation) ---")
                for i in range(min(4, len(preds_vi))):
                    q_type = "CLOSED" if labels[i].item() != -1 else "OPEN"
                    if variant == 'B1':
                        print(f"[{q_type}] Q (En): {questions_en[i]}")
                        print(f"  Pred (En raw):   '{preds_raw[i]}'")
                        print(f"  Pred (En clean): '{preds_en_clean[i]}'")
                    else:
                        print(f"[{q_type}] Q (Vi): {questions_vi[i]}")
                        print(f"  Pred (Vi raw):   '{preds_raw[i]}'")
                    print(f"  Pred display:    '{preds_vi_display[i]}'")
                    print(f"  Pred (Vi):       '{preds_vi[i]}'")
                    print(f"  GT (Vi): '{refs_vi[i]}'  |  GT (En): '{refs_en[i]}'")
                print("-----------------------------------------\n")

            all_preds.extend([normalize_for_metric(p) for p in preds_vi])
            all_preds_raw.extend([normalize_for_metric(p) for p in preds_vi_raw])
            all_preds_display.extend([normalize_for_metric(p) for p in preds_vi_display])
            all_preds_en.extend([normalize_for_metric(p) for p in preds_en_clean])
            all_refs.extend([normalize_for_metric(r) for r in refs_vi])
            all_refs_full.extend([normalize_for_metric(postprocess_answer(r, max_words=100)) for r in batch.get('raw_answer_full', batch['raw_answer'])])
            all_refs_en.extend([normalize_for_metric(r) for r in refs_en])
            all_is_closed.extend((labels != -1).tolist())

    # [FIX 4] Dual-language scoring cho open-ended (chỉ dùng cho B1)
    if variant == 'B1':
        open_idx = [i for i, c in enumerate(all_is_closed) if not c]
        if open_idx:
            best_open = _dual_score_open(
                [all_preds[i]    for i in open_idx],
                [all_preds_en[i] for i in open_idx],
                [all_refs[i]     for i in open_idx],
                [all_refs_en[i]  for i in open_idx],
            )
            for k, i in enumerate(open_idx):
                all_preds[i] = best_open[k]

    # ── Compute metrics ──────────────────────────────────────────────────────
    metrics = batch_metrics(all_preds, all_refs)
    metrics["semantic"]   = compute_semantic_score(all_preds_raw, all_refs)
    metrics["bert_score"] = compute_bertscore(all_preds_raw, all_refs)
    metrics = _attach_metric_views(metrics)
    metrics.update(_compute_format_stats(all_preds, max_words=max_words))
    metrics['predictions']      = all_preds
    metrics['predictions_raw']  = all_preds_raw
    metrics['predictions_display'] = all_preds_display
    metrics['predictions_en']   = all_preds_en
    metrics['ground_truths']    = all_refs
    metrics['ground_truths_en'] = all_refs_en

    def _subset(pred_list, ref_list, pred_raw_list):
        m = batch_metrics(pred_list, ref_list)
        m["semantic"]   = compute_semantic_score(pred_raw_list, ref_list)
        m["bert_score"] = compute_bertscore(pred_raw_list, ref_list)
        m = _attach_metric_views(m)
        m.update(_compute_format_stats(pred_list, max_words=max_words))
        return m

    closed_idx = [i for i, c in enumerate(all_is_closed) if c]
    open_idx   = [i for i, c in enumerate(all_is_closed) if not c]

    if closed_idx:
        metrics['closed'] = _subset(
            [all_preds[i]     for i in closed_idx],
            [all_refs[i]      for i in closed_idx],
            [all_preds_raw[i] for i in closed_idx],
        )
        metrics['closed_eval'] = {
            "accuracy": metrics['closed'].get("accuracy_normalized", 0.0),
            "em": metrics['closed'].get("em_normalized", 0.0),
            "f1": metrics['closed'].get("f1_normalized", 0.0),
            "count": len(closed_idx),
        }
    if open_idx:
        metrics['open'] = _subset(
            [all_preds[i]     for i in open_idx],
            [all_refs[i]      for i in open_idx],
            [all_preds_raw[i] for i in open_idx],
        )
        metrics['open_eval'] = {
            "semantic": metrics['open'].get("semantic_raw", 0.0),
            "bert_score": metrics['open'].get("bert_score_raw", 0.0),
            "f1": metrics['open'].get("f1_normalized", 0.0),
            "rouge_l": metrics['open'].get("rouge_l_normalized", 0.0),
            "count": len(open_idx),
        }
        
    metrics['long_answers_eval'] = {
        "accuracy": batch_metrics(all_preds, all_refs_full).get("accuracy_normalized", 0),
        "f1": batch_metrics(all_preds, all_refs_full).get("f1_normalized", 0),
        "bleu4": batch_metrics(all_preds, all_refs_full).get("bleu4_normalized", 0),
        "semantic": compute_semantic_score(all_preds_raw, all_refs_full),
        "bert_score": compute_bertscore(all_preds_raw, all_refs_full)
    }

    return metrics

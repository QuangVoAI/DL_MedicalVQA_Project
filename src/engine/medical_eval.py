import torch
from tqdm import tqdm
from src.utils.metrics import batch_metrics
from src.utils.text_utils import is_medical_term_compliant, postprocess_answer

def normalize_for_metric(text: str) -> str:
    return text.strip().lower()


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
            return evaluate_multimodal_vqa(model, dataloader, self.device, self.processor, beam_width)

def evaluate_vqa(model, dataloader, device, tokenizer, beam_width=1, max_len=32, max_words=10):
    model.eval()
    all_preds = []
    all_refs = []
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
            preds_text = [
                postprocess_answer(t, max_words=max_words)
                for t in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            ]
            
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
                        print(f"  Pred: '{preds_text[i]}'")
                        print(f"  GT  : '{batch['raw_answer'][i]}'")
                        if is_closed: shown_closed += 1
                        else: shown_open += 1
                    if shown_closed >= 2 and shown_open >= 2:
                        break
                print("--------------------------\n")
            
            all_preds.extend([normalize_for_metric(p) for p in preds_text])
            # [CRITICAL FIX] Dùng đáp án Tiếng Việt để chấm điểm
            all_refs.extend([normalize_for_metric(postprocess_answer(r, max_words=max_words)) for r in batch['raw_answer']])
            is_closed = (batch['label_closed'] != -1).tolist()
            all_is_closed.extend(is_closed)

    metrics = batch_metrics(all_preds, all_refs)
    metrics.update(_compute_format_stats(all_preds, max_words=max_words))
    metrics['predictions'] = all_preds
    metrics['ground_truths'] = all_refs
    
    closed_preds = [p for p, c in zip(all_preds, all_is_closed) if c]
    closed_refs = [r for r, c in zip(all_refs, all_is_closed) if c]
    if closed_preds:
        metrics['closed'] = batch_metrics(closed_preds, closed_refs)
        metrics['closed'].update(_compute_format_stats(closed_preds, max_words=max_words))
    open_preds = [p for p, c in zip(all_preds, all_is_closed) if not c]
    open_refs = [r for r, c in zip(all_refs, all_is_closed) if not c]
    if open_preds:
        metrics['open'] = batch_metrics(open_preds, open_refs)
        metrics['open'].update(_compute_format_stats(open_preds, max_words=max_words))
    return metrics

def evaluate_multimodal_vqa(model, dataloader, device, processor, beam_width=1, max_words=10):
    model.eval()
    all_preds = []
    all_refs = []
    all_is_closed = []
    
    # Khởi tạo Translator cho Hướng B (Zero-shot)
    from src.utils.translator import MedicalTranslator
    translator = MedicalTranslator(device=device.type)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Multimodal"):
            raw_images = batch.get('raw_image')
            # Lấy câu hỏi tiếng Việt để dịch (đảm bảo tính nhất quán cho bài toán Tiếng Việt)
            questions_vi = batch.get('raw_questions')
            
            # Bước 1: Dịch Vi -> En
            questions_en = translator.translate_vi2en(questions_vi)
            
            # Bao bọc vào Prompt Template chuẩn của LLaVA-1.5 (PHẢI bằng tiếng Anh)
            prompts = [
                f"USER: <image>\n{q}\nAnswer with standard medical terminology, concise, at most {max_words} words. ASSISTANT:"
                for q in questions_en
            ]
            
            if raw_images is not None:
                inputs = processor(text=prompts, images=raw_images, return_tensors="pt", padding=True).to(device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            else:
                # Fallback
                inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

            output_ids = model.generate(
                **inputs, 
                max_new_tokens=max_words + 4,
                do_sample=False, 
                num_beams=beam_width, 
                early_stopping=True if beam_width > 1 else False
            )
            input_token_len = inputs.input_ids.shape[1]
            new_tokens = output_ids[:, input_token_len:]
            preds_en = processor.batch_decode(new_tokens, skip_special_tokens=True)
            
            # Bước 2: Dịch En -> Vi để có kết quả Tiếng Việt như user yêu cầu
            preds_vi = [postprocess_answer(pred, max_words=max_words) for pred in translator.translate_en2vi(preds_en)]
            
            # Debug mẫu đầu tiên
            if len(all_preds) == 0:
                print("\n--- DEBUG B1 (Zero-shot + Translation) ---")
                print(f"Q (Vi): {questions_vi[0]}")
                print(f"Q (En): {questions_en[0]}")
                print(f"Pred (En): {preds_en[0]}")
                print(f"Pred (Vi): {preds_vi[0]}")
                print(f"GT (Vi): {batch['raw_answer'][0]}")
                print("------------------------------------------\n")

            all_preds.extend([normalize_for_metric(p) for p in preds_vi])
            all_refs.extend([normalize_for_metric(postprocess_answer(r, max_words=max_words)) for r in batch['raw_answer']])
            is_closed = (batch['label_closed'] != -1).tolist()
            all_is_closed.extend(is_closed)

    metrics = batch_metrics(all_preds, all_refs)
    metrics.update(_compute_format_stats(all_preds, max_words=max_words))
    metrics['predictions'] = all_preds
    metrics['ground_truths'] = all_refs
    
    closed_preds = [p for p, c in zip(all_preds, all_is_closed) if c]
    closed_refs = [r for r, c in zip(all_refs, all_is_closed) if c]
    if closed_preds:
        metrics['closed'] = batch_metrics(closed_preds, closed_refs)
        metrics['closed'].update(_compute_format_stats(closed_preds, max_words=max_words))
    open_preds = [p for p, c in zip(all_preds, all_is_closed) if not c]
    open_refs = [r for r, c in zip(all_refs, all_is_closed) if not c]
    if open_preds:
        metrics['open'] = batch_metrics(open_preds, open_refs)
        metrics['open'].update(_compute_format_stats(open_preds, max_words=max_words))
    return metrics

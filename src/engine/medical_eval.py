import torch
from tqdm import tqdm
from src.utils.metrics import batch_metrics

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

def evaluate_vqa(model, dataloader, device, tokenizer, beam_width=1):
    model.eval()
    all_preds = []
    all_refs = []
    all_is_closed = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Sử dụng hàm generate chính thức
            logits_open = model.generate(images, input_ids, attention_mask, beam_width=beam_width)
            pred_ids = torch.argmax(logits_open, dim=-1)
            preds_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            
            # Debug: In ra mẫu đầu tiên để kiểm tra
            if len(all_preds) == 0:
                print("\n--- DEBUG PREDICTIONS ---")
                for i in range(min(3, len(preds_text))):
                    print(f"Q: {batch['raw_questions'][i]}")
                    print(f"Pred: '{preds_text[i]}'")
                    print(f"GT  : '{batch['raw_answer'][i]}'")
                print("--------------------------\n")
            
            all_preds.extend(preds_text)
            all_refs.extend(batch['raw_answer'])
            is_closed = (batch['label_closed'] != -1).tolist()
            all_is_closed.extend(is_closed)

    metrics = batch_metrics(all_preds, all_refs)
    metrics['predictions'] = all_preds
    metrics['ground_truths'] = all_refs
    
    closed_preds = [p for p, c in zip(all_preds, all_is_closed) if c]
    closed_refs = [r for r, c in zip(all_refs, all_is_closed) if c]
    if closed_preds: metrics['closed'] = batch_metrics(closed_preds, closed_refs)
    open_preds = [p for p, c in zip(all_preds, all_is_closed) if not c]
    open_refs = [r for r, c in zip(all_refs, all_is_closed) if not c]
    if open_preds: metrics['open'] = batch_metrics(open_preds, open_refs)
    return metrics

def evaluate_multimodal_vqa(model, dataloader, device, processor, beam_width=1):
    # ... (giữ nguyên code cũ)
    model.eval()
    all_preds = []
    all_refs = []
    all_is_closed = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Multimodal"):
            raw_images = batch.get('raw_images')
            questions = batch.get('raw_questions')
            
            if raw_images is None:
                inputs = {"pixel_values": batch['image'].to(device), "input_ids": batch['input_ids'].to(device)}
            else:
                inputs = processor(text=questions, images=raw_images, return_tensors="pt", padding=True).to(device)

            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=beam_width, early_stopping=True if beam_width > 1 else False)
            input_token_len = inputs.input_ids.shape[1]
            new_tokens = output_ids[:, input_token_len:]
            preds_text = processor.batch_decode(new_tokens, skip_special_tokens=True)
            
            all_preds.extend(preds_text)
            all_refs.extend(batch['raw_answer'])
            is_closed = (batch['label_closed'] != -1).tolist()
            all_is_closed.extend(is_closed)

    metrics = batch_metrics(all_preds, all_refs)
    metrics['predictions'] = all_preds
    metrics['ground_truths'] = all_refs
    
    closed_preds = [p for p, c in zip(all_preds, all_is_closed) if c]
    closed_refs = [r for r, c in zip(all_refs, all_is_closed) if c]
    if closed_preds: metrics['closed'] = batch_metrics(closed_preds, closed_refs)
    open_preds = [p for p, c in zip(all_preds, all_is_closed) if not c]
    open_refs = [r for r, c in zip(all_refs, all_is_closed) if not c]
    if open_preds: metrics['open'] = batch_metrics(open_preds, open_refs)
    return metrics

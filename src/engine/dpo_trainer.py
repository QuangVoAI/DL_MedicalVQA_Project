import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

from src.utils.text_utils import get_target_answer, normalize_answer

_VI_TO_EN_ANSWER_MAP = {
    "có": "yes",
    "không": "no",
    "bình thường": "normal",
    "bất thường": "abnormal",
    "gan": "liver",
    "phổi": "lung",
    "tim": "heart",
    "não": "brain",
    "thận": "kidney",
    "bàng quang": "bladder",
    "lách": "spleen",
    "phía trên bên trái": "upper left",
    "phía trên bên phải": "upper right",
    "phía dưới bên trái": "lower left",
    "phía dưới bên phải": "lower right",
}


def _is_closed_question(question: str, answer: str) -> bool:
    q = normalize_answer(question)
    a = normalize_answer(answer)
    return (
        a in {"có", "không"}
        or q.endswith(" không")
        or " bình thường " in f" {q} "
        or " có " in f" {q} "
    )


def _englishify_answer(question: str, answer: str) -> str:
    q = normalize_answer(question)
    a = normalize_answer(answer)
    translated = a
    for vi, en in sorted(_VI_TO_EN_ANSWER_MAP.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(vi, en)

    if _is_closed_question(question, answer):
        if a == "có":
            if "bình thường" in q:
                return "Yes, the image appears normal."
            return "Yes, the finding is present."
        if a == "không":
            if "bình thường" in q:
                return "No, the image is not normal."
            return "No, the finding is absent."

    if any(term in q for term in ["ở đâu", "vi tri", "where"]):
        return f"The structure is located at {translated}."
    return f"The answer is {translated}."


def _flip_closed_answer(answer: str) -> str:
    a = normalize_answer(answer)
    if a == "có":
        return "không"
    if a == "không":
        return "có"
    return a


def _build_rejected_candidates(data: list[dict], idx: int, chosen: str) -> list[str]:
    item = data[idx]
    question = item.get("question_vi", item.get("question", ""))
    candidates = []

    english_verbose = _englishify_answer(question, chosen)
    if english_verbose:
        candidates.append(english_verbose)

    if _is_closed_question(question, chosen):
        flipped = _flip_closed_answer(chosen)
        if flipped and flipped != normalize_answer(chosen):
            candidates.append(flipped)
            candidates.append(_englishify_answer(question, flipped))
    else:
        next_answer = get_target_answer(data[(idx + 1) % len(data)], max_words=10)
        if next_answer and normalize_answer(next_answer) != normalize_answer(chosen):
            candidates.append(next_answer)
        candidates.append(f"{question} {chosen}")

    deduped = []
    chosen_norm = normalize_answer(chosen)
    seen = set()
    for candidate in candidates:
        candidate_norm = normalize_answer(candidate)
        if not candidate_norm or candidate_norm == chosen_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        deduped.append(candidate)
    return deduped

def create_preference_data(vqa_json_path, output_path, num_pairs=200):
    """
    Tạo dữ liệu Preference (Chosen vs Rejected) cho DPO.
    Trong Medical VQA, 'Rejected' thường là các câu trả lời bị hallucination hoặc sai thuật ngữ y khoa.
    """
    with open(vqa_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pref_data = []
    for i in range(len(data)):
        item = data[i]
        chosen = get_target_answer(item, max_words=10)
        rejected_candidates = _build_rejected_candidates(data, i, chosen)

        for rejected in rejected_candidates:
            pref_data.append({
                "image": item.get("image_name") or item.get("image"),
                "source_idx": i,
                "question": item["question_vi"],
                "chosen": chosen,
                "rejected": rejected
            })
            if len(pref_data) >= num_pairs:
                break
        if len(pref_data) >= num_pairs:
            break
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pref_data, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] Đã tạo {len(pref_data)} cặp preference dữ liệu tại {output_path}")
    return pref_data

class MedicalDPOTrainer:
    """
    Trainer cho Direct Preference Optimization (DPO) trên LLaVA-Med.
    Giúp tối ưu hóa mô hình dựa trên các cặp preference dữ liệu y tế.
    """
    def __init__(self, model, reference_model, train_loader, optimizer, device, config):
        self.model = model
        self.reference_model = reference_model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.beta = config.get('dpo_beta', 0.1)

    def get_log_probs(self, logits, labels):
        """
        Tính log probabilities cho các sequence.
        logits: [batch, seq_len, vocab]
        labels: [batch, seq_len]
        """
        # Shift logits và labels để khớp (next token prediction)
        log_probs = F.log_softmax(logits, dim=-1)
        # Lấy log prob của các token đúng
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # Chỉ lấy các token không phải padding (giả định mask > 0)
        return (per_token_logps * (labels != 0)).sum(-1)

    def compute_loss(self, policy_chosen_logps, policy_rejected_logps, 
                     reference_chosen_logps, reference_rejected_logps):
        """
        Tính DPO loss theo công thức: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        # Thêm các chỉ số để theo dõi (rewards)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return loss, chosen_rewards, rejected_rewards

    def train(self, epochs=3):
        print(f"[INFO] Bắt đầu huấn luyện DPO (beta={self.beta})...")
        self.model.train()
        self.reference_model.eval()
        # Freeze reference model để tiết kiệm VRAM (Quan trọng cho T4)
        for param in self.reference_model.parameters():
            param.requires_grad_(False)
            
        print(f"[INFO] DPO Trainer Ready ({self.device})")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0  # Đã thêm dòng khởi tạo total_loss tại đây
            pbar = tqdm(self.train_loader, desc=f"DPO Epoch {epoch+1}")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                
                # Tính Logits cho Chosen và Rejected (Sử dụng Duck Typing/Safe Forward)
                try:
                    # Case: LLaVA-style multimodal model
                    outputs_w = self.model(input_ids=chosen_ids, pixel_values=images, labels=chosen_ids)
                    outputs_l = self.model(input_ids=rejected_ids, pixel_values=images, labels=rejected_ids)
                    logits_w = outputs_w.logits
                    logits_l = outputs_l.logits
                except Exception:
                    # Fallback: Modular model (A1/A2 style)
                    _, logits_w = self.model(images, chosen_ids)
                    _, logits_l = self.model(images, rejected_ids)
                
                # 2. Forward Reference Model (No Grad)
                with torch.no_grad():
                    try:
                        # Multimodal case
                        outputs_ref_w = self.reference_model(input_ids=chosen_ids, pixel_values=images, labels=chosen_ids)
                        outputs_ref_l = self.reference_model(input_ids=rejected_ids, pixel_values=images, labels=rejected_ids)
                        ref_logits_w = outputs_ref_w.logits
                        ref_logits_l = outputs_ref_l.logits
                    except Exception:
                        # Modular case
                        _, ref_logits_w = self.reference_model(images, chosen_ids)
                        _, ref_logits_l = self.reference_model(images, rejected_ids)
                
                # 3. Tính log probs
                logps_w = self.get_log_probs(logits_w, chosen_ids)
                logps_l = self.get_log_probs(logits_l, rejected_ids)
                ref_logps_w = self.get_log_probs(ref_logits_w, chosen_ids)
                ref_logps_l = self.get_log_probs(ref_logits_l, rejected_ids)
                
                # 4. Tính Loss
                loss, _, _ = self.compute_loss(logps_w, logps_l, ref_logps_w, ref_logps_l)
                
                # 5. Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
            print(f"Epoch {epoch+1} | DPO Loss: {total_loss/len(self.train_loader):.4f}")

import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import random

from src.utils.text_utils import get_target_answer, normalize_answer

def _is_closed_question(question: str, answer: str) -> bool:
    q = normalize_answer(question)
    a = normalize_answer(answer)
    return (
        a in {"có", "không"}
        or q.endswith(" không")
        or " bình thường " in f" {q} "
        or " có " in f" {q} "
    )


def _flip_closed_answer(answer: str) -> str:
    a = normalize_answer(answer)
    if a == "có":
        return "không"
    if a == "không":
        return "có"
    return a


def _answer_category(question: str, answer: str) -> str:
    q = normalize_answer(question)
    a = normalize_answer(answer)
    if _is_closed_question(question, answer):
        return "closed"
    if any(term in q for term in ["ở đâu", "vi tri", "where"]):
        return "location"
    if any(term in a for term in ["trái", "phải", "trên", "dưới", "giữa", "bên"]):
        return "location"
    if any(term in a for term in ["mặt phẳng", "ngang", "vành", "dọc"]):
        return "plane"
    if any(term in a for term in ["gan", "phổi", "tim", "não", "thận", "lách", "bàng quang", "khí quản", "trung thất"]):
        return "organ"
    return "finding"


def _build_answer_pools(data: list[dict], max_words: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    question_to_answers = {}
    category_to_answers = {}
    for item in data:
        question = item.get("question_vi", item.get("question", ""))
        answer = get_target_answer(item, max_words=max_words)
        if not question or not answer:
            continue
        q_norm = normalize_answer(question)
        a_norm = normalize_answer(answer)
        category = _answer_category(question, answer)
        question_to_answers.setdefault(q_norm, [])
        if a_norm not in question_to_answers[q_norm]:
            question_to_answers[q_norm].append(a_norm)
        category_to_answers.setdefault(category, [])
        if a_norm not in category_to_answers[category]:
            category_to_answers[category].append(a_norm)
    return question_to_answers, category_to_answers


def _build_rejected_candidates(
    data: list[dict],
    idx: int,
    chosen: str,
    question_to_answers: dict[str, list[str]],
    category_to_answers: dict[str, list[str]],
) -> list[str]:
    item = data[idx]
    question = item.get("question_vi", item.get("question", ""))
    question_norm = normalize_answer(question)
    chosen_norm = normalize_answer(chosen)
    category = _answer_category(question, chosen)
    candidates = []

    if _is_closed_question(question, chosen):
        flipped = _flip_closed_answer(chosen)
        if flipped and flipped != chosen_norm:
            candidates.append(flipped)
    else:
        for answer in question_to_answers.get(question_norm, []):
            if answer != chosen_norm:
                candidates.append(answer)
        for answer in category_to_answers.get(category, []):
            if answer != chosen_norm:
                candidates.append(answer)
    deduped = []
    seen = set()
    for candidate in candidates:
        candidate_norm = normalize_answer(candidate)
        if not candidate_norm or candidate_norm == chosen_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        deduped.append(candidate_norm)
    return deduped

def _build_pair_record(item: dict, source_idx: int, chosen: str, rejected: str) -> dict:
    return {
        "image": item.get("image_name") or item.get("image"),
        "source_idx": source_idx,
        "question": item["question_vi"],
        "chosen": chosen,
        "rejected": rejected,
        "answer_type": _answer_category(item["question_vi"], chosen),
    }


def _round_robin_merge(grouped_pairs: dict[str, list[dict]], target_count: int) -> list[dict]:
    ordered_groups = sorted(grouped_pairs.keys())
    merged = []
    while len(merged) < target_count:
        progressed = False
        for group in ordered_groups:
            if grouped_pairs[group]:
                merged.append(grouped_pairs[group].pop())
                progressed = True
                if len(merged) >= target_count:
                    break
        if not progressed:
            break
    return merged


def create_preference_data(
    vqa_json_path,
    output_path,
    num_pairs=400,
    closed_ratio=0.6,
    max_answer_words=6,
    seed=42,
):
    """
    Tạo dữ liệu Preference (Chosen vs Rejected) cho DPO.
    Trong Medical VQA, 'Rejected' thường là các câu trả lời bị hallucination hoặc sai thuật ngữ y khoa.
    """
    with open(vqa_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question_to_answers, category_to_answers = _build_answer_pools(data, max_words=max_answer_words)
    rng = random.Random(seed)
    closed_pairs = []
    open_pairs_by_group = {"location": [], "plane": [], "organ": [], "finding": []}

    for i in range(len(data)):
        item = data[i]
        chosen = get_target_answer(item, max_words=max_answer_words)
        chosen_norm = normalize_answer(chosen)
        if not chosen_norm or len(chosen_norm.split()) > max_answer_words:
            continue
        rejected_candidates = _build_rejected_candidates(
            data,
            i,
            chosen_norm,
            question_to_answers=question_to_answers,
            category_to_answers=category_to_answers,
        )
        category = _answer_category(item["question_vi"], chosen_norm)

        for rejected in rejected_candidates:
            if len(rejected.split()) > max_answer_words:
                continue
            pair = _build_pair_record(item, i, chosen_norm, rejected)
            if category == "closed":
                closed_pairs.append(pair)
            elif category in open_pairs_by_group:
                open_pairs_by_group[category].append(pair)

    rng.shuffle(closed_pairs)
    for pairs in open_pairs_by_group.values():
        rng.shuffle(pairs)

    target_closed = min(len(closed_pairs), int(round(num_pairs * closed_ratio)))
    target_open = max(0, num_pairs - target_closed)
    sampled_closed = closed_pairs[:target_closed]
    sampled_open = _round_robin_merge(open_pairs_by_group, target_open)
    pref_data = sampled_closed + sampled_open
    rng.shuffle(pref_data)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pref_data, f, ensure_ascii=False, indent=2)
    
    print(
        f"[SUCCESS] Đã tạo {len(pref_data)} cặp preference dữ liệu tại {output_path} "
        f"(closed={len(sampled_closed)}, open={len(sampled_open)})"
    )
    preview_count = min(30, len(pref_data))
    if preview_count:
        print(f"[INFO] Preview {preview_count} cặp preference đầu tiên để kiểm tra nhanh:")
        for idx, pair in enumerate(pref_data[:preview_count], start=1):
            print(
                f"  [{idx:02d}] type={pair.get('answer_type')} | "
                f"Q={pair.get('question')} | chosen={pair.get('chosen')} | rejected={pair.get('rejected')}"
            )
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

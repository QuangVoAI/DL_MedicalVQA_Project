"""Evaluation metrics for VQA: Accuracy, EM, F1, BLEU-1~4, METEOR, and Semantic Score."""

from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("[INFO] Đang tự động tải bộ từ điển NLTK WordNet cho METEOR score...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# 1. Semantic Score (SentenceTransformer)
try:
    from sentence_transformers import SentenceTransformer, util
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    semantic_model = None
    print(f"[WARNING] Could not load SentenceTransformer: {e}")

# 2. BERTScore
try:
    from bert_score import BERTScorer
    # Ép sử dụng model multilingual để tránh lỗi attribute của Tokenizer trên Python 3.12
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scorer = BERTScorer(model_type="bert-base-multilingual-cased", device=device)
except ImportError:
    print("[WARNING] Thư viện bert_score chưa được cài đặt.")
    bert_scorer = None
except Exception as e:
    bert_scorer = None
    print(f"[WARNING] Could not load BERTScorer: {e}")

# 3. ROUGE-L
try:
    from rouge_score import rouge_scorer as rs
    rouge_l_scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
except Exception as e:
    rouge_l_scorer = None
    print(f"[WARNING] Could not load rouge-score: {e}")

from .text_utils import normalize_answer, majority_answer

def compute_rouge_l(pred: str, refs) -> float:
    """Tính ROUGE-L (Lấy MAX over multiple refs)."""
    if not rouge_l_scorer: return 0.0
    if isinstance(refs, str): refs = [refs]
    best_rouge = 0.0
    for r in refs:
        score = rouge_l_scorer.score(normalize_answer(r), normalize_answer(pred))['rougeL'].fmeasure
        best_rouge = max(best_rouge, score)
    return best_rouge

def compute_bertscore(preds: list[str], refs: list) -> float:
    """Tính BERTScore cho cả batch."""
    if not bert_scorer or not preds or not refs:
        return 0.0
    
    clean_preds = [normalize_answer(p) if normalize_answer(p).strip() else "." for p in preds]
    clean_refs = [majority_answer(r) if isinstance(r, list) else normalize_answer(r) for r in refs]
    clean_refs = [r if r.strip() else "." for r in clean_refs]
    
    try:
        # Tăng tốc bằng cách tắt idf nếu cần
        P, R, F1 = bert_scorer.score(clean_preds, clean_refs)
        return float(F1.mean().item())
    except Exception as e:
        print(f"[WARNING] BERTScore error: {e}")
        return 0.0

def compute_exact_match(pred: str, refs) -> float:
    """So khớp chính xác lấy MAX (soft match over multiple refs)."""
    if isinstance(refs, str): refs = [refs]
    return float(any(normalize_answer(pred) == normalize_answer(r) for r in refs))

def compute_f1(pred: str, refs) -> float:
    """Tính F1-score ở mức độ token. Lấy MAX over multiple refs."""
    if isinstance(refs, str): refs = [refs]
    best_f1 = 0.0
    p_toks = normalize_answer(pred).split()
    for r in refs:
        r_toks = normalize_answer(r).split()
        if not p_toks or not r_toks:
            f1 = float(p_toks == r_toks)
        else:
            common = Counter(p_toks) & Counter(r_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                precision = num_same / len(p_toks)
                recall = num_same / len(r_toks)
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

def compute_bleu(pred: str, refs) -> dict[str, float]:
    """Tính BLEU from 1 đến 4 sử dụng corpus-level refs."""
    if isinstance(refs, str): refs = [refs]
    smoothie = SmoothingFunction().method4
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]
    
    if not p_toks or not r_toks_list:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    weights = [
        (1, 0, 0, 0),          # BLEU-1
        (0.5, 0.5, 0, 0),      # BLEU-2
        (0.33, 0.33, 0.33, 0), # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    
    return {
        f"bleu{i+1}": sentence_bleu(r_toks_list, p_toks, weights=w, smoothing_function=smoothie)
        for i, w in enumerate(weights)
    }

def compute_meteor(pred: str, refs) -> float:
    """Tính METEOR score (hỗ trợ N refs)."""
    if isinstance(refs, str): refs = [refs]
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]
    if not p_toks or not r_toks_list:
        return 0.0
    return _nltk_meteor(r_toks_list, p_toks)

def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """
    Tính VQA Accuracy mềm: min(#người_cùng_đáp_án / 3, 1.0).
    Using cho các tập dữ liệu có nhiều người gắn nhãn (như A-OKVQA).
    """
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)
    
    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)

def compute_semantic_score(preds: list[str], refs: list) -> float:
    """Tính điểm tương đồng ngữ nghĩa bằng Cosine Similarity."""
    if not semantic_model or not preds or not refs:
        return 0.0
    
    clean_preds = [normalize_answer(p) for p in preds]
    # Take the most representative string if it's a list for semantic comparison
    clean_refs = [majority_answer(r) if isinstance(r, list) else normalize_answer(r) for r in refs]
    
    # Encode to Vector (Embeddings)
    pred_embs = semantic_model.encode(clean_preds, convert_to_tensor=True, show_progress_bar=False)
    ref_embs = semantic_model.encode(clean_refs, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute Cosine distance matrix and take diagonal (1-to-1 comparison)
    cosine_scores = util.cos_sim(pred_embs, ref_embs)
    scores = torch.diag(cosine_scores)
    
    return float(scores.mean().item())

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đo lường trên batch."""
    results = {
        "accuracy": [], "em": [], "f1": [], "meteor": [],
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [],
        "rouge_l": []
    }
    
    for pred, ref in zip(predictions, references):
        # Pass full refs list to compute_f1, compute_bleu to maximize score
        results["accuracy"].append(compute_vqa_accuracy(pred, ref))
        results["em"].append(compute_exact_match(pred, ref))
        results["f1"].append(compute_f1(pred, ref))
        results["meteor"].append(compute_meteor(pred, ref))
        results["rouge_l"].append(compute_rouge_l(pred, ref))
        
        bleus = compute_bleu(pred, ref)
        for k, v in bleus.items():
            results[k].append(v)
            
    # Average traditional metrics
    final_metrics = {k: float(np.mean(v)) for k, v in results.items()}
    
    # Compute Semantic Score and BERTScore for entire batch
    final_metrics["semantic"] = compute_semantic_score(predictions, references)
    final_metrics["bert_score"] = compute_bertscore(predictions, references)
    
    return final_metrics

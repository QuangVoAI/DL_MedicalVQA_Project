"""
Medical VQA — Complete Data Processing Pipeline
================================================
Pipeline:
 1. Tải SLAKE + VQA-RAD từ HuggingFace
 2. Gộp & shuffle (seed=42)
 3. Dịch question + answer → tiếng Việt (Ollama local, Mac M4 optimised)
   - Dictionary-Enhanced Prompting (thuật ngữ y tế chuẩn)
   - Yes/No rule-based (không gọi LLM, tiết kiệm ~50% thời gian)
   - Output validation (phát hiện output lẫn tiếng Trung/Anh)
 4. Paraphrase augmentation (sinh thêm 1 câu VI cho mỗi mẫu)
 5. Back-translation QA (dịch ngược VI→EN, tính overlap score)
 6. Chia train/val/test 80/10/10
 7. Push lên HuggingFace Hub

Cách dùng:
 # Cài deps
 pip install datasets tqdm requests

 # Test 5 mẫu (không cần Ollama lâu)
 python data_pipeline.py --dry_run

 # Chạy đầy đủ, không push HF
 python data_pipeline.py --no_push

 # Chạy đầy đủ + push
 export HF_TOKEN=os.environ.get("HF_TOKEN", "")
 python data_pipeline.py --hf_repo "SpringWang08/medical-vqa-vi"

 # Dùng model nhỏ hơn nếu RAM < 16GB
 python data_pipeline.py --model qwen2.5:7b --no_push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict


# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"  # đổi sang qwen2.5:7b nếu RAM < 16 GB
CHECKPOINT  = "data/translate_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# TỪ ĐIỂN Y TẾ EN → VI (dictionary-enhanced prompting)
# ─────────────────────────────────────────────────────────────────────────────

MED_DICT: dict[str, str] = {
  # ── Giải phẫu cơ bản ──────────────────────────────────────────────────
  "lobe": "thùy",
  "right lobe": "thùy phải",
  "left lobe": "thùy trái",
  "upper lobe": "thùy trên",
  "lower lobe": "thùy dưới",
  "middle lobe": "thùy giữa",
  "lung": "phổi",
  "lungs": "phổi",
  "right lung": "phổi phải",
  "left lung": "phổi trái",
  "heart": "tim",
  "cardiac": "tim",
  "aorta": "động mạch chủ",
  "pericardial": "màng ngoài tim",
  "vascular": "mạch máu",
  "trachea": "khí quản",
  "diaphragm": "cơ hoành",
  "abdomen": "bụng",
  "liver": "gan",
  "spleen": "lách",
  "kidney": "thận",
  "gallbladder": "túi mật",
  "pancreas": "tụy",
  "appendix": "ruột thừa",
  "bowel": "ruột",
  "colon": "đại tràng",
  "stomach": "dạ dày",
  "chest": "ngực",
  "neck": "cổ",
  "shoulder": "vai",
  "wrist": "cổ tay",
  "ankle": "mắt cá chân",
  "thyroid": "tuyến giáp",
  "lymph node": "hạch bạch huyết",
  "spine": "cột sống",
  "pelvis": "xương chậu",
  "femur": "xương đùi",
  "tibia": "xương chày",
  "rib": "xương sườn",
  "vertebra": "đốt sống",
  "joint": "khớp",
  # ── Não / Thần kinh ───────────────────────────────────────────────────
  "brain": "não",
  "head": "đầu",
  "skull": "hộp sọ",
  "cortex": "vỏ não",
  "cerebral cortex": "vỏ não đại não",
  "medulla": "tủy",
  "cerebellum": "tiểu não",
  "temporal": "thái dương",
  "parietal": "đỉnh",
  "frontal": "trán",
  "occipital": "chẩm",
  # ── Bệnh lý / Tổn thương ──────────────────────────────────────────────
  "pneumonia": "viêm phổi",
  "pleural effusion": "tràn dịch màng phổi",
  "atelectasis": "xẹp phổi",
  "consolidation": "đông đặc",
  "infiltrate": "thâm nhiễm",
  "pneumothorax": "tràn khí màng phổi",
  "emphysema": "khí phế thũng",
  "bronchitis": "viêm phế quản",
  "cardiomegaly": "tim to",
  "fracture": "gãy xương",
  "scoliosis": "vẹo cột sống",
  "osteoporosis": "loãng xương",
  "arthritis": "viêm khớp",
  "dislocation": "trật khớp",
  "hemorrhage": "xuất huyết",
  "stroke": "đột quỵ",
  "cerebral edema": "phù não",
  "brain edema": "phù não",
  "infarction": "nhồi máu",
  "hematoma": "máu tụ",
  "aneurysm": "phình mạch",
  "stenosis": "hẹp",
  "thrombosis": "huyết khối",
  "ischemia": "thiếu máu cục bộ",
  "tumor": "khối u",
  "mass": "khối u",
  "nodule": "nốt",
  "lesion": "tổn thương",
  "abnormality": "bất thường",
  "opacity": "đục mờ",
  "edema": "phù nề",
  "calcification": "vôi hóa",
  "effusion": "tràn dịch",
  "shadow": "bóng mờ",
  # ── Hình ảnh học ──────────────────────────────────────────────────────
  "modality": "phương thức chụp",
  "organ system": "hệ cơ quan",
  "imaging": "hình ảnh",
  "scan": "ảnh chụp",
  "sagittal": "mặt phẳng dọc",
  "coronal": "mặt phẳng trán",
  "axial": "mặt phẳng ngang",
  "plane": "mặt phẳng",
  "view": "góc nhìn",
  "section": "lát cắt",
  "slice": "lát cắt",
  # ── Hình thái / Mô tả ─────────────────────────────────────────────────
  "u-shaped": "hình chữ U",
  "c-shaped": "hình chữ C",
  "round": "tròn",
  "oval": "bầu dục",
  "irregular": "không đều",
  "homogeneous": "đồng nhất",
  "heterogeneous": "không đồng nhất",
  "density": "mật độ",
  # ── Vị trí tương đối ──────────────────────────────────────────────────
  "bilateral": "hai bên",
  "unilateral": "một bên",
  "ipsilateral": "cùng bên",
  "contralateral": "đối bên",
  "anterior": "phía trước",
  "posterior": "phía sau",
  "lateral": "bên",
  "medial": "giữa",
  "superior": "trên",
  "inferior": "dưới",
  "proximal": "gần",
  "distal": "xa",
  "central": "trung tâm",
  "peripheral": "ngoại vi",
  # ── Trạng thái chung ──────────────────────────────────────────────────
  "normal": "bình thường",
  "abnormal": "bất thường",
}

# Tập Yes / No — không cần gọi LLM
YES_SET: set[str] = {"yes", "true", "present", "positive", "1", "correct"}
NO_SET: set[str] = {"no", "false", "absent", "negative", "0", "incorrect"}

# Regex dấu thanh điệu tiếng Việt
VI_DIACRITIC = re.compile(
  r"[àáảãạăắặẳẵằâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
  r"ùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẮẶẲẴẰÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌ"
  r"ÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Phát hiện tiếng Trung bằng Unicode
# ─────────────────────────────────────────────────────────────────────────────

def is_chinese(text: str) -> bool:
  """True nếu câu chứa >= 3 ký tự CJK (tránh false positive với ký hiệu)."""
  count = sum(
    1 for ch in text
    if "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
    or "\u3400" <= ch <= "\u4dbf"  # Extension A
    or "\uf900" <= ch <= "\ufaff"  # CJK Compatibility Ideographs
  )
  return count >= 3


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — Validate output là tiếng Việt hợp lệ
# ─────────────────────────────────────────────────────────────────────────────

# Tập hợp các từ tiếng Việt/thuật ngữ y khoa hợp lệ nhưng hoàn toàn KHÔNG CÓ DẤU
VALID_NO_DIACRITIC_WORDS = frozenset({
  "gan", "tim", "tay", "vai", "u", "nang", "to", "sau", "trong", "nam",
  "hai", "ba", "tai", "da", "cao", "suy",
  "phim", "tia", "x", "ray", "scan", "ct", "mri", "ph", "mmhg", "spo2",
  "ecg", "ekg", "icu", "pet", "us"
})

def is_valid_vi(text: str, original: str) -> bool:
  """
  True nếu text trông như tiếng Việt hợp lệ:
   - Không rỗng, không chứa CJK
   - Không giống hệt tiếng Anh gốc
   - Phải có dấu tiếng Việt, NẾU KHÔNG CÓ DẤU thì phải thuộc danh sách từ ngoại lệ (gan, tim, CT...)
  """
  if not text or len(text.strip()) < 2:
    return False
  if is_chinese(text):
    return False
  if text.strip().lower() == original.strip().lower():
    return False
    
  # Nếu câu có chứa dấu/ký tự đặc thù tiếng Việt -> Hợp lệ
  if bool(VI_DIACRITIC.search(text)):
    return True
    
  # NẾU KHÔNG CÓ DẤU:
  # 1. Chỉ chấp nhận câu ngắn (<= 3 từ)
  words = text.lower().split()
  if len(words) > 3:
    return False
    
  # 2. Bắt buộc MỌI từ trong câu phải nằm trong whitelist không dấu
  # (Tránh lọt các từ tiếng Anh lười dịch như "liver", "right side")
  return all(w in VALID_NO_DIACRITIC_WORDS for w in words)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_Q_PROMPT = """\
Bạn là chuyên gia dịch thuật y tế (Anh → Việt).

QUY TẮC BẮT BUỘC:
1. Giữ nguyên tiếng Anh: CT scan, MRI, X-ray, pH, mmHg, SpO2, tên thuốc.
2. Dùng từ điển dưới đây, ghi tiếng Anh trong ngoặc lần đầu xuất hiện.
  TỪ ĐIỂN: {term_dict}
3. Câu hỏi tự nhiên, ngắn gọn (≤ 15 từ), đúng cú pháp tiếng Việt.
4. TRẢ VỀ JSON duy nhất: {{"translation": "..."}}

CÂU GỐC: {text}"""

_A_PROMPT = """\
Bạn là chuyên gia dịch thuật y tế (Anh → Việt).

QUY TẮC BẮT BUỘC:
1. Giữ nguyên tiếng Anh: CT scan, MRI, X-ray, pH, mmHg, SpO2, tên thuốc.
2. Dùng từ điển dưới đây.
  TỪ ĐIỂN: {term_dict}
3. Câu trả lời ngắn gọn (≤ 10 từ).
4. TRẢ VỀ JSON duy nhất: {{"translation": "..."}}

CÂU GỐC: {text}"""

_PARA_Q_PROMPT = """\
Bạn là một chuyên gia ngôn ngữ y tế tiếng Việt.
Nhiệm vụ: Viết lại (paraphrase) câu hỏi y khoa dưới đây thành 4 cách diễn đạt KHÁC NHAU.
Yêu cầu: 
- Giữ nguyên nghĩa y khoa và các thuật ngữ. 
- Đảo cấu trúc câu hoặc dùng từ đồng nghĩa tự nhiên.
Câu hỏi gốc: {question}
TRẢ VỀ ĐỊNH DẠNG JSON DUY NHẤT (key 'variants' là mảng chứa 4 chuỗi): {{"variants": ["cách 1", "cách 2", "cách 3", "cách 4"]}}"""

_PARA_A_PROMPT = """\
Bạn là một chuyên gia ngôn ngữ y tế tiếng Việt.
Nhiệm vụ: Viết ra 4 biến thể KHÁC NHAU của câu trả lời dưới đây (kết hợp cả trả lời ngắn và câu trả lời đầy đủ).
Yêu cầu:
- Giữ nguyên ý nghĩa y khoa so với đáp án gốc. KHÔNG ĐƯỢC bịa thêm thông tin.
- Có thể dùng từ đồng nghĩa tự nhiên.
Câu hỏi tham khảo: {question}
Đáp án gốc: {answer}
TRẢ VỀ ĐỊNH DẠNG JSON DUY NHẤT (key 'variants' là mảng chứa 4 chuỗi): {{"variants": ["biến thể 1", "biến thể 2", "biến thể 3", "biến thể 4"]}}"""

_EXPAND_PROMPT = """\
Chuyển câu trả lời ngắn thành một câu hoàn chỉnh, tự nhiên và đa dạng cách diễn đạt.
YÊU CẦU BẮT BUỘC: 
1. TRẢ LỜI HOÀN TOÀN BẰNG TIẾNG VIỆT.
2. Câu trả lời phải CỰC KỲ NGẮN GỌN (TỐI ĐA 10 TỪ).
3. KHÔNG lặp đi lặp lại một kiểu mở bài. Hãy trả lời trực tiếp.
4. TUYỆT ĐỐI KHÔNG tự bịa thêm thông tin ngoài Đáp án gốc.

Câu hỏi: {question}
Đáp án gốc: {answer}
TRẢ VỀ JSON duy nhất: {{"translation": "..."}}"""

_BT_PROMPT = """\
Translate the following Vietnamese medical question back to English.
Return JSON only: {{"translation": "..."}}

Vietnamese: {question_vi}"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_terms(text: str) -> str:
  """Tìm thuật ngữ y tế trong câu → chuỗi "en=vi, ..." để inject vào prompt."""
  t = text.lower()
  found: list[str] = []
  # Sắp xếp multi-word trước để tránh "lung" match trong "right lung"
  for en, vi in sorted(MED_DICT.items(), key=lambda x: -len(x[0])):
    if en in t and not any(en in prev for prev in found):
      found.append(f"{en}={vi}")
  return ", ".join(found) if found else "Không có thuật ngữ đặc biệt."


def _post_process(text: str) -> str:
  """Chuẩn hoá viết hoa các ký hiệu y tế, xoá dấu nháy thừa."""
  for w in ["CT", "MRI", "X-ray", "pH", "mmHg", "SpO2", "ECG", "EKG", "ICU"]:
    text = re.sub(r"\b" + re.escape(w) + r"\b", w, text, flags=re.IGNORECASE)
  return text.strip().strip('"')


def _call_ollama(
  prompt: str,
  temperature: float = 0.0,
  max_tokens: int = 150,
  retries: int = 3,
) -> str:
  """Gọi Ollama, trả về string (đã parse JSON nếu được)."""
  payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": temperature, "num_predict": max_tokens},
  }
  for attempt in range(retries):
    try:
      r = requests.post(OLLAMA_URL, json=payload, timeout=60)
      raw = r.json().get("response", "{}").strip()
      try:
        parsed = json.loads(raw)
        # Lấy value đầu tiên trong dict nếu key không rõ
        for key in ("translation", "paraphrase"):
          if key in parsed:
            return str(parsed[key])
        return raw
      except json.JSONDecodeError:
        return raw
    except Exception:
      time.sleep(2 ** attempt)
  return ""


def _token_overlap(a: str, b: str) -> float:
  """BLEU-1 đơn giản: tỷ lệ từ chung / max độ dài."""
  ta, tb = set(a.lower().split()), set(b.lower().split())
  if not ta or not tb:
    return 0.0
  return len(ta & tb) / max(len(ta), len(tb))


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def translate_question(text: str, retries: int = 3) -> tuple[str, bool]:
  """
  Dịch câu hỏi tiếng Anh → tiếng Việt.
  Trả về (translation, is_valid).
  """
  if not text.strip():
    return "", False
  term_dict = _extract_terms(text)
  prompt = _Q_PROMPT.format(text=text, term_dict=term_dict)
  for _ in range(retries):
    raw = _call_ollama(prompt)
    result = _post_process(raw)
    if is_valid_vi(result, text):
      return result, True
  return "", False


def translate_answer(text: str) -> tuple[str, bool]:
  """
  Dịch câu trả lời.
  Yes/No → rule-based (không gọi LLM).
  Câu dài → gọi LLM.
  """
  if not text.strip():
    return "", False
  t = text.strip().lower()
  # Rule-based Yes/No — nhanh, chính xác 100%
  if t in YES_SET:
    return "Có", True
  if t in NO_SET:
    return "Không", True
  # Câu trả lời ngắn 1 từ (VD: "Right", "Head", "MRI")
  if len(t.split()) == 1:
    # Thử tra từ điển trước
    vi = MED_DICT.get(t)
    if vi:
      return vi, True
  # Gọi LLM cho câu dài hơn
  term_dict = _extract_terms(text)
  prompt = _A_PROMPT.format(text=text, term_dict=term_dict)
  for _ in range(3):
    raw = _call_ollama(prompt, max_tokens=80)
    result = _post_process(raw)
    if is_valid_vi(result, text):
      return result, True
  return text, False  # fallback giữ nguyên tiếng Anh


def expand_answer(question_vi: str, answer_vi: str) -> str:
  """Phóng to câu trả lời ngắn thành câu giao tiếp hoàn chỉnh."""
  if not question_vi.strip() or not answer_vi.strip():
    return answer_vi
  if len(answer_vi.split()) > 7:
    return answer_vi
  prompt = _EXPAND_PROMPT.format(question=question_vi, answer=answer_vi)
  raw = _call_ollama(prompt, temperature=0.5, max_tokens=100) # Temp=0.5 để đa dạng hóa
  result = _post_process(raw)
  
  # Fallback nếu LLM bịa ra tiếng Trung hoặc lỗi ngôn ngữ
  if is_chinese(result):
    return answer_vi
    
  return result


def generate_variants(prompt: str, original_valid: str) -> list[str]:
  """Hàm gọi Ollama chung để sinh ra mảng các biến thể (variants)."""
  payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.7, "num_predict": 200},
  }
  for _ in range(3):
    try:
      r = requests.post(OLLAMA_URL, json=payload, timeout=60)
      parsed = json.loads(r.json().get("response", "{}"))
      variants = parsed.get("variants", [])
      if isinstance(variants, list) and len(variants) > 0:
        # Xóa dấu nháy, khoảng trắng và đảm bảo là tiếng Việt hợp lệ
        cleaned = [_post_process(str(v)) for v in variants if is_valid_vi(str(v), original_valid)]
        # Bỏ các câu trùng nhau
        unique_variants = list(set(cleaned))
        # Trả về tối đa 4 câu
        return unique_variants[:4]
    except Exception:
      time.sleep(1)
  return []

def paraphrase_question(question_vi: str) -> list[str]:
  if not question_vi.strip():
    return []
  prompt = _PARA_Q_PROMPT.format(question=question_vi)
  return generate_variants(prompt, original_valid=question_vi)

def paraphrase_answer(question_vi: str, answer_vi: str) -> list[str]:
  if not question_vi.strip() or not answer_vi.strip():
    return []
    
  t = answer_vi.lower()
  # Nếu là Có/Không, tự hardcode các biến thể (vì AI sinh sẽ dễ bịa hoặc lỗi)
  if t == "có":
    return ["Có.", "Đúng vậy.", "Chính xác.", "Đúng thế."]
  if t == "không":
    return ["Không.", "Sai.", "Không phải.", "Hoàn toàn không."]
    
  prompt = _PARA_A_PROMPT.format(question=question_vi, answer=answer_vi)
  return generate_variants(prompt, original_valid=answer_vi)


def back_translate(question_vi: str) -> tuple[str, float]:
  """
  Dịch ngược VI → EN, tính token overlap với câu gốc EN.
  Trả về (back_translation_text, overlap_score).
  """
  if not question_vi.strip():
    return "", 0.0
  prompt = _BT_PROMPT.format(question_vi=question_vi)
  raw = _call_ollama(prompt, max_tokens=100)
  return _post_process(raw), 0.0  # score sẽ tính sau khi có EN gốc


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1 + 2: LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def load_slake() -> list[dict]:
  """
  [PATCH 1] Dùng Unicode detection thay vì q_lang field
  vì BoKelvin/SLAKE không export trường đó đầy đủ.
  """
  print("[1/5] Tải SLAKE từ HuggingFace...")
  ds = load_dataset("BoKelvin/SLAKE", split="train")
  rows, skipped = [], 0
  for item in ds:
    q = item.get("question", "")
    a = str(item.get("answer", ""))
    # Lọc câu Trung Quốc
    if is_chinese(q) or is_chinese(a):
      skipped += 1
      continue
    a_type = item.get("answer_type", "OPEN")
    if isinstance(a_type, str):
      a_type = a_type.upper()
    else:
      a_type = "CLOSED" if a.lower() in YES_SET | NO_SET else "OPEN"
    rows.append({
      "id":      f"slake_{item.get('qid', len(rows))}",
      "source":    "slake",
      "image_name":  item.get("img_name", ""),
      "question":   q,
      "answer":    a,
      "answer_type": a_type,
      "content_type": str(item.get("content_type", "")),
      "modality":   str(item.get("modality", "")),
      "location":   str(item.get("location", "")),
    })
  print(f" → {len(rows)} mẫu tiếng Anh | đã lọc {skipped} câu Trung Quốc")
  return rows


def load_vqa_rad() -> list[dict]:
  print("[1/5] Tải VQA-RAD từ HuggingFace...")
  ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
  rows = []
  for i, item in enumerate(ds):
    a = str(item.get("answer", ""))
    a_type = "CLOSED" if a.lower() in YES_SET | NO_SET else "OPEN"
    rows.append({
      "id":      f"vqarad_{i}",
      "source":    "vqa-rad",
      "image_name":  item.get("image_name", f"rad_{i}.jpg"),
      "question":   item.get("question", ""),
      "answer":    a,
      "answer_type": a_type,
      "content_type": str(item.get("question_type", "")),
      "modality":   "",
      "location":   "",
    })
  print(f" → {len(rows)} mẫu VQA-RAD")
  return rows


def merge_and_shuffle(slake: list, vqarad: list) -> list:
  merged = slake + vqarad
  random.seed(42)
  random.shuffle(merged)
  print(
    f"[2/5] Merged: {len(merged)} mẫu "
    f"({len(slake)} SLAKE + {len(vqarad)} VQA-RAD)"
  )
  return merged


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3 + 4 + 5: DỊCH + AUGMENT + QA
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
  try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    has = any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    if not has:
      print(f"⚠️ Chưa có model. Chạy: ollama pull {OLLAMA_MODEL}")
      return False
    print(f"✅ Ollama OK — model: {OLLAMA_MODEL}")
    return True
  except Exception:
    print("❌ Không kết nối được Ollama. Hãy mở app Ollama trước!")
    return False


def process_dataset(
  data: list,
  do_expand: bool = True,
  do_paraphrase: bool = True,
  do_back_translate: bool = True,
  bt_threshold: float = 0.3,
  checkpoint_path: str = CHECKPOINT,
  batch_log: int = 50,
) -> list:
  """
  Với mỗi mẫu:
   - Dịch question_vi + answer_vi (có validate output)
   - Sinh paraphrase_vi      (nếu do_paraphrase=True)
   - Back-translation + score   (nếu do_back_translate=True)
   - Gắn low_quality=True nếu score < bt_threshold
  Checkpoint tự động mỗi batch_log mẫu để resume khi bị ngắt.
  """
  # Load checkpoint
  done: dict = {}
  if os.path.exists(checkpoint_path):
    with open(checkpoint_path, encoding="utf-8") as f:
      done = json.load(f)
    print(f"[3/5] Resume: đã có {len(done)} mục trong checkpoint")

  def _save():
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
      json.dump(done, f, ensure_ascii=False, indent=2)

  to_do = [row for row in data if row["id"] not in done]
  print(f"[3/5] Cần xử lý: {len(to_do)} mẫu | đã bỏ qua: {len(data)-len(to_do)}")

  low_q_count = 0

  for i, row in enumerate(tqdm(to_do, desc="Dịch + augment")):
    rid = row["id"]

    # ── Dịch câu hỏi ──────────────────────────────────────────────────
    q_vi, q_valid = translate_question(row["question"])

    # ── Dịch câu trả lời ──────────────────────────────────────────────
    a_vi, a_valid = translate_answer(row["answer"])

    # ── Phóng to câu trả lời ──────────────────────────────────────────
    a_full_vi = ""
    if do_expand and a_valid and a_vi:
      a_full_vi = expand_answer(q_vi, a_vi)

    # ── Data Augmentation: Paraphrase ─────────────────────────────────
    para_questions_vi = []
    if do_paraphrase and q_valid and q_vi:
      para_questions_vi = paraphrase_question(q_vi)
      
    para_answers_vi = []
    if do_paraphrase and a_valid and a_vi:
      para_answers_vi = paraphrase_answer(q_vi, a_vi)

    # ── Back-translation QA ───────────────────────────────────────────
    bt_text = ""
    bt_score = 1.0
    low_q  = False
    if do_back_translate and q_valid and q_vi:
      bt_text, _ = back_translate(q_vi)
      bt_score  = _token_overlap(row["question"], bt_text)
      low_q   = bt_score < bt_threshold
      if low_q:
        low_q_count += 1

    done[rid] = {
      "question_vi":     q_vi,
      "question_vi_valid":  q_valid,
      "answer_vi":      a_vi,
      "answer_vi_valid":   a_valid,
      "answer_full_vi":    a_full_vi,
      "paraphrase_questions": para_questions_vi, # Mảng chứa ~4 câu hỏi biến thể
      "paraphrase_answers":  para_answers_vi,  # Mảng chứa ~4 câu trả lời biến thể
      "back_translation_en": bt_text,
      "bt_score":       round(bt_score, 3),
      "low_quality":     low_q,
    }

    if (i + 1) % batch_log == 0:
      _save()
      tqdm.write(
        f" [{i+1}/{len(to_do)}] low_quality so far: {low_q_count}"
      )

  _save()

  # Gắn kết quả vào từng row
  for row in data:
    row.update(done.get(row["id"], {}))

  total = len(data)
  print(
    f"[3/5] ✅ Xong! "
    f"Low quality: {low_q_count}/{total} "
    f"({low_q_count/max(total,1)*100:.1f}%)"
  )
  return data


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: SPLIT + PUSH
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(data: list) -> dict[str, list]:
  from collections import defaultdict
  
  # Gom nhóm dữ liệu theo tên ảnh (để đảm bảo không rò rỉ ảnh giữa các tập)
  images = defaultdict(list)
  for row in data:
    images[row["image_name"]].append(row)
    
  image_names = list(images.keys())
  random.seed(42)
  random.shuffle(image_names)
  
  # Yêu cầu: Chia train/val/test 80/10/10 và ảnh không trùng với train.
  num_images = len(image_names)
  n_train = int(num_images * 0.8)
  n_val  = int(num_images * 0.1)
  
  train_images = image_names[:n_train]
  val_images  = image_names[n_train : n_train + n_val]
  test_images = image_names[n_train + n_val:]
  
  splits = {"train": [], "validation": [], "test": []}
  
  for img in test_images:
    splits["test"].extend(images[img])
  for img in val_images:
    splits["validation"].extend(images[img])
  for img in train_images:
    splits["train"].extend(images[img])
    
  print(
    f"[4/5] Split (Image-disjoint) → "
    f"train: {len(splits['train'])} mẫu ({len(train_images)} ảnh) | "
    f"val: {len(splits['validation'])} mẫu ({len(val_images)} ảnh) | "
    f"test: {len(splits['test'])} mẫu ({len(test_images)} ảnh)"
  )
  return splits


def push_to_hub(splits: dict[str, list], repo_id: str) -> None:
  token = os.environ.get("HF_TOKEN")
  if not token:
    print(
      "⚠️ Chưa set HF_TOKEN — bỏ qua bước push.\n"
      "  Để push, chạy: export HF_TOKEN='hf_...'"
    )
    return
  hf_dict = DatasetDict(
    {k: Dataset.from_list(v) for k, v in splits.items()}
  )
  print(f"[5/5] Đang push lên: {repo_id} ...")
  hf_dict.push_to_hub(repo_id=repo_id, token=token, private=False)
  print(f"✅ Done! https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# THỐNG KÊ CUỐI
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(data: list) -> None:
  total  = len(data)
  closed = sum(1 for r in data if r.get("answer_type") == "CLOSED")
  low_q  = sum(1 for r in data if r.get("low_quality"))
  has_para = sum(1 for r in data if r.get("paraphrase_vi"))
  q_ok  = sum(1 for r in data if r.get("question_vi_valid"))
  a_ok  = sum(1 for r in data if r.get("answer_vi_valid"))
  slake_n = sum(1 for r in data if r["source"] == "slake")
  rad_n  = sum(1 for r in data if r["source"] == "vqa-rad")

  bar = "─" * 46
  print(f"\n{bar}")
  print(f"  THỐNG KÊ DATASET")
  print(bar)
  print(f" Tổng mẫu     : {total:>6}")
  print(f" SLAKE      : {slake_n:>6} ({slake_n/max(total,1)*100:.1f}%)")
  print(f" VQA-RAD     : {rad_n:>6} ({rad_n/max(total,1)*100:.1f}%)")
  print(bar)
  print(f" Closed (yes/no) : {closed:>6} ({closed/max(total,1)*100:.1f}%)")
  print(f" Open       : {total-closed:>6} ({(total-closed)/max(total,1)*100:.1f}%)")
  print(bar)
  print(f" question_vi OK  : {q_ok:>6} ({q_ok/max(total,1)*100:.1f}%)")
  print(f" answer_vi OK   : {a_ok:>6} ({a_ok/max(total,1)*100:.1f}%)")
  print(f" Có paraphrase  : {has_para:>6} ({has_para/max(total,1)*100:.1f}%)")
  print(f" Low quality (BT) : {low_q:>6} ({low_q/max(total,1)*100:.1f}%)")
  print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
  global OLLAMA_MODEL
  parser = argparse.ArgumentParser(
    description="Medical VQA Data Pipeline — Mac M4 / CUDA"
  )
  parser.add_argument(
    "--hf_repo", default="YOUR_USERNAME/medical-vqa-vi",
    help="HuggingFace dataset repo ID"
  )
  parser.add_argument(
    "--dry_run", action="store_true",
    help="Chỉ chạy 5 mẫu để test nhanh"
  )
  parser.add_argument(
    "--no_push", action="store_true",
    help="Không push lên HuggingFace"
  )
  parser.add_argument(
    "--no_paraphrase", action="store_true",
    help="Bỏ qua paraphrase augmentation"
  )
  parser.add_argument(
    "--no_back_translate", action="store_true",
    help="Bỏ qua back-translation QA"
  )
  parser.add_argument(
    "--bt_threshold", type=float, default=0.3,
    help="Ngưỡng back-translation overlap score (mặc định: 0.3)"
  )
  parser.add_argument(
    "--model", default=OLLAMA_MODEL,
    help=f"Ollama model name (mặc định: {OLLAMA_MODEL})"
  )
  parser.add_argument(
    "--checkpoint", default=CHECKPOINT,
    help="Đường dẫn file checkpoint"
  )
  args = parser.parse_args()

  OLLAMA_MODEL = args.model # type: ignore[assignment]

  # ── 1+2: Load & merge ────────────────────────────────────────────────
  slake = load_slake()
  vqarad = load_vqa_rad()
  merged = merge_and_shuffle(slake, vqarad)

  if args.dry_run:
    merged = merged[:5]
    print(f"[DRY RUN] Chỉ xử lý {len(merged)} mẫu.")

  # ── 3+4+5: Translate + augment ───────────────────────────────────────
  if not check_ollama():
    print("Pipeline dừng — Ollama chưa sẵn sàng.")
    return

  merged = process_dataset(
    merged,
    do_paraphrase   = not args.no_paraphrase,
    do_back_translate = not args.no_back_translate,
    bt_threshold    = args.bt_threshold,
    checkpoint_path  = args.checkpoint,
  )

  # ── Lưu JSON local ───────────────────────────────────────────────────
  out_path = Path("data/merged_vqa_vi.json")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
  print(f"\n[*] Đã lưu: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

  print_stats(merged)

  # ── 6: Split + push ──────────────────────────────────────────────────
  if not args.dry_run:
    splits = split_dataset(merged)
    if not args.no_push:
      push_to_hub(splits, repo_id=args.hf_repo)
    else:
      # Lưu từng split ra file riêng để tiện dùng
      for name, rows in splits.items():
        p = Path(f"data/{name}.json")
        with open(p, "w", encoding="utf-8") as f:
          json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[*] Lưu split '{name}': {p}")


if __name__ == "__main__":
  main()
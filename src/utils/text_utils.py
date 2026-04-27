import re
from collections import Counter
from underthesea import text_normalize

def normalize_answer(text: str) -> str:
    """
    Chuẩn hóa câu trả lời: Unicode normalization, viết thường, xóa dấu câu thừa.
    """
    if not text:
        return ""
    
    # Chuẩn hóa Unicode (hòa vs hoà, v.v.)
    text = text_normalize(str(text))
    
    text = text.lower().strip()
    # Xóa dấu câu ở cuối (nếu có)
    text = re.sub(r'[.\!?]$', '', text)
    return text.strip()

def majority_answer(answers: list[str]) -> str:
    """
    Trả về câu trả lời xuất hiện nhiều nhất trong danh sách.
    """
    if not answers:
        return ""
    if isinstance(answers, str):
        return answers
    counts = Counter([normalize_answer(a) for a in answers])
    return counts.most_common(1)[0][0]

def clean_vqa_output(text: str) -> str:
    """
    Làm sạch output từ PhoBERT tokenizer:
    - Xóa subword prefix (@@, ##)
    - Xóa khoảng trắng thừa
    - Capitalize đầu câu
    """
    if not text:
        return ""
    # Xóa subword prefix của PhoBERT (@@) và BERT (##)
    text = re.sub(r'@@\s?', '', text)
    text = re.sub(r'##_?', '', text)
    # Xóa ký tự thừa
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize đầu câu
    return text[:1].upper() + text[1:] if text else text

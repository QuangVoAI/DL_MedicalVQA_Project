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

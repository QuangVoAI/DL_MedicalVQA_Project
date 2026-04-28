import re
from collections import Counter

from underthesea import text_normalize as uts_text_normalize, word_tokenize

_MEDICAL_TERM_MAP = {
    "xray": "x-quang",
    "x ray": "x-quang",
    "x-ray": "x-quang",
    "x quang": "x-quang",
    "mri scan": "mri",
    "mr": "mri",
    "ct scan": "ct",
    "ct-scan": "ct",
    "cat scan": "ct",
    "computed tomography": "ct",
    "transverse  plane": "mặt phẳng ngang",
    "transverse plane": "mặt phẳng ngang",
    "coronal plane": "mặt phẳng vành",
    "sagittal plane": "mặt phẳng dọc",
    "elliptical": "hình elip",
    "spleen": "lách",
    "liver": "gan",
    "lung": "phổi",
    "lungs": "phổi",
    "heart": "tim",
    "brain": "não",
    "kidney": "thận",
    "bladder": "bàng quang",
    "cardiomegaly": "tim to",
}

_NON_CANONICAL_ALIASES = {
    "xray",
    "x ray",
    "x-ray",
    "x quang",
    "mri scan",
    "mr",
    "ct scan",
    "ct-scan",
    "cat scan",
    "computed tomography",
    "transverse plane",
    "coronal plane",
    "sagittal plane",
    "elliptical",
    "spleen",
    "liver",
    "lung",
    "lungs",
    "heart",
    "brain",
    "kidney",
    "bladder",
    "cardiomegaly",
}


def text_normalize(text: str) -> str:
    """Wrapper để chuẩn hóa Unicode và spacing cho tiếng Việt."""
    if not text:
        return ""
    return uts_text_normalize(str(text))


def normalize_answer(text: str) -> str:
    """
    Chuẩn hóa đáp án về dạng canonical để train/eval ổn định.
    """
    if not text:
        return ""

    text = text_normalize(str(text))
    text = text.replace("_", " ")
    text = text.lower().strip()
    text = re.sub(r"[@#]{1,2}", " ", text)
    text = re.sub(r"[“”\"']", "", text)
    text = re.sub(r"[,:;!?()\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for src, dst in sorted(_MEDICAL_TERM_MAP.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.\-]+$", "", text).strip()
    return text


def _tokenize_vietnamese_words(text: str) -> list[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []
    try:
        tokens = word_tokenize(normalized)
        return [token.strip() for token in tokens if token and token.strip()]
    except Exception:
        return normalized.split()


def count_words(text: str) -> int:
    return len(_tokenize_vietnamese_words(text))


def _trim_to_max_words(text: str, max_words: int) -> str:
    words = _tokenize_vietnamese_words(text)
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _choose_best_answer_text(answer_vi: str, answer_full_vi: str, max_words: int) -> str:
    short_answer = normalize_answer(answer_vi)
    full_answer = normalize_answer(answer_full_vi)

    if short_answer and count_words(short_answer) <= max_words:
        return short_answer
    if full_answer:
        return _trim_to_max_words(full_answer, max_words)
    return _trim_to_max_words(short_answer, max_words)


def get_target_answer(item: dict, max_words: int = 10) -> str:
    """
    Chọn target answer ngắn, chuẩn hóa và không vượt quá số từ cho phép.
    """
    answer_vi = item.get("answer_vi", "")
    answer_full_vi = item.get("answer_full_vi", "")
    answer = _choose_best_answer_text(answer_vi, answer_full_vi, max_words=max_words)
    if answer:
        return answer
    fallback = item.get("answer", "")
    return _trim_to_max_words(fallback, max_words)


def postprocess_answer(text: str, max_words: int = 10) -> str:
    """
    Chuẩn hóa output model và cắt ngắn về tối đa `max_words`.
    Không mở rộng câu trả lời để tránh làm xấu exact match.
    """
    if not text:
        return ""
    text = clean_vqa_output(text)
    text = normalize_answer(text)
    return _trim_to_max_words(text, max_words=max_words)


def is_medical_term_compliant(text: str) -> bool:
    """
    Heuristic nhẹ: không còn alias y khoa phổ biến chưa canonicalize.
    """
    normalized = normalize_answer(text)
    if not normalized:
        return False
    for alias in _NON_CANONICAL_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return False
    return True


def majority_answer(answers: list[str]) -> str:
    """
    Trả về câu trả lời xuất hiện nhiều nhất trong danh sách.
    """
    if not answers:
        return ""
    if isinstance(answers, str):
        return normalize_answer(answers)
    counts = Counter([normalize_answer(a) for a in answers])
    return counts.most_common(1)[0][0]


def clean_vqa_output(text: str) -> str:
    """
    Làm sạch output từ tokenizer trước khi postprocess.
    """
    if not text:
        return ""
    text = re.sub(r"@@\s?", "", text)
    text = re.sub(r"##_?", "", text)
    text = re.sub(r"\b(answer|response|assistant|trả lời)\b\s*:?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

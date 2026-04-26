import re
import collections

def normalize_answer(s):
    """
    Chuẩn hóa câu trả lời: viết thường, bỏ dấu câu, bỏ mạo từ...
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def majority_answer(answer_list):
    """
    Lấy câu trả lời xuất hiện nhiều nhất trong danh sách (Voting).
    """
    if not answer_list:
        return ""
    count = collections.Counter([normalize_answer(a) for a in answer_list])
    return count.most_common(1)[0][0]

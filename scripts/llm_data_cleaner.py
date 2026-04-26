import json
import requests
import os
from tqdm import tqdm

# Cấu hình Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b" # Hoặc model bạn đang dùng
INPUT_FILE = "data/merged_vqa_vi_cleaned.json"

PROMPT_TEMPLATE = """Bạn là một chuyên gia chẩn đoán hình ảnh.
Hãy dịch câu hỏi và câu trả lời y khoa sau đây sang tiếng Việt chuẩn chuyên ngành và tạo ra 4 biến thể (paraphrase) cho mỗi câu.

CÂU GỐC (TIẾNG ANH):
Question: {en_q}
Answer: {en_a}

YÊU CẦU TRẢ VỀ ĐỊNH DẠNG JSON:
{{
  "question_vi": "Bản dịch câu hỏi chuẩn y khoa",
  "paraphrase_questions": ["Biến thể 1", "Biến thể 2", "Biến thể 3", "Biến thể 4"],
  "paraphrase_answers": ["Biến thể 1", "Biến thể 2", "Biến thể 3", "Biến thể 4"],
  "back_translation_en": "Dịch ngược lại câu hỏi sang tiếng Anh"
}}"""

def call_qwen(en_q, en_a):
    prompt = PROMPT_TEMPLATE.format(en_q=en_q, en_a=en_a)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return json.loads(r.json().get("response", "{}"))
    except Exception as e:
        print(f"[WARNING] Lỗi Qwen: {e}")
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Đang bắt đầu làm sạch dữ liệu bằng {MODEL_NAME}...")
    
    # Chỉ xử lý các mẫu cần thiết hoặc bạn có thể chọn một khoảng cụ thể
    # Ở đây tôi sẽ demo xử lý các mẫu mà bạn cảm thấy chưa ổn
    for i in tqdm(range(len(data))): # Xử lý toàn bộ 6712 mẫu
        item = data[i]
        res = call_qwen(item['question'], item['answer'])
        if res:
            item['question_vi'] = res.get('question_vi', item['question_vi'])
            item['paraphrase_questions'] = res.get('paraphrase_questions', [])
            item['paraphrase_answers'] = res.get('paraphrase_answers', [])
            item['back_translation_en'] = res.get('back_translation_en', item['question'])

        # Lưu tạm sau mỗi 10 mẫu để tránh mất dữ liệu
        if i % 10 == 0:
            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("[SUCCESS] Đã làm sạch dữ liệu thành công bằng Qwen!")

if __name__ == "__main__":
    main()

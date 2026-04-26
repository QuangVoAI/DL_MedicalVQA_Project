import json
import random
import os

def create_manual_test_set(input_path="data/judge_results.json", output_path="data/manual_test_50.json", num_samples=50):
    """
    Trích xuất ngẫu nhiên 50 mẫu để thực hiện Human Review (Kiểm tra thủ công).
    """
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy {input_path}. Hãy chạy llm_judge_eval.py trước.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_keys = list(data.keys())
    # Chọn ngẫu nhiên 50 ID
    selected_keys = random.sample(all_keys, min(num_samples, len(all_keys)))
    
    manual_data = []
    for key in selected_keys:
        item = data[key]
        # Tạo cấu trúc để bạn dễ dàng sửa tay
        manual_data.append({
            "id": key,
            "image": item["original_data"].get("image_name"),
            "question_en": item["original_data"].get("back_translation_en"),
            "question_vi_ai": item["original_data"].get("question_vi"),
            "question_vi_human": "", # CHỖ NÀY BẠN SẼ ĐIỀN CÂU BẠN TỰ SỬA
            "answer_vi_ai": item["original_data"].get("answer_vi"),
            "answer_vi_human": "",   # CHỖ NÀY BẠN SẼ ĐIỀN CÂU BẠN TỰ SỬA
            "notes": ""              # Ghi chú tại sao bạn sửa (nếu có)
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manual_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã tạo file: {output_path}")
    print(f"👉 Nhiệm vụ của bạn: Mở file này ra và điền vào các trường '_human' để hoàn tất yêu cầu đề bài.")

if __name__ == "__main__":
    create_manual_test_set()

import json
import random
import os

def load_predictions(file_path):
    """Load JSON predictions."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Không tìm thấy file: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def manual_review(samples, preds_b2, preds_dpo, num_samples=20):
    """
    So sánh SFT (B2) vs DPO. Lưu lại sở thích dựa trên tính chính xác y khoa.
    """
    results = {"B2_wins": 0, "DPO_wins": 0, "Tie": 0}
    
    # Lấy các index ngẫu nhiên
    indices = list(range(len(samples)))
    random.shuffle(indices)
    review_indices = indices[:min(num_samples, len(samples))]
    
    print("\n" + "="*50)
    print(f"BẮT ĐẦU PHIÊN ĐÁNH GIÁ THỦ CÔNG ({len(review_indices)} câu hỏi)")
    print("Mục tiêu: Đánh giá xem DPO có sinh ra câu trả lời tốt hơn B2 không.")
    print("="*50)
    
    for i, idx in enumerate(review_indices):
        sample = samples[idx]
        b2_ans = preds_b2[idx].get("predicted", "") if idx < len(preds_b2) else "N/A"
        dpo_ans = preds_dpo[idx].get("predicted", "") if idx < len(preds_dpo) else "N/A"
        
        # Ground Truth
        q_en = sample.get("question", sample.get("raw_questions", ""))
        gt_en = sample.get("answer", sample.get("raw_answers", ""))
        gt_vi = sample.get("answer_vi", "")
        
        print(f"\n[Câu {i+1}/{len(review_indices)}]")
        print(f"Câu hỏi (En): {q_en}")
        print(f"Đáp án chuẩn (Vi): {gt_vi}")
        print("-" * 30)
        
        # Randomize order to prevent bias (Blind Test)
        is_b2_first = random.choice([True, False])
        
        if is_b2_first:
            print(f"Mô hình 1: {b2_ans}")
            print(f"Mô hình 2: {dpo_ans}")
        else:
            print(f"Mô hình 1: {dpo_ans}")
            print(f"Mô hình 2: {b2_ans}")
            
        print("-" * 30)
        choice = ""
        while choice not in ['1', '2', '3']:
            choice = input("Mô hình nào tốt hơn? (1: Mô hình 1 | 2: Mô hình 2 | 3: Hòa): ").strip()
            
        if choice == '3':
            results["Tie"] += 1
        elif (choice == '1' and is_b2_first) or (choice == '2' and not is_b2_first):
            results["B2_wins"] += 1
        else:
            results["DPO_wins"] += 1
            
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ THỦ CÔNG (BLIND TEST)")
    print("="*50)
    print(f"B2 thắng:  {results['B2_wins']}")
    print(f"DPO thắng: {results['DPO_wins']}")
    print(f"Hòa:       {results['Tie']}")
    print("="*50)
    
    if results['DPO_wins'] > results['B2_wins']:
        print("=> Kết luận: DPO ĐÃ CẢI THIỆN ĐƯỢC CHẤT LƯỢNG SINH VĂN BẢN (RLHF hoạt động tốt!)")
    elif results['DPO_wins'] < results['B2_wins']:
        print("=> Kết luận: DPO sinh ra kết quả kém hơn B2 (Cần chỉnh lại tham số Beta hoặc dữ liệu Preference).")
    else:
        print("=> Kết luận: B2 và DPO không có sự chênh lệch rõ rệt.")
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/vqa_rad.json", help="Path to ground truth dataset")
    parser.add_argument("--b2", type=str, default="results/predictions/B2_predictions.json")
    parser.add_argument("--dpo", type=str, default="results/predictions/DPO_predictions.json")
    parser.add_argument("--n", type=int, default=20, help="Số lượng câu cần đánh giá")
    args = parser.parse_args()
    
    # Load data
    samples = load_predictions(args.data)
    preds_b2 = load_predictions(args.b2)
    preds_dpo = load_predictions(args.dpo)
    
    if samples and preds_b2 and preds_dpo:
        manual_review(samples, preds_b2, preds_dpo, num_samples=args.n)
    else:
        print("Vui lòng chạy đánh giá và lưu kết quả predict của B2 và DPO ra file JSON trước khi dùng script này.")

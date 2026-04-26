import os
import json
from datasets import load_dataset, Dataset, DatasetDict, Image
from huggingface_hub import snapshot_download
from tqdm import tqdm
from pathlib import Path

# CẤU HÌNH
JSON_PATH = "data/merged_vqa_vi.json"
HF_REPO = "SpringWang08/medical-vqa-vi"
TOKEN = os.environ.get("HF_TOKEN", "") # Dùng token bạn đã cung cấp

def push_with_images():
    print("📥 Bước 1: Đang tải toàn bộ file ảnh SLAKE từ Hugging Face (Snapshot)...")
    # Tải toàn bộ repo Slake về thư mục tạm
    slake_dir = snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset")
    
    # GIẢI NÉN ẢNH SLAKE
    slake_img_dir = Path(slake_dir) / "unzipped_imgs"
    if not slake_img_dir.exists():
        zip_path = Path(slake_dir) / "imgs.zip"
        if zip_path.exists():
            import zipfile
            print(f"📦 Đang giải nén {zip_path}... (việc này có thể mất vài phút)")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(slake_img_dir)
            print("✅ Giải nén thành công.")
    
    print("📥 Bước 2: Tải bộ VQA-RAD chuẩn (đã có sẵn cột Image)...")
    vqarad_ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    
    # Tạo cache cho VQA-RAD bằng QUESTION (vì không có image_name)
    vqarad_cache = {item['question'].lower().strip(): item['image'] for item in tqdm(vqarad_ds, desc="Caching VQA-RAD")}

    print("📝 Bước 3: Khớp bản dịch với file ảnh thực tế...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        translated_data = json.load(f)

    final_rows = []
    for row in tqdm(translated_data, desc="Merging"):
        source = row['source']
        img_name = row['image_name']
        
        if source == "slake":
            # Tìm trong thư mục vừa giải nén
            possible_paths = [
                slake_img_dir / img_name,
                slake_img_dir / "imgs" / img_name
            ]
            
            found_path = None
            for p in possible_paths:
                if p.exists():
                    found_path = str(p)
                    break
            
            if found_path:
                row['image'] = found_path # Datasets sẽ tự load từ path này
                final_rows.append(row)
        
        elif source == "vqa-rad":
            q_key = row['question'].lower().strip()
            if q_key in vqarad_cache:
                row['image'] = vqarad_cache[q_key]
                final_rows.append(row)

    print(f"✅ Đã chuẩn bị xong {len(final_rows)} mẫu dữ liệu kèm ảnh.")

    # 4. Định nghĩa cấu trúc dữ liệu (Features) để tránh lỗi ArrowTypeError
    from datasets import Features, Value, List as fList, Image as fImage
    features = Features({
        "image": fImage(),
        "question_vi": Value("string"),
        "answer_vi": Value("string"),
        "answer_full_vi": Value("string"),
        "id": Value("string"),
        "source": Value("string"),
        "modality": Value("string"),
        "location": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "answer_type": Value("string"),
        "content_type": Value("string"),
        "paraphrase_questions": fList(Value("string")),
        "paraphrase_answers": fList(Value("string")),
        "image_name": Value("string")
    })

    # Tạo Dataset với cấu trúc đã định nghĩa
    # Chúng ta lọc bỏ các cột dư thừa ngay từ bước tạo list để khớp với features
    final_rows_cleaned = []
    for row in final_rows:
        clean_row = {k: row[k] for k in features.keys() if k in row}
        final_rows_cleaned.append(clean_row)

    ds = Dataset.from_list(final_rows_cleaned, features=features)

    print("⚖️ Bước 5: Chia tập Train/Val/Test...")
    train_test = ds.train_test_split(test_size=0.2, seed=42)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)

    final_ds_dict = DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })

    print(f"🚀 Bước 6: Đẩy lên Hub: {HF_REPO}")
    final_ds_dict.push_to_hub(HF_REPO, token=TOKEN)
    print(f"🎉 THÀNH CÔNG! Dataset của bạn hiện đã có đầy đủ ảnh.")

if __name__ == "__main__":
    push_with_images()

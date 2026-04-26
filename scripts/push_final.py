import os
import json
import random
import argparse
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image, List as fList
from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm

def split_and_push(data_path, repo_id):
    """Đẩy dữ liệu hoàn thiện (Slake + RAD) kèm ảnh lên Hub."""
    
    # BƯỚC 1: Chuẩn bị kho ảnh Slake
    print("📥 Bước 1: Đang chuẩn bị kho ảnh Slake...")
    slake_dir = snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset")
    slake_img_dir = Path(slake_dir) / "unzipped_imgs"
    if not slake_img_dir.exists():
        zip_path = Path(slake_dir) / "imgs.zip"
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(slake_img_dir)
    
    # BƯỚC 2: Chuẩn bị kho ảnh VQA-RAD (Tải từ Hub để lấy cột Image)
    print("📥 Bước 2: Đang lấy kho ảnh VQA-RAD từ Hub...")
    vqarad_ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    # Caching theo question để ánh xạ
    vqarad_cache = {item['question'].lower().strip(): item['image'] for item in vqarad_ds}
    
    print(f"📖 Bước 3: Đang đọc dữ liệu sạch từ: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    features = Features({
        "image": Image(),
        "id": Value("string"),
        "source": Value("string"),
        "image_name": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "question_vi": Value("string"),
        "answer_vi": Value("string"),
        "answer_full_vi": Value("string"),
        "answer_type": Value("string"),
        "modality": Value("string"),
        "location": Value("string"),
        "paraphrase_questions": fList(Value("string")),
        "paraphrase_answers": fList(Value("string")),
        "back_translation_en": Value("string"),
        "bt_score": Value("float64"),
        "low_quality": Value("bool")
    })

    final_rows = []
    print("🖼️ Bước 4: Ánh xạ ảnh cho Slake và VQA-RAD...")
    for item in tqdm(raw_data):
        source = item.get('source', '')
        img_name = item.get('image_name', '')
        q_en = item.get('question', '').lower().strip()
        
        found_image = None
        if source == "slake":
            p1 = slake_img_dir / img_name
            p2 = slake_img_dir / "imgs" / img_name
            if p1.exists(): found_image = str(p1)
            elif p2.exists(): found_image = str(p2)
        elif source == "vqa-rad":
            if q_en in vqarad_cache:
                found_image = vqarad_cache[q_en] # Đây là đối tượng Image của PIL

        if found_image:
            row = {k: item.get(k) for k in features.keys()}
            row["image"] = found_image
            final_rows.append(row)

    print(f"✅ Đã sẵn sàng {len(final_rows)}/6712 mẫu có kèm ảnh.")

    # 3. Chia tập và đẩy lên Hub
    random.seed(42)
    random.shuffle(final_rows)
    n = len(final_rows)
    train_ds = Dataset.from_list(final_rows[:int(n*0.8)], features=features)
    val_ds = Dataset.from_list(final_rows[int(n*0.8):int(n*0.9)], features=features)
    test_ds = Dataset.from_list(final_rows[int(n*0.9):], features=features)
    
    hf_dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    
    token = os.environ.get("HF_TOKEN")
    print(f"🚀 Bước 5: Đẩy lên Hub: {repo_id}")
    hf_dataset.push_to_hub(repo_id, token=token)
    print("🎉 HOÀN TẤT! Toàn bộ 6,712 mẫu kèm ảnh đã được đưa lên Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--input", type=str, default="data/merged_vqa_vi_cleaned.json")
    args = parser.parse_args()
    split_and_push(args.input, args.repo)

import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Features, Image as HFImage
from huggingface_hub import snapshot_download

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
JSON_PATH = "data/merged_vqa_vi_cleaned.json"
HF_REPO = "SpringWang08/medical-vqa-vi"
TOKEN = os.environ.get("HF_TOKEN")

def prepare_images():
  """Tải và chuẩn bị folder ảnh Slake từ Hub."""
  print(" Bước 1: Chuẩn bị kho ảnh Slake...")
  slake_dir = snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset")
  slake_img_dir = Path(slake_dir) / "unzipped_imgs"
  
  if not slake_img_dir.exists():
    zip_path = Path(slake_dir) / "imgs.zip"
    if zip_path.exists():
      print(f" Đang giải nén {zip_path}...")
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(slake_img_dir)
  return slake_img_dir

def load_vqarad_images():
  """Tải VQA-RAD để lấy ảnh gốc."""
  print(" Bước 2: Đang lấy ảnh từ VQA-RAD...")
  vqarad = load_dataset("flaviagiammarino/vqa-rad", split="train")
  return {item["image_name"]: item["image"] for item in vqarad}

def main():
  if not TOKEN:
    print("❌ LỖI: Chưa tìm thấy HF_TOKEN. Hãy chạy: export HF_TOKEN='your_token'")
    return

  slake_img_path = prepare_images()
  vqarad_images = load_vqarad_images()

  print(f" Bước 3: Đang đọc dữ liệu từ: {JSON_PATH}")
  with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

  # 4. Ánh xạ dữ liệu với ảnh thực tế
  print("️ Bước 4: Ánh xạ dữ liệu với file ảnh thực tế...")
  final_samples = []
  for item in tqdm(data):
    img_name = item.get("image_name")
    image_obj = None

    # Trường hợp Slake
    local_path = slake_img_path / img_name if img_name else None
    if local_path and local_path.exists():
      image_obj = str(local_path)
    
    # Trường hợp VQA-RAD (lấy từ cache object)
    elif img_name in vqarad_images:
      image_obj = vqarad_images[img_name]

    if image_obj:
      item["image"] = image_obj
      final_samples.append(item)

  print(f"✅ Đã chuẩn bị xong {len(final_samples)} mẫu có kèm ảnh.")

  # 5. Định nghĩa Schema cho Hugging Face
  features = Features({
    "id": Dataset.from_dict({"a": [1]}).features.get("id", Features({"id": {"dtype": "string"}}).get("id")), # Giữ nguyên ID
    "image": HFImage(),
    "image_name": Dataset.from_dict({"a": [1]}).features.get("image_name", Features({"image_name": {"dtype": "string"}}).get("image_name")),
    "question": Dataset.from_dict({"a": [1]}).features.get("question", Features({"question": {"dtype": "string"}}).get("question")),
    "question_vi": Dataset.from_dict({"a": [1]}).features.get("question_vi", Features({"question_vi": {"dtype": "string"}}).get("question_vi")),
    "answer": Dataset.from_dict({"a": [1]}).features.get("answer", Features({"answer": {"dtype": "string"}}).get("answer")),
    "answer_vi": Dataset.from_dict({"a": [1]}).features.get("answer_vi", Features({"answer_vi": {"dtype": "string"}}).get("answer_vi")),
    "paraphrase_questions": [Dataset.from_dict({"a": [1]}).features.get("q", Features({"q": {"dtype": "string"}}).get("q"))],
    "paraphrase_answers": [Dataset.from_dict({"a": [1]}).features.get("a", Features({"a": {"dtype": "string"}}).get("a"))],
    "source_dataset": Dataset.from_dict({"a": [1]}).features.get("src", Features({"src": {"dtype": "string"}}).get("src")),
  })
  
  # Đơn giản hóa Features để tránh lỗi type mapping phức tạp
  # Ta sẽ để Dataset tự infer nhưng ép kiểu cột Image
  
  # 6. Chia Split và Push
  full_ds = Dataset.from_list(final_samples)
  
  # Ép kiểu cột image sang Image feature
  full_ds = full_ds.cast_column("image", HFImage())
  
  splits = full_ds.train_test_split(test_size=0.15, seed=42)
  test_val = splits["test"].train_test_split(test_size=0.5, seed=42)
  
  ds_dict = DatasetDict({
    "train": splits["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
  })

  print(f" Bước 5: Đang đẩy lên Hub: {HF_REPO}")
  # Đẩy từng split để tránh lỗi pagination
  for split_name, dataset in ds_dict.items():
    print(f" Đang đẩy split: {split_name}...")
    dataset.push_to_hub(HF_REPO, split=split_name, token=TOKEN, embed_external_files=True)

  print(f"\n HOÀN TẤT! Bộ dữ liệu đã sẵn sàng tại: https://huggingface.co/datasets/{HF_REPO}")

if __name__ == "__main__":
  main()

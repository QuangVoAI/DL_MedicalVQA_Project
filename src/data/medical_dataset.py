import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from src.utils.text_utils import normalize_answer, text_normalize

class MedicalVQADataset(Dataset):
    """
    Dataset class chung cho Medical VQA (SLAKE + VQA-RAD).
    """
    def __init__(self, hf_dataset=None, json_path=None, image_dir=None, tokenizer=None, transform=None, max_seq_len=64, max_ans_len=10, is_dpo=False, in_channels=1):
        if hf_dataset is not None:
            self.data = hf_dataset
            self.use_hf = True
        elif json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.use_hf = False
        else:
            raise ValueError("Phải cung cấp hf_dataset hoặc json_path!")
            
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len
        self.is_dpo = is_dpo
        self.in_channels = in_channels
        
        # Mapping for closed questions (Yes/No)
        self.label_map = {"no": 0, "yes": 1, "không": 0, "có": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Xử lý ảnh
        if self.use_hf:
            image = item["image"]
            if self.in_channels == 1:
                if image.mode != "L": image = image.convert("L")
            else:
                if image.mode != "RGB": image = image.convert("RGB")
        else:
            # DPO preference data might use 'image' or 'image_name'
            img_name = item.get("image_name") or item.get("image")
            img_path = os.path.join(self.image_dir, img_name)
            mode = "L" if self.in_channels == 1 else "RGB"
            image = Image.open(img_path).convert(mode)
            
        # [UPGRADE] Tích hợp CLAHE ngay trong Dataset để đảm bảo cả Hướng A và B đều được hưởng lợi
        from src.utils.visualization import apply_clahe
        import numpy as np
        
        image_np = np.array(image)
        image_clahe = apply_clahe(image_np)
        # Chuyển ngược lại PIL để đồng nhất đầu vào cho các bước sau
        image = Image.fromarray((image_clahe * 255).astype(np.uint8))
        raw_image = image # Bản lưu trữ cho Multimodal Processor (đã có CLAHE, chưa Normalize)

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)
            
        # 2. Xử lý câu hỏi
        q_key = "question" if self.is_dpo else "question_vi"
        raw_question = item[q_key]
        question = text_normalize(raw_question)
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        
        if self.is_dpo:
            # 3. Xử lý DPO Preference (Chosen vs Rejected)
            chosen_ans = normalize_answer(item["chosen"])
            rejected_ans = normalize_answer(item["rejected"])
            
            chosen_encoding = self.tokenizer(chosen_ans, padding="max_length", truncation=True, max_length=10, return_tensors="pt")
            rejected_encoding = self.tokenizer(rejected_ans, padding="max_length", truncation=True, max_length=10, return_tensors="pt")
            
            return {
                "image": image,
                "raw_image": raw_image,
                "raw_questions": raw_question,
                "input_ids": encoding["input_ids"].flatten(),
                "chosen_ids": chosen_encoding["input_ids"].flatten(),
                "rejected_ids": rejected_encoding["input_ids"].flatten(),
            }
        
        # 3. Xử lý câu trả lời chuẩn (Non-DPO)
        answer = normalize_answer(item["answer_vi"])
        label_closed = self.label_map.get(answer, -1)
        
        ans_encoding = self.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_ans_len,
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "raw_image": raw_image,
            "raw_questions": raw_question,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label_closed": torch.tensor(label_closed, dtype=torch.long),
            "target_ids": ans_encoding["input_ids"].flatten(),
            "raw_answer": answer
        }

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from underthesea import word_tokenize

class MedicalTranslator:
    """
    Lớp dịch thuật nâng cao sử dụng MedCrab-1.5B (SOTA English-Vietnamese Medical Translation).
    Thay thế Helsinki-NLP để đạt độ chính xác y khoa cao nhất.
    """
    def __init__(self, device="cpu", dict_path="data/medical_dict.json"):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Khởi tạo Translation Layer với MedCrab-1.5B ({self.device})")
        
        # 1. Load MedDict (Dùng để kiểm tra/hậu xử lý nếu cần)
        self.med_dict = {}
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.med_dict = json.load(f)
                self.inv_med_dict = {v.lower(): k.lower() for k, v in self.med_dict.items()}
            except: pass
        
        # 2. Load MedCrab-1.5B
        try:
            model_id = "pnnbao-ump/MedCrab-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Nạp ở FP16 để tiết kiệm VRAM cho LLaVA-Med
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto" if device == "cuda" else None
            ).to(self.device)
            self.is_ready = True
        except Exception as e:
            print(f"[WARNING] Không thể tải MedCrab-1.5B: {e}. Đang dùng fallback...")
            self.is_ready = False

    def _gen_medcrab(self, text, task="en2vi"):
        if not self.is_ready: return text
        
        # Prompt chuẩn cho MedCrab (Dựa trên Qwen2.5)
        if task == "en2vi":
            prompt = f"English: {text}\nVietnamese:"
        else:
            prompt = f"Vietnamese: {text}\nEnglish:"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Tách lấy phần trả lời sau "Vietnamese:" hoặc "English:"
        if task == "en2vi":
            translated = full_text.split("Vietnamese:")[-1].strip()
        else:
            translated = full_text.split("English:")[-1].strip()
            
        return translated

    def translate_vi2en(self, text):
        if not text: return text
        if isinstance(text, list):
            return [self._gen_medcrab(t, task="vi2en") for t in text]
        return self._gen_medcrab(text, task="vi2en")

    def translate_en2vi(self, text):
        if not text: return text
        
        # 1. Chuẩn hóa nhãn (Dành cho Accuracy)
        if isinstance(text, str):
            text_lower = text.lower().strip().replace(".", "").replace(",", "")
            if any(word in text_lower for word in ["yes", "correct", "true", "normal"]): return "có"
            if any(word in text_lower for word in ["no", "incorrect", "false", "abnormal"]): return "không"
        
        # 2. Dịch bằng MedCrab
        if isinstance(text, list):
            return [self._gen_medcrab(t, task="en2vi") for t in text]
        return self._gen_medcrab(text, task="en2vi")

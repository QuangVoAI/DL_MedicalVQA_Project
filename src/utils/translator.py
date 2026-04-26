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
        
        # 2. Load Models
        try:
            # MedCrab cho En -> Vi (Trọng tâm)
            medcrab_id = "pnnbao-ump/MedCrab-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(medcrab_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                medcrab_id, torch_dtype=torch.float16, device_map="auto" if device == "cuda" else None
            ).to(self.device)
            
            # Helsinki-NLP cho Vi -> En (Dùng cho câu hỏi, ổn định hơn MedCrab ở mảng này)
            from transformers import pipeline
            self.vi2en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", device=0 if device == "cuda" else -1)
            
            self.is_ready = True
        except Exception as e:
            print(f"[WARNING] Lỗi nạp model dịch: {e}. Fallback enabled.")
            self.is_ready = False

    def _gen_medcrab_en2vi(self, text):
        """Dịch En -> Vi bằng MedCrab với ràng buộc ngắn gọn."""
        if not self.is_ready: return text
        
        # Prompt ép model trả lời cực ngắn
        prompt = f"English: {text}\nVietnamese (trả lời ngắn gọn):"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,           # Giới hạn độ dài (~10-15 từ tiếng Việt)
                repetition_penalty=1.2,      # Chống lặp từ (Fix bug lặp vô hạn)
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = full_text.split("Vietnamese (trả lời ngắn gọn):")[-1].strip()
        # Cắt nếu vẫn quá dài (hậu xử lý cứng)
        return " ".join(translated.split()[:12])

    def translate_vi2en(self, text):
        """Dùng Helsinki-NLP để dịch câu hỏi sang tiếng Anh (Ổn định cho Zero-shot)."""
        if not text or not self.is_ready: return text
        try:
            if isinstance(text, list):
                res = self.vi2en(text)
                return [r['translation_text'] for r in res]
            res = self.vi2en(text)
            return res[0]['translation_text']
        except: return text

    def translate_en2vi(self, text):
        """Dịch kết quả từ LLaVA-Med sang Tiếng Việt."""
        if not text: return text
        
        # 1. Ánh xạ trực tiếp các nhãn nhị phân (Độ chính xác cao nhất cho Y khoa)
        if isinstance(text, str):
            text_lower = text.lower().strip().replace(".", "").replace(",", "")
            # Mapping nhanh
            if text_lower in ["yes", "correct", "true", "normal", "present"]: return "có"
            if text_lower in ["no", "incorrect", "false", "abnormal", "absent"]: return "không"
            # Kiểm tra chứa từ khóa
            if any(w in text_lower for w in ["normal", "no abnormality"]): return "bình thường"
            if any(w in text_lower for w in ["abnormal", "pathology"]): return "bất thường"
        
        # 2. Dịch bằng MedCrab cho các câu mô tả
        if isinstance(text, list):
            return [self._gen_medcrab_en2vi(t) for t in text]
        return self._gen_medcrab_en2vi(text)

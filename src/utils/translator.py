import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class MedicalTranslator:
    """
    Lớp dịch thuật nâng cao với cơ chế Lazy Loading để tránh OOM.
    """
    def __init__(self, device="cpu", dict_path="data/medical_dict.json"):
        self.device_type = device
        # Tối ưu cho Dual GPU: Nếu có 2 GPU, đẩy Translator sang GPU thứ 2 (cuda:1)
        if torch.cuda.device_count() > 1:
            self.device = torch.device("cuda:1")
            print(f"[INFO] Dual-GPU detected. Moving Translator to {self.device}")
        else:
            self.device = torch.device("cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu")
        
        self.dict_path = dict_path
        
        self.tokenizer = None
        self.model = None
        self.vi2en = None
        self.is_ready = False
        self.med_dict = {}
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.med_dict = json.load(f)
            except: pass

    def _lazy_load(self):
        """Chỉ nạp model khi thực sự cần dịch."""
        if self.is_ready: return
        
        print(f"[INFO] Đang nạp Translation Models (Lazy Load)...")
        try:
            # 1. MedCrab (4-bit)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            medcrab_id = "pnnbao-ump/MedCrab-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(medcrab_id)
            
            # Nếu dùng CUDA, ép model vào đúng GPU đã chọn (ví dụ cuda:1)
            d_map = {"": self.device} if self.device.type == "cuda" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                medcrab_id, 
                quantization_config=bnb_config,
                device_map=d_map,
                low_cpu_mem_usage=True
            )
            
            # 2. Helsinki-NLP (Ép chạy trên CPU để tiết kiệm VRAM cho LLaVA)
            from transformers import pipeline
            try:
                # Thử các task name khác nhau tùy theo phiên bản transformers
                self.vi2en = pipeline("translation_vi_to_en", model="Helsinki-NLP/opus-mt-vi-en", device=-1)
            except:
                try:
                    self.vi2en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", device=-1)
                except Exception as e_inner:
                    print(f"[DEBUG] Cố gắng nạp pipeline tự động...")
                    self.vi2en = pipeline(model="Helsinki-NLP/opus-mt-vi-en", device=-1)
            
            self.is_ready = True
            print("[INFO] Translation Layer đã sẵn sàng.")
        except Exception as e:
            print(f"[WARNING] Lỗi Lazy Load: {e}")

    def _gen_medcrab_en2vi(self, text):
        self._lazy_load()
        if not self.is_ready: return text
        
        prompt = f"English: {text}\nVietnamese (trả lời ngắn gọn):"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                repetition_penalty=1.2,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = full_text.split("Vietnamese (trả lời ngắn gọn):")[-1].strip()
        return " ".join(translated.split()[:12])

    def translate_vi2en(self, text):
        self._lazy_load()
        if not text or not self.is_ready: return text
        try:
            if isinstance(text, list):
                res = self.vi2en(text)
                return [r['translation_text'] for r in res]
            res = self.vi2en(text)
            return res[0]['translation_text']
        except: return text

    def translate_en2vi(self, text):
        if not text: return text
        
        if isinstance(text, str):
            text_lower = text.lower().strip().replace(".", "").replace(",", "")
            if text_lower in ["yes", "correct", "true", "normal", "present"]: return "có"
            if text_lower in ["no", "incorrect", "false", "abnormal", "absent"]: return "không"
            if any(w in text_lower for w in ["normal", "no abnormality"]): return "bình thường"
            if any(w in text_lower for w in ["abnormal", "pathology"]): return "bất thường"
        
        if isinstance(text, list):
            return [self._gen_medcrab_en2vi(t) for t in text]
        return self._gen_medcrab_en2vi(text)

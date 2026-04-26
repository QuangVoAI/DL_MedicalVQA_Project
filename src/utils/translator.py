import torch
from transformers import pipeline

class MedicalTranslator:
    """
    Lớp dịch thuật hỗ trợ Hướng B (Zero-shot) chuyển đổi Vi <-> En.
    Sử dụng mô hình Helsinki-NLP chạy local trên GPU/CPU.
    """
    def __init__(self, device="cpu"):
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        print(f"[INFO] Khởi tạo Translation Layer trên: {'GPU' if self.device == 0 else 'CPU'}")
        
        try:
            # Vi -> En
            self.vi2en = pipeline("translation_vi_to_en", model="Helsinki-NLP/opus-mt-vi-en", device=self.device)
            # En -> Vi
            self.en2vi = pipeline("translation_en_to_vi", model="Helsinki-NLP/opus-mt-en-vi", device=self.device)
        except Exception as e:
            print(f"[WARNING] Không thể load mô hình dịch thuật: {e}")
            self.vi2en = None
            self.en2vi = None

    def translate_vi2en(self, text):
        if not self.vi2en or not text: return text
        if isinstance(text, list):
            res = self.vi2en(text)
            return [r['translation_text'] for r in res]
        return self.vi2en(text)[0]['translation_text']

    def translate_en2vi(self, text):
        if not self.en2vi or not text: return text
        if isinstance(text, list):
            res = self.en2vi(text)
            return [r['translation_text'] for r in res]
        # Xử lý các câu trả lời ngắn đặc thù y tế
        text_lower = text.lower().strip()
        if text_lower in ["yes", "it is yes"]: return "có"
        if text_lower in ["no", "it is no"]: return "không"
        
        return self.en2vi(text)[0]['translation_text']

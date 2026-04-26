import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MedicalTranslator:
    """
    Lớp dịch thuật hỗ trợ Hướng B (Zero-shot) chuyển đổi Vi <-> En.
    Nạp mô hình trực tiếp từ Hugging Face Hub (Helsinki-NLP).
    """
    def __init__(self, device="cpu"):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Khởi tạo Translation Layer trực tiếp từ HuggingFace Hub ({self.device})")
        
        try:
            # Vi -> En (Tải trực tiếp từ HF)
            self.vi2en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
            self.vi2en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en").to(self.device)
            
            # En -> Vi (Tải trực tiếp từ HF)
            self.en2vi_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
            self.en2vi_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-vi").to(self.device)
            
            self.is_ready = True
        except Exception as e:
            print(f"[WARNING] Không thể tải mô hình từ HuggingFace: {e}")
            self.is_ready = False

    def _translate(self, text, model, tokenizer):
        if not self.is_ready or not text: return text
        
        is_single = isinstance(text, str)
        if is_single: text = [text]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        results = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return results[0] if is_single else results

    def translate_vi2en(self, text):
        return self._translate(text, self.vi2en_model, self.vi2en_tokenizer)

    def translate_en2vi(self, text):
        # Tiền xử lý các câu trả lời ngắn
        if isinstance(text, str):
            text_lower = text.lower().strip()
            if text_lower in ["yes", "it is yes", "correct"]: return "có"
            if text_lower in ["no", "it is no", "incorrect"]: return "không"
        
        return self._translate(text, self.en2vi_model, self.en2vi_tokenizer)

import torch
import json
import os

from src.utils.text_utils import postprocess_answer

class MedicalTranslator:
    """
    Dịch thuật y tế với cơ chế Lazy Loading + Independent Fallback.
    - Vi→En: MarianMT (Helsinki-NLP) trên CPU
    - En→Vi: MedCrab-1.5B (4-bit) trên GPU phụ (nếu có)
    Mỗi model load độc lập — nếu 1 cái fail, cái kia vẫn hoạt động.
    """
    def __init__(self, device="cpu", dict_path="data/medical_dict.json"):
        self.device_str = device  # "cuda" hoặc "cpu"
        
        # Chọn GPU: nếu Dual GPU → dùng cuda:1, nếu Single → dùng cuda:0
        if torch.cuda.is_available() and device == "cuda":
            if torch.cuda.device_count() > 1:
                self.gpu_device = torch.device("cuda:1")
                print(f"[INFO] Dual-GPU detected → Translator on {self.gpu_device}")
            else:
                self.gpu_device = torch.device("cuda:0")
        else:
            self.gpu_device = torch.device("cpu")
        
        # State flags
        self._load_attempted = False
        self._vi2en_ready = False
        self._en2vi_ready = False
        
        # Models (lazy)
        self._vi2en_model = None
        self._vi2en_tokenizer = None
        self._en2vi_model = None
        self._en2vi_tokenizer = None
        
        # Medical dictionary
        self.med_dict = {}
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.med_dict = json.load(f)
            except:
                pass

    def _lazy_load(self):
        """Nạp models. Chỉ gọi 1 lần duy nhất."""
        if self._load_attempted:
            return
        self._load_attempted = True
        print("[INFO] Đang nạp Translation Models (Lazy Load)...")
        
        # ── 1. Helsinki-NLP Vi→En (Chạy trên CPU, nhẹ ~300MB) ──
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            vi2en_id = "Helsinki-NLP/opus-mt-vi-en"
            self._vi2en_tokenizer = AutoTokenizer.from_pretrained(vi2en_id)
            self._vi2en_model = AutoModelForSeq2SeqLM.from_pretrained(vi2en_id).to("cpu")
            self._vi2en_model.eval()
            self._vi2en_ready = True
            print("[INFO] ✅ Helsinki-NLP (Vi→En) đã sẵn sàng trên CPU")
        except Exception as e:
            print(f"[WARNING] ❌ Helsinki-NLP load thất bại: {e}")
        
        # ── 2. MedCrab En→Vi (4-bit trên GPU) ──
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            medcrab_id = "pnnbao-ump/MedCrab-1.5B"
            self._en2vi_tokenizer = AutoTokenizer.from_pretrained(medcrab_id)
            
            d_map = {"": self.gpu_device} if self.gpu_device.type == "cuda" else None
            self._en2vi_model = AutoModelForCausalLM.from_pretrained(
                medcrab_id,
                quantization_config=bnb_config,
                device_map=d_map,
                low_cpu_mem_usage=True
            )
            self._en2vi_model.eval()
            self._en2vi_ready = True
            print(f"[INFO] ✅ MedCrab-1.5B (En→Vi) đã sẵn sàng trên {self.gpu_device}")
        except Exception as e:
            print(f"[WARNING] ❌ MedCrab load thất bại: {e}")

    # ── Vi → En ──
    def translate_vi2en(self, text):
        """Dịch câu hỏi Tiếng Việt sang Tiếng Anh."""
        if not text:
            return text
        self._lazy_load()
        
        if not self._vi2en_ready:
            # Fallback: trả về nguyên văn (LLaVA vẫn hiểu được một phần)
            return text
        
        try:
            texts = text if isinstance(text, list) else [text]
            results = []
            for t in texts:
                inputs = self._vi2en_tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    output_ids = self._vi2en_model.generate(**inputs, max_new_tokens=128)
                translated = self._vi2en_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append(translated)
            return results if isinstance(text, list) else results[0]
        except Exception as e:
            print(f"[WARNING] Vi→En error: {e}")
            return text

    # ── En → Vi ──
    def translate_en2vi(self, text):
        """Dịch kết quả từ LLaVA-Med sang Tiếng Việt."""
        if not text:
            return text
        
        # 1. Ánh xạ trực tiếp nhãn nhị phân (nhanh + chính xác 100%)
        if isinstance(text, str):
            t = text.lower().strip().rstrip(".").rstrip(",").strip()
            
            # Xử lý các câu trả lời dài bắt đầu bằng Yes/No của LLaVA (vd: "No, the image does not...")
            if t.startswith("yes"):
                return "có"
            if t.startswith("no"):
                return "không"
                
            # Exact match trước
            direct_map = {
                "true": "có", "false": "không",
                "correct": "có", "incorrect": "không",
                "present": "có", "absent": "không",
                "normal": "bình thường", "abnormal": "bất thường",
            }
            if t in direct_map:
                return direct_map[t]
        
        # 2. Dịch bằng MedCrab
        self._lazy_load()
        if not self._en2vi_ready:
            if isinstance(text, list):
                return text
            return text
        
        if isinstance(text, list):
            return [self._medcrab_translate(t) for t in text]
        return self._medcrab_translate(text)

    def _medcrab_translate(self, text):
        """Dịch 1 câu En→Vi bằng MedCrab với ràng buộc ngắn gọn."""
        # Kiểm tra ánh xạ trực tiếp trước
        t = text.lower().strip().rstrip(".").rstrip(",").strip()
        direct_map = {
            "yes": "có", "no": "không",
            "normal": "bình thường", "abnormal": "bất thường",
        }
        if t in direct_map:
            return direct_map[t]
        
        try:
            prompt = f"English: {text}\nVietnamese (trả lời ngắn gọn):"
            inputs = self._en2vi_tokenizer(prompt, return_tensors="pt").to(self.gpu_device)
            
            with torch.no_grad():
                outputs = self._en2vi_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    repetition_penalty=1.2,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._en2vi_tokenizer.eos_token_id
                )
            
            full_text = self._en2vi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = full_text.split("Vietnamese (trả lời ngắn gọn):")[-1].strip()
            return translated
        except Exception as e:
            print(f"[WARNING] En→Vi error: {e}")
            return text

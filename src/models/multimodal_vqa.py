import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MultimodalVQA:
    """
    Wrapper cho LLaVA-Med-7B tích hợp QLoRA 4-bit để huấn luyện trên Kaggle.
    Sử dụng kiến trúc LLaVA-1.5 (microsoft/llava-med-v1.5-7b).
    """
    def __init__(self, model_id="microsoft/llava-med-v1.5-7b"):
        self.model_id = model_id
        
        # 1. Cấu hình Quantization 4-bit (Tiết kiệm VRAM)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 2. Cấu hình LoRA (Chỉ huấn luyện một phần nhỏ tham số)
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Các lớp attention
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def load_model(self):
        print(f"[INFO] Đang tải LLaVA-Med-v1.5-7B với chế độ 4-bit...")
        processor = LlavaProcessor.from_pretrained(self.model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        
        # Chuẩn bị mô hình cho PEFT
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        
        model.print_trainable_parameters()
        return model, processor

    def generate_prompt_vi(self, question_en):
        """
        Hàm hỗ trợ tạo prompt cho LLaVA-Med (EN). 
        Nhớ dùng Translation Layer trước khi gọi hàm này.
        """
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions about the medical image. USER: <image>\n{question_en} ASSISTANT:"
        return prompt

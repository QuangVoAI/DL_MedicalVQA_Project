import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MultimodalVQA:
    """
    Wrapper cho LLaVA-Med-7B tích hợp QLoRA 4-bit để huấn luyện trên Kaggle.
    Sử dụng kiến trúc LLaVA-1.5 (microsoft/llava-med-v1.5-7b).
    """
    def __init__(
        self,
        model_id="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
    ):
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
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def load_model(self):
        print(f"[INFO] Đang tải LLaVA-Med-v1.5-7B với chế độ 4-bit...")
        processor = LlavaProcessor.from_pretrained(self.model_id)
        processor.tokenizer.padding_side = "left" # Bắt buộc cho decoder-only models
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto"
        )

        model.config.use_cache = False

        # Chuẩn bị mô hình cho PEFT
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        model.print_trainable_parameters()
        return model, processor

    def generate_prompt_vi(self, question_en):
        """
        Hàm hỗ trợ tạo prompt cho LLaVA-Med (EN). 
        Nhớ dùng Translation Layer trước khi gọi hàm này.
        """
        return self.build_instruction_prompt(question_en, language="en", include_answer=False)

    def build_instruction_prompt(self, question, language="vi", include_answer=False):
        """
        Prompt thống nhất cho zero-shot, SFT và demo.
        """
        if language == "vi":
            instruction = "Chi tra loi bang tieng Viet, khong dung tieng Anh, thuat ngu y khoa chuan, ngan gon, toi da 10 tu."
        else:
            instruction = "Answer with standard medical terminology, concise, at most 10 words."
        suffix = " ASSISTANT:" if not include_answer else ""
        return f"USER: <image>\n{question}\n{instruction}{suffix}"

import os
from dataclasses import dataclass

import torch

from src.utils.text_utils import postprocess_answer


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class RewriteConfig:
    enabled: bool = False
    model_id: str = ""
    use_4bit: bool = True
    max_new_tokens: int = 28
    max_words: int = 10


class MedicalAnswerRewriter:
    """
    Rewrite lớp cuối cho VQA output.

    Mục tiêu:
    - Giữ nguyên ý nghĩa gốc.
    - Làm câu trả lời tự nhiên và đầy đủ hơn một chút.
    - Vẫn giới hạn tối đa số từ theo cấu hình.

    Mô hình này không thay thế VQA model chính.
    """

    def __init__(self, config: RewriteConfig | None = None) -> None:
        self.config = config or self._load_config()
        self._load_attempted = False
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _load_config() -> RewriteConfig:
        model_id = (
            os.getenv("ANSWER_REWRITE_MODEL_ID", "").strip()
            or os.getenv("QWEN_REWRITE_MODEL_ID", "").strip()
            or "Qwen/Qwen2.5-14B-Instruct"
        )
        enabled = _as_bool(os.getenv("ANSWER_REWRITE_ENABLED"), default=True)
        use_4bit = _as_bool(os.getenv("ANSWER_REWRITE_USE_4BIT"), default=True)
        max_new_tokens = int(os.getenv("ANSWER_REWRITE_MAX_NEW_TOKENS", "28"))
        max_words = int(os.getenv("ANSWER_REWRITE_MAX_WORDS", "10"))
        return RewriteConfig(
            enabled=enabled,
            model_id=model_id,
            use_4bit=use_4bit,
            max_new_tokens=max_new_tokens,
            max_words=max_words,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.model_id)

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def ready(self) -> bool:
        return self._ready

    def _lazy_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self.enabled:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            hf_token = (
                os.getenv("ANSWER_REWRITE_HF_TOKEN", "").strip()
                or os.getenv("HF_TOKEN", "").strip()
                or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
                or None
            )

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True, token=hf_token)
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if self._device.type == "cuda":
                if self.config.use_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                    except Exception as exc:
                        print(f"[WARNING] Rewrite 4-bit config unavailable, falling back to bf16: {exc}")
                        model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            if hf_token is not None:
                model_kwargs["token"] = hf_token

            model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._ready = True
            print(f"[INFO] ✅ Answer rewriter ready: {self.config.model_id}")
        except Exception as exc:
            self._ready = False
            print(f"[WARNING] ❌ Answer rewriter load failed: {exc}")

    def _build_messages(self, question: str, answer: str, language: str = "vi") -> list[dict[str, str]]:
        system_prompt = (
            "Bạn là bộ biên tập câu trả lời cho hệ thống Medical VQA. "
            "Nhiệm vụ của bạn là viết lại câu trả lời gốc thành một câu ngắn, tự nhiên, "
            "rõ nghĩa hơn nhưng KHÔNG thêm thông tin mới ngoài nội dung đã có. "
            "Giới hạn tối đa 10 từ. Chỉ trả về câu trả lời cuối cùng."
        )
        if language.lower().startswith("en"):
            system_prompt = (
                "You are an editor for a Medical VQA system. "
                "Rewrite the raw answer into a short, natural, clearer sentence "
                "without adding facts beyond the original answer. "
                "Use at most 10 words. Return only the final answer."
            )

        examples = [
            {
                "question": "Ảnh này có tràn dịch màng phổi không?",
                "answer": "không",
                "rewrite": "Không, không có tràn dịch màng phổi.",
            },
            {
                "question": "Hình ảnh có tim to không?",
                "answer": "có",
                "rewrite": "Có, tim to.",
            },
            {
                "question": "Đây là loại ảnh gì?",
                "answer": "x quang ngực",
                "rewrite": "X-quang ngực.",
            },
        ]

        if language.lower().startswith("en"):
            examples = [
                {
                    "question": "Is there pleural effusion?",
                    "answer": "no",
                    "rewrite": "No, no pleural effusion.",
                },
                {
                    "question": "Is the heart enlarged?",
                    "answer": "yes",
                    "rewrite": "Yes, enlarged heart.",
                },
                {
                    "question": "What modality is this?",
                    "answer": "chest x ray",
                    "rewrite": "Chest X-ray.",
                },
            ]

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for ex in examples:
            messages.append(
                {
                    "role": "user",
                    "content": f"Câu hỏi: {ex['question']}\nĐáp án gốc: {ex['answer']}",
                }
            )
            messages.append({"role": "assistant", "content": ex["rewrite"]})

        user_prompt = f"Câu hỏi: {question}\nĐáp án gốc: {answer}\nViết lại ngắn gọn, tự nhiên, không thêm thông tin mới."
        if language.lower().startswith("en"):
            user_prompt = (
                f"Question: {question}\nRaw answer: {answer}\n"
                "Rewrite it into a short, natural answer without adding new facts."
            )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def rewrite(self, question: str, answer: str, language: str = "vi") -> str:
        """
        Rewrite câu trả lời để tự nhiên hơn.
        Nếu rewrite model không sẵn sàng, trả về output đã postprocess.
        """
        if not answer:
            return ""

        self._lazy_load()
        fallback = postprocess_answer(answer, max_words=self.config.max_words)
        if not self.enabled or not self._ready:
            return fallback

        try:
            messages = self._build_messages(question=question, answer=answer, language=language)
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.05,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated = self._tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
            cleaned = postprocess_answer(generated, max_words=self.config.max_words)
            return cleaned or fallback
        except Exception as exc:
            print(f"[WARNING] Rewrite failed: {exc}")
            return fallback

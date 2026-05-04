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


_REWRITE_STYLE_BY_MODEL = {
    "A1": {
        "vi": "Diễn đạt đơn giản, trực tiếp, gần với đáp án gốc.",
        "en": "Use simple, direct wording close to the raw answer.",
    },
    "A2": {
        "vi": "Diễn đạt như một quan sát ngắn trên hình ảnh.",
        "en": "Word it as a short imaging observation.",
    },
    "B1": {
        "vi": "Diễn đạt tự nhiên, mềm hơn, dễ đọc.",
        "en": "Use natural, softer, easy-to-read wording.",
    },
    "B2": {
        "vi": "Diễn đạt hay hơn A1/A2, theo phong cách lâm sàng súc tích.",
        "en": "Use stronger concise clinical wording than A1/A2.",
    },
    "DPO": {
        "vi": "Diễn đạt hay nhất theo hướng thận trọng, chuyên nghiệp.",
        "en": "Use the most careful, professional wording.",
    },
    "PPO": {
        "vi": "Diễn đạt hay nhất theo hướng rõ ràng, mạch lạc.",
        "en": "Use the clearest, most polished wording.",
    },
    "SOUP": {
        "vi": "Diễn đạt cân bằng giữa lâm sàng, thận trọng và rõ ràng.",
        "en": "Use balanced clinical, careful, and clear wording.",
    },
}


_MODEL_SPECIFIC_EXAMPLES = {
    "A1": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, có khối u.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, there is a mass.",
        },
    },
    "A2": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, thấy khối u trên ảnh.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, a mass is seen.",
        },
    },
    "B2": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, hình ảnh gợi ý khối u.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, imaging suggests a mass.",
        },
    },
    "DPO": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, có dấu hiệu gợi ý khối u.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, findings suggest a mass.",
        },
    },
    "PPO": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, kết quả gợi ý khối u rõ.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, results clearly suggest a mass.",
        },
    },
    "SOUP": {
        "vi": {
            "question": "Ảnh có khối u không?",
            "answer": "có",
            "rewrite": "Có, hình ảnh gợi ý khối u rõ.",
        },
        "en": {
            "question": "Is there a mass?",
            "answer": "yes",
            "rewrite": "Yes, imaging clearly suggests a mass.",
        },
    },
}


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
            print(f"[INFO] Answer rewriter ready: {self.config.model_id}")
        except Exception as exc:
            self._ready = False
            print(f"[WARNING] Answer rewriter load failed: {exc}")

    def _get_style_instruction(self, source_model: str | None, language: str) -> str:
        if not source_model:
            return ""
        style = _REWRITE_STYLE_BY_MODEL.get(source_model.upper())
        if not style:
            return ""
        lang_key = "en" if language.lower().startswith("en") else "vi"
        return style[lang_key]

    def _get_model_specific_example(self, source_model: str | None, language: str) -> dict[str, str] | None:
        if not source_model:
            return None
        examples = _MODEL_SPECIFIC_EXAMPLES.get(source_model.upper())
        if not examples:
            return None
        lang_key = "en" if language.lower().startswith("en") else "vi"
        return examples[lang_key]

    def _build_messages(
        self,
        question: str,
        answer: str,
        language: str = "vi",
        source_model: str | None = None,
    ) -> list[dict[str, str]]:
        style_instruction = self._get_style_instruction(source_model, language)
        model_example = self._get_model_specific_example(source_model, language)
        system_prompt = (
            "Bạn là bộ biên tập câu trả lời cho hệ thống Medical VQA. "
            "Nhiệm vụ của bạn là mở rộng đáp án gốc thành một câu trả lời đầy đủ, "
            "tự nhiên và rõ nghĩa hơn, nhưng vẫn phải bám sát đáp án gốc. "
            "KHÔNG thêm thông tin y khoa mới, KHÔNG suy diễn ngoài đáp án gốc. "
            "Có thể dùng câu hỏi để xác định đối tượng y khoa đang được hỏi, "
            "nhưng đáp án gốc quyết định ý nghĩa đúng/sai/có/không. "
            "Nếu nhiều model có cùng đáp án gốc, vẫn dùng phong cách riêng của model hiện tại. "
            "CÂU TRẢ LỜI BẮT BUỘC PHẢI DƯỚI 10 TỪ, ÍT NHẤT 3 TỪ. "
            "Chỉ trả về câu trả lời cuối cùng."
        )
        if style_instruction:
            system_prompt += f" Phong cách riêng cho model này: {style_instruction}"

        if language.lower().startswith("en"):
            system_prompt = (
                "You are an editor for a Medical VQA system. "
                "Expand the raw answer into a fuller, natural, clearer answer "
                "while staying strictly based on the raw answer. "
                "Do not add new medical facts or infer beyond the raw answer. "
                "You may use the question to identify the medical target, "
                "but the raw answer controls yes/no/presence/absence. "
                "If several models share the same raw answer, still use this model's wording style. "
                "THE ANSWER MUST BE UNDER 10 WORDS and at least 3 words. "
                "Return only the final answer."
            )
            if style_instruction:
                system_prompt += f" Model-specific wording style: {style_instruction}"

        examples = [
            {
                "question": "Ảnh này có tràn dịch màng phổi không?",
                "answer": "không",
                "rewrite": "Không, không thấy tràn dịch màng phổi.",
            },
            {
                "question": "Hình ảnh có tim to không?",
                "answer": "có",
                "rewrite": "Có, hình ảnh cho thấy tim to.",
            },
            {
                "question": "Đây là loại ảnh gì?",
                "answer": "x quang ngực",
                "rewrite": "Đây là ảnh X-quang ngực.",
            },
        ]

        if language.lower().startswith("en"):
            examples = [
                {
                    "question": "Is there pleural effusion?",
                    "answer": "no",
                    "rewrite": "No, pleural effusion is not seen.",
                },
                {
                    "question": "Is the heart enlarged?",
                    "answer": "yes",
                    "rewrite": "Yes, the heart appears enlarged.",
                },
                {
                    "question": "What modality is this?",
                    "answer": "chest x ray",
                    "rewrite": "This is a chest X-ray.",
                },
            ]

        if model_example:
            examples.append(model_example)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for ex in examples:
            messages.append(
                {
                    "role": "user",
                    "content": f"Câu hỏi: {ex['question']}\nĐáp án gốc: {ex['answer']}",
                }
            )
            messages.append({"role": "assistant", "content": ex["rewrite"]})

        user_prompt = (
            f"Câu hỏi: {question}\n"
            f"Đáp án gốc: {answer}\n"
            f"Model nguồn: {source_model or 'unknown'}\n"
            "Viết lại thành câu đầy đủ hơn, tự nhiên hơn, dưới 10 từ. "
            "CHỈ DÙNG THÔNG TIN TỪ ĐÁP ÁN GỐC."
        )
        if style_instruction:
            user_prompt += f"\nPhong cách diễn đạt: {style_instruction}"

        if language.lower().startswith("en"):
            user_prompt = (
                f"Question: {question}\nRaw answer: {answer}\n"
                f"Source model: {source_model or 'unknown'}\n"
                "Rewrite it as a fuller, natural answer under 10 words. "
                "Use only information from the raw answer."
            )
            if style_instruction:
                user_prompt += f"\nWording style: {style_instruction}"
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def rewrite(
        self,
        question: str,
        answer: str,
        language: str = "vi",
        source_model: str | None = None,
    ) -> str:
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
            messages = self._build_messages(
                question=question,
                answer=answer,
                language=language,
                source_model=source_model,
            )
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


def rewrite_final_answer(
    question: str,
    answer: str,
    language: str = "vi",
    source_model: str | None = None,
) -> str:
    """
    Helper tiện dùng trong notebook / web.
    """
    rewriter = MedicalAnswerRewriter()
    return rewriter.rewrite(
        question=question,
        answer=answer,
        language=language,
        source_model=source_model,
    )

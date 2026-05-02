import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import yaml
import argparse
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# [Bypass CVE-2025-32434] Bỏ qua yêu cầu nâng cấp PyTorch 2.6 của transformers
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

# [Bypass FSDPModule Error] Sửa lỗi thư viện trl import FSDPModule trên PyTorch cũ
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, "FSDPModule"):
    fsdp.FSDPModule = fsdp.FullyShardedDataParallel

import csv
import json
from datetime import datetime
from pathlib import Path
from PIL import Image

from datasets import load_dataset
# Import các thành phần từ thư mục src
from src.models.medical_vqa_model import MedicalVQAModelA
from src.models.multimodal_vqa import MultimodalVQA
from src.utils.visualization import MedicalImageTransform as MedicalTransform
from src.data.medical_dataset import MedicalVQADataset
from src.utils.text_utils import get_target_answer, normalize_answer, postprocess_answer


def build_training_arguments(training_arguments_cls, **kwargs):
    """Create TrainingArguments across transformers versions."""
    if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
        alias_kwargs = dict(kwargs)
        alias_kwargs["eval_strategy"] = alias_kwargs.pop("evaluation_strategy")
        try:
            return training_arguments_cls(**alias_kwargs)
        except TypeError as exc:
            if "eval_strategy" not in str(exc):
                raise

    return training_arguments_cls(**kwargs)


def vqa_collate_fn(batch):
    """Hàm gom batch tùy chỉnh để xử lý ảnh PIL và raw text."""
    elem = batch[0]
    collated = {}
    for key in elem.keys():
        if key in ['image', 'input_ids', 'attention_mask', 'label_closed', 'target_ids', 'chosen_ids', 'rejected_ids']:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            # Giữ nguyên list cho PIL images và raw text
            collated[key] = [item[key] for item in batch]
    return collated


def flatten_dict(data, parent_key="", sep="."):
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, (list, tuple)):
            continue
        else:
            items[new_key] = value
    return items


def create_history_dir(base_log_dir, variant):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join(base_log_dir, "history", variant, timestamp)
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


def save_history_records(history_dir, records):
    os.makedirs(history_dir, exist_ok=True)
    json_path = os.path.join(history_dir, "history.json")
    csv_path = os.path.join(history_dir, "history.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    flat_rows = [flatten_dict(record) for record in records]
    if flat_rows:
        fieldnames = sorted({key for row in flat_rows for key in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def select_best_adapter_checkpoint(checkpoint_root: str):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục checkpoint: {checkpoint_root}")

    checkpoint_dirs = sorted(
        p for p in checkpoint_root.glob("checkpoint-*")
        if (p / "adapter_config.json").exists()
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"Không có adapter checkpoint trong {checkpoint_root}")

    for state_file in sorted(checkpoint_root.glob("checkpoint-*/trainer_state.json"), reverse=True):
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        best_path = state.get("best_model_checkpoint")
        if best_path:
            best_dir = Path(best_path.replace("./", ""))
            if not best_dir.is_absolute():
                best_dir = Path.cwd() / best_dir
            if (best_dir / "adapter_config.json").exists():
                return best_dir.resolve()

    return checkpoint_dirs[-1].resolve()


def build_dpo_instruction_prompt(question: str, max_words: int = 10) -> str:
    question = str(question or "").strip()
    instruction = (
        "Chi tra loi bang tieng Viet. "
        "Khong dung tieng Anh. "
        "Khong lap lai cau hoi. "
        "Khong mo ta hinh anh chung chung. "
        f"Chi tra loi truc tiep dap an, toi da {max_words} tu."
    )
    return f"USER: <image>\n{question}\n{instruction} ASSISTANT:"


def load_latest_variant_metrics(history_root: str, variant: str) -> dict | None:
    variant_dir = Path(history_root) / variant
    if not variant_dir.exists():
        return None
    history_files = sorted(variant_dir.glob("*/history.json"))
    if not history_files:
        return None
    for history_file in reversed(history_files):
        try:
            records = json.loads(history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if records:
            return records[-1]
    return None


def evaluate_dpo_acceptance(b2_metrics: dict | None, dpo_metrics: dict) -> dict:
    if not b2_metrics:
        return {
            "status": "unknown",
            "reason": "missing_b2_metrics",
            "summary": "Khong tim thay metrics B2 de doi chieu.",
        }

    def pct_delta(key: str) -> float | None:
        b2_val = b2_metrics.get(key)
        dpo_val = dpo_metrics.get(key)
        if b2_val is None or dpo_val is None:
            return None
        return (dpo_val - b2_val) * 100.0

    deltas = {
        "accuracy": pct_delta("val_accuracy_normalized"),
        "f1": pct_delta("val_f1_normalized"),
        "bleu4": pct_delta("val_bleu4_normalized"),
        "closed_acc": pct_delta("val_closed_accuracy"),
        "open_semantic": pct_delta("val_open_semantic"),
        "open_bert": pct_delta("val_open_bertscore"),
    }
    failed_drop = any(
        delta is not None and delta < -1.0
        for delta in (deltas["accuracy"], deltas["f1"], deltas["bleu4"])
    )
    closed_ok = (
        b2_metrics.get("val_closed_accuracy") is not None
        and dpo_metrics.get("val_closed_accuracy") is not None
        and dpo_metrics["val_closed_accuracy"] >= b2_metrics["val_closed_accuracy"]
    )
    open_ok = (
        b2_metrics.get("val_open_semantic") is not None
        and dpo_metrics.get("val_open_semantic") is not None
        and b2_metrics.get("val_open_bertscore") is not None
        and dpo_metrics.get("val_open_bertscore") is not None
        and dpo_metrics["val_open_semantic"] >= b2_metrics["val_open_semantic"]
        and (dpo_metrics["val_open_bertscore"] - b2_metrics["val_open_bertscore"]) * 100.0 >= -0.3
    )
    accepted = (not failed_drop) and (closed_ok or open_ok)
    def _fmt(delta: float | None) -> str:
        return "N/A" if delta is None else f"{delta:.2f}"
    summary = (
        f"DPO vs B2 deltas (pp): Acc={_fmt(deltas['accuracy'])} | F1={_fmt(deltas['f1'])} | "
        f"BLEU={_fmt(deltas['bleu4'])} | Closed={_fmt(deltas['closed_acc'])} | "
        f"OpenSem={_fmt(deltas['open_semantic'])} | OpenBERT={_fmt(deltas['open_bert'])}"
    )
    return {
        "status": "accepted" if accepted else "failed",
        "reason": "criteria_met" if accepted else "metric_drop_or_no_gain",
        "summary": summary,
        "deltas_pp": deltas,
        "closed_ok": closed_ok,
        "open_ok": open_ok,
    }


def evaluate_refinement_acceptance(base_metrics: dict | None, rl_metrics: dict) -> dict:
    return evaluate_dpo_acceptance(base_metrics, rl_metrics)


def sanitize_dpo_completion(question: str, answer: str, max_words: int = 10) -> str:
    question_norm = normalize_answer(question)
    answer_norm = postprocess_answer(answer, max_words=max_words)

    if answer_norm in {"yes", "có"}:
        return "có"
    if answer_norm in {"no", "không"}:
        return "không"

    is_closed = any(
        pattern in question_norm
        for pattern in ["bình thường", "bat thuong", "normal", "abnormal"]
    ) or question_norm.endswith(" không") or " có " in f" {question_norm} "

    if is_closed:
        if any(token in answer_norm for token in ["không", "no", "not normal", "abnormal"]):
            return "không"
        if any(token in answer_norm for token in ["có", "yes", "bình thường", "normal", "present", "detected"]):
            return "có"

    return answer_norm


def resolve_dpo_image(item: dict, hf_train_data=None, image_dir: str | None = None):
    source_idx = item.get("source_idx")
    if hf_train_data is not None and source_idx is not None and 0 <= int(source_idx) < len(hf_train_data):
        img = hf_train_data[int(source_idx)].get("image")
        if img is not None and getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        return img

    image_name = item.get("image")
    if image_name and image_dir:
        img_path = os.path.join(image_dir, image_name)
        if os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")
    return None


def infer_closed_answer_type(item: dict, answer: str | None = None) -> bool:
    answer_norm = normalize_answer(answer if answer is not None else get_target_answer(item))
    answer_type = str(item.get("answer_type", "")).strip().upper()
    label_closed = item.get("label_closed", None)
    if answer_type == "CLOSED" or label_closed in (0, 1):
        return True
    return answer_norm in {"có", "không", "yes", "no"}


def move_model_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_multimodal_completion_batch(processor, prompts, completions, images, max_length=None):
    full_texts = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)]
    batch = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prompt_batch = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    completion_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
    for i, prompt_len in enumerate(prompt_lengths.tolist()):
        token_positions = batch["attention_mask"][i].nonzero(as_tuple=True)[0]
        completion_mask[i, token_positions[prompt_len:]] = 1

    if max_length is not None and batch["input_ids"].shape[1] > max_length:
        batch["input_ids"] = batch["input_ids"][:, :max_length]
        batch["attention_mask"] = batch["attention_mask"][:, :max_length]
        completion_mask = completion_mask[:, :max_length]
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in batch:
                batch[key] = batch[key][:, :max_length]

    return batch, completion_mask


def compute_masked_sequence_logprobs(model, batch, completion_mask):
    model_inputs = move_model_batch_to_device(batch, next(model.parameters()).device)
    completion_mask = completion_mask.to(model_inputs["input_ids"].device)
    outputs = model(**model_inputs)
    logits = outputs.logits[:, :-1, :]
    labels = model_inputs["input_ids"][:, 1:]
    token_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs * token_mask
    denom = token_mask.sum(dim=1).clamp_min(1.0)
    seq_log_probs = masked_log_probs.sum(dim=1) / denom

    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    seq_entropy = (token_entropy * token_mask).sum(dim=1) / denom
    return seq_log_probs, seq_entropy


def compute_single_open_reward(pred: str, ref: str) -> tuple[float, dict]:
    from src.utils.metrics import compute_exact_match, compute_f1, compute_rouge_l
    from src.utils import metrics as metrics_module

    norm_pred = normalize_answer(pred) or "."
    norm_ref = normalize_answer(ref) or "."
    exact = compute_exact_match(norm_pred, norm_ref)
    f1 = compute_f1(norm_pred, norm_ref)
    rouge_l = compute_rouge_l(norm_pred, norm_ref)

    bert = 0.0
    scorer = getattr(metrics_module, "bert_scorer", None)
    if scorer is not None:
        try:
            _, _, bert_f1 = scorer.score([norm_pred], [norm_ref])
            bert = float(bert_f1.mean().item())
        except Exception:
            bert = 0.0

    blended = (0.55 * bert) + (0.30 * f1) + (0.10 * rouge_l) + (0.05 * exact)
    reward = (2.0 * blended) - 1.0
    return reward, {
        "bert": bert,
        "f1": f1,
        "rouge_l": rouge_l,
        "exact": exact,
        "blended": blended,
    }

def train(args):
    # 1. Load Cấu hình
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ── WandB Setup ──────────────────────────────────────────────────────────
    _wandb_cfg = config.get("wandb", {})
    _use_wandb = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE"))

    if _use_wandb:
        _api_key = os.environ.get("WANDB_API_KEY")
        if _api_key:
            wandb.login(key=_api_key)

        # Offline mode: set WANDB_MODE=offline hoặc config wandb.offline: true
        _offline = _wandb_cfg.get("offline", False) or \
                   os.environ.get("WANDB_MODE", "").lower() == "offline"
        if _offline:
            os.environ["WANDB_MODE"] = "offline"
            print("[INFO] WandB chạy ở chế độ OFFLINE (sync sau bằng: wandb sync)")

        # Tags theo variant từ YAML
        _tags = _wandb_cfg.get("tags", {}).get(args.variant, [])

        # Rich config ghi đầy đủ thông tin experiment
        _run_config = {
            # ── Model architecture ──
            "variant":               args.variant,
            "decoder_type":          config["model_a"].get("decoder_type"),
            "image_encoder":         config["model_a"].get("image_encoder"),
            "text_encoder":          config["model_a"].get("text_encoder"),
            "hidden_size":           config["model_a"].get("hidden_size"),
            "transformer_heads":     config["model_a"].get("transformer_heads"),
            "transformer_ff_dim":    config["model_a"].get("transformer_ff_dim"),
            "transformer_layers":    config["model_a"].get("transformer_decoder_layers"),
            "norm_first":            config["model_a"].get("transformer_norm_first"),
            "freeze_phobert_layers": config["model_a"].get("freeze_phobert_layers"),
            # ── Training ──
            "learning_rate":         config["train"].get("learning_rate"),
            "phobert_lr":            config["train"].get("phobert_lr"),
            "vision_lr":             config["train"].get("vision_lr"),
            "batch_size":            config["train"].get("batch_size"),
            "grad_accum_steps":      config["train"].get("gradient_accumulation_steps"),
            "effective_batch":       config["train"].get("batch_size", 32) *
                                     config["train"].get("gradient_accumulation_steps", 1),
            "label_smoothing":       config["train"].get("label_smoothing"),
            "open_loss_weight":      config["train"].get("open_loss_weight"),
            "warmup_epochs":         config["train"].get("warmup_epochs"),
            "scheduler":             config["train"].get("scheduler"),
            "patience":              config["train"].get("patience"),
            "use_amp":               config["train"].get("use_amp"),
            # ── Data ──
            "dataset":               config["data"].get("dataset_name"),
            "max_question_len":      config["data"].get("max_question_len"),
            "max_answer_len":        config["data"].get("max_answer_len"),
            # ── Eval ──
            "beam_width":            config["eval"].get("beam_width_a") if args.variant in ("A1", "A2")
                                     else config["eval"].get("beam_width_b"),
        }

        # Thêm hardware info
        if torch.cuda.is_available():
            _run_config["gpu_name"]    = torch.cuda.get_device_name(0)
            _run_config["gpu_count"]   = torch.cuda.device_count()
            _run_config["vram_gb"]     = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

        _entity = _wandb_cfg.get("entity") or None   # None = WandB dùng default entity

        wandb.init(
            project=_wandb_cfg.get("project", "MedicalVQA-Vietnam"),
            entity=_entity,
            name=f"{args.variant}-{datetime.now().strftime('%m%d-%H%M')}",
            group=_wandb_cfg.get("group", "DL-Final"),
            job_type=_wandb_cfg.get("job_type", "train"),
            tags=_tags,
            notes=_wandb_cfg.get("notes", ""),
            config=_run_config,
            save_code=_wandb_cfg.get("save_code", True),
            reinit="finish_previous",    # Kết thúc run trước nếu chạy nhiều variant liên tiếp
        )
        print(f"[INFO] ✅ WandB run: {wandb.run.url}")

        # Watch model gradients nếu được bật
        if _wandb_cfg.get("watch_model", False):
            # model chưa khởi tạo ở đây — hook sẽ được gọi sau khi model được tạo
            os.environ["_WANDB_WATCH_PENDING"] = "1"
    else:
        print("[INFO] WandB không được cấu hình (thiếu WANDB_API_KEY) — bỏ qua logging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị sử dụng: {device}")
    history_dir = create_history_dir(config.get("log_dir", "logs/medical_vqa"), args.variant)
    print(f"[INFO] Lưu training history tại: {history_dir}")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model_a']['phobert_model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    transform = MedicalTransform(size=config['data']['image_size'])
    answer_max_words = int(config['data'].get('answer_max_words', 10))
    
    # Nạp dữ liệu từ HuggingFace Hub hoặc cục bộ
    hf_repo = config['data'].get('hf_dataset')
    use_hf_splits = bool(config['data'].get('use_hf_splits', True))
    if hf_repo and use_hf_splits:
        print(f"[INFO] Đang tải dữ liệu từ Hub: {hf_repo}")
        dataset_dict = load_dataset(hf_repo)
        
        if args.debug:
            print("[WARNING] DEBUG MODE: Chỉ lấy 20 mẫu để chạy thử.")
            dataset_dict['train'] = dataset_dict['train'].select(range(min(20, len(dataset_dict['train']))))
            config['train']['epochs'] = 2
            config['train']['batch_size'] = 2
            
        train_ds = MedicalVQADataset(
            hf_dataset=dataset_dict['train'], 
            tokenizer=tokenizer, 
            transform=transform, 
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len'],
            answer_max_words=answer_max_words
        )
        val_ds = MedicalVQADataset(
            hf_dataset=dataset_dict['validation'], 
            tokenizer=tokenizer, 
            transform=transform, 
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len'],
            answer_max_words=answer_max_words
        )
    else:
        vqa_path = config['data']['vqa_json']
        print(f"[INFO] Đang tải dữ liệu cục bộ từ: {vqa_path}")
        full_dataset = MedicalVQADataset(
            json_path=vqa_path,
            image_dir=config['data']['image_dir'],
            tokenizer=tokenizer,
            transform=transform,
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len'],
            answer_max_words=answer_max_words
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        collate_fn=vqa_collate_fn,
        num_workers=config['train'].get('num_workers', 0),
        pin_memory=config['train'].get('pin_memory', False)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['train']['eval_batch_size'] if 'eval_batch_size' in config['train'] else 8, 
        collate_fn=vqa_collate_fn
    )

    # 3. Khởi tạo Mô hình dựa trên Variant
    if args.variant in ['A1', 'A2']:
        decoder_type = "lstm" if args.variant == 'A1' else "transformer"
        model = MedicalVQAModelA(
            decoder_type=decoder_type, 
            vocab_size=len(tokenizer),
            hidden_size=config['model_a'].get('hidden_size', 768),
            phobert_model=config['model_a'].get('phobert_model', "vinai/phobert-base")
        ).to(device)

        # Log model param count lên WandB
        if wandb.run:
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.config.update({
                "total_params_M":     round(total_params / 1e6, 2),
                "trainable_params_M": round(trainable_params / 1e6, 2),
            })
            print(f"[INFO] Tổng params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")
            # wandb.watch: chỉ bật nếu log_gradients: true
            if _wandb_cfg.get("log_gradients", False):
                wandb.watch(model, log="gradients",
                            log_freq=_wandb_cfg.get("log_freq", 50))
        
        # Thiết lập Optimizer với Differential Learning Rate
        optimizer = optim.AdamW([
            {'params': model.image_encoder.parameters(), 'lr': float(config['train']['vision_lr'])},
            {'params': model.text_encoder.parameters(), 'lr': float(config['train']['phobert_lr'])},
            {'params': model.fusion.parameters(), 'lr': float(config['train']['learning_rate'])},
            {'params': model.decoder.parameters(), 'lr': float(config['train']['learning_rate'])}
        ])
        
        # [CRITICAL FIX] Dùng Cosine Schedule với Warmup, step theo batch thay vì epoch
        from transformers import get_cosine_schedule_with_warmup
        # Use a_epochs for Direction A models (A1, A2), otherwise use default epochs
        if args.variant in ['A1', 'A2']:
            epochs = config['train'].get('a_epochs', config['train']['epochs'])
        else:
            epochs = config['train']['epochs']
        warmup_epochs = config['train'].get('warmup_epochs', 5)
        accumulation_steps = config['train'].get('gradient_accumulation_steps', 2)
        total_steps = epochs * len(train_loader) // max(accumulation_steps, 1)
        warmup_steps = warmup_epochs * len(train_loader) // max(accumulation_steps, 1)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )        
        # Khởi tạo Trainer với pad_token_id và beam_width từ config
        beam_width = config['eval'].get('beam_width_a', 5)
        from src.engine.trainer import MedicalVQATrainer
        trainer = MedicalVQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config={
                **config,
                'variant': args.variant,
                'history_dir': history_dir,
                # Pass tunable open-loss weight so trainer doesn't use hardcoded value
                'open_loss_weight': config['train'].get('open_loss_weight', 2.0),
            },
            pad_token_id=tokenizer.pad_token_id,
            beam_width=beam_width
        )
        print(f"[INFO] Beam Width cho Hướng A: {beam_width}")

        print(f"[INFO] Bắt đầu huấn luyện cấu hình {args.variant} ({epochs} epochs)...")
        trainer.train(epochs, tokenizer=tokenizer)
        if wandb.run:
            wandb.finish()
        return

    elif args.variant == 'PPO':
        from src.engine.medical_eval import evaluate_multimodal_vqa

        ppo_cfg = config.get('ppo', {})
        ppo_answer_max_words = int(ppo_cfg.get('max_answer_words', min(answer_max_words, 6)))
        wrapper = MultimodalVQA(
            model_id=config['model_b']['model_name'],
            lora_r=int(config['model_b'].get('lora_r', 16)),
            lora_alpha=int(config['model_b'].get('lora_alpha', 32)),
            lora_dropout=float(config['model_b'].get('lora_dropout', 0.05)),
            lora_target_modules=config['model_b'].get('lora_target_modules'),
        )
        b2_checkpoint = select_best_adapter_checkpoint(config['train'].get('b2_output_dir', './checkpoints/B2'))
        print(f"[INFO] PPO sẽ khởi tạo từ B2 checkpoint: {b2_checkpoint}")
        model, processor = wrapper.load_model(adapter_path=str(b2_checkpoint), is_trainable=True)

        if not ppo_cfg.get('train_mlp_lora', False):
            frozen_lora = 0
            for name, param in model.named_parameters():
                if "lora_" in name and any(proj in name for proj in ("gate_proj", "up_proj", "down_proj")):
                    param.requires_grad = False
                    frozen_lora += param.numel()
            print(f"[INFO] PPO đang freeze LoRA MLP để giảm VRAM: {frozen_lora:,} tham số")
            model.print_trainable_parameters()

        def _build_ppo_source():
            if hf_repo:
                return dataset_dict['train'], dataset_dict['train']
            if hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "data"):
                subset_indices = getattr(train_ds, "indices", list(range(len(train_ds.dataset.data))))
                local_items = [train_ds.dataset.data[i] for i in subset_indices]
                return local_items, None
            raise ValueError("Khong the truy cap raw train data de tao PPO rollout set.")

        def _prepare_ppo_records(raw_items, num_samples: int, closed_ratio: float):
            closed_records = []
            open_records = []
            for idx in range(len(raw_items)):
                item = raw_items[idx]
                question = str(item.get("question_vi", item.get("question", ""))).strip()
                target = get_target_answer(item, max_words=ppo_answer_max_words)
                if not question or not target:
                    continue
                record = {
                    "question": question,
                    "target": target,
                    "source_idx": idx,
                    "image": item.get("image_name"),
                    "is_closed": infer_closed_answer_type(item, target),
                }
                if record["is_closed"]:
                    closed_records.append(record)
                else:
                    open_records.append(record)

            rng = random.Random(int(config.get("seed", 42)))
            rng.shuffle(closed_records)
            rng.shuffle(open_records)

            target_closed = min(len(closed_records), int(round(num_samples * closed_ratio)))
            target_open = min(len(open_records), max(0, num_samples - target_closed))

            selected = closed_records[:target_closed] + open_records[:target_open]
            rng.shuffle(selected)
            return selected

        raw_train_source, hf_train_source = _build_ppo_source()
        ppo_records = _prepare_ppo_records(
            raw_train_source,
            num_samples=int(ppo_cfg.get('num_samples', 192)),
            closed_ratio=float(ppo_cfg.get('closed_ratio', 0.5)),
        )
        if not ppo_records:
            raise ValueError("Khong tao duoc PPO rollout set hop le.")
        print(f"[INFO] PPO rollout set: {len(ppo_records)} mau")

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=float(ppo_cfg.get('learning_rate', 5.0e-7)),
            weight_decay=float(ppo_cfg.get('weight_decay', 0.0)),
        )
        rollout_batch_size = max(1, int(ppo_cfg.get('rollout_batch_size', 2)))
        total_updates = max(1, (len(ppo_records) + rollout_batch_size - 1) // rollout_batch_size)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)

        ppo_history = []
        eos = processor.tokenizer.eos_token or ""
        max_seq_length = max(int(config['train'].get('dpo_max_length', 768)), 768)
        grad_clip = float(config['train'].get('grad_clip', 1.0))
        entropy_coef = float(ppo_cfg.get('entropy_coef', 0.001))
        clip_range = float(ppo_cfg.get('clip_range', 0.2))
        max_new_tokens = int(ppo_cfg.get('max_new_tokens', 12))
        temperature = float(ppo_cfg.get('temperature', 0.8))
        top_p = float(ppo_cfg.get('top_p', 0.9))
        closed_positive = float(ppo_cfg.get('closed_positive_reward', 1.0))
        closed_negative = float(ppo_cfg.get('closed_negative_reward', -1.0))

        print("[INFO] Bắt đầu huấn luyện PPO-style refinement...")
        model.train()
        for update_idx in range(total_updates):
            batch_records = ppo_records[update_idx * rollout_batch_size:(update_idx + 1) * rollout_batch_size]
            prompts, images, questions, targets, closed_flags = [], [], [], [], []
            for record in batch_records:
                image = resolve_dpo_image(
                    record,
                    hf_train_data=hf_train_source,
                    image_dir=config['data'].get('image_dir'),
                )
                if image is None:
                    continue
                prompts.append(build_dpo_instruction_prompt(record["question"], max_words=ppo_answer_max_words))
                images.append(image)
                questions.append(record["question"])
                targets.append(record["target"])
                closed_flags.append(record["is_closed"])

            if not prompts:
                continue

            generation_inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            generation_inputs = move_model_batch_to_device(generation_inputs, next(model.parameters()).device)
            if "pixel_values" in generation_inputs:
                generation_inputs["pixel_values"] = generation_inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                generated_ids = model.generate(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            prompt_token_len = generation_inputs["input_ids"].shape[1]
            generated_texts = processor.batch_decode(
                generated_ids[:, prompt_token_len:],
                skip_special_tokens=True,
            )

            sampled_answers = []
            rewards = []
            reward_breakdown = []
            for question, target, is_closed, raw_output in zip(questions, targets, closed_flags, generated_texts):
                pred = sanitize_dpo_completion(question, raw_output, max_words=ppo_answer_max_words)
                if not pred:
                    pred = "không" if is_closed else "không rõ"
                sampled_answers.append(pred)
                if is_closed:
                    reward = closed_positive if normalize_answer(pred) == normalize_answer(target) else closed_negative
                    rewards.append(reward)
                    reward_breakdown.append({"exact": float(reward > 0), "reward": reward})
                else:
                    reward, details = compute_single_open_reward(pred, target)
                    rewards.append(reward)
                    reward_breakdown.append(details | {"reward": reward})

            completion_texts = [f" {pred}{eos}" for pred in sampled_answers]
            rollout_batch, rollout_mask = build_multimodal_completion_batch(
                processor,
                prompts,
                completion_texts,
                images,
                max_length=max_seq_length,
            )

            with torch.no_grad():
                old_seq_log_probs, _ = compute_masked_sequence_logprobs(model, rollout_batch, rollout_mask)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=old_seq_log_probs.device)
            if reward_tensor.numel() > 1:
                advantages = reward_tensor - reward_tensor.mean()
                advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)
            else:
                advantages = reward_tensor

            optimizer.zero_grad(set_to_none=True)
            new_seq_log_probs, entropy = compute_masked_sequence_logprobs(model, rollout_batch, rollout_mask)
            ratios = torch.exp(new_seq_log_probs - old_seq_log_probs.detach())
            clipped_ratios = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
            surrogate_1 = ratios * advantages
            surrogate_2 = clipped_ratios * advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            entropy_bonus = entropy.mean()
            loss = policy_loss - (entropy_coef * entropy_bonus)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            scheduler.step()

            closed_rewards = [r for r, is_closed in zip(rewards, closed_flags) if is_closed]
            open_rewards = [r for r, is_closed in zip(rewards, closed_flags) if not is_closed]
            log_record = {
                "epoch": 1,
                "update": update_idx + 1,
                "train_loss": float(loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "entropy": float(entropy_bonus.detach().cpu().item()),
                "avg_reward": float(sum(rewards) / len(rewards)),
                "avg_closed_reward": float(sum(closed_rewards) / len(closed_rewards)) if closed_rewards else None,
                "avg_open_reward": float(sum(open_rewards) / len(open_rewards)) if open_rewards else None,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "sample_predictions": sampled_answers[:2],
                "sample_targets": targets[:2],
                "reward_breakdown": reward_breakdown[:2],
            }
            ppo_history.append(log_record)

            if wandb.run:
                wandb.log({
                    "ppo/train_loss": log_record["train_loss"],
                    "ppo/policy_loss": log_record["policy_loss"],
                    "ppo/entropy": log_record["entropy"],
                    "ppo/avg_reward": log_record["avg_reward"],
                    "ppo/avg_closed_reward": log_record["avg_closed_reward"],
                    "ppo/avg_open_reward": log_record["avg_open_reward"],
                    "ppo/learning_rate": log_record["learning_rate"],
                    "ppo/update": log_record["update"],
                })

            del generation_inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_ppo_dir = Path("checkpoints/PPO/final_adapter")
        final_ppo_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_ppo_dir))
        processor.save_pretrained(str(final_ppo_dir))
        with open("checkpoints/medical_vqa_ppo_from.txt", "w", encoding="utf-8") as f:
            f.write(str(b2_checkpoint))

        print("[INFO] Đang chạy đánh giá nghiệm thu trên tập Validation cho PPO...")
        model.eval()
        metrics = evaluate_multimodal_vqa(
            model,
            val_loader,
            device,
            processor,
            beam_width=config['eval'].get('beam_width_b', 1),
            beam_width_closed=config['eval'].get('beam_width_b_closed', 1),
            beam_width_open=config['eval'].get('beam_width_b_open', config['eval'].get('beam_width_b', 1)),
            max_new_tokens_closed=config['eval'].get('max_new_tokens_b_closed', 4),
            max_new_tokens_open=config['eval'].get('max_new_tokens_b_open', answer_max_words + 6),
            generation_batch_size=config['eval'].get('generation_batch_size_b', 1),
            max_words=answer_max_words,
            variant='PPO'
        )

        closed_eval = metrics.get('closed_eval', {})
        open_eval = metrics.get('open_eval', {})
        ppo_history.append({
            "epoch": 1,
            "val_accuracy_normalized": metrics.get('accuracy_normalized'),
            "val_f1_normalized": metrics.get('f1_normalized'),
            "val_bleu4_normalized": metrics.get('bleu4_normalized'),
            "val_bert_score_raw": metrics.get('bert_score_raw'),
            "val_semantic_raw": metrics.get('semantic_raw'),
            "val_closed_accuracy": closed_eval.get('accuracy', 0),
            "val_closed_em": closed_eval.get('em', 0),
            "val_closed_f1": closed_eval.get('f1', 0),
            "val_open_semantic": open_eval.get('semantic', 0),
            "val_open_bertscore": open_eval.get('bert_score', 0),
            "val_open_f1": open_eval.get('f1', 0),
            "val_open_rouge_l": open_eval.get('rouge_l', 0),
        })

        b2_metrics = load_latest_variant_metrics(os.path.join(config['log_dir'], "history"), "B2")
        ppo_acceptance = evaluate_refinement_acceptance(b2_metrics, ppo_history[-1])
        ppo_history[-1]["ppo_acceptance"] = ppo_acceptance
        print(f"[INFO] {ppo_acceptance['summary']}")
        if ppo_acceptance["status"] == "accepted":
            print("[SUCCESS] PPO accepted: dat tieu chi refinement nhe tren B2.")
        elif ppo_acceptance["status"] == "failed":
            print("[WARN] PPO failed, keep B2. Khong khuyen nghi tiep tuc tuning them.")

        os.makedirs("checkpoints/PPO", exist_ok=True)
        with open("checkpoints/PPO/acceptance_summary.json", "w", encoding="utf-8") as f:
            json.dump(ppo_acceptance, f, ensure_ascii=False, indent=2)

        save_history_records(history_dir, ppo_history)
        print("[SUCCESS] Đã lưu checkpoint và metrics PPO.")
        return

    elif args.variant == 'DPO':
        from trl import DPOTrainer
        try:
            from trl import DPOConfig
        except ImportError:
            DPOConfig = None
        from transformers import TrainingArguments
        from datasets import Dataset as HFDataset
        import inspect
        import json
        
        dpo_answer_max_words = int(config.get('dpo', {}).get('max_answer_words', min(answer_max_words, 6)))
        wrapper = MultimodalVQA(
            model_id=config['model_b']['model_name'],
            lora_r=int(config['model_b'].get('lora_r', 16)),
            lora_alpha=int(config['model_b'].get('lora_alpha', 32)),
            lora_dropout=float(config['model_b'].get('lora_dropout', 0.05)),
            lora_target_modules=config['model_b'].get('lora_target_modules'),
        )
        b2_checkpoint = select_best_adapter_checkpoint(config['train'].get('b2_output_dir', './checkpoints/B2'))
        print(f"[INFO] DPO sẽ khởi tạo từ B2 checkpoint: {b2_checkpoint}")
        model, processor = wrapper.load_model(adapter_path=str(b2_checkpoint), is_trainable=True)
        if not config['train'].get('dpo_train_mlp_lora', False):
            frozen_lora = 0
            for name, param in model.named_parameters():
                if "lora_" in name and any(proj in name for proj in ("gate_proj", "up_proj", "down_proj")):
                    param.requires_grad = False
                    frozen_lora += param.numel()
            print(f"[INFO] DPO đang freeze LoRA MLP để giảm VRAM: {frozen_lora:,} tham số")
            model.print_trainable_parameters()
        
        # Tạo/Load Preference Data
        pref_json = config.get('dpo', {}).get('preference_data', 'data/preference_data_slake.json')
        force_rebuild_pref = bool(config.get('dpo', {}).get('force_rebuild_preference_data', False))
        if force_rebuild_pref and os.path.exists(pref_json):
            print(f"[INFO] Dang xoa preference data cu de tao lai theo cau hinh hien tai: {pref_json}")
            os.remove(pref_json)

        if not os.path.exists(pref_json):
            print(f"[INFO] Chưa có preference data. Đang tự động tạo từ training data...")
            from src.engine.dpo_trainer import create_preference_data
            if hf_repo:
                raw_data = [{"question_vi": item["question_vi"], "answer_vi": get_target_answer(item, max_words=dpo_answer_max_words), 
                             "image_name": item.get("image_name"),
                             "source_idx": i} 
                            for i, item in enumerate(dataset_dict['train'])]
                tmp_json = "data/tmp_train_for_dpo.json"
                os.makedirs("data", exist_ok=True)
                with open(tmp_json, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, ensure_ascii=False, indent=2)
                create_preference_data(
                    tmp_json,
                    pref_json,
                    num_pairs=int(config.get('dpo', {}).get('num_pairs', 400)),
                    closed_ratio=float(config.get('dpo', {}).get('closed_ratio', 0.6)),
                    max_answer_words=dpo_answer_max_words,
                )
            else:
                create_preference_data(
                    config['data']['vqa_json'],
                    pref_json,
                    num_pairs=int(config.get('dpo', {}).get('num_pairs', 400)),
                    closed_ratio=float(config.get('dpo', {}).get('closed_ratio', 0.6)),
                    max_answer_words=dpo_answer_max_words,
                )
        
        # Đọc file JSON preference data
        with open(pref_json, 'r', encoding='utf-8') as f:
            pref_data = json.load(f)

        if hf_repo and any("source_idx" not in item for item in pref_data):
            print("[INFO] Preference data cu khong co source_idx. Dang tao lai de giu lien ket image cho DPO...")
            from src.engine.dpo_trainer import create_preference_data
            raw_data = [{"question_vi": item["question_vi"], "answer_vi": get_target_answer(item, max_words=dpo_answer_max_words),
                         "image_name": item.get("image_name"), "source_idx": i}
                        for i, item in enumerate(dataset_dict['train'])]
            tmp_json = "data/tmp_train_for_dpo.json"
            with open(tmp_json, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)
            create_preference_data(
                tmp_json,
                pref_json,
                num_pairs=int(config.get('dpo', {}).get('num_pairs', 400)),
                closed_ratio=float(config.get('dpo', {}).get('closed_ratio', 0.6)),
                max_answer_words=dpo_answer_max_words,
            )
            with open(pref_json, 'r', encoding='utf-8') as f:
                pref_data = json.load(f)
            
        # Chuẩn bị HF Dataset cho DPOTrainer (yêu cầu cột: prompt, chosen, rejected)
        prompts, chosens, rejecteds, images = [], [], [], []
        eos = processor.tokenizer.eos_token or ""
        filtered_pairs = 0
        for item in pref_data:
            q = item.get("question", "")
            chosen = sanitize_dpo_completion(q, item.get("chosen", ""), max_words=dpo_answer_max_words)
            rejected = sanitize_dpo_completion(q, item.get("rejected", ""), max_words=dpo_answer_max_words)
            image = resolve_dpo_image(
                item,
                hf_train_data=dataset_dict['train'] if hf_repo else None,
                image_dir=config['data'].get('image_dir'),
            )

            if not chosen or not rejected or chosen == rejected or image is None:
                filtered_pairs += 1
                continue

            prompts.append(build_dpo_instruction_prompt(q, max_words=dpo_answer_max_words))
            chosens.append(f" {chosen}{eos}")
            rejecteds.append(f" {rejected}{eos}")
            images.append(image)

        if not prompts:
            raise ValueError("Khong con cap preference hop le sau khi sanitize DPO data.")
        if filtered_pairs:
            print(f"[INFO] Da bo qua {filtered_pairs} cap preference khong hop le sau sanitize.")

        dpo_hf_dataset = HFDataset.from_dict({
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
            "image": images,
        })

        class MultimodalDPODataCollator:
            def __init__(self, processor, max_length=None):
                self.processor = processor
                self.tokenizer = processor.tokenizer
                # LLaVA expands a single <image> placeholder into hundreds of visual tokens.
                # If max_length is too small, the processor truncates those tokens and raises
                # "image token count" mismatch. Keep a safe floor for multimodal DPO.
                self.max_length = max(max_length or 0, 768) if max_length is not None else None

            def __call__(self, examples):
                prompts = [example["prompt"] for example in examples]
                chosens = [example["chosen"] for example in examples]
                rejecteds = [example["rejected"] for example in examples]
                images = [example["image"] for example in examples]

                full_texts = [f"{prompt}{chosen}" for prompt, chosen in zip(prompts, chosens)]
                full_texts.extend(f"{prompt}{rejected}" for prompt, rejected in zip(prompts, rejecteds))
                repeated_prompts = prompts + prompts
                repeated_images = images + images

                batch = self.processor(
                    text=full_texts,
                    images=repeated_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )

                prompt_batch = self.processor(
                    text=repeated_prompts,
                    images=repeated_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )

                completion_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
                prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
                for i, prompt_len in enumerate(prompt_lengths.tolist()):
                    token_positions = batch["attention_mask"][i].nonzero(as_tuple=True)[0]
                    completion_mask[i, token_positions[prompt_len:]] = 1

                if self.max_length is not None and batch["input_ids"].shape[1] > self.max_length:
                    batch["input_ids"] = batch["input_ids"][:, :self.max_length]
                    batch["attention_mask"] = batch["attention_mask"][:, :self.max_length]
                    completion_mask = completion_mask[:, :self.max_length]
                    for key in ("token_type_ids", "mm_token_type_ids"):
                        if key in batch:
                            batch[key] = batch[key][:, :self.max_length]

                batch["completion_mask"] = completion_mask
                return batch
        
        dpo_sequence_limits = {
            "max_length": max(int(config['train'].get('dpo_max_length', 768)), 768),
            "max_prompt_length": int(config['train'].get('dpo_max_prompt_length', 96)),
            "max_completion_length": int(config['train'].get('dpo_max_completion_length', 24)),
        }
        training_args_dict = {
            "output_dir": "./checkpoints/DPO",
            "per_device_train_batch_size": int(config['train'].get('dpo_batch_size', 1)),
            "gradient_accumulation_steps": int(config['train'].get('dpo_gradient_accumulation_steps', 8)),
            "num_train_epochs": config['train'].get('dpo_epochs', 1),
            "learning_rate": float(config.get('dpo', {}).get('learning_rate', 1.0e-6)),
            "lr_scheduler_type": "cosine",       # [OPTIMIZED] Giúp hội tụ mượt mà hơn
            "warmup_ratio": 0.1,                 # [OPTIMIZED] Tránh sốc gradient ở epoch đầu
            "bf16": True,
            "remove_unused_columns": False,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "optim": config['train'].get('dpo_optim', 'paged_adamw_8bit'),
            "gradient_checkpointing": True,
        }
        
        if DPOConfig is not None:
            training_args_dict["beta"] = float(config.get('dpo', {}).get('beta', 0.1))
            dpo_config_params = set(inspect.signature(DPOConfig.__init__).parameters)
            for key, value in dpo_sequence_limits.items():
                if key in dpo_config_params:
                    training_args_dict[key] = value
            training_args = DPOConfig(**training_args_dict)
        else:
            training_args = build_training_arguments(TrainingArguments, **training_args_dict)
            training_args.model_init_kwargs = None
        
        dpo_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": dpo_hf_dataset,
            "data_collator": MultimodalDPODataCollator(processor, max_length=dpo_sequence_limits["max_length"]),
        }
        dpo_trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters)
        for key, value in dpo_sequence_limits.items():
            if key in dpo_trainer_params:
                dpo_kwargs[key] = value
        
        try:
            print("[INFO] Thử khởi tạo DPOTrainer với processing_class...")
            trainer = DPOTrainer(**dpo_kwargs, processing_class=processor)
        except TypeError:
            try:
                trainer = DPOTrainer(**dpo_kwargs, tokenizer=processor)
            except TypeError:
                trainer = DPOTrainer(**dpo_kwargs, tokenizer=processor.tokenizer)

        print("[INFO] Bắt đầu huấn luyện DPO...")
        trainer.train()
        os.makedirs("checkpoints", exist_ok=True)
        final_dpo_dir = Path("checkpoints/DPO/final_adapter")
        final_dpo_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dpo_dir))
        processor.save_pretrained(str(final_dpo_dir))
        with open("checkpoints/medical_vqa_dpo_from.txt", "w", encoding="utf-8") as f:
            f.write(str(b2_checkpoint))

        # [FIX] Đánh giá DPO sau khi train xong để có Accuracy, F1, BLEU cho biểu đồ so sánh
        from src.engine.medical_eval import evaluate_multimodal_vqa
        print("[INFO] Đang chạy đánh giá nghiệm thu trên tập Validation cho DPO...")
        model.eval()
        metrics = evaluate_multimodal_vqa(
            model, 
            val_loader, 
            device, 
            processor, 
            beam_width=config['eval'].get('beam_width_b', 1),
            beam_width_closed=config['eval'].get('beam_width_b_closed', 1),
            beam_width_open=config['eval'].get('beam_width_b_open', config['eval'].get('beam_width_b', 1)),
            max_new_tokens_closed=config['eval'].get('max_new_tokens_b_closed', 4),
            max_new_tokens_open=config['eval'].get('max_new_tokens_b_open', answer_max_words + 6),
            generation_batch_size=config['eval'].get('generation_batch_size_b', 1),
            max_words=answer_max_words,
            variant='DPO'
        )
        
        closed_eval = metrics.get('closed_eval', {})
        open_eval = metrics.get('open_eval', {})

        print(f"\n[RESULT DPO - CLOSED QUESTIONS]")
        print(f"Count: {closed_eval.get('count', 0)}")
        print(f"Accuracy: {closed_eval.get('accuracy', 0):.4f}")
        print(f"EM: {closed_eval.get('em', 0):.4f}")
        print(f"F1: {closed_eval.get('f1', 0):.4f}")

        print(f"\n[RESULT DPO - OPEN QUESTIONS]")
        print(f"Count: {open_eval.get('count', 0)}")
        print(f"Semantic: {open_eval.get('semantic', 0):.4f}")
        print(f"BERTScore: {open_eval.get('bert_score', 0):.4f}")
        print(f"F1: {open_eval.get('f1', 0):.4f}")
        print(f"ROUGE-L: {open_eval.get('rouge_l', 0):.4f}")
        
        final_epoch = training_args.num_train_epochs
        trainer.state.log_history.append({
            "epoch": final_epoch,
            "val_accuracy_normalized": metrics.get('accuracy_normalized'),
            "val_f1_normalized": metrics.get('f1_normalized'),
            "val_bleu4_normalized": metrics.get('bleu4_normalized'),
            "val_bert_score_raw": metrics.get('bert_score_raw'),
            "val_semantic_raw": metrics.get('semantic_raw'),
            "val_closed_accuracy": closed_eval.get('accuracy', 0),
            "val_closed_em": closed_eval.get('em', 0),
            "val_closed_f1": closed_eval.get('f1', 0),
            "val_open_semantic": open_eval.get('semantic', 0),
            "val_open_bertscore": open_eval.get('bert_score', 0),
            "val_open_f1": open_eval.get('f1', 0),
            "val_open_rouge_l": open_eval.get('rouge_l', 0),
        })
        b2_metrics = load_latest_variant_metrics(os.path.join(config['log_dir'], "history"), "B2")
        dpo_acceptance = evaluate_dpo_acceptance(b2_metrics, trainer.state.log_history[-1])
        trainer.state.log_history[-1]["dpo_acceptance"] = dpo_acceptance
        print(f"[INFO] {dpo_acceptance['summary']}")
        if dpo_acceptance["status"] == "accepted":
            print("[SUCCESS] DPO accepted: dat tieu chi refinement nhe tren B2.")
        elif dpo_acceptance["status"] == "failed":
            print("[WARN] DPO failed, keep B2. Khong khuyen nghi tiep tuc tuning them.")
        os.makedirs("checkpoints/DPO", exist_ok=True)
        with open("checkpoints/DPO/acceptance_summary.json", "w", encoding="utf-8") as f:
            json.dump(dpo_acceptance, f, ensure_ascii=False, indent=2)
        
        save_history_records(history_dir, trainer.state.log_history)
        print("[SUCCESS] Đã lưu checkpoint và metrics DPO.")
        return

    elif args.variant == 'B2':
        # Fine-tuning LLaVA-Med
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset as HFDataset
        
        wrapper = MultimodalVQA(
            model_id=config['model_b']['model_name'],
            lora_r=int(config['model_b'].get('lora_r', 16)),
            lora_alpha=int(config['model_b'].get('lora_alpha', 32)),
            lora_dropout=float(config['model_b'].get('lora_dropout', 0.05)),
            lora_target_modules=config['model_b'].get('lora_target_modules'),
        )
        model, processor = wrapper.load_model()
        
        def make_sft_dataset(raw_ds):
            prompts = []
            answers = []
            texts = []
            images = []
            for i in range(len(raw_ds)):
                item = raw_ds[i]
                if isinstance(item, dict):
                    q = item.get("question_vi", item.get("question", item.get("raw_questions", "")))
                    a = get_target_answer(item, max_words=answer_max_words)
                    answer_type = str(item.get("answer_type", "")).upper()
                    label_closed = item.get("label_closed", None)
                    if answer_type == "CLOSED" or label_closed in (0, 1) or a in {"có", "không", "yes", "no"}:
                        a_norm = str(a).strip().lower()
                        a = "không" if a_norm in {"không", "khong", "no", "false", "absent"} else "có"
                    prompt = wrapper.build_instruction_prompt(q, language="vi", include_answer=False)
                    prompts.append(prompt)
                    answers.append(a)
                    eos = processor.tokenizer.eos_token or ""
                    texts.append(f"{prompt} {a}{eos}")
                    
                    img = item.get("image", None)
                    if img is not None:
                        if img.mode != "RGB": img = img.convert("RGB")
                    images.append(img)
            return HFDataset.from_dict({"prompt": prompts, "answer": answers, "text": texts, "image": images})
        
        if hf_repo:
            sft_train = make_sft_dataset(dataset_dict['train'])
            sft_val = make_sft_dataset(dataset_dict['validation'])
        else:
            sft_train = make_sft_dataset(train_ds)
            sft_val = make_sft_dataset(val_ds)
            
        class MultimodalDataCollator:
            def __init__(self, processor, max_length=None):
                self.processor = processor
                self.tokenizer = processor.tokenizer
                self.max_length = max_length
            def __call__(self, examples):
                texts = [example["text"] for example in examples]
                prompts = [example["prompt"] for example in examples]
                images = [example["image"] for example in examples]

                batch = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                labels = batch["input_ids"].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Mask the full prompt so SFT loss is computed only on the answer.
                # Searching for "ASSISTANT:" token ids is brittle because tokenization can
                # split the separator differently across models.
                prompt_batch = self.processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
                for i, prompt_len in enumerate(prompt_lengths.tolist()):
                    token_positions = batch["attention_mask"][i].nonzero(as_tuple=True)[0]
                    labels[i, token_positions[:prompt_len]] = -100
                
                batch["labels"] = labels
                # Remove text and image lists as Trainer only wants tensors
                return batch

        b2_micro_batch = int(config['train'].get('b2_batch_size', 1))
        b2_grad_accum = int(config['train'].get('b2_gradient_accumulation_steps', max(config['train'].get('gradient_accumulation_steps', 2), 1)))
        b2_max_length = int(config['train'].get('b2_max_length', config['data'].get('max_question_len', 64) + config['data'].get('max_answer_len', 20) + 32))
        
        training_args = build_training_arguments(
            TrainingArguments,
            output_dir="./checkpoints/B2",
            per_device_train_batch_size=b2_micro_batch,
            per_device_eval_batch_size=int(config['train'].get('b2_eval_batch_size', 1)),
            gradient_accumulation_steps=b2_grad_accum,
            num_train_epochs=config['train'].get('epochs', 3),
            learning_rate=float(config['train'].get('b2_lr', 2.0e-5)),
            lr_scheduler_type="cosine",
            warmup_steps=int(config['train'].get('b2_warmup_steps', 50)),
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            optim=config['train'].get('b2_optim', 'paged_adamw_8bit'),
            max_grad_norm=float(config['train'].get('grad_clip', 1.0)),
            dataloader_num_workers=int(config['train'].get('b2_num_workers', 4)),
            dataloader_pin_memory=bool(config['train'].get('pin_memory', True)),
            load_best_model_at_end=config['train'].get('b2_load_best_model_at_end', True),
            metric_for_best_model=config['train'].get('b2_metric_for_best', 'eval_loss'),
            greater_is_better=False,
        )

        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=sft_train,
            eval_dataset=sft_val,
            data_collator=MultimodalDataCollator(processor, max_length=b2_max_length)
        )
            
        trainer.train()
        
        # [FIX] Đánh giá B2 sau khi train xong để có Accuracy, F1, BLEU cho biểu đồ so sánh
        from src.engine.medical_eval import evaluate_multimodal_vqa
        print("[INFO] Đang chạy đánh giá nghiệm thu trên tập Validation cho B2...")
        # Đưa model về evaluation mode
        model.eval()
        metrics = evaluate_multimodal_vqa(
            model, 
            val_loader, 
            device, 
            processor, 
            beam_width=config['eval'].get('beam_width_b', 1),
            beam_width_closed=config['eval'].get('beam_width_b_closed', 1),
            beam_width_open=config['eval'].get('beam_width_b_open', config['eval'].get('beam_width_b', 1)),
            max_new_tokens_closed=config['eval'].get('max_new_tokens_b_closed', 4),
            max_new_tokens_open=config['eval'].get('max_new_tokens_b_open', answer_max_words + 6),
            generation_batch_size=config['eval'].get('generation_batch_size_b', 1),
            max_words=answer_max_words,
            variant='B2'
        )
        
        closed_eval = metrics.get('closed_eval', {})
        open_eval = metrics.get('open_eval', {})

        print(f"\n[RESULT B2 - CLOSED QUESTIONS]")
        print(f"Count: {closed_eval.get('count', 0)}")
        print(f"Accuracy: {closed_eval.get('accuracy', 0):.4f}")
        print(f"EM: {closed_eval.get('em', 0):.4f}")
        print(f"F1: {closed_eval.get('f1', 0):.4f}")

        print(f"\n[RESULT B2 - OPEN QUESTIONS]")
        print(f"Count: {open_eval.get('count', 0)}")
        print(f"Semantic: {open_eval.get('semantic', 0):.4f}")
        print(f"BERTScore: {open_eval.get('bert_score', 0):.4f}")
        print(f"F1: {open_eval.get('f1', 0):.4f}")
        print(f"ROUGE-L: {open_eval.get('rouge_l', 0):.4f}")
        
        if 'long_answers_eval' in metrics:
            print(f"\n[RESULT B2 - LONG METRICS]")
            print(f"Accuracy: {metrics['long_answers_eval'].get('accuracy', 0):.4f}")
            print(f"F1: {metrics['long_answers_eval'].get('f1', 0):.4f}")
            print(f"Semantic: {metrics['long_answers_eval'].get('semantic', 0):.4f}")
            print(f"BERTScore: {metrics['long_answers_eval'].get('bert_score', 0):.4f}")
            
            # Gắn thêm vào log_history cho wandb
            trainer.state.log_history.append({
                "epoch": training_args.num_train_epochs,
                "val_long_accuracy": metrics['long_answers_eval'].get('accuracy', 0),
                "val_long_f1": metrics['long_answers_eval'].get('f1', 0),
                "val_long_semantic": metrics['long_answers_eval'].get('semantic', 0),
                "val_long_bertscore": metrics['long_answers_eval'].get('bert_score', 0),
            })
            
        # Gắn kết quả vào history để compare_models.py đọc được
        final_epoch = training_args.num_train_epochs
        trainer.state.log_history.append({
            "epoch": final_epoch,
            "val_accuracy_normalized": metrics.get('accuracy_normalized'),
            "val_f1_normalized": metrics.get('f1_normalized'),
            "val_bleu4_normalized": metrics.get('bleu4_normalized'),
            "val_bert_score_raw": metrics.get('bert_score_raw'),
            "val_semantic_raw": metrics.get('semantic_raw'),
            "val_closed_accuracy": closed_eval.get('accuracy', 0),
            "val_closed_em": closed_eval.get('em', 0),
            "val_closed_f1": closed_eval.get('f1', 0),
            "val_open_semantic": open_eval.get('semantic', 0),
            "val_open_bertscore": open_eval.get('bert_score', 0),
            "val_open_f1": open_eval.get('f1', 0),
            "val_open_rouge_l": open_eval.get('rouge_l', 0),
        })
        
        save_history_records(history_dir, trainer.state.log_history)
        return

    elif args.variant == 'B1':
        # Zero-shot Evaluation cho Hướng B
        from src.engine.medical_eval import evaluate_multimodal_vqa
        
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        beam_width = config['eval'].get('beam_width_b', 1)
        print(f"[INFO] Bắt đầu đánh giá B1 với Beam Width = {beam_width}...")
        
        metrics = evaluate_multimodal_vqa(
            model, 
            val_loader, 
            device, 
            processor, 
            beam_width=beam_width,
            beam_width_closed=config['eval'].get('beam_width_b_closed', beam_width),
            beam_width_open=config['eval'].get('beam_width_b_open', beam_width),
            max_new_tokens_closed=config['eval'].get('max_new_tokens_b_closed', 4),
            max_new_tokens_open=config['eval'].get('max_new_tokens_b_open', answer_max_words + 6),
            generation_batch_size=config['eval'].get('generation_batch_size_b', 1),
            max_words=answer_max_words,
            variant='B1'
        )
        
        closed_eval = metrics.get('closed_eval', {})
        open_eval = metrics.get('open_eval', {})

        print(f"\n[RESULT B1 - CLOSED QUESTIONS]")
        print(f"Count: {closed_eval.get('count', 0)}")
        print(f"Accuracy: {closed_eval.get('accuracy', 0):.4f}")
        print(f"EM: {closed_eval.get('em', 0):.4f}")
        print(f"F1: {closed_eval.get('f1', 0):.4f}")

        print(f"\n[RESULT B1 - OPEN QUESTIONS]")
        print(f"Count: {open_eval.get('count', 0)}")
        print(f"Semantic: {open_eval.get('semantic', 0):.4f}")
        print(f"BERTScore: {open_eval.get('bert_score', 0):.4f}")
        print(f"F1: {open_eval.get('f1', 0):.4f}")
        print(f"ROUGE-L: {open_eval.get('rouge_l', 0):.4f}")
        
        if 'long_answers_eval' in metrics:
            print(f"\n[RESULT B1 - LONG METRICS]")
            print(f"Accuracy: {metrics['long_answers_eval'].get('accuracy', 0):.4f}")
            print(f"F1: {metrics['long_answers_eval'].get('f1', 0):.4f}")
            print(f"Semantic: {metrics['long_answers_eval'].get('semantic', 0):.4f}")
            print(f"BERTScore: {metrics['long_answers_eval'].get('bert_score', 0):.4f}")
        # [FIX] Lưu dưới dạng record có 'epoch' để compare_models.py có thể parse
        save_history_records(history_dir, [{
            "epoch": 1,
            "variant": "B1",
            "beam_width": beam_width,
            "train_loss": 0.0,   # zero-shot không có train loss
            "val_accuracy_normalized": float(metrics.get('accuracy_normalized', metrics.get('accuracy', 0))),
            "val_f1_normalized":       float(metrics.get('f1_normalized', metrics.get('f1', 0))),
            "val_bleu4_normalized":    float(metrics.get('bleu4_normalized', metrics.get('bleu4', 0))),
            "val_bert_score_raw":      float(metrics.get('bert_score_raw', metrics.get('bert_score', 0))),
            "val_semantic_raw":        float(metrics.get('semantic_raw', metrics.get('semantic', 0))),
            "val_closed_accuracy":     float(closed_eval.get('accuracy', 0)),
            "val_closed_em":           float(closed_eval.get('em', 0)),
            "val_closed_f1":           float(closed_eval.get('f1', 0)),
            "val_open_semantic":       float(open_eval.get('semantic', 0)),
            "val_open_bertscore":      float(open_eval.get('bert_score', 0)),
            "val_open_f1":             float(open_eval.get('f1', 0)),
            "val_open_rouge_l":        float(open_eval.get('rouge_l', 0)),
            "metrics": metrics,
        }])
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/medical_vqa.yaml")
    parser.add_argument("--variant",     type=str, choices=['A1', 'A2', 'B1', 'B2', 'DPO', 'PPO'], required=True)
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no_compare",  action="store_true",
                        help="Bỏ qua vẽ chart so sánh 5 model sau khi train xong")
    args = parser.parse_args()
    train(args)

    # Auto-generate comparison charts after training
    if not args.no_compare:
        import subprocess, sys
        log_dir  = "logs/medical_vqa/history"
        out_dir  = "results/charts"
        print(f"\n[INFO] 📊 Tự động vẽ biểu đồ so sánh 5 model → {out_dir}/")
        try:
            subprocess.run(
                [sys.executable, "scripts/compare_models.py",
                 "--log_dir", log_dir, "--out", out_dir],
                check=False
            )
        except Exception as e:
            print(f"[WARNING] compare_models.py thất bại: {e}")
            print("  Chạy thủ công: python scripts/compare_models.py")

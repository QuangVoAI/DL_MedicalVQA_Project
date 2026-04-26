import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import yaml
import argparse
import os

from datasets import load_dataset
# Import các thành phần từ thư mục src
from src.models.medical_vqa_model import MedicalVQAModelA
from src.models.multimodal_vqa import MultimodalVQA
from src.utils.visualization import MedicalImageTransform as MedicalTransform
from src.data.medical_dataset import MedicalVQADataset
from src.utils.metrics import batch_metrics

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

def train(args):
    # 1. Load Cấu hình
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Khởi tạo WandB
    if os.environ.get('WANDB_API_KEY'):
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(
            project='MedicalVQA-Vietnam',
            name=f'Variant-{args.variant}',
            config=config
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị sử dụng: {device}")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model_a']['phobert_model'])
    transform = MedicalTransform(size=config['data']['image_size'])
    
    # Nạp dữ liệu từ HuggingFace Hub hoặc cục bộ
    hf_repo = config['data'].get('hf_dataset')
    if hf_repo:
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
            max_ans_len=config['data']['max_answer_len']
        )
        val_ds = MedicalVQADataset(
            hf_dataset=dataset_dict['validation'], 
            tokenizer=tokenizer, 
            transform=transform, 
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len']
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
            max_ans_len=config['data']['max_answer_len']
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
        batch_size=config['train'].get('eval_batch_size', 8), 
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
        
        # Thiết lập Optimizer với Differential Learning Rate
        optimizer = optim.AdamW([
            {'params': model.image_encoder.parameters(), 'lr': float(config['train']['vision_lr'])},
            {'params': model.text_encoder.parameters(), 'lr': float(config['train']['phobert_lr'])},
            {'params': model.fusion.parameters(), 'lr': float(config['train']['learning_rate'])},
            {'params': model.decoder.parameters(), 'lr': float(config['train']['learning_rate'])}
        ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
        
        # Khởi tạo Trainer với pad_token_id được sửa lỗi
        from src.engine.trainer import MedicalVQATrainer
        trainer = MedicalVQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config={**config, 'variant': args.variant},
            pad_token_id=tokenizer.pad_token_id
        )

        print(f"[INFO] Bắt đầu huấn luyện cấu hình {args.variant}...")
        trainer.train(config['train']['epochs'], tokenizer=tokenizer)
        return

    elif args.variant == 'DPO':
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        # Load Reference Model cho DPO
        ref_wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        reference_model, _ = ref_wrapper.load_model()
        
        optimizer = optim.AdamW(model.parameters(), lr=float(config['train'].get('dpo_lr', 5e-7)))
        
        from src.engine.dpo_trainer import MedicalDPOTrainer
        trainer = MedicalDPOTrainer(
            model=model,
            reference_model=reference_model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config
        )
        trainer.train(epochs=config['train'].get('dpo_epochs', 3))
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/medical_vqa_dpo.pth")
        return

    elif args.variant == 'B2':
        # Fine-tuning LLaVA-Med (SFT)
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from torch.utils.data import Dataset as TorchDataset
        
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        # Wrapper dataset: SFTTrainer cần field 'text' + column_names
        class SFTTextDataset(TorchDataset):
            column_names = ["text"]
            def __init__(self, hf_ds):
                self.data = hf_ds
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                q = item.get("question_vi", item.get("question", ""))
                a = item.get("answer_vi", item.get("answer", ""))
                text = f"USER: <image>\n{q} ASSISTANT: {a}"
                return {"text": text}
        
        # Lấy raw HF dataset (chưa qua MedicalVQADataset)
        sft_train = SFTTextDataset(dataset_dict['train'] if hf_repo else train_ds.dataset)
        sft_val = SFTTextDataset(dataset_dict['validation'] if hf_repo else val_ds.dataset)
        
        training_args = TrainingArguments(
            output_dir="./checkpoints/B2",
            per_device_train_batch_size=config['train']['batch_size'],
            num_train_epochs=config['train'].get('epochs', 3),
            fp16=True,
            remove_unused_columns=False,
            logging_steps=10
        )

        # Khởi tạo SFTTrainer với cơ chế fallback cho phiên bản TRL
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": sft_train,
            "eval_dataset": sft_val,
        }
        
        try:
            print("[INFO] Thử khởi tạo SFTTrainer với processing_class...")
            trainer = SFTTrainer(**trainer_kwargs, processing_class=processor)
        except TypeError:
            try:
                print("[INFO] Fallback: Thử với tokenizer...")
                trainer = SFTTrainer(**trainer_kwargs, tokenizer=processor)
            except TypeError:
                print("[INFO] Fallback: Thử với tokenizer.tokenizer...")
                trainer = SFTTrainer(**trainer_kwargs, tokenizer=processor.tokenizer)
            
        trainer.train()
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
            beam_width=beam_width
        )
        
        print(f"\n[RESULT B1]")
        print(f"Accuracy: {metrics.get('accuracy', metrics.get('vqa_accuracy', 0)):.4f}")
        print(f"F1: {metrics.get('f1', 0):.4f}")
        print(f"BLEU-4: {metrics.get('bleu4', 0):.4f}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/medical_vqa.yaml")
    parser.add_argument("--variant", type=str, choices=['A1', 'A2', 'B1', 'B2', 'DPO'], required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    train(args)
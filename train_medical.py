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
        from trl import DPOTrainer
        from transformers import TrainingArguments
        from datasets import Dataset as HFDataset
        import json
        
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        # Tạo/Load Preference Data
        pref_json = config.get('dpo', {}).get('preference_data', 'data/preference_data_slake.json')
        if not os.path.exists(pref_json):
            print(f"[INFO] Chưa có preference data. Đang tự động tạo từ training data...")
            from src.engine.dpo_trainer import create_preference_data
            if hf_repo:
                raw_data = [{"question_vi": item["question_vi"], "answer_vi": item["answer_vi"], 
                             "image_name": item.get("image_name", f"img_{i}.png")} 
                            for i, item in enumerate(dataset_dict['train'])]
                tmp_json = "data/tmp_train_for_dpo.json"
                os.makedirs("data", exist_ok=True)
                with open(tmp_json, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, ensure_ascii=False, indent=2)
                create_preference_data(tmp_json, pref_json, num_pairs=200)
            else:
                create_preference_data(config['data']['vqa_json'], pref_json, num_pairs=200)
        
        # Đọc file JSON preference data
        with open(pref_json, 'r', encoding='utf-8') as f:
            pref_data = json.load(f)
            
        # Chuẩn bị HF Dataset cho DPOTrainer (yêu cầu cột: prompt, chosen, rejected)
        prompts, chosens, rejecteds = [], [], []
        for item in pref_data:
            q = item.get("question", "")
            # LLaVA cần prompt định dạng chứa <image>
            prompts.append(f"USER: <image>\n{q} ASSISTANT:")
            # Kèm theo một khoảng trắng ở đầu (quy ước của TRL cho chosen/rejected)
            chosens.append(f" {item.get('chosen', '')}")
            rejecteds.append(f" {item.get('rejected', '')}")
            
        dpo_hf_dataset = HFDataset.from_dict({
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds
        })
        
        training_args = TrainingArguments(
            output_dir="./checkpoints/DPO",
            per_device_train_batch_size=config['train'].get('dpo_batch_size', 2),
            num_train_epochs=config['train'].get('dpo_epochs', 3),
            bf16=True, # LLaVA 4-bit dùng bfloat16
            remove_unused_columns=False,
            logging_steps=10
        )
        
        dpo_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": dpo_hf_dataset,
        }
        
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
        torch.save(model.state_dict(), f"checkpoints/medical_vqa_dpo.pth")
        print("[SUCCESS] Đã lưu checkpoint DPO.")
        return

    elif args.variant == 'B2':
        # Fine-tuning LLaVA-Med (SFT)
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset as HFDataset
        
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        # Chuyển đổi sang HF Dataset với field 'text' mà SFTTrainer yêu cầu
        def make_sft_dataset(raw_ds):
            texts = []
            for i in range(len(raw_ds)):
                item = raw_ds[i]
                # HF dataset trả về dict, còn MedicalVQADataset trả về dict khác
                if isinstance(item, dict):
                    q = item.get("question_vi", item.get("question", item.get("raw_questions", "")))
                    a = item.get("answer_vi", item.get("answer", item.get("raw_answer", "")))
                else:
                    q, a = "", ""
                texts.append(f"USER: <image>\n{q} ASSISTANT: {a}")
            return HFDataset.from_dict({"text": texts})
        
        if hf_repo:
            sft_train = make_sft_dataset(dataset_dict['train'])
            sft_val = make_sft_dataset(dataset_dict['validation'])
        else:
            sft_train = make_sft_dataset(train_ds)
            sft_val = make_sft_dataset(val_ds)
        
        training_args = TrainingArguments(
            output_dir="./checkpoints/B2",
            per_device_train_batch_size=config['train']['batch_size'],
            num_train_epochs=config['train'].get('epochs', 3),
            bf16=True,
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
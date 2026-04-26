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
# Import các linh kiện từ thư mục src
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

    # WandB Initialization
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
    
    # Logic nạp dữ liệu: Ưu tiên Hub nếu có hf_dataset
    hf_repo = config['data'].get('hf_dataset')
    if hf_repo:
        print(f"[INFO] Đang tải dữ liệu từ Hub: {hf_repo}")
        dataset_dict = load_dataset(hf_repo)
        
        if args.debug:
            print("[WARNING] DEBUG MODE: Chỉ lấy 20 mẫu từ Hub để test.")
            # Cắt giảm tập train cho nhanh
            train_idx = min(20, len(dataset_dict['train']))
            dataset_dict['train'] = dataset_dict['train'].select(range(train_idx))
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
        test_ds = MedicalVQADataset(
            hf_dataset=dataset_dict['test'], 
            tokenizer=tokenizer, 
            transform=transform, 
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len']
        )
    else:
        vqa_path = config['data']['vqa_json']
        if not os.path.exists(vqa_path):
            # Fallback nếu không tìm thấy file _cleaned
            vqa_path = vqa_path.replace("_cleaned", "")
            
        print(f"[INFO] Đang tải dữ liệu cục bộ từ: {vqa_path}")
        full_dataset = MedicalVQADataset(
            json_path=vqa_path,
            image_dir=config['data']['image_dir'],
            tokenizer=tokenizer,
            transform=transform,
            max_seq_len=config['data']['max_question_len'],
            max_ans_len=config['data']['max_answer_len']
        )
        if args.debug:
            print("[WARNING] DEBUG MODE: Chỉ lấy 20 mẫu từ dữ liệu cục bộ để test.")
            full_dataset.data = full_dataset.data[:20]
            config['train']['epochs'] = 2
            config['train']['batch_size'] = 2

        # Chia train/val/test 80/10/10
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Cập nhật DataLoader với collate_fn
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        collate_fn=vqa_collate_fn,
        num_workers=config['train'].get('num_workers', 0),
        pin_memory=config['train'].get('pin_memory', False)
    )
    eval_batch_size = config['train'].get('eval_batch_size', config['train']['batch_size'])
    val_loader = DataLoader(
        val_ds, 
        batch_size=eval_batch_size, 
        collate_fn=vqa_collate_fn,
        num_workers=config['train'].get('num_workers', 0),
        pin_memory=config['train'].get('pin_memory', False)
    )

    # 3. Khởi tạo Mô hình dựa trên Variant
    vocab_size = len(tokenizer)
    if args.variant in ['A1', 'A2']:
        decoder_type = "lstm" if args.variant == 'A1' else "transformer"
        model = MedicalVQAModelA(
            decoder_type=decoder_type, 
            vocab_size=vocab_size,
            hidden_size=config['model_a'].get('hidden_size', 768),
            phobert_model=config['model_a'].get('phobert_model', "vinai/phobert-base")
        ).to(device)
        
        # Tách parameter groups cho PhoBERT (Fine-tune nhẹ) và Decoder (Train mạnh)
        phobert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "text_encoder" in name:
                phobert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {"params": phobert_params, "lr": float(config['train'].get('phobert_lr', 1e-5))},
            {"params": other_params, "lr": float(config['train']['learning_rate'])}
        ])
        print(f"[INFO] Differential LR: PhoBERT ({config['train'].get('phobert_lr', 1e-5)}) | Other ({config['train']['learning_rate']})")
    
    elif args.variant in ['B1', 'B2']:
        # Hướng B: Multimodal Pretrained (LLaVA-Med)
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        optimizer = optim.AdamW(model.parameters(), lr=float(config['model_b'].get('learning_rate', 5e-5)))
    
    elif args.variant == 'DPO':
        # Hướng DPO: Tối ưu hóa dựa trên preference (thường bắt đầu từ B2)
        print("[INFO] Đang chuẩn bị cho huấn luyện DPO...")
        wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model, processor = wrapper.load_model()
        
        # Thử load checkpoint B2 nếu có
        b2_ckpt = "checkpoints/medical_vqa_B2.pth"
        if os.path.exists(b2_ckpt):
            print(f"[INFO] Đang tải checkpoint B2 làm base cho DPO: {b2_ckpt}")
            model.load_state_dict(torch.load(b2_ckpt, map_location=device))
        else:
            print("[WARNING] Cảnh báo: Không tìm thấy checkpoint B2. DPO sẽ chạy trên model gốc (không khuyến nghị).")
        
        # Load reference model (bản sao đóng băng của policy model tại thời điểm bắt đầu)
        ref_wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        reference_model, _ = ref_wrapper.load_model()
        if os.path.exists(b2_ckpt):
            reference_model.load_state_dict(torch.load(b2_ckpt, map_location=device))
        
        optimizer = optim.AdamW(model.parameters(), lr=float(config['train'].get('dpo_lr', 5e-7)))
        
        # Load preference data
        pref_json = "data/preference_data_slake.json"
        if not os.path.exists(pref_json):
             from src.engine.dpo_trainer import create_preference_data
             create_preference_data(config['data']['vqa_json'], pref_json)
             
        # Ở đây ta sử dụng MedicalVQADataset với chế độ DPO
        train_ds = MedicalVQADataset(
            json_path=pref_json, 
            image_dir=config['data']['image_dir'], 
            tokenizer=tokenizer, 
            transform=transform,
            is_dpo=True
        )
        train_loader = DataLoader(train_ds, batch_size=config['train'].get('dpo_batch_size', 2), shuffle=True)

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
        
        # [FIX 1] Thêm lệnh return để chặn luồng chạy rơi tự do xuống khối A1/A2
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/medical_vqa_dpo.pth")
        print(f"[SUCCESS] Đã lưu checkpoint DPO: checkpoints/medical_vqa_dpo.pth")
        return

    # [FIX 3] Đưa khối SFTTrainer của B2 lên trước khối Đánh giá (Eval)
    if args.variant == 'B2':
        print("[INFO] Đang khởi tạo SFTTrainer cho B2 (Fine-tuning LLaVA-Med)...")
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./checkpoints/B2",
            per_device_train_batch_size=config['train']['batch_size'],
            learning_rate=float(config['model_b'].get('learning_rate', 2e-5)),
            num_train_epochs=config['train'].get('epochs', 3), # Fix bug thiếu tham số 'train'
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
            fp16=True,
        )

        def formatting_func(example):
             return [f"USER: <image>\n{q} ASSISTANT: {a}" for q, a in zip(example['raw_questions'], example['raw_answer'])]

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=processor.tokenizer,
            packing=False,
        )
        trainer.train()
        print("[SUCCESS] Đã hoàn thành Fine-tuning B2.")

    # 4. Chạy Zero-shot / Eval cho Hướng B (B1 và B2 sau khi đã fine-tune)
    if args.variant in ['B1', 'B2']:
        from src.engine.medical_eval import evaluate_multimodal_vqa
        from src.utils.translator import MedicalTranslator
        
        print(f"[INFO] Bắt đầu đánh giá biến thể {args.variant} (LLaVA-Med)...")
        translator = MedicalTranslator(device=device)
        
        # [FIX] Sử dụng tên lớp chính xác: MultimodalVQA
        model_wrapper = MultimodalVQA(model_id=config['model_b']['model_name'])
        model_b, processor_b = model_wrapper.load_model()
        
        beam_width = config['eval'].get('beam_width_b', 1)
        print(f"[INFO] Sử dụng Beam Width = {beam_width} cho Hướng B")
        
        metrics = evaluate_multimodal_vqa(
            model_b, 
            val_loader, 
            device, 
            processor_b, 
            translator, 
            beam_width=beam_width
        )
        
        print(f"\n[RESULT {args.variant}]")
        print(f"Accuracy: {metrics.get('vqa_accuracy', 0):.4f}")
        print(f"F1: {metrics.get('f1', 0):.4f}")
        print(f"BLEU-4: {metrics.get('bleu', 0):.4f}")
        return

    # --- Hướng A: Huấn luyện Modular Models (A1, A2) ---
    if args.variant in ['A1', 'A2']:
        beam_width = config['eval'].get('beam_width_a', 5)
        print(f"[INFO] Sử dụng Beam Width = {beam_width} cho Hướng A")
        
        # Gom chung vòng lặp logic (đã xóa đoạn return sớm dư thừa ở code cũ)
        from src.engine.trainer import MedicalVQATrainer
        
        optimizer = optim.AdamW([
            {'params': model.image_encoder.parameters(), 'lr': float(config['train']['vision_lr'])},
            {'params': model.text_encoder.parameters(), 'lr': float(config['train']['phobert_lr'])},
            {'params': model.fusion.parameters(), 'lr': float(config['train']['learning_rate'])},
            {'params': model.decoder.parameters(), 'lr': float(config['train']['learning_rate'])}
        ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['train']['epochs'],
            eta_min=float(config['train'].get('eta_min', 1e-6))
        )
        
        trainer = MedicalVQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            pad_token_id=tokenizer.pad_token_id
        )

        print(f"[INFO] Bắt đầu huấn luyện cấu hình {args.variant}...")
        best_val_acc = 0
        patience = config['train'].get('patience', 3)
        counter = 0
        
        for epoch in range(config['train'].get('epochs', 10)):
            train_loss = trainer.train_epoch(epoch + 1)
            val_metrics = trainer.val_epoch(tokenizer, epoch=epoch + 1)
            val_acc = val_metrics['accuracy']
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/medical_vqa_{args.variant}_best.pth")
                print(f"[SUCCESS] Epoch {epoch+1}: Đã lưu model tốt nhất mới (Acc: {val_acc:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"[STOP] Early stopping tại epoch {epoch+1} do Accuracy không cải thiện.")
                    break

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/medical_vqa_{args.variant}_final.pth")
        print(f"[SUCCESS] Đã hoàn thành huấn luyện {args.variant}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/medical_vqa.yaml")
    parser.add_argument("--variant", type=str, choices=['A1', 'A2', 'B1', 'B2', 'DPO'], required=True)
    parser.add_argument("--debug", action="store_true", help="Chạy chế độ debug thử nghiệm")
    args = parser.parse_args()
    train(args)
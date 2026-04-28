import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import csv
import json

class MedicalVQATrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config, scheduler=None, pad_token_id=0, beam_width=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.beam_width = beam_width
        
        # [FIX] Đặt class weights thủ công để boost class "có" (index 1) giảm class imbalance
        self.criterion_closed = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.5]).to(device)
        )
        print("[INFO] Closed-head class weights manually set: không=1.0, có=2.5")
        
        self.criterion_open = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, 
            label_smoothing=config['train'].get('label_smoothing', 0.0)
        )
        
        # AMP (Automatic Mixed Precision)
        self.use_amp = config['train'].get('use_amp', False) and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = []

    @staticmethod
    def _flatten_dict(data, parent_key="", sep="."):
        items = {}
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            if isinstance(value, dict):
                items.update(MedicalVQATrainer._flatten_dict(value, new_key, sep=sep))
            elif isinstance(value, (list, tuple)):
                continue
            else:
                items[new_key] = value
        return items

    def save_history(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "history.json")
        csv_path = os.path.join(output_dir, "history.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        flat_rows = [self._flatten_dict(row) for row in self.history]
        if flat_rows:
            fieldnames = sorted({key for row in flat_rows for key in row.keys()})
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)

    @staticmethod
    def _compute_closed_weights(train_loader):
        """Đếm phân phối Yes/No và tính inverse frequency weights."""
        counts = {0: 0, 1: 0}  # 0=không, 1=có
        for batch in train_loader:
            labels = batch['label_closed']
            for lbl in labels:
                v = lbl.item()
                if v in counts:
                    counts[v] += 1
        
        total = counts[0] + counts[1]
        if total == 0:
            return torch.ones(2)
        
        # Inverse frequency: class ít mẫu → weight cao hơn
        w0 = total / (2 * max(counts[0], 1))
        w1 = total / (2 * max(counts[1], 1))
        weights = torch.tensor([w0, w1], dtype=torch.float32)
        print(f"[INFO] Closed question distribution: không={counts[0]}, có={counts[1]}")
        return weights

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # [OPTIMIZATION] Gradient accumulation for larger effective batch size
        accumulation_steps = self.config['train'].get('gradient_accumulation_steps', 2)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            label_closed = batch['label_closed'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Zero gradients only at the beginning or after optimizer step
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Sử dụng AMP Autocast
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Teacher Forcing: Input là <s> A B, Target là A B </s>
                decoder_input = target_ids[:, :-1]
                decoder_target = target_ids[:, 1:]
                
                logits_closed, logits_open = self.model(images, input_ids, attention_mask, decoder_input)
                
                # Loss calculation
                loss = 0
                mask_closed = (label_closed != -1)
                if mask_closed.any():
                    loss += self.criterion_closed(logits_closed[mask_closed], label_closed[mask_closed])
                
                # Phân tách Loss Generator để chống Mode Collapse (Lười biếng)
                vocab_size = logits_open.size(-1)
                mask_open = (label_closed == -1)
                
                # 1. Câu hỏi Yes/No: Giảm trọng số xuống cực thấp (0.1) để model không bị thiên vị
                if mask_closed.any():
                    loss_gen_closed = self.criterion_open(logits_open[mask_closed].reshape(-1, vocab_size), decoder_target[mask_closed].reshape(-1))
                    loss += loss_gen_closed * 0.1
                    
                # 2. Câu hỏi Mở: Tăng trọng số + Length Penalty + Coverage Penalty
                if mask_open.any():
                    open_logits = logits_open[mask_open]
                    open_targets = decoder_target[mask_open]
                    loss_gen_open = self.criterion_open(open_logits.reshape(-1, vocab_size), open_targets.reshape(-1))
                    
                    # Length penalty: phạt nếu model sinh quá ít token có nghĩa
                    pred_lengths = (open_targets != self.criterion_open.ignore_index).float().sum(dim=-1).mean()
                    length_penalty = torch.clamp(1.0 - pred_lengths / 15.0, min=0.0)
                    
                    # Coverage penalty: phạt nếu model lặp từ (tập trung quá vào 1 token)
                    probs = torch.softmax(open_logits, dim=-1)  # [N, seq, vocab]
                    coverage = probs.sum(dim=1)  # [N, vocab] — tổng xác suất mỗi token qua các bước
                    coverage_loss = torch.clamp(coverage - 1.0, min=0.0).mean()  # phạt nếu > 1 lần
                    
                    loss += (loss_gen_open + 0.3 * length_penalty + 0.1 * coverage_loss) * 3.0
                
                # [OPTIMIZATION] Normalize loss by accumulation steps for proper gradient scaling
                loss = loss / accumulation_steps
            
            # Backward với GradScaler
            self.scaler.scale(loss).backward()
            
            # [OPTIMIZATION] Update weights only after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient Clipping
                if self.config['train'].get('grad_clip'):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # [CRITICAL FIX] Step scheduler sau mỗi batch thay vì epoch để warmup mượt hơn
                if self.scheduler:
                    self.scheduler.step()
                
            total_loss += loss.item() * accumulation_steps
            # [FIX] Log LR cho từng param group — hiển thị decoder LR (group cuối) trên progress bar
            decoder_lr = self.optimizer.param_groups[-1]['lr']
            vision_lr = self.optimizer.param_groups[0]['lr']
            if wandb.run: 
                wandb.log({
                    "batch_loss": loss.item(),
                    "lr_vision": vision_lr,
                    "lr_decoder": decoder_lr,
                })
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "dec_lr": f"{decoder_lr:.1e}", "vis_lr": f"{vision_lr:.1e}"})
            
        return total_loss / len(self.train_loader)


    def val_epoch(self, tokenizer, epoch=0):
        """
        Thực hiện đánh giá trên tập Validation sau mỗi Epoch.
        """
        from src.engine.medical_eval import evaluate_vqa
        max_ans_len = self.config.get('data', {}).get('max_answer_len', 32)
        max_words = self.config.get('data', {}).get('answer_max_words', 10)
        print(f"\n🔍 Đang chạy Validation cho Epoch {epoch} (max_ans_len={max_ans_len})...")
        metrics = evaluate_vqa(
            self.model,
            self.val_loader,
            self.device,
            tokenizer,
            beam_width=self.beam_width,
            max_len=max_ans_len,
            max_words=max_words
        )
        
        # In các metrics quan trọng
        print(
            f"[METRICS] Accuracy: {metrics.get('accuracy_normalized', metrics['accuracy']):.4f} | "
            f"F1: {metrics.get('f1_normalized', metrics['f1']):.4f} | "
            f"BLEU-4: {metrics.get('bleu4_normalized', metrics['bleu4']):.4f}"
        )
        
        if wandb.run:
            wandb.log({
                "epoch": epoch,
                "val_accuracy": metrics["accuracy"],
                "val_accuracy_normalized": metrics.get("accuracy_normalized", metrics["accuracy"]),
                "val_f1": metrics["f1"],
                "val_f1_normalized": metrics.get("f1_normalized", metrics["f1"]),
                "val_bleu4": metrics["bleu4"],
                "val_bleu4_normalized": metrics.get("bleu4_normalized", metrics["bleu4"]),
                "val_bert_score": metrics.get("bert_score", 0),
                "val_bert_score_raw": metrics.get("bert_score_raw", metrics.get("bert_score", 0)),
                "val_semantic_raw": metrics.get("semantic_raw", metrics.get("semantic", 0)),
            })

        return metrics

    def train(self, epochs, tokenizer=None):
        best_val_acc = 0.0
        patience = self.config['train'].get('patience', 10)
        counter = 0
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        history_dir = self.config.get("history_dir")
        
        print(f"[INFO] Bắt đầu huấn luyện trong {epochs} epochs...")
        
        # Log to WandB if available
        if wandb.run is not None:
            wandb.config.update({
                'total_epochs': epochs,
                'patience': patience,
                'variant': self.config.get('variant', 'Unknown'),
                'device': str(self.device),
                'use_amp': self.use_amp,
            })
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.val_epoch(tokenizer, epoch=epoch)
            
            val_acc = metrics.get('accuracy_normalized', metrics.get('accuracy', 0))
            is_best = val_acc > best_val_acc
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_accuracy": float(metrics.get("accuracy", 0.0)),
                "val_accuracy_normalized": float(metrics.get("accuracy_normalized", metrics.get("accuracy", 0.0))),
                "val_f1": float(metrics.get("f1", 0.0)),
                "val_f1_normalized": float(metrics.get("f1_normalized", metrics.get("f1", 0.0))),
                "val_bleu4": float(metrics.get("bleu4", 0.0)),
                "val_bleu4_normalized": float(metrics.get("bleu4_normalized", metrics.get("bleu4", 0.0))),
                "val_bert_score": float(metrics.get("bert_score", 0.0)),
                "val_bert_score_raw": float(metrics.get("bert_score_raw", metrics.get("bert_score", 0.0))),
                "val_semantic_raw": float(metrics.get("semantic_raw", metrics.get("semantic", 0.0))),
                "best_so_far": bool(is_best),
                "metrics": metrics,
            }
            self.history.append(epoch_record)
            
            # Kiểm tra và Lưu Best Checkpoint
            if is_best:
                best_val_acc = val_acc
                counter = 0
                variant = self.config.get('variant', 'A')
                save_path = os.path.join(ckpt_dir, f"medical_vqa_{variant}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                
                # [CRITICAL FIX] Lưu checkpoint riêng hỗ trợ resume để không bị mất warmup/scheduler state
                resume_path = os.path.join(ckpt_dir, f"medical_vqa_{variant}_resume.pth")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                torch.save(checkpoint, resume_path)
                
                print(f"🌟 Best model saved with Accuracy: {val_acc:.4f}")
            else:
                counter += 1
            if history_dir:
                self.save_history(history_dir)
            if counter >= patience:
                print(f"🛑 Early stopping tại epoch {epoch}!")
                break
                    
        print("[INFO] Huấn luyện hoàn tất.")
        if history_dir:
            self.save_history(history_dir)
        return self.history

import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os

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
        
        self.criterion_closed = nn.CrossEntropyLoss()
        self.criterion_open = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, 
            label_smoothing=config['train'].get('label_smoothing', 0.0)
        )
        
        # AMP (Automatic Mixed Precision)
        self.use_amp = config['train'].get('use_amp', False) and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            label_closed = batch['label_closed'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
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
                
                # Loss generator (Open-ended) - So khớp với decoder_target
                vocab_size = logits_open.size(-1)
                loss += self.criterion_open(logits_open.reshape(-1, vocab_size), decoder_target.reshape(-1))
            
            # Backward với GradScaler
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            if self.config['train'].get('grad_clip'):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            if wandb.run: wandb.log({"batch_loss": loss.item()})
            pbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]['lr']})
            
        # Step scheduler sau mỗi epoch
        if self.scheduler:
            self.scheduler.step()
            
        return total_loss / len(self.train_loader)


    def val_epoch(self, tokenizer, epoch=0):
        """
        Thực hiện đánh giá trên tập Validation sau mỗi Epoch.
        """
        from src.engine.medical_eval import evaluate_vqa
        print(f"\n🔍 Đang chạy Validation cho Epoch {epoch}...")
        metrics = evaluate_vqa(self.model, self.val_loader, self.device, tokenizer, beam_width=self.beam_width)
        
        # In các metrics quan trọng
        print(f"[METRICS] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | BLEU-4: {metrics['bleu4']:.4f}")
        
        if wandb.run:
            wandb.log({
                "epoch": epoch,
                "val_accuracy": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "val_bleu4": metrics["bleu4"],
                "val_bert_score": metrics.get("bert_score", 0)
            })

        return metrics

    def train(self, epochs, tokenizer=None):
        best_val_acc = 0.0
        patience = self.config['train'].get('patience', 10)
        counter = 0
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        print(f"[INFO] Bắt đầu huấn luyện trong {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.val_epoch(tokenizer, epoch=epoch)
            
            val_acc = metrics.get('accuracy', 0)
            
            # Kiểm tra và Lưu Best Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                variant = self.config.get('variant', 'A')
                save_path = os.path.join(ckpt_dir, f"medical_vqa_{variant}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"🌟 Best model saved with Accuracy: {val_acc:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"🛑 Early stopping tại epoch {epoch}!")
                    break
                    
        print("[INFO] Huấn luyện hoàn tất.")

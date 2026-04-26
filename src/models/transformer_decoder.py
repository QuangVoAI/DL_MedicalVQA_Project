import torch
import torch.nn as nn

class MedicalVQADecoder(nn.Module):
    def __init__(self, decoder_type="transformer", vocab_size=30000, hidden_size=768):
        super(MedicalVQADecoder, self).__init__()
        self.decoder_type = decoder_type.lower()
        self.vocab_size = vocab_size
        
        # Nhánh 1: Classifier cho Yes/No
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        
        # Nhánh 2: Generator (Hỗ trợ Seq2Seq training)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        if self.decoder_type == "lstm":
            self.generator = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.generator = nn.TransformerDecoder(decoder_layer, num_layers=3)
            
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def generate(self, fused_features, beam_width=1, max_len=10):
        """
        Sinh câu trả lời sử dụng Greedy Search hoặc Beam Search.
        """
        if beam_width <= 1:
            return self._greedy_search(fused_features, max_len)
        else:
            return self._beam_search(fused_features, beam_width, max_len)

    def _greedy_search(self, fused_features, max_len):
        batch_size = fused_features.size(0)
        device = fused_features.device
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device) # BOS
        all_logits = []
        h_state = None

        for _ in range(max_len):
            curr_emb = self.embedding(generated)
            if self.decoder_type == "lstm":
                if h_state is None:
                    h0 = fused_features.transpose(0, 1).contiguous()
                    h_state = (h0, torch.zeros_like(h0))
                outputs, h_state = self.generator(curr_emb, h_state)
            else:
                outputs = self.generator(curr_emb, fused_features)
            
            next_logits = self.output_layer(outputs[:, -1:, :])
            all_logits.append(next_logits)
            next_token = torch.argmax(next_logits, dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
        return torch.cat(all_logits, dim=1)

    def _beam_search(self, fused_features, beam_width, max_len, repetition_penalty=1.2):
        """
        Triển khai Beam Search với cơ chế Repetition Penalty để chống lặp từ.
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        vocab_size = self.vocab_size
        
        final_logits = []

        for b in range(batch_size):
            feat = fused_features[b:b+1] # [1, 1, hidden]
            # (sequence, score, h_state if lstm)
            beams = [(torch.zeros((1, 1), dtype=torch.long, device=device), 0.0, None)]
            
            for _ in range(max_len):
                new_beams = []
                for seq, score, h_state in beams:
                    # Nếu token cuối là EOS (id=2), giữ nguyên và chuyển vào vòng sau
                    if seq[0, -1].item() == 2:
                        new_beams.append((seq, score, h_state))
                        continue
                        
                    curr_emb = self.embedding(seq)
                    if self.decoder_type == "lstm":
                        if h_state is None:
                            h0 = feat.transpose(0, 1).contiguous()
                            h_state = (h0, torch.zeros_like(h0))
                        outputs, next_h = self.generator(curr_emb, h_state)
                    else:
                        outputs = self.generator(curr_emb, feat)
                        next_h = None
                    
                    logits = self.output_layer(outputs[:, -1:, :]).squeeze() # [Vocab]
                    
                    # --- Repetition Penalty ---
                    for token_id in set(seq[0].tolist()):
                        if token_id in [0, 2]: continue # Bỏ qua BOS/EOS
                        if logits[token_id] < 0:
                            logits[token_id] *= repetition_penalty
                        else:
                            logits[token_id] /= repetition_penalty
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, topk_ids[i].view(1, 1)], dim=1)
                        new_beams.append((new_seq, score + topk_log_probs[i].item(), next_h))
                
                # Chọn Top-K beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]
                
                # Nếu tất cả beams đều kết thúc bằng EOS, dừng sớm
                if all(b[0][0, -1].item() == 2 for b in beams):
                    break
            
            # Lấy beam tốt nhất
            best_seq = beams[0][0][:, 1:] # [1, max_len]
            # Convert IDs back to "one-hot" style logits for compatibility
            one_hot = torch.zeros((1, max_len, vocab_size), device=device)
            one_hot.scatter_(2, best_seq.unsqueeze(-1), 1.0)
            final_logits.append(one_hot)

        return torch.cat(final_logits, dim=0)

    def forward(self, fused_features, target_ids=None, beam_width=1):
        """
        fused_features: [B, 1, 768] (Đặc trưng ảnh + câu hỏi)
        target_ids: [B, SeqLen] (Dùng cho Teacher Forcing khi train)
        """
        # 1. Nhánh phân loại
        logits_closed = self.classifier_head(fused_features.squeeze(1))
        
        # 2. Nhánh sinh câu trả lời (Generator)
        if target_ids is not None:
            # Chế độ Training: Sử dụng Teacher Forcing
            target_emb = self.embedding(target_ids) # [B, SeqLen, 768]
            
            if self.decoder_type == "lstm":
                h0 = fused_features.transpose(0, 1).contiguous()
                c0 = torch.zeros_like(h0)
                outputs, _ = self.generator(target_emb, (h0, c0))
            else:
                outputs = self.generator(target_emb, fused_features)
                
            logits_open = self.output_layer(outputs) # [B, SeqLen, Vocab]
        else:
            # Chế độ Inference
            logits_open = self.generate(fused_features, beam_width=beam_width)
            
        return logits_closed, logits_open

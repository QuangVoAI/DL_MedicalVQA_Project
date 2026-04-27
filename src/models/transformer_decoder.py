import torch
import torch.nn as nn

class MedicalVQADecoder(nn.Module):
    def __init__(self, decoder_type="transformer", vocab_size=30000, hidden_size=768, pretrained_embeddings=None):
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
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        if self.decoder_type == "lstm":
            self.generator = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.generator = nn.TransformerDecoder(decoder_layer, num_layers=3)
            
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def generate(self, fused_features, beam_width=1, max_len=10):
        """Sinh câu trả lời. Trả về token IDs [B, max_len]."""
        if beam_width <= 1:
            return self._greedy_search(fused_features, max_len)
        else:
            return self._beam_search(fused_features, beam_width, max_len)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _greedy_search(self, fused_features, max_len):
        """
        [FIX] Sửa lỗi LSTM bị feed lại toàn bộ chuỗi mỗi bước.
        LSTM chỉ cần token cuối cùng + h_state mang ngữ cảnh.
        Trả về token IDs [B, max_len] thay vì logits.
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)  # BOS
        h_state = None

        for _ in range(max_len):
            if self.decoder_type == "lstm":
                # [FIX] Chỉ feed token cuối cùng, h_state giữ toàn bộ ngữ cảnh
                curr_emb = self.embedding(generated[:, -1:])  # [B, 1, 768]
                if h_state is None:
                    h0 = fused_features.transpose(0, 1).contiguous()
                    h_state = (h0, torch.zeros_like(h0))
                outputs, h_state = self.generator(curr_emb, h_state)
            else:
                # Transformer: cần toàn bộ chuỗi cho causal self-attention
                curr_emb = self.embedding(generated)
                tgt_mask = self._generate_square_subsequent_mask(generated.size(1)).to(device)
                outputs = self.generator(curr_emb, fused_features, tgt_mask=tgt_mask)
            
            next_logits = self.output_layer(outputs[:, -1:, :])
            next_token = torch.argmax(next_logits, dim=-1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated[:, 1:]  # Bỏ BOS, trả về token IDs [B, max_len]

    def _beam_search(self, fused_features, beam_width, max_len, repetition_penalty=1.2, alpha=0.7):
        """
        Beam Search với Length Normalization + Repetition Penalty.
        alpha: length normalization factor (0=không normalize, 1=hoàn toàn)
               0.6–0.8 khuyến khích sinh câu dài hơn.
        Trả về token IDs [B, max_len].
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        all_results = []

        for b in range(batch_size):
            feat = fused_features[b:b+1]  # [1, 1, hidden]
            beams = [(torch.zeros((1, 1), dtype=torch.long, device=device), 0.0, None)]
            
            for step in range(max_len):
                new_beams = []
                for seq, score, h_state in beams:
                    if seq[0, -1].item() == 2:  # EOS
                        new_beams.append((seq, score, h_state))
                        continue
                    
                    if self.decoder_type == "lstm":
                        # [FIX] Chỉ feed token cuối cùng
                        curr_emb = self.embedding(seq[:, -1:])
                        if h_state is None:
                            h0 = feat.transpose(0, 1).contiguous()
                            h_state = (h0, torch.zeros_like(h0))
                        outputs, next_h = self.generator(curr_emb, h_state)
                    else:
                        curr_emb = self.embedding(seq)
                        tgt_mask = self._generate_square_subsequent_mask(seq.size(1)).to(device)
                        outputs = self.generator(curr_emb, feat, tgt_mask=tgt_mask)
                        next_h = None
                    
                    logits = self.output_layer(outputs[:, -1:, :]).squeeze()
                    
                    # Repetition Penalty
                    for token_id in set(seq[0].tolist()):
                        if token_id in [0, 2]: continue
                        if logits[token_id] < 0:
                            logits[token_id] *= repetition_penalty
                        else:
                            logits[token_id] /= repetition_penalty
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, topk_ids[i].view(1, 1)], dim=1)
                        new_beams.append((new_seq, score + topk_log_probs[i].item(), next_h))
                
                # [FIX] Length Normalization: chia score cho độ dài^alpha
                # Tránh phạt câu dài vì cumulative log_prob tự nhiên giảm theo độ dài
                def _normalized_score(beam):
                    seq_len = beam[0].size(1) - 1  # trừ BOS
                    return beam[1] / (max(seq_len, 1) ** alpha)
                
                new_beams.sort(key=_normalized_score, reverse=True)
                beams = new_beams[:beam_width]
                
                if all(bm[0][0, -1].item() == 2 for bm in beams):
                    break
            
            # Chọn beam tốt nhất theo normalized score
            beams.sort(key=_normalized_score, reverse=True)
            best_seq = beams[0][0][:, 1:]  # Bỏ BOS
            # Pad hoặc cắt về max_len
            if best_seq.size(1) < max_len:
                pad = torch.zeros((1, max_len - best_seq.size(1)), dtype=torch.long, device=device)
                best_seq = torch.cat([best_seq, pad], dim=1)
            else:
                best_seq = best_seq[:, :max_len]
            all_results.append(best_seq)

        return torch.cat(all_results, dim=0)  # [B, max_len] token IDs

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
                # Transformer cần causal mask để không nhìn trước tương lai
                tgt_mask = self._generate_square_subsequent_mask(target_ids.size(1)).to(target_ids.device)
                # fused_features: [B, 1, 768] -> memory cho Transformer
                outputs = self.generator(target_emb, fused_features, tgt_mask=tgt_mask)
                
            logits_open = self.output_layer(outputs) # [B, SeqLen, Vocab]
        else:
            # Chế độ Inference — trả về token IDs
            logits_open = self.generate(fused_features, beam_width=beam_width)
            
        return logits_closed, logits_open

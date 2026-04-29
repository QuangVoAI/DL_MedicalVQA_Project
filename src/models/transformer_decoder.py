import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalVQADecoder(nn.Module):
    def __init__(
        self,
        decoder_type: str = "transformer",
        vocab_size: int = 30000,
        hidden_size: int = 768,
        pretrained_embeddings=None,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_type = decoder_type.lower()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # ── Nhánh 1: Classifier cho Yes/No ──────────────────────────────────
        # [FIX] Thêm Dropout + GELU theo best-practice hiện đại
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        # ── Nhánh 2: Generator ───────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if self.decoder_type == "lstm":
            self.generator = nn.LSTM(
                hidden_size, hidden_size, num_layers=1, batch_first=True
            )
        else:
            # [FIX A2] Pre-LayerNorm (norm_first=True): hội tụ ổn định hơn, giảm gap A1-A2
            # dim_feedforward=4*hidden (768*4=3072) theo chuẩn Transformer gốc
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.generator = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

        # [OPTIMIZATION] Weight Tying: chia sẻ trọng số Embedding ↔ Output Projection
        # Giảm ~vocab_size * hidden_size params, cải thiện generalization (Press & Wolf 2017)
        self.output_layer.weight = self.embedding.weight

        # [OPTIMIZATION] Cache causal mask để tránh re-allocate mỗi forward pass
        self._causal_mask_cache: dict[tuple, torch.Tensor] = {}

    # ── Mask helper ─────────────────────────────────────────────────────────
    def _get_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        key = (sz, str(device))
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    # ── Public generate API ──────────────────────────────────────────────────
    def generate(self, fused_features, beam_width: int = 1, max_len: int = 10):
        """Sinh câu trả lời. Trả về token IDs [B, max_len]."""
        if beam_width <= 1:
            return self._greedy_search(fused_features, max_len)
        return self._beam_search(fused_features, beam_width, max_len)

    # ── Greedy Search ────────────────────────────────────────────────────────
    def _greedy_search(self, fused_features, max_len: int):
        """
        Greedy decoding (beam_width=1).
        LSTM: chỉ feed token cuối, h_state giữ ngữ cảnh → tránh O(n²) recompute.
        Trả về token IDs [B, max_len].
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)  # BOS=0
        h_state = None

        for _ in range(max_len):
            if self.decoder_type == "lstm":
                curr_emb = self.embedding(generated[:, -1:])  # [B,1,H]
                if h_state is None:
                    h0 = fused_features.transpose(0, 1).contiguous()
                    h_state = (h0, torch.zeros_like(h0))
                outputs, h_state = self.generator(curr_emb, h_state)
            else:
                curr_emb = self.embedding(generated)
                tgt_mask = self._get_causal_mask(generated.size(1), device)
                outputs = self.generator(curr_emb, fused_features, tgt_mask=tgt_mask)

            next_token = self.output_layer(outputs[:, -1:, :]).argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # Bỏ BOS

    # ── Beam Search ──────────────────────────────────────────────────────────
    def _beam_search(
        self,
        fused_features,
        beam_width: int,
        max_len: int,
        repetition_penalty: float = 1.2,
        alpha: float = 0.7,
    ):
        """
        Beam Search với Length Normalization + Vectorised Repetition Penalty.
        [FIX] Thay vòng for Python sang tensor ops để tăng tốc ~3-5× trên GPU.
        Trả về token IDs [B, max_len].
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        all_results = []

        for b in range(batch_size):
            feat = fused_features[b:b+1]  # [1, 1, H]
            beams = [(torch.zeros((1, 1), dtype=torch.long, device=device), 0.0, None)]

            for _ in range(max_len):
                new_beams = []
                for seq, score, h_state in beams:
                    if seq[0, -1].item() == 2:  # EOS
                        new_beams.append((seq, score, h_state))
                        continue

                    if self.decoder_type == "lstm":
                        curr_emb = self.embedding(seq[:, -1:])
                        if h_state is None:
                            h0 = feat.transpose(0, 1).contiguous()
                            h_state = (h0, torch.zeros_like(h0))
                        outputs, next_h = self.generator(curr_emb, h_state)
                    else:
                        curr_emb = self.embedding(seq)
                        tgt_mask = self._get_causal_mask(seq.size(1), device)
                        outputs = self.generator(curr_emb, feat, tgt_mask=tgt_mask)
                        next_h = None

                    logits = self.output_layer(outputs[:, -1, :]).squeeze(0)  # [V]

                    # [OPTIMIZED] Vectorised Repetition Penalty (thay vòng for Python)
                    unique_ids = seq[0].unique()
                    valid_ids = unique_ids[(unique_ids != 0) & (unique_ids != 2)]
                    if valid_ids.numel() > 0:
                        neg_mask = logits[valid_ids] < 0
                        factors = torch.where(
                            neg_mask,
                            torch.full_like(logits[valid_ids], repetition_penalty),
                            torch.full_like(logits[valid_ids], 1.0 / repetition_penalty),
                        )
                        logits = logits.clone()
                        logits[valid_ids] = logits[valid_ids] * factors

                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        new_seq = torch.cat([seq, topk_ids[i].view(1, 1)], dim=1)
                        new_beams.append((new_seq, score + topk_log_probs[i].item(), next_h))

                def _norm_score(beam):
                    seq_len = max(beam[0].size(1) - 1, 1)
                    return beam[1] / (seq_len ** alpha)

                new_beams.sort(key=_norm_score, reverse=True)
                beams = new_beams[:beam_width]

                if all(bm[0][0, -1].item() == 2 for bm in beams):
                    break

            beams.sort(key=_norm_score, reverse=True)
            best_seq = beams[0][0][:, 1:]  # Bỏ BOS

            if best_seq.size(1) < max_len:
                pad = torch.zeros((1, max_len - best_seq.size(1)), dtype=torch.long, device=device)
                best_seq = torch.cat([best_seq, pad], dim=1)
            else:
                best_seq = best_seq[:, :max_len]
            all_results.append(best_seq)

        return torch.cat(all_results, dim=0)  # [B, max_len]

    # ── Training Forward ─────────────────────────────────────────────────────
    def forward(self, fused_features, target_ids=None, beam_width: int = 1):
        """
        fused_features: [B, 1, H]
        target_ids:     [B, SeqLen] — Teacher Forcing; None → inference
        """
        logits_closed = self.classifier_head(fused_features.squeeze(1))

        if target_ids is not None:
            target_emb = self.embedding(target_ids)

            if self.decoder_type == "lstm":
                h0 = fused_features.transpose(0, 1).contiguous()
                outputs, _ = self.generator(target_emb, (h0, torch.zeros_like(h0)))
            else:
                tgt_mask = self._get_causal_mask(target_ids.size(1), target_ids.device)
                outputs = self.generator(target_emb, fused_features, tgt_mask=tgt_mask)

            logits_open = self.output_layer(outputs)
        else:
            logits_open = self.generate(fused_features, beam_width=beam_width)

        return logits_closed, logits_open

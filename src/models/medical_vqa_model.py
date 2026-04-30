import torch
import torch.nn as nn
from .encoder import MedicalImageEncoder
from .phobert_encoder import PhoBERTEncoder
from .transformer_decoder import MedicalVQADecoder

class CoAttentionFusion(nn.Module):
    """
    Cơ chế Co-Attention giúp mô hình tập trung vào các vùng ảnh và từ ngữ liên quan lẫn nhau.
    """
    def __init__(self, hidden_size=768, nhead=8):
        super(CoAttentionFusion, self).__init__()
        # Cross-modal attention: Ảnh hỏi Chữ và Chữ hỏi Ảnh
        self.v2t_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)
        self.t2v_attn = nn.MultiheadAttention(hidden_size, nhead, batch_first=True)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, v_feats, t_feats):
        # v_feats: [B, 49, 768] — KHÔNG cần unsqueeze nữa
        t_seq = t_feats.unsqueeze(1)  # [B, 1, 768] — text vẫn giữ
        
        # Parallel Co-Attention
        v_fused, _ = self.v2t_attn(v_feats, t_seq, t_seq)
        t_fused, _ = self.t2v_attn(t_seq, v_feats, v_feats)
        
        # v_fused: [B, 49, 768] → pool về [B, 1, 768] trước khi concat
        v_fused = v_fused.mean(dim=1, keepdim=True)
        
        # Kết hợp thông tin từ cả hai hướng
        combined = torch.cat([v_fused, t_fused], dim=-1) # [B, 1, 1536]
        return self.fusion_layer(combined) # [B, 1, 768]

class MedicalVQAModelA(nn.Module):
    """
    Kiến trúc rời (Hướng A) cho Medical VQA Tiếng Việt.
    Sử dụng DenseNet-121 (XRV) + PhoBERT + Co-Attention + Dual-Head Decoder.
    """
    def __init__(self, decoder_type="transformer", vocab_size=30000, hidden_size=768, phobert_model=None, **kwargs):
        super(MedicalVQAModelA, self).__init__()
        
        # 1. Image Encoder (DenseNet-121 XRV)
        self.image_encoder = MedicalImageEncoder(pretrained=True)
        
        # 2. Text Encoder (PhoBERT)
        self.text_encoder = PhoBERTEncoder(model_name=phobert_model) if phobert_model else PhoBERTEncoder()
        
        # 3. Fusion Layer (Co-Attention Fusion)
        self.fusion = CoAttentionFusion(hidden_size=hidden_size, nhead=8)
        
        # 4. Trích xuất pretrained embeddings từ PhoBERT cho Decoder
        phobert_embeddings = self.text_encoder.bert.embeddings.word_embeddings.weight
        actual_vocab_size = phobert_embeddings.size(0)
        
        # 5. Decoder (LSTM / Transformer)
        self.decoder = MedicalVQADecoder(
            decoder_type=decoder_type, 
            vocab_size=actual_vocab_size,
            hidden_size=hidden_size,
            pretrained_embeddings=phobert_embeddings
        )

    def forward(self, images, input_ids, attention_mask, labels_open=None, labels_closed=None):
        v_feats = self.image_encoder(images)
        t_feats = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(v_feats, t_feats)
        
        logits_closed, logits_open = self.decoder(fused, labels_open)

        return logits_closed, logits_open

    def generate(self, images, input_ids, attention_mask, beam_width=1, max_len=10):
        """
        Giao diện chuyên biệt cho quá trình Inference (chỉ trả token IDs cho open-ended).
        """
        v_feats = self.image_encoder(images)
        t_feats = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(v_feats, t_feats)
        
        return self.decoder.generate(fused, beam_width=beam_width, max_len=max_len)

    def inference(self, images, input_ids, attention_mask, beam_width=1, max_len=10):
        """
        [NEW] Trả về CẢ HAI dual-head outputs:
        - logits_closed: [B, 2] — dùng cho câu Yes/No (classifier head)
        - generated_ids: [B, max_len] — dùng cho câu mở (generative head)
        """
        v_feats = self.image_encoder(images)
        t_feats = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(v_feats, t_feats)
        
        logits_closed = self.decoder.classifier_head(fused.squeeze(1))  # [B, 2]
        generated_ids = self.decoder.generate(fused, beam_width=beam_width, max_len=max_len)  # [B, max_len]
        
        return logits_closed, generated_ids

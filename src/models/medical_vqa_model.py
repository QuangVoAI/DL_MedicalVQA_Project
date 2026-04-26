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
        # v_feats, t_feats: [B, 768]
        v_seq = v_feats.unsqueeze(1) # [B, 1, 768]
        t_seq = t_feats.unsqueeze(1) # [B, 1, 768]
        
        # Parallel Co-Attention
        v_fused, _ = self.v2t_attn(v_seq, t_seq, t_seq)
        t_fused, _ = self.t2v_attn(t_seq, v_seq, v_seq)
        
        # Kết hợp thông tin từ cả hai hướng
        combined = torch.cat([v_fused, t_fused], dim=-1) # [B, 1, 1536]
        return self.fusion_layer(combined) # [B, 1, 768]

class MedicalVQAModelA(nn.Module):
    """
    Kiến trúc rời (Hướng A) cho Medical VQA Tiếng Việt.
    Sử dụng DenseNet-121 (XRV) + PhoBERT + Co-Attention + Dual-Head Decoder.
    """
    def __init__(self, decoder_type="transformer", vocab_size=30000):
        super(MedicalVQAModelA, self).__init__()
        
        # 1. Image Encoder (DenseNet-121 XRV)
        self.image_encoder = MedicalImageEncoder(pretrained=True)
        
        # 2. Text Encoder (PhoBERT)
        self.text_encoder = PhoBERTEncoder()
        
        # 3. Fusion Layer (Co-Attention Fusion)
        self.fusion = CoAttentionFusion(hidden_size=768, nhead=8)
        
        # 4. Decoder (LSTM / Transformer)
        self.decoder = MedicalVQADecoder(
            decoder_type=decoder_type, 
            vocab_size=vocab_size,
            hidden_size=768
        )

    def forward(self, images, input_ids, attention_mask, target_ids=None, beam_width=1):
        # Visual features: [B, 768]
        v_feats = self.image_encoder(images)
        
        # Text features: [B, 768]
        t_feats = self.text_encoder(input_ids, attention_mask)
        
        # Fusion sử dụng Co-Attention
        fused = self.fusion(v_feats, t_feats) # [B, 1, 768]
        
        # Decoding với hỗ trợ Beam Search
        logits_closed, logits_open = self.decoder(fused, target_ids, beam_width=beam_width)
        
        return logits_closed, logits_open

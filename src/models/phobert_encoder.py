import torch.nn as nn
from transformers import AutoModel

class PhoBERTEncoder(nn.Module):
    """
    Text Encoder sử dụng PhoBERT pretrained.
    Hỗ trợ tiếng Việt tốt nhất cho Medical VQA.
    """
    def __init__(self, model_name="vinai/phobert-base", freeze_layers=10):
        super(PhoBERTEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Đóng băng các lớp Transformer đầu tiên nếu cần
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy [CLS] token đại diện cho toàn bộ câu hỏi
        return outputs.last_hidden_state[:, 0, :]

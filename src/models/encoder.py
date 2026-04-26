import torch
import torch.nn as nn
import torchxrayvision as xrv

class MedicalImageEncoder(nn.Module):
    """
    SOTA Image Encoder sử dụng DenseNet-121 (TorchXRayVision)
    Pretrained trên 200K+ ảnh X-ray (CheXpert, NIH, v.v.)
    """
    def __init__(self, pretrained=True):
        super(MedicalImageEncoder, self).__init__()
        if pretrained:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        else:
            self.model = xrv.models.DenseNet(weights=None)
            
        self.model.classifier = nn.Identity() # Bỏ lớp phân loại 
        self.projector = nn.Linear(1024, 768) # Map về dimension của PhoBERT
        
    def forward(self, x):
        features = self.model(x)
        return self.projector(features)

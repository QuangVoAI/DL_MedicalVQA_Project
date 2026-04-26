import cv2
import numpy as np
import torch
from torchvision import transforms

def apply_clahe(img_array):
    """
    Áp dụng Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Giúp tăng cường độ tương phản cục bộ cho ảnh X-ray.
    """
    # Nếu ảnh đang ở dạng float [0, 1], chuyển về uint8 [0, 255]
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Xử lý cho ảnh xám (Grayscale)
    if len(img_array.shape) == 2:
        img_clahe = clahe.apply(img_array)
    # Xử lý cho ảnh màu (RGB) - Chuyển sang LAB để giữ màu sắc
    else:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        img_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        
    return img_clahe.astype(np.float32) / 255.0

class MedicalImageTransform:
    """
    Custom transform tích hợp CLAHE và chuẩn hóa cho Medical VQA (Hướng A - XRV).
    """
    def __init__(self, size=224):
        self.resize = transforms.Resize((size, size))
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img):
        # 1. Resize
        img = self.resize(img)
        
        # 2. Apply CLAHE (Tăng cường độ tương phản y tế)
        img_np = np.array(img)
        img_clahe = apply_clahe(img_np) # Trả về ảnh [0, 1]
        
        # 3. Chuyển sang Tensor 1 kênh cho DenseNet XRV
        # img_clahe shape: [224, 224]
        img_tensor = torch.from_numpy(img_clahe).unsqueeze(0) # [1, 224, 224]
            
        # 4. Chuẩn hóa về dải [-1, 1]. 
        # Ghi chú: XRV thường cảnh báo dải [-1024, 1024] nhưng với ảnh 8-bit, 
        # dải [-1, 1] là đủ để mô hình hoạt động ổn định.
        return self.normalize(img_tensor)

"""
Enhanced medical image data augmentation with domain-aware techniques.
Optimized for X-ray, MRI, CT images while preserving diagnostic information.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance
import cv2


class MedicalImageAugmentation:
    """
    Enhanced augmentation pipeline for medical images.
    
    Key features:
    - Preserves diagnostic information (no aggressive transforms)
    - Domain-specific augmentations (CLAHE, noise, perspective)
    - Randomized parameter selection for diversity
    """
    
    def __init__(self, size=224, aggressive_mode=False):
        self.size = size
        self.aggressive_mode = aggressive_mode
        
    def clahe_augmentation(self, image_np):
        """
        Contrast Limited Adaptive Histogram Equalization.
        Improves contrast while preventing noise amplification.
        """
        if isinstance(image_np, Image.Image):
            image_np = np.array(image_np)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        augmented = clahe.apply(image_np)
        
        # Convert back to PIL
        return Image.fromarray(augmented)
    
    def gaussian_noise(self, image_np, noise_level=0.02):
        """Add realistic Gaussian noise (simulate sensor noise)."""
        if isinstance(image_np, Image.Image):
            image_np = np.array(image_np, dtype=np.float32)
        else:
            image_np = image_np.astype(np.float32)
        
        noise = np.random.normal(0, noise_level * 255, image_np.shape)
        noisy = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def random_rotation(self, image, max_angle=10):
        """Small random rotation (clinical context allows)."""
        angle = np.random.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor=0, expand=False)
    
    def random_elastic_deformation(self, image_np, alpha=30, sigma=5, p=0.5):
        """
        Elastic deformations for data augmentation.
        Simulates slight anatomical variations.
        """
        if np.random.random() > p:
            return image_np if isinstance(image_np, Image.Image) else Image.fromarray(image_np)
        
        if isinstance(image_np, Image.Image):
            image_np = np.array(image_np, dtype=np.float32)
        
        h, w = image_np.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma
        
        # Blur displacement fields
        from scipy import ndimage
        dx = ndimage.gaussian_filter(dx, sigma=sigma) * alpha
        dy = ndimage.gaussian_filter(dy, sigma=sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.array([y + dy, x + dx])
        
        # Apply displacement
        deformed = ndimage.map_coordinates(
            image_np, indices, order=1, mode='constant', cval=0
        )
        
        return Image.fromarray(np.uint8(np.clip(deformed, 0, 255)))
    
    def random_brightness_contrast(self, image, brightness=0.15, contrast=0.2):
        """Adjust brightness and contrast with medical image constraints."""
        # Brightness
        if np.random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(1 - brightness, 1 + brightness)
            image = enhancer.enhance(factor)
        
        # Contrast
        if np.random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(1 - contrast, 1 + contrast)
            image = enhancer.enhance(factor)
        
        return image
    
    def get_augmentation_pipeline(self):
        """
        Build augmentation pipeline with medical-specific constraints.
        
        Returns:
            torchvision.transforms.Compose instance
        """
        augmentations = []
        
        # Resize to target size
        augmentations.append(transforms.Resize((self.size, self.size)))
        
        # Medical-specific augmentations
        augmentations.extend([
            # Small rotations (±10°)
            transforms.RandomRotation(degrees=10, fill=0),
            
            # Elastic deformations (anatomical variations)
            transforms.RandomAffine(degrees=0, shear=5),
            
            # Brightness/Contrast adjustments (imaging variations)
            transforms.ColorJitter(brightness=0.1, contrast=0.15),
            
            # Random horizontal flip (medical acceptable)
            transforms.RandomHorizontalFlip(p=0.2),
            
            # Slight zoom effect
            transforms.RandomResizedCrop(self.size, scale=(0.95, 1.05)),
        ])
        
        if self.aggressive_mode:
            # Add more aggressive augmentations for better regularization
            augmentations.extend([
                # Random erasing (occlusion/artifacts)
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
                # Gaussian blur (motion blur simulation)
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
        
        # Convert to tensor and normalize
        augmentations.extend([
            transforms.ToTensor(),
            # ImageNet normalization adjusted for medical images
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(augmentations)
    
    def get_inference_transforms(self):
        """Non-augmented transforms for inference."""
        return transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class ClinicalAwareAugmentation:
    """
    Advanced augmentation that maintains clinical validity.
    Uses domain knowledge about medical image characteristics.
    """
    
    def __init__(self, size=224):
        self.size = size
        self.medical_aug = MedicalImageAugmentation(size)
    
    def __call__(self, image):
        """Apply augmentations in sequence."""
        # 1. CLAHE enhancement (preserve structure)
        if np.random.random() > 0.5:
            image = self.medical_aug.clahe_augmentation(image)
        
        # 2. Geometric augmentations
        if np.random.random() > 0.5:
            image = self.medical_aug.random_rotation(image, max_angle=10)
        
        # 3. Brightness/Contrast
        image = self.medical_aug.random_brightness_contrast(image)
        
        # 4. Noise addition (low probability, small magnitude)
        if np.random.random() > 0.8:
            image = self.medical_aug.gaussian_noise(image, noise_level=0.01)
        
        # 5. Standard transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transform(image)

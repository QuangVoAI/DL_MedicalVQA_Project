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
    
    def random_rotation(self, image, max_angle=3):
        """
        VERY LIMITED rotation to simulate positioning error only.
        
        ⚠️  MEDICAL SAFETY NOTE:
        - Large rotations (>5°) create UNREALISTIC images
        - Radiological findings change with orientation
        - Limited to ±2-3° to model minor positioning variations
        - NOT to simulate actual anatomical variations
        
        Use Case: Simulating patient positioning error during imaging
        NOT Use Case: Creating augmented variants of same diagnosis
        """
        # CRITICAL: max_angle capped at 3° for medical safety
        safe_angle = min(abs(max_angle), 3.0)
        angle = np.random.uniform(-safe_angle, safe_angle)
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
        Build augmentation pipeline with MEDICALLY SAFE constraints.
        
        ⚠️  CRITICAL MEDICAL SAFETY NOTES:
        
        DISALLOWED Augmentations:
        ❌ Large rotations (>5°) - Changes diagnosis interpretation
        ❌ Flips - X-ray orientation must be preserved (PA vs AP)
        ❌ Elastic deformations >2% - Can obscure pathology
        
        ALLOWED Augmentations:
        ✅ Brightness/Contrast - Simulates imaging device variations
        ✅ Noise - Simulates sensor noise in real devices
        ✅ Very small rotations (±2-3°) - Positioning error only
        ✅ Minimal shear - Slight patient positioning variation
        
        Rationale:
        In radiology, image ORIENTATION and POSITION are clinically significant.
        We augment to handle IMAGING VARIATIONS, not create fake diagnoses.
        
        Returns:
            torchvision.transforms.Compose instance
        """
        augmentations = []
        
        # Resize to target size
        augmentations.append(transforms.Resize((self.size, self.size)))
        
        # MEDICALLY SAFE augmentations only
        augmentations.extend([
            # SAFE: Very small rotations (±2-3° only to model positioning error)
            # NOT for creating anatomically different images
            transforms.RandomRotation(degrees=2, fill=0),
            
            # SAFE: Minimal shear (±2°) for slight patient positioning variations
            # Keep minimal to avoid changing anatomical structure
            transforms.RandomAffine(degrees=0, shear=2, fill=0),
            
            # SAFE: Brightness/Contrast adjustments (imaging equipment variation)
            # Medical devices produce images with different intensities
            transforms.ColorJitter(brightness=0.1, contrast=0.15),
            
            # ❌ REMOVED: Random horizontal flip
            # transforms.RandomHorizontalFlip(p=0.2)
            # Reason: PA (posterior-anterior) vs AP (anterior-posterior) X-ray
            # are different views with different diagnostic interpretations
            
            # SAFE: Slight zoom (±2-3% only for focus variation)
            transforms.RandomResizedCrop(self.size, scale=(0.97, 1.03)),
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
        """Apply augmentations in sequence - CLINICALLY VALID ONLY."""
        # 1. CLAHE enhancement (preserve structure)
        if np.random.random() > 0.5:
            image = self.medical_aug.clahe_augmentation(image)
        
        # [REMOVED] 2. Geometric augmentations - NO ROTATION!
        # Rotation changes radiological orientation which is diagnostically critical
        # if np.random.random() > 0.5:
        #     image = self.medical_aug.random_rotation(image, max_angle=10)  # ❌ INVALID
        
        # 2. Brightness/Contrast (minimal, realistic variations)
        image = self.medical_aug.random_brightness_contrast(image, brightness=0.08, contrast=0.12)
        
        # 3. Noise addition (very low probability, small magnitude - simulator sensor noise)
        if np.random.random() > 0.85:
            image = self.medical_aug.gaussian_noise(image, noise_level=0.005)
        
        # 4. Standard transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transform(image)

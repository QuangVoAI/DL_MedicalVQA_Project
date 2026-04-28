"""
Discriminative learning rates for different model layers.
Earlier layers (pretrained) get lower LR to preserve learned features.
Later layers get higher LR for task-specific adaptation.
"""

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def create_discriminative_optimizer(model, config):
    """
    Create optimizer with discriminative learning rates.
    
    Layer groups and their learning rates:
    - Image Encoder (pretrained XRV): 1e-5 (preserve medical features)
    - Text Encoder (PhoBERT): 1e-5 (preserve language understanding)
    - Fusion layer (co-attention): 1e-4 (moderate adaptation)
    - Decoder (task-specific): 1e-3 (heavy adaptation)
    
    Args:
        model: Model with parameter groups
        config: Config dict with learning rates
    
    Returns:
        Optimizer with layer-specific learning rates
    """
    
    # Define parameter groups with different learning rates
    param_groups = []
    
    base_lr = float(config['train'].get('learning_rate', 3e-4))
    vision_lr = float(config['train'].get('vision_lr', 1e-5))
    phobert_lr = float(config['train'].get('phobert_lr', 1e-5))
    
    # Group 1: Image Encoder (lowest LR)
    if hasattr(model, 'image_encoder'):
        param_groups.append({
            'params': model.image_encoder.parameters(),
            'lr': vision_lr,
            'name': 'image_encoder'
        })
    
    # Group 2: Text Encoder (low LR)
    if hasattr(model, 'text_encoder'):
        param_groups.append({
            'params': model.text_encoder.parameters(),
            'lr': phobert_lr,
            'name': 'text_encoder'
        })
    
    # Group 3: Fusion/Attention layers (medium LR)
    fusion_params = []
    if hasattr(model, 'fusion'):
        fusion_params.extend(model.fusion.parameters())
    if hasattr(model, 'co_attention'):
        fusion_params.extend(model.co_attention.parameters())
    if hasattr(model, 'spatial_attention'):
        fusion_params.extend(model.spatial_attention.parameters())
    
    if fusion_params:
        param_groups.append({
            'params': fusion_params,
            'lr': base_lr * 0.5,  # 50% of base LR
            'name': 'fusion'
        })
    
    # Group 4: Decoder (highest LR)
    decoder_params = []
    if hasattr(model, 'decoder'):
        decoder_params.extend(model.decoder.parameters())
    if hasattr(model, 'open_head'):
        decoder_params.extend(model.open_head.parameters())
    if hasattr(model, 'closed_head'):
        decoder_params.extend(model.closed_head.parameters())
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,  # Full base LR
            'name': 'decoder'
        })
    
    # Group 5: Any remaining parameters
    # Collect all params that aren't in above groups
    all_params = set(model.parameters())
    grouped_params = set()
    for group in param_groups:
        grouped_params.update(group['params'])
    
    remaining_params = [p for p in all_params if p not in grouped_params]
    if remaining_params:
        param_groups.append({
            'params': remaining_params,
            'lr': base_lr * 0.1,  # 10% of base LR for safety
            'name': 'remaining'
        })
    
    # Create optimizer
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config['train'].get('weight_decay', 0.01)
    )
    
    # Log layer learning rates
    print("[INFO] Discriminative Learning Rates Setup:")
    for group in param_groups:
        param_count = sum(p.numel() for p in group['params'])
        print(f"  {group['name']:15s}: LR={group['lr']:.2e}, Params={param_count:,}")
    
    return optimizer


def create_scheduler_with_warmup(optimizer, num_training_steps, config):
    """
    Create cosine scheduler with warmup.
    
    Args:
        optimizer: Optimizer instance
        num_training_steps: Total training steps
        config: Config dict
    
    Returns:
        LambdaLR scheduler with warmup
    """
    
    warmup_steps = int(num_training_steps * config['train'].get('warmup_steps_ratio', 0.1))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,  # 0.5 = cosine goes from 1 to 0
        last_epoch=-1
    )
    
    print(f"[INFO] Scheduler: Cosine with warmup")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/num_training_steps*100:.1f}%)")
    print(f"  Total steps: {num_training_steps}")
    
    return scheduler


def get_current_learning_rates(optimizer):
    """Get current learning rate for each parameter group."""
    lrs = {}
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('name', f'group_{i}')
        lrs[name] = param_group['lr']
    return lrs

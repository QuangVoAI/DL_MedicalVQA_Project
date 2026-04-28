# ═══════════════════════════════════════════════════════════════════════
# WandB Configuration for Medical VQA Training Monitoring
# ═══════════════════════════════════════════════════════════════════════

## QUICK START:

### 1. Create WandB Account
   Go to: https://wandb.ai/
   Sign up with GitHub or Email

### 2. Get API Key
   Go to: https://wandb.ai/settings/profile
   Copy your API key

### 3. Set Environment Variable
   export WANDB_API_KEY="your_api_key_here"
   # Or in Jupyter:
   import os
   os.environ['WANDB_API_KEY'] = 'your_api_key_here'

### 4. Run Training
   python train_medical.py --variant A1
   # Automatically logs to WandB!

## WHAT GETS LOGGED:

✅ Training Metrics (per epoch):
   - train_loss
   - train_accuracy
   - train_bleu
   - train_rouge
   - train_bertscore

✅ Validation Metrics (per epoch):
   - val_loss
   - val_accuracy
   - val_bleu
   - val_rouge
   - val_bertscore

✅ Model Info:
   - Number of parameters
   - Model architecture
   - Config settings

✅ Hardware:
   - GPU usage
   - Memory
   - Training time

✅ Learning Rate:
   - Current LR per epoch
   - Warmup schedule

## MONITORING DASHBOARD:

View live at: https://wandb.ai/QuangVoAI/MedicalVQA-Vietnam

Features:
- Real-time loss graphs
- Metric comparison across variants
- Training progress
- System resource monitoring
- Hyperparameter tracking
- Model checkpoints

## ADVANCED:

Save Checkpoints to WandB:
   wandb.save('checkpoint.pt')

Log Custom Metrics:
   wandb.log({'custom_metric': value, 'epoch': epoch})

Compare Models:
   Visit: https://wandb.ai/QuangVoAI/MedicalVQA-Vietnam/reports

## OFFLINE MODE:

If you don't have internet:
   export WANDB_MODE=offline
   python train_medical.py --variant A1
   # Saves locally, can sync later

## TIPS:

1. Set descriptive run names:
   wandb.init(..., name="A2_50epochs_final")

2. Add tags for easy filtering:
   wandb.init(..., tags=["production", "50-epochs"])

3. Create reports with charts:
   Use WandB UI to create custom reports

4. Compare multiple runs:
   Group runs by config/variant

═══════════════════════════════════════════════════════════════════════

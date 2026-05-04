---
title: Medical VQA
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Vietnamese Medical VQA

Medical Visual Question Answering project for Vietnamese medical images. The system compares modular VQA models, LLaVA-Med based models, alignment variants, and a merged adapter created with Model Soup.

Demo Space: https://springwang08-medical-vqa.hf.space

## Team

- Võ Xuân Quang - 523H0173
- Hoàng Xuân Thành - 523H0178

## Overview

The project uses a Vietnamese medical VQA dataset built from SLAKE and VQA-RAD. Images and QA pairs are hosted on Hugging Face:

- Dataset: `SpringWang08/medical-vqa-vi`
- Base multimodal model: `chaoyinshe/llava-med-v1.5-mistral-7b-hf`

The app supports image upload, Vietnamese question input, model selection, and side-by-side prediction comparison.

## Model Variants

| Variant | Description | Artifact |
|---|---|---|
| A1 LSTM | DenseNet-121 XRV + PhoBERT + LSTM decoder | `SpringWang08/medical-vqa-a1` |
| A2 Transformer | DenseNet-121 XRV + PhoBERT + Transformer decoder | `SpringWang08/medical-vqa-a2` |
| B1 Zero-shot | LLaVA-Med without fine-tuning | `chaoyinshe/llava-med-v1.5-mistral-7b-hf` |
| B2 Fine-tuned | LLaVA-Med + LoRA SFT | `SpringWang08/medical-vqa-b2` |
| DPO Alignment | B2 refined with Direct Preference Optimization | `SpringWang08/medical-vqa-dpo` |
| PPO RL refinement | B2 refined with PPO-style reward optimization | `SpringWang08/medical-vqa-ppo` |
| SOUP Model Soup | Merged LoRA adapter from B2, DPO, and PPO | `SpringWang08/medical-vqa-soup` |

## Project Structure

```text
.
├── app.py                    # Gradio demo used by Hugging Face Space
├── configs/
│   └── medical_vqa.yaml      # Dataset, model, training, and eval config
├── src/
│   ├── data/                 # Dataset wrapper
│   ├── engine/               # Training and evaluation logic
│   ├── models/               # Direction A and LLaVA-Med model wrappers
│   └── utils/                # Text processing, translation, rewrite, metrics
├── scripts/                  # Data and utility scripts kept for reproducibility
├── web/                      # FastAPI web implementation
├── train_medical.py          # Training entry point
├── requirements.txt
└── Dockerfile
```

Large runtime artifacts are intentionally not committed:

- `checkpoints/`
- `logs/`
- `results/`
- `scratch/`
- `Test_Prediction/`

## Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For LLaVA-Med variants (`B1`, `B2`, `DPO`, `PPO`, `SOUP`), use a CUDA GPU. The deployed Space currently requests `l4x1`.

## Run The Gradio Demo

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

The first request can be slow because model weights and adapters are downloaded from Hugging Face Hub. Later requests reuse the cache.

## Train Models

Direction A:

```bash
python train_medical.py --config configs/medical_vqa.yaml --variant A1
python train_medical.py --config configs/medical_vqa.yaml --variant A2
```

Direction B:

```bash
python train_medical.py --config configs/medical_vqa.yaml --variant B1
python train_medical.py --config configs/medical_vqa.yaml --variant B2
python train_medical.py --config configs/medical_vqa.yaml --variant DPO
python train_medical.py --config configs/medical_vqa.yaml --variant PPO
```

`B1` is zero-shot evaluation. `B2`, `DPO`, and `PPO` require the LLaVA-Med base model and LoRA/QLoRA dependencies.

## Inference Flow

For each selected model, the app runs:

```text
image + question -> raw prediction -> normalized prediction -> rewritten answer
```

The final result table intentionally shows only:

```text
Model | Prediction
```

The rewrite layer uses `MedicalAnswerRewriter` and keeps answers under the configured word limit. If the rewrite model cannot load, the app falls back to the normalized prediction.

## Hugging Face Space

The Space uses Docker and launches `app.py` on port `7860`.

```bash
docker build -t medical-vqa .
docker run --rm --gpus all -p 7860:7860 medical-vqa
```

Space metadata is stored at the top of this README so Hugging Face can build the Docker app correctly.

## Dataset And Test Samples

Main dataset:

```text
SpringWang08/medical-vqa-vi
```

Local test samples for manual prediction can be generated separately and are ignored by Git:

```text
Test_Prediction/
```

## References

- SLAKE: https://arxiv.org/abs/2102.09542
- VQA-RAD: https://www.nature.com/articles/sdata2018189
- PhoBERT: https://arxiv.org/abs/2003.00744
- DPO: https://arxiv.org/abs/2305.18290
- LLaVA-Med: https://arxiv.org/abs/2306.00890

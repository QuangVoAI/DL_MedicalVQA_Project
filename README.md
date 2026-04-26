<p align="center">
 <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained">
 <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python">
 <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg" alt="PyTorch">
 <img src="https://img.shields.io/badge/SOTA-Medical--VQA-orange.svg" alt="SOTA">
</p>

## Nhóm thực hiện
*  **Võ Xuân Quang** (MSSV: 523H0173) 
*  **Hoàng Xuân Thành** (MSSV: 523H0178)

Hệ thống **Visual Question Answering (VQA) Y tế** sử dụng tiếng Việt, xây dựng trên tập dữ liệu **SLAKE + VQA-RAD** đã được dịch sang tiếng Việt bằng kỹ thuật **Dictionary-Enhanced Prompting** (SOTA En→Vi, arXiv 2509.15640).


## ️ Kiến trúc

| Cấu hình | Image Encoder | Text Encoder | Answer Decoder | Ghi chú |
|---|---|---|---|---|
| **A1** | **DenseNet-121 (XRV)** | PhoBERT | LSTM + Bahdanau | So sánh Decoder (1) |
| **A2** | **DenseNet-121 (XRV)** | PhoBERT | **Transformer Decoder** | So sánh Decoder (2) |
| **B1** | **LLaVA-Med-7B** | — | — | Zero-shot (Multimodal Pretrained) |
| **B2** | **LLaVA-Med-7B** | — | — | Fine-tuned (QLoRA 4-bit) + DPO |

> [!NOTE]
> **Sự khác biệt về chiến lược giải mã:**
> - **Hướng A (Closed-Vocab):** Sử dụng bộ từ vựng cố định được xây dựng từ tập huấn luyện. Phù hợp cho các câu trả lời ngắn, chuẩn hóa nhưng giới hạn khả năng sinh từ mới cho các câu hỏi mở (Open-ended).
> - **Hướng B (Open-Vocabulary):** Sử dụng cơ chế Generative (LLM-based), cho phép sinh các câu trả lời linh hoạt, mô tả chi tiết và có khả năng suy luận vượt ra ngoài các cụm từ có sẵn trong tập train.

**Cải tiến SOTA tích hợp:**
1. **Medical Backbone:** Sử dụng `torchxrayvision` (DenseNet-121) pretrained trên 200K+ ảnh X-ray.
2. **Custom Dual-Head:** Tối ưu hóa bằng cách tách nhánh Classifier (Yes/No) và Generator (LSTM/Transformer).
3. **Image Enhancement:** Thuật toán CLAHE tăng cường độ tương phản y tế.
4. **RLHF/DPO:** Huấn luyện bổ sung với 200 cặp dữ liệu preference.
5. **Đánh giá đa tầng:** Kết hợp tự động + LLM-as-a-judge + **Human Evaluation (Bắt buộc)**.

---

## Cấu trúc báo cáo & Sản phẩm
- **Báo cáo (15-20 trang):** Gồm các chương độc lập về Dữ liệu, Kiến trúc, Phương pháp đánh giá và Thực nghiệm.
- **GitHub:** Mã nguồn sạch, kèm README hướng dẫn.
- **HuggingFace:** Dataset sạch (`judge_results.json`) và Model Checkpoints.
- **Demo:** Giao diện Web tương tác bằng Gradio/Streamlit.

---

## Cấu trúc thư mục (Final)
```text
DL_MedicalVQA_Project/
├── configs/
│  └── medical_vqa.yaml     # Toàn bộ cấu hình (dataset, model, training, eval)
├── data/             # Dữ liệu (KHÔNG commit lên git)
│  ├── merged_vqa_vi.json     # Output sau dịch thuật (Train/Val/Test ID)
│  ├── test_in_domain.json    # Test Set 1 (In-Distribution): Trích từ SLAKE + VQA-RAD
│  ├── test_ood_vqamed.json    # Test Set 2 (Out-of-Distribution): Trích từ VQA-MED
│  └── preference_data_slake.json # DPO preference data
├── checkpoints/          # Model weights (KHÔNG commit)
├── logs/             # Training logs
├── scripts/
│  ├── data_pipeline.py      # Sinh dữ liệu, Paraphrase, Test Set 1 (ID)
│  ├── prepare_ood_test.py    # Tạo Test Set 2 (OOD) từ tập VQA-MED
│  └── llm_judge_eval.py     # Chấm điểm Semantic QA bằng Qwen-Plus API
├── src/
│  ├── config.py         # Dataclass config loader
│  ├── data/
│  │  ├── medical_dataset.py   # PyTorch Dataset cho SLAKE+VQA-RAD
│  │  └── translate_med_vqa.py # Pipeline dịch thuật 6 bước
│  ├── engine/
│  │  ├── trainer.py       # Training loop (A1/A2)
│  │  ├── medical_eval.py    # VQA Acc, BLEU, ROUGE, BERTScore, LLM-judge
│  │  └── dpo_trainer.py     # DPO training + preference data generator
│  ├── models/
│  │  ├── encoder.py       # CNNEncoder (DenseNet)
│  │  ├── phobert_encoder.py   # ViHealthBERT Text Encoder
│  │  ├── attention.py      # BahdanauAttention + SpatialAttention
│  │  ├── medical_vqa_model.py  # MedicalVQAModelA + CoAttentionFusion
│  │  ├── transformer_decoder.py # Transformer Decoder + Beam Search
│  │  └── multimodal_vqa.py   # Hướng B: LLaVA-Med wrapper
│  └── utils/
│    ├── metrics.py       # BLEU, ROUGE, METEOR, BERTScore
│    ├── helpers.py       # Tiện ích chung
│    └── visualization.py    # GradCAM, Radar chart, Confusion Matrix
├── app.py             # File chạy giao diện Demo Web
└── train_medical.py        # Entry point: train A1/A2/B1/B2/all
```

---

## Chiến lược Đánh giá Chéo (Cross-Dataset Evaluation)
Để chứng minh khả năng tổng quát hóa của mô hình và bám sát yêu cầu "Tập test chuẩn bị thủ công", hệ thống sử dụng 2 tập Test riêng biệt:
1. **Test Set 1 (In-Distribution):** Trích xuất ~60 ảnh (Image-disjoint) từ SLAKE + VQA-RAD để đảm bảo bảo toàn điểm số an toàn (Baseline).
2. **Test Set 2 (Out-of-Distribution):** Trích xuất ~50 ảnh thủ công từ **VQA-MED** (chỉ lấy X-Quang, MRI, CT). Dùng để kiểm tra khả năng chống chịu sự dịch chuyển miền dữ liệu (Domain Shift), được đánh giá tự động bằng **LLM-as-a-judge (Qwen-Plus API)**.

---

## Hướng dẫn chạy

### Yêu cầu Phần cứng
* **Hướng A:** Khả thi trên GPU phổ thông (T4 16GB VRAM, RTX 3060/4060) hoặc CPU (thời gian huấn luyện dài hơn).
* **Hướng B & DPO:** Yêu cầu GPU tối thiểu 16GB VRAM (Khuyến nghị sử dụng Kaggle P100/T4x2 hoặc Google Colab Pro) để chạy mô hình đa phương thức cùng kỹ thuật lượng tử hóa QLoRA 4-bit.

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Dịch thuật dataset (SLAKE + VQA-RAD → Tiếng Việt)

```bash
# Dịch VQA-RAD
python src/data/translate_med_vqa.py \
  --api_key "YOUR_GEMINI_API_KEY" \
  --dataset vqa-rad \
  --output data/translated_vqa_rad.json

# Dịch SLAKE
python src/data/translate_med_vqa.py \
  --api_key "YOUR_GEMINI_API_KEY" \
  --dataset slake \
  --output data/translated_slake.json

# Merge 2 file lại thành merged_vqa_vi.json (thủ công hoặc dùng script)
```

### 3. Tạo tập test thủ công (bắt buộc theo đề bài)

```bash
python scripts/create_manual_test.py \
  --input data/merged_vqa_vi.json \
  --output data/manual_test_set.json \
  --n_images 60
```

### 4. Huấn luyện 4 cấu hình bắt buộc

```bash
# Hướng A — Kiến trúc rời rạc
python train_medical.py --config configs/medical_vqa.yaml --variant A1
python train_medical.py --config configs/medical_vqa.yaml --variant A2

# Hướng B — Multimodal Pretrained
python train_medical.py --config configs/medical_vqa.yaml --variant B1 # Zero-shot
python train_medical.py --config configs/medical_vqa.yaml --variant B2 # LoRA fine-tune
```

### 5. Tạo DPO Preference Data & huấn luyện DPO

```bash
# Tạo preference data từ SLAKE format
python src/engine/dpo_trainer.py \
  --input data/merged_vqa_vi.json \
  --output data/preference_data_slake.json \
  --num_pairs 200

# DPO training (chạy sau B2)
python train_medical.py --config configs/medical_vqa.yaml --variant DPO
```

### 6. Khởi động Web Demo

```bash
python app.py
```

---

## Kết quả kỳ vọng

| Model | VQA-RAD Closed | VQA-RAD Open | SLAKE Acc |
|---|---|---|---|
| A1 (LSTM) | ~65–68% | ~50–53% | ~74–76% |
| A2 (Transformer + Beam Search) | ~68–72% | ~53–57% | ~76–79% |
| B1 (LLaVA-Med-7B Zero-shot) | ~62–68% | ~40–48% | ~70–75% |
| B2 (LLaVA-Med-7B + LoRA) | ~82–88% | ~62–70% | ~85–92% |

---

## Tài liệu tham khảo

- SLAKE Dataset: [PolyU, ACL 2021](https://arxiv.org/abs/2102.09542)
- VQA-RAD: [Lau et al., Nature Scientific Data 2018](https://www.nature.com/articles/sdata2018189)
- Dictionary-Enhanced Prompting: arXiv 2509.15640
- Co-Attention Fusion: [Kim et al., NeurIPS 2018](https://arxiv.org/abs/1805.07932)
- DPO: [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- PhoBERT: [Nguyen & Nguyen, EMNLP 2020](https://arxiv.org/abs/2003.00744)
```
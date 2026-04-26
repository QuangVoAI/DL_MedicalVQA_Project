# 📝 Tài liệu Kỹ thuật: Mô hình Baseline (Cấu hình A1)

Tài liệu này mô tả chi tiết thiết lập mô hình mốc (Baseline) cho dự án Medical VQA Tiếng Việt. Baseline được sử dụng để thiết lập một mức hiệu năng cơ bản, từ đó đánh giá sự cải tiến của các kiến trúc phức tạp hơn (Transformer, Multimodal).

## 1. Kiến trúc Mô hình (Architecture)
Mô hình Baseline sử dụng phương pháp **Rời rạc hóa (Modular Approach)** với các thành phần sau:

| Thành phần | Công nghệ sử dụng | Lý do lựa chọn |
|---|---|---|
| **Image Encoder** | **DenseNet-121 (XRV)** | Pretrained chuyên biệt trên 200,000+ ảnh X-quang, MRI (torchxrayvision). |
| **Text Encoder** | **PhoBERT-base** | Mô hình ngôn ngữ SOTA cho tiếng Việt, giúp hiểu ngữ cảnh y khoa bản địa. |
| **Fusion Layer** | **Linear Concatenation** | Gộp đặc trưng ảnh và văn bản (768 + 768) qua lớp tuyến tính để tạo vector hội tụ. |
| **Answer Decoder** | **LSTM (RNN)** | Mô hình giải mã chuỗi cổ điển, phù hợp làm mốc so sánh cho Transformer Decoder. |

## 2. Thông số Huấn luyện (Hyperparameters)
Để đảm bảo tính công bằng, Baseline được huấn luyện với các thông số tiêu chuẩn:
- **Optimizer:** AdamW (Learning Rate: 1e-4)
- **Loss Function:** Dual-CrossEntropy (Phân loại Yes/No + Sinh câu trả lời Open)
- **Batch Size:** 16 - 32 (Tùy thuộc vào VRAM)
- **Epochs:** 10 - 20
- **Sequence Length:** 10 tokens (Trả lời ngắn gọn theo yêu cầu y tế)

## 3. Quy trình đánh giá (Evaluation)
Mô hình Baseline sẽ được đánh giá trên 2 tập dữ liệu:
1. **In-Domain (ID):** Tập test trích từ SLAKE/VQA-RAD.
2. **Out-of-Distribution (OOD):** Tập test thủ công từ VQA-MED.

**Các chỉ số đo lường:**
- **Accuracy:** Cho các câu hỏi đóng (Yes/No).
- **BLEU-4 / ROUGE-L:** Cho các câu hỏi mở mô tả bệnh lý.
- **BERTScore:** Đánh giá độ tương đồng về ngữ nghĩa y khoa.

## 4. Mục tiêu của Baseline
- Xác định khả năng xử lý tiếng Việt của PhoBERT trong miền y khoa.
- Kiểm tra xem cơ chế LSTM có đủ khả năng ghi nhớ các đặc trưng hình ảnh phức tạp hay không.
- Làm căn cứ để chứng minh hiệu quả của cơ chế **Attention** và **Transformer** trong các cấu hình A2, B2.

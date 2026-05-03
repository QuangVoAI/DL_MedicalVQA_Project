Mình đã rà lại toàn bộ source + tài liệu mô tả chính của project, và có 2 điểm rất quan trọng trước khi bạn viết báo cáo:

README/baseline/optimization docs có vài chỗ đã cũ hoặc mô tả “định hướng” hơn là code đang có thật.
Báo cáo nên bám vào code thực tế trong repo, đặc biệt là:

train_medical.py

src/models/medical_vqa_model.py

src/models/transformer_decoder.py

src/models/multimodal_vqa.py

src/engine/trainer.py

src/engine/medical_eval.py

web/main.py

configs/medical_vqa.yaml




Một số lệch cần sửa trong báo cáo:


README nhắc app.py, nhưng demo thực tế là web/main.py.

README nhắc src/data/translate_med_vqa.py, nhưng pipeline dịch thực tế nằm ở scripts/data_pipeline.py và src/utils/translator.py.

README nhắc scripts/prepare_ood_test.py, nhưng file thực tế để tạo tập test thủ công là scripts/create_manual_test.py.

OPTIMIZATION_REPORT.md nhắc một số file như src/utils/optimized_metrics.py và src/utils/medical_augmentation.py, nhưng chúng không có trong snapshot repo hiện tại, nên không nên viết vào báo cáo như là code đã có.


Dưới đây là nội dung báo cáo mình khuyên bạn dùng.

1. Tóm tắt đề tài

Đề tài xây dựng hệ thống Medical Visual Question Answering tiếng Việt trên bộ dữ liệu SLAKE và VQA-RAD đã được dịch sang tiếng Việt. Mục tiêu của project là tạo ra một mô hình có khả năng trả lời câu hỏi y khoa dựa trên ảnh chẩn đoán bằng cả hai hướng: hướng rời rạc truyền thống với encoder-decoder, và hướng sinh tự do dựa trên mô hình đa phương thức lớn. Hệ thống được thiết kế để xử lý cả câu hỏi đóng dạng Yes/No lẫn câu hỏi mở mô tả tổn thương, vị trí, phương thức chụp và cơ quan.

2. Cơ sở dữ liệu

Project sử dụng hai nguồn chính:


SLAKE, một dataset y khoa đa ngôn ngữ có chú thích ngữ nghĩa.

VQA-RAD, dataset câu hỏi trả lời cho ảnh X-quang và chẩn đoán hình ảnh.


Dữ liệu gốc được chuẩn hóa sang tiếng Việt, gắn nhãn theo kiểu câu hỏi đóng/mở, và được lưu thành bộ dữ liệu đã merge để train/validation/test. Một pipeline khác được dùng để tạo tập test thủ công nhằm đánh giá thực tế và phục vụ human review.

3. Cơ sở lý thuyết và kiến thức sử dụng

Hệ thống này kết hợp nhiều mảng kiến thức:


Computer Vision: dùng CNN DenseNet-121 làm image encoder, có tối ưu riêng cho ảnh y khoa.

NLP tiếng Việt: dùng PhoBERT để biểu diễn câu hỏi tiếng Việt.

Multimodal learning: dùng co-attention/cross-attention để trộn đặc trưng ảnh và văn bản.

Sequence generation: dùng LSTM và Transformer Decoder để sinh câu trả lời.

Efficient fine-tuning: dùng LoRA và QLoRA cho LLaVA-Med.

RLHF/alignment: dùng DPO và PPO để tinh chỉnh đầu ra theo preference y khoa.

Evaluation NLP: dùng Accuracy, EM, F1, BLEU, ROUGE-L, METEOR, BERTScore và semantic similarity.


4. Kiến trúc hệ thống

Project tách thành hai hướng:



Hướng A là mô hình modular:


Image encoder: DenseNet-121 từ TorchXRayVision.

Text encoder: PhoBERT.

Fusion: co-attention.

Decoder: hai biến thể, A1 là LSTM, A2 là Transformer Decoder.

Output head: tách nhánh closed-head cho câu trả lời Yes/No và open-head cho câu trả lời sinh tự do.





Hướng B là mô hình generative:


Dùng LLaVA-Med 7B làm nền tảng.

B1 là zero-shot.

B2 là fine-tuned bằng LoRA/QLoRA.

DPO và PPO là các bước tinh chỉnh bổ sung để cải thiện độ phù hợp với preference y khoa.





5. Luồng dữ liệu

Dữ liệu đi qua các bước:


Chuẩn hóa câu hỏi và câu trả lời.

Dịch sang tiếng Việt bằng pipeline translation có từ điển y khoa.

Làm sạch output và canonicalize các thuật ngữ y khoa.

Tạo train/validation/test.

Tạo preference pairs cho DPO.

Tạo tập test thủ công để kiểm tra thủ công hoặc làm benchmark bổ sung.


File trung tâm cho phần này là:


src/data/medical_dataset.py

src/utils/text_utils.py

src/utils/translator.py

scripts/data_pipeline.py

scripts/create_manual_test.py


6. Mô hình A1/A2

Trong src/models/medical_vqa_model.py, mô hình A dùng DenseNet-121 để trích đặc trưng không gian của ảnh và PhoBERT để mã hóa câu hỏi. Đặc trưng ảnh và text được đưa vào lớp co-attention để học tương tác liên miền. Sau đó decoder sinh hai đầu ra:


classifier head cho câu hỏi đóng.

generator head cho câu hỏi mở.


A1 dùng LSTM decoder, phù hợp làm baseline tuần tự.

A2 thay LSTM bằng Transformer Decoder, cho khả năng mô hình hóa phụ thuộc dài hơn và thường cho kết quả tốt hơn trên câu hỏi mở.

MedicalVQADecoder trong src/models/transformer_decoder.py còn có các điểm đáng chú ý:


weight tying giữa embedding và output projection.

beam search có length normalization.

causal mask cache.

tách training/inference rõ ràng.


7. Mô hình B1/B2/DPO/PPO

Trong src/models/multimodal_vqa.py, LLaVA-Med được nạp với 4-bit quantization và LoRA để giảm VRAM. Đây là lựa chọn phù hợp nếu muốn fine-tune mô hình lớn trên phần cứng giới hạn.

Trong train_medical.py, B2 được train bằng SFT với prompt tiếng Việt, còn DPO và PPO là các bước refinement:


B2 học từ cặp prompt-answer chuẩn.

DPO học từ preference data gồm chosen/rejected.

PPO dùng reward từ câu trả lời sinh ra, nhấn mạnh consistency và semantic match.


8. Huấn luyện

Trong src/engine/trainer.py, training loop của hướng A có các kỹ thuật:


AMP mixed precision.

gradient accumulation.

dynamic class weights cho nhãn Yes/No.

cosine scheduler với warmup.

label smoothing cho nhánh open.

early stopping theo patience.


Loss cũng được tách theo hai nhánh:


closed loss cho câu hỏi đóng.

open loss cho câu hỏi mở, kèm penalty để tránh model quá ngắn hoặc quá “chỉ đoán một token”.


Trong configs/medical_vqa.yaml, các biến thể A1/A2/B1/B2/DPO/PPO được cấu hình riêng, bao gồm batch size, learning rate, beam width, số token tối đa và các tham số LoRA/QLoRA.

9. Tiền xử lý ảnh

src/utils/visualization.py chứa MedicalImageTransform, hiện thực:


resize ảnh.

áp dụng CLAHE để tăng tương phản cục bộ.

chuyển sang tensor 1 kênh.

scale theo dải phù hợp cho XRayVision.


Trong tài liệu safety, project nhấn mạnh không nên dùng augmentation nguy hiểm như flip lớn hay rotation lớn đối với ảnh y khoa. Tuy nhiên trong code hiện tại, phần augmentation thực tế chủ yếu là CLAHE và normalization, nên báo cáo nên mô tả đúng như vậy.

10. Đánh giá

src/engine/medical_eval.py là file đánh giá quan trọng nhất. Nó tách rõ:


prediction raw.

prediction normalized.

closed vs open.

long-answer evaluation.


Cách đánh giá này rất hợp lý cho Medical VQA vì:


câu hỏi đóng cần so khớp nhãn chuẩn.

câu hỏi mở cần đánh giá ngữ nghĩa, không chỉ exact match.


Các metric dùng trong repo:


Accuracy, EM, F1 cho câu trả lời ngắn.

BLEU-1/2/3/4, ROUGE-L, METEOR cho sinh tự do.

BERTScore và semantic score để đo độ gần về nghĩa.

human review và LLM-judge để kiểm tra chất lượng dịch thuật và câu trả lời.


11. Demo web

web/main.py xây dựng FastAPI server để:


upload ảnh.

nhập câu hỏi.

chạy so sánh giữa A1, A2, B1, B2, DPO, PPO.

cache model.

rewrite câu trả lời đầu ra bằng một layer phụ.


Phần này rất phù hợp để đưa vào báo cáo như “hệ thống triển khai thực nghiệm” hoặc “giao diện minh họa mô hình”.

12. Kết luận kỹ thuật

Điểm mạnh lớn nhất của project là không chỉ xây model, mà còn xây đủ pipeline hoàn chỉnh:


dữ liệu,

dịch thuật,

preprocessing,

training,

evaluation,

alignment,

web demo,

logging với WandB,

xuất biểu đồ so sánh.


Điều này giúp báo cáo có thể viết theo hướng một hệ thống end-to-end cho Medical VQA tiếng Việt, chứ không phải chỉ là một mô hình đơn lẻ.

13. Phần nên đưa thẳng vào báo cáo

Bạn có thể viết phần “đóng góp chính” như sau:


Xây dựng pipeline Medical VQA tiếng Việt từ hai dataset y khoa lớn là SLAKE và VQA-RAD.

Thiết kế kiến trúc modular với DenseNet-121, PhoBERT và co-attention cho hướng truyền thống.

Thiết kế hướng generative với LLaVA-Med và fine-tuning bằng LoRA/QLoRA.

Bổ sung DPO/PPO để cải thiện alignment và tính y khoa của câu trả lời.

Xây dựng hệ thống đánh giá đa tầng kết hợp metric tự động, LLM-as-a-judge và human review.

Triển khai web demo phục vụ thử nghiệm và so sánh nhiều biến thể mô hình.


14. Tài liệu tham khảo nên trích

Dưới đây là danh sách paper/link chuẩn để bạn đưa vào báo cáo:


SLAKE: arXiv 2102.09542

VQA-RAD: Nature Scientific Data 2018

DenseNet: arXiv 1608.06993

Bahdanau attention: arXiv 1409.0473

Transformer: arXiv 1706.03762

Co-attention: arXiv 1606.00061

PhoBERT: arXiv 2003.00744

Medical VQA survey: arXiv 2111.10056

LLaVA: arXiv 2304.08485

LLaVA-Med: arXiv 2306.00890

LoRA: arXiv 2106.09685

QLoRA: arXiv 2305.14314

DPO: arXiv 2305.18290

PPO: arXiv 1707.06347

BERTScore: arXiv 1904.09675

Dictionary-enhanced prompting cho MT/domain adaptation: arXiv 2402.15061
## Medical VQA Web

Thư mục này chứa FastAPI + web UI để:

- upload ảnh
- nhập câu hỏi VQA
- chạy dự đoán
- so sánh 6 model: `A1`, `A2`, `B1`, `B2`, `DPO`, `PPO`

### Chạy server

Từ thư mục gốc project:

```bash
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
```

Nếu muốn preload toàn bộ model khi startup trên GPU:

```bash
WEB_PRELOAD_MODELS=1 uvicorn web.main:app --host 0.0.0.0 --port 8000
```

Khi chạy trên GPU, nên để `--workers 1` để tránh mỗi worker nạp một bản model riêng.

### Chạy bằng Docker

Build image:

```bash
docker build -t medical-vqa-web .
```

Run container trên máy có GPU:

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e WEB_PRELOAD_MODELS=1 \
  -v medical-vqa-hf-cache:/hf_cache \
  medical-vqa-web
```

Nếu muốn chạy lại nhanh hơn, giữ volume cache `medical-vqa-hf-cache` để không tải lại model Hugging Face mỗi lần.

### Tùy chọn: rewrite output bằng Qwen

Lớp rewrite hiện đã bật mặc định và sẽ tự thử load Qwen từ Hugging Face Hub khi server khởi động.
Nếu bạn muốn đổi sang model repo khác trên Hub, đặt thêm các biến môi trường sau:

```bash
ANSWER_REWRITE_ENABLED=1
ANSWER_REWRITE_MODEL_ID=Qwen/Qwen2.5-14B-Instruct
ANSWER_REWRITE_USE_4BIT=1
ANSWER_REWRITE_MAX_NEW_TOKENS=28
ANSWER_REWRITE_MAX_WORDS=10
ANSWER_REWRITE_HF_TOKEN=hf_...
```

Lớp này chỉ rewrite phần output hiển thị, không thay thế model VQA chính. Nếu model rewrite không load được, hệ thống sẽ tự fallback về output hiện tại.

Mở:

```text
http://localhost:8000
```

### API

- `GET /health`
  - kiểm tra trạng thái server và artifact khả dụng
- `GET /v1/models`
  - trả metadata 6 model
- `POST /v1/predict`
  - form-data:
    - `question`: câu hỏi VQA
    - `image`: ảnh đầu vào
    - `model_name` hoặc `model_names`:
      - nếu bỏ trống thì chạy toàn bộ 6 model
      - `model_names` nhận chuỗi JSON list hoặc chuỗi phân tách bằng dấu phẩy

### Artifact cần có

- `A1`: `checkpoints/medical_vqa_A1_best.pth`
- `A2`: `checkpoints/medical_vqa_A2_best.pth`
- `B1`: model base từ `model_b.model_name` trong `configs/medical_vqa.yaml`
- `B2`: checkpoint tốt nhất trong `checkpoints/B2/checkpoint-*`
- `DPO`: `checkpoints/DPO/final_adapter` hoặc `checkpoints/DPO/checkpoint-25`
- `PPO`: `checkpoints/PPO/final_adapter`

### Lưu ý

- `B1`, `B2`, `DPO`, `PPO` cần CUDA để chạy ổn trong cấu hình hiện tại.
- Nếu một model chưa có artifact hoặc không đủ điều kiện chạy, UI vẫn hiển thị lỗi riêng cho model đó thay vì làm hỏng toàn bộ request.
- Web giữ model trong cache sau lần load đầu tiên, nên request sau sẽ nhanh hơn đáng kể.

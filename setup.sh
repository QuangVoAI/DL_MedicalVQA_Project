#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# setup.sh — Medical VQA Environment Setup
# Hỗ trợ: Vast.ai (CUDA), Google Colab, local macOS (CPU/MPS)
#
# Cách dùng:
#   chmod +x setup.sh && bash setup.sh
#   bash setup.sh --colab        # Google Colab mode (skip git config)
#   bash setup.sh --offline      # Offline mode (không sync WandB)
#   bash setup.sh --skip-nltk    # Bỏ qua download NLTK data
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────────────────────
COLAB_MODE=0
OFFLINE_MODE=0
SKIP_NLTK=0
for arg in "$@"; do
  case $arg in
    --colab)    COLAB_MODE=1  ;;
    --offline)  OFFLINE_MODE=1 ;;
    --skip-nltk) SKIP_NLTK=1 ;;
  esac
done

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🏥  Medical VQA — Environment Setup"
echo "  Project: DL Final 523H0173 & 523H0178"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── 1. Python version check ──────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)

info "Python $PY_VER tại: $($PYTHON -c 'import sys; print(sys.executable)')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  error "Cần Python ≥ 3.10 (hiện tại: $PY_VER)"
fi

# ── 2. GPU detection ─────────────────────────────────────────────────────────
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "True" ]; then
  GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
  VRAM=$($PYTHON -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1))" 2>/dev/null || echo "?")
  info "GPU: $GPU_NAME | VRAM: ${VRAM}GB"
else
  warn "Không phát hiện CUDA GPU — training sẽ rất chậm trên CPU"
fi

# ── 3. Install pip packages ──────────────────────────────────────────────────
info "Cài đặt dependencies từ requirements.txt..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
  error "Không tìm thấy $REQ_FILE"
fi

# Nâng pip trước
$PYTHON -m pip install --upgrade pip --quiet

# Cài main requirements (quiet để giảm noise)
$PYTHON -m pip install -r "$REQ_FILE" --quiet || {
  warn "Cài đặt silent thất bại, thử với verbose..."
  $PYTHON -m pip install -r "$REQ_FILE"
}

# wandb (cần version chính xác)
$PYTHON -m pip install "wandb>=0.16.0" --quiet
info "✅ Dependencies đã cài xong"

# ── 4. NLTK data download ─────────────────────────────────────────────────────
if [ "$SKIP_NLTK" -eq 0 ]; then
  info "Tải NLTK data (punkt, wordnet)..."
  $PYTHON -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f'  [WARN] NLTK {pkg}: {e}')
print('  NLTK data OK')
"
fi

# ── 5. Python path configuration ─────────────────────────────────────────────
info "Cấu hình Python path..."

# Tạo .pth file để Python tự động thêm project root vào sys.path
SITE_PACKAGES=$($PYTHON -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || \
                $PYTHON -c "import site; print(site.getusersitepackages())")
PTH_FILE="$SITE_PACKAGES/medical_vqa.pth"

echo "$SCRIPT_DIR" > "$PTH_FILE" && \
  info "✅ Path cấu hình tại: $PTH_FILE" || \
  warn "Không thể ghi vào site-packages, thử export PYTHONPATH thủ công."

# Cũng export PYTHONPATH trong session hiện tại
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
info "PYTHONPATH = $PYTHONPATH"

# ── 6. .env file ─────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  warn "Đã tạo .env từ .env.example — Hãy điền WANDB_API_KEY!"
fi

if [ -f "$ENV_FILE" ]; then
  # Source .env (bỏ qua comment và dòng trống)
  set -a
  source <(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$') 2>/dev/null || true
  set +a
  info ".env đã được load"
fi

# ── 7. WandB login ───────────────────────────────────────────────────────────
if [ "$OFFLINE_MODE" -eq 1 ]; then
  export WANDB_MODE=offline
  info "WandB: OFFLINE mode (sync sau bằng: wandb sync)"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  $PYTHON -m wandb login "$WANDB_API_KEY" --relogin --quiet 2>/dev/null && \
    info "✅ WandB logged in (entity: SpringWang08)" || \
    warn "WandB login thất bại — kiểm tra WANDB_API_KEY"
else
  warn "WANDB_API_KEY chưa được set — WandB sẽ bị bỏ qua khi training"
  warn "  Set bằng: export WANDB_API_KEY=your_key"
  warn "  Hoặc điền vào file .env"
fi

# ── 8. HuggingFace login ─────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  $PYTHON -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)" 2>/dev/null && \
    info "✅ HuggingFace logged in" || \
    warn "HF login thất bại — dataset công khai vẫn tải được"
else
  warn "HF_TOKEN chưa được set (không cần nếu dataset là public)"
fi

# ── 9. Tạo thư mục cần thiết ─────────────────────────────────────────────────
info "Tạo thư mục dự án..."
for dir in checkpoints logs/history results/charts data scripts; do
  mkdir -p "$SCRIPT_DIR/$dir"
done
info "✅ Thư mục sẵn sàng"

# ── 10. Smoke test import ─────────────────────────────────────────────────────
info "Kiểm tra imports..."
$PYTHON - <<'PYEOF'
import sys, importlib
ok, fail = [], []
checks = [
    ("torch",             "PyTorch"),
    ("torchvision",       "TorchVision"),
    ("transformers",      "Transformers"),
    ("datasets",          "HF Datasets"),
    ("peft",              "PEFT (LoRA)"),
    ("trl",               "TRL (SFT/DPO)"),
    ("wandb",             "WandB"),
    ("nltk",              "NLTK"),
    ("bert_score",        "BERTScore"),
    ("rouge_score",       "ROUGE"),
    ("sklearn",           "Scikit-learn"),
    ("matplotlib",        "Matplotlib"),
    ("yaml",              "PyYAML"),
    ("dotenv",            "python-dotenv"),
    ("cv2",               "OpenCV"),
]
for mod, name in checks:
    try:
        importlib.import_module(mod)
        ok.append(name)
    except ImportError:
        fail.append(name)

print(f"  ✅ OK ({len(ok)}): {', '.join(ok)}")
if fail:
    print(f"  ❌ MISSING ({len(fail)}): {', '.join(fail)}")
    sys.exit(1)
PYEOF

# ── 11. Kiểm tra src modules ─────────────────────────────────────────────────
info "Kiểm tra src modules..."
$PYTHON - <<'PYEOF'
import sys
checks = [
    "src.models.medical_vqa_model",
    "src.models.transformer_decoder",
    "src.engine.trainer",
    "src.engine.medical_eval",
    "src.data.medical_dataset",
    "src.utils.text_utils",
    "src.utils.translator",
]
ok, fail = [], []
for mod in checks:
    try:
        __import__(mod)
        ok.append(mod.split(".")[-1])
    except Exception as e:
        fail.append(f"{mod.split('.')[-1]} ({e})")

print(f"  ✅ src OK ({len(ok)}): {', '.join(ok)}")
if fail:
    print(f"  ❌ src FAIL ({len(fail)}): {', '.join(fail)}")
PYEOF

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅  Setup hoàn tất!"
echo ""
echo "  Tiếp theo:"
echo "    export WANDB_API_KEY=your_key    # nếu chưa có"
echo "    python train_medical.py --variant A1"
echo "    python train_medical.py --variant A2"
echo "    python train_medical.py --variant B1"
echo "    python train_medical.py --variant B2"
echo "    python train_medical.py --variant DPO"
echo ""
echo "  So sánh 5 model sau khi train xong:"
echo "    python scripts/compare_models.py"
echo "════════════════════════════════════════════════════════════"
echo ""

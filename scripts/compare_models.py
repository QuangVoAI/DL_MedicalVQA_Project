"""
compare_models.py — Vẽ biểu đồ so sánh 5 variant sau khi training xong.

Cách dùng:
    python scripts/compare_models.py                        # auto-tìm tất cả history
    python scripts/compare_models.py --log_dir logs/history # chỉ định thư mục
    python scripts/compare_models.py --out results/charts   # thư mục lưu chart

Tự động tìm file history.json theo pattern:
    logs/history/{VARIANT}/{timestamp}/history.json
"""

import argparse
import json
import os
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Cấu hình ────────────────────────────────────────────────────────────────

VARIANTS = ["A1", "A2", "B1", "B2", "DPO"]

COLORS = {
    "A1":  "#2ecc71",   # xanh lá
    "A2":  "#3498db",   # xanh dương
    "B1":  "#e67e22",   # cam
    "B2":  "#9b59b6",   # tím
    "DPO": "#e74c3c",   # đỏ
}

MARKERS = {
    "A1": "o", "A2": "s", "B1": "^", "B2": "D", "DPO": "P"
}

METRICS_LABELS = {
    "val_accuracy_normalized": "Accuracy",
    "val_f1_normalized":       "F1 Score",
    "val_bleu4_normalized":    "BLEU-4",
    "val_bert_score_raw":      "BERTScore",
    "val_semantic_raw":        "Semantic Score",
    "train_loss":              "Train Loss",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_history(log_dir: str, variant: str) -> dict | None:
    """
    Tìm file history.json mới nhất cho một variant.
    Hỗ trợ cả 2 format:
      • logs/history/{VARIANT}/{timestamp}/history.json  (MedicalVQATrainer)
      • logs/history/{VARIANT}/history.json              (flat)
    """
    patterns = [
        os.path.join(log_dir, variant, "**", "history.json"),
        os.path.join(log_dir, variant, "history.json"),
        os.path.join(log_dir, "**", variant, "**", "history.json"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))

    if not found:
        return None

    # Lấy file mới nhất theo mtime
    latest = max(found, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[✓] {variant}: {latest} ({len(data)} records)")
        return {"path": latest, "records": data}
    except Exception as e:
        print(f"[✗] {variant}: đọc thất bại — {e}")
        return None


def extract_series(records: list, key: str) -> tuple[list, list]:
    """Trích xuất (epochs, values) từ list records."""
    epochs, values = [], []
    for r in records:
        # Hỗ trợ cả HuggingFace log format (có 'epoch' float) và MedicalVQATrainer format
        epoch = r.get("epoch")
        if epoch is None:
            continue
        val = r.get(key)
        if val is None:
            # Thử alias cho HF SFTTrainer/DPOTrainer logs
            aliases = {
                "val_accuracy_normalized": ["eval_accuracy", "eval_vqa_accuracy"],
                "val_f1_normalized":       ["eval_f1"],
                "val_bleu4_normalized":    ["eval_bleu4", "eval_bleu"],
                "val_bert_score_raw":      ["eval_bertscore", "eval_bert_score"],
                "val_semantic_raw":        ["eval_semantic"],
                "train_loss":              ["loss", "train/loss"],
            }
            for alias in aliases.get(key, []):
                val = r.get(alias)
                if val is not None:
                    break
        if val is not None:
            epochs.append(float(epoch))
            values.append(float(val))
    return epochs, values


def get_best_metric(records: list, key: str) -> float | None:
    """Trả về giá trị tốt nhất của một metric."""
    _, values = extract_series(records, key)
    if not values:
        return None
    return max(values) if key != "train_loss" else min(values)


# ─── Plot functions ───────────────────────────────────────────────────────────

def plot_metric_curves(all_data: dict, metric_key: str, output_dir: str):
    """Vẽ đường cong một metric cho tất cả variant."""
    label = METRICS_LABELS.get(metric_key, metric_key)
    minimize = metric_key == "train_loss"

    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = 0
    for variant, info in all_data.items():
        if info is None:
            continue
        epochs, values = extract_series(info["records"], metric_key)
        if not epochs:
            continue

        ax.plot(
            epochs, values,
            color=COLORS[variant], linewidth=2.5,
            marker=MARKERS[variant], markersize=7,
            label=f"{variant} (best={min(values) if minimize else max(values):.3f})"
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"[SKIP] {label}: không có dữ liệu")
        return

    ax.set_title(f"{label} — So sánh 5 Variant", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if metric_key != "train_loss":
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = os.path.join(output_dir, f"compare_{metric_key}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")


def plot_final_bar(all_data: dict, output_dir: str):
    """
    Bar chart so sánh kết quả cuối (best) của từng model
    trên 4 metrics: Accuracy, F1, BLEU-4, BERTScore.
    """
    metric_keys   = ["val_accuracy_normalized", "val_f1_normalized",
                     "val_bleu4_normalized",    "val_bert_score_raw"]
    metric_labels = ["Accuracy", "F1", "BLEU-4", "BERTScore"]

    variants_with_data = [v for v in VARIANTS if all_data.get(v)]
    if not variants_with_data:
        print("[SKIP] Final bar chart: không có dữ liệu")
        return

    x   = np.arange(len(metric_labels))
    w   = 0.8 / len(variants_with_data)

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, variant in enumerate(variants_with_data):
        info   = all_data[variant]
        values = [get_best_metric(info["records"], k) or 0.0 for k in metric_keys]
        offset = (i - len(variants_with_data) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, values, w, label=variant,
                        color=COLORS[variant], alpha=0.88)
        # Hiển thị số liệu trên đầu cột
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold"
                )

    ax.set_title("Kết quả tốt nhất — So sánh 5 Variant",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    fname = os.path.join(output_dir, "compare_final_bar.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")


def plot_radar(all_data: dict, output_dir: str):
    """Radar chart so sánh 5 model trên 5 chiều."""
    metric_keys   = ["val_accuracy_normalized", "val_f1_normalized",
                     "val_bleu4_normalized",    "val_bert_score_raw",
                     "val_semantic_raw"]
    metric_labels = ["Accuracy", "F1", "BLEU-4", "BERTScore", "Semantic"]

    variants_with_data = [v for v in VARIANTS if all_data.get(v)]
    if len(variants_with_data) < 2:
        return

    N      = len(metric_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for variant in variants_with_data:
        info   = all_data[variant]
        values = [get_best_metric(info["records"], k) or 0.0 for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5,
                color=COLORS[variant], label=variant, marker=MARKERS[variant])
        ax.fill(angles, values, alpha=0.08, color=COLORS[variant])

    ax.set_title("Radar — So sánh 5 Variant (Best per Metric)",
                 fontsize=14, fontweight="bold", y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    fig.tight_layout()

    fname = os.path.join(output_dir, "compare_radar.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")


def plot_loss_comparison(all_data: dict, output_dir: str):
    """Train Loss của tất cả variant trên cùng trục."""
    plot_metric_curves(all_data, "train_loss", output_dir)


def print_summary_table(all_data: dict):
    """In bảng tóm tắt ra console."""
    metric_keys   = ["val_accuracy_normalized", "val_f1_normalized",
                     "val_bleu4_normalized",    "val_bert_score_raw",
                     "val_semantic_raw"]
    metric_short  = ["Accuracy", "F1", "BLEU-4", "BERT", "Semantic"]

    header = f"{'Model':<8}" + "".join(f"{m:>12}" for m in metric_short)
    print("\n" + "═" * (8 + 12 * len(metric_short)))
    print("  📊  FINAL COMPARISON — ALL VARIANTS")
    print("═" * (8 + 12 * len(metric_short)))
    print(f"  {header}")
    print("─" * (8 + 12 * len(metric_short)))

    for variant in VARIANTS:
        info = all_data.get(variant)
        if info is None:
            print(f"  {variant:<8}" + "".join(f"{'N/A':>12}" for _ in metric_keys))
            continue
        row = f"  {variant:<8}"
        for k in metric_keys:
            best = get_best_metric(info["records"], k)
            row += f"{best:>12.2%}" if best is not None else f"{'N/A':>12}"
        print(row)

    print("═" * (8 + 12 * len(metric_short)) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="So sánh 5 variant Medical VQA")
    parser.add_argument("--log_dir", default="logs/history",
                        help="Thư mục gốc chứa history (default: logs/history)")
    parser.add_argument("--out", default="results/charts",
                        help="Thư mục lưu biểu đồ (default: results/charts)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n[INFO] Tìm history tại: {args.log_dir}")
    print("─" * 60)

    # Thu thập dữ liệu từ tất cả variant
    all_data: dict = {}
    for variant in VARIANTS:
        all_data[variant] = find_latest_history(args.log_dir, variant)

    available = [v for v in VARIANTS if all_data[v]]
    print(f"\n[INFO] Có dữ liệu: {available}")
    if not available:
        print("[ERROR] Không tìm thấy bất kỳ history.json nào. Hãy train trước!")
        return

    print(f"\n[INFO] Đang vẽ biểu đồ → {args.out}/")
    print("─" * 60)

    # 1. Accuracy curves
    plot_metric_curves(all_data, "val_accuracy_normalized", args.out)
    # 2. F1 curves
    plot_metric_curves(all_data, "val_f1_normalized", args.out)
    # 3. BLEU-4 curves
    plot_metric_curves(all_data, "val_bleu4_normalized", args.out)
    # 4. Train loss
    plot_loss_comparison(all_data, args.out)
    # 5. BERTScore
    plot_metric_curves(all_data, "val_bert_score_raw", args.out)
    # 6. Bar chart tổng hợp
    plot_final_bar(all_data, args.out)
    # 7. Radar chart
    plot_radar(all_data, args.out)

    # In bảng tóm tắt
    print_summary_table(all_data)

    print(f"[DONE] Tất cả biểu đồ đã lưu tại: {args.out}/")
    charts = glob.glob(os.path.join(args.out, "compare_*.png"))
    for c in sorted(charts):
        print(f"  📊 {os.path.basename(c)}")


if __name__ == "__main__":
    main()

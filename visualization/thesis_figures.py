"""
Main thesis figure generator.
Reads results/combined_results.json and produces comparison charts.

Usage:
    python -m visualization.thesis_figures
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, OUTPUT_DIR, FIGURES_DIR

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

HOG_COLOR  = "#4C72B0"
YOLO_COLOR = "#DD8452"


def load_results() -> dict:
    path = os.path.join(OUTPUT_DIR, "combined_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run run_evaluation.py first.\nExpected: {path}")
    with open(path) as f:
        return json.load(f)


def fig_map_comparison(results: dict):
    hog_map  = results["HOG_SVM"]["mAP"]
    yolo_map = results["YOLOv8n"]["mAP"]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(
        ["HOG + SVM", "YOLOv8n"],
        [hog_map * 100, yolo_map * 100],
        color=[HOG_COLOR, YOLO_COLOR],
        width=0.45, edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Overall Detection Accuracy (mAP@0.5)")
    ax.set_ylim(0, max(hog_map, yolo_map) * 100 * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "01_map_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_per_class_ap(results: dict):
    hog_aps  = [results["HOG_SVM"]["per_class_ap"].get(c, 0) * 100 for c in CLASSES]
    yolo_aps = [results["YOLOv8n"]["per_class_ap"].get(c, 0) * 100 for c in CLASSES]

    x     = np.arange(len(CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x - width/2, hog_aps,  width, label="HOG + SVM",  color=HOG_COLOR,  edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width/2, yolo_aps, width, label="YOLOv8n",    color=YOLO_COLOR, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax.set_ylabel("AP@0.5 (%)")
    ax.set_title("Per-Class Average Precision (AP@0.5)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)

    out = os.path.join(FIGURES_DIR, "02_per_class_ap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_speed_accuracy(results: dict):
    data = [
        ("HOG + SVM", results["HOG_SVM"]["mAP"] * 100, results["HOG_SVM"]["mean_ms"], HOG_COLOR),
        ("YOLOv8n",   results["YOLOv8n"]["mAP"] * 100, results["YOLOv8n"]["mean_ms"], YOLO_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, acc, speed, color in data:
        ax.scatter(speed, acc, s=180, color=color, zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(label, (speed, acc),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Mean Inference Time (ms / image, CPU)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Speed vs Accuracy Trade-off (CPU)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "03_speed_accuracy.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_comparison_table(results: dict):
    hog  = results["HOG_SVM"]
    yolo = results["YOLOv8n"]

    rows = [
        ["Metric",              "HOG + SVM",                             "YOLOv8n"],
        ["mAP@0.5",             f"{hog['mAP']*100:.1f}%",               f"{yolo['mAP']*100:.1f}%"],
        ["Inference (ms/img)",  f"{hog['mean_ms']:.0f}",                f"{yolo['mean_ms']:.0f}"],
        ["Throughput (FPS)",    f"{hog['fps']:.1f}",                     f"{yolo['fps']:.1f}"],
        ["RAM delta (MB)",      f"{hog['ram_delta_mb']:.0f}",            f"{yolo['ram_delta_mb']:.0f}"],
        ["Model size (MB)",     f"{hog.get('model_size_mb',0):.1f}",     f"{yolo.get('model_size_mb',0):.1f}"],
        ["Feature extraction",  "Hand-crafted (HOG)",                    "Learned (CNN)"],
        ["Training data needed","Yes (positives + negatives)",            "Yes (bounding boxes)"],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    col_widths = [0.35, 0.32, 0.33]
    table = ax.table(
        cellText  = [r for r in rows[1:]],
        colLabels = rows[0],
        cellLoc   = "center",
        loc       = "center",
        colWidths = col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(3):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows)):
        bg = "#F0F3F4" if i % 2 == 0 else "white"
        for j in range(3):
            table[(i, j)].set_facecolor(bg)

    ax.set_title("Table: Detector Comparison Summary", pad=10, fontsize=11, fontweight="bold")

    out = os.path.join(FIGURES_DIR, "04_comparison_table.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_pr_curves(results: dict):
    hog_pc  = results["HOG_SVM"].get("per_class_pr", {})
    yolo_pc = results["YOLOv8n"].get("per_class_pr", {})

    ranked = sorted(CLASSES,
                    key=lambda c: (results["HOG_SVM"]["per_class_ap"].get(c, 0) +
                                   results["YOLOv8n"]["per_class_ap"].get(c, 0)),
                    reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Precision-Recall Curves (selected classes)", fontsize=13, fontweight="bold")

    for ax, cls in zip(axes.flat, ranked):
        hog_data  = hog_pc.get(cls, {})
        yolo_data = yolo_pc.get(cls, {})

        if hog_data:
            ax.plot(hog_data["recall"], hog_data["precision"],
                    color=HOG_COLOR, lw=1.5,
                    label=f"HOG+SVM  AP={results['HOG_SVM']['per_class_ap'].get(cls,0)*100:.1f}%")
        if yolo_data:
            ax.plot(yolo_data["recall"], yolo_data["precision"],
                    color=YOLO_COLOR, lw=1.5,
                    label=f"YOLOv8n  AP={results['YOLOv8n']['per_class_ap'].get(cls,0)*100:.1f}%")

        ax.set_title(cls.capitalize())
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "05_pr_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def draw_detections(img_bgr: np.ndarray, detections: list[dict],
                    color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on an image copy."""
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.get('class_name', det.get('cls',''))}: {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fig_detection_examples(sample_image_paths: list[str],
                           hog_classifiers_path: str,
                           yolo_weights_path: str):
    """Render example images with HOG+SVM (left) and YOLOv8n (right) detections."""
    import pickle
    from classical.hog_svm_detector import sliding_window_detect, nms
    from ultralytics import YOLO

    classifiers = pickle.load(open(hog_classifiers_path, "rb"))
    yolo_model  = YOLO(yolo_weights_path)

    n = min(3, len(sample_image_paths))
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Detection Examples: HOG+SVM (left) vs. YOLOv8n (right)",
                 fontsize=12, fontweight="bold")

    for row, img_path in enumerate(sample_image_paths[:n]):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hog_dets = []
        for cls, pipe in classifiers.items():
            d = sliding_window_detect(img, pipe, threshold=0.4)
            d = nms(d)
            for det in d:
                det["class_name"] = cls
            hog_dets.extend(d)
        hog_vis = draw_detections(img_rgb, hog_dets, color=(70, 130, 180))

        res = yolo_model(img_path, verbose=False, device="cpu")[0]
        yolo_dets = []
        for box in res.boxes:
            yolo_dets.append({
                "bbox":       [int(v) for v in box.xyxy[0].tolist()],
                "score":      float(box.conf.item()),
                "class_name": CLASSES[int(box.cls.item())],
            })
        yolo_vis = draw_detections(img_rgb, yolo_dets, color=(221, 132, 82))

        axes[row][0].imshow(hog_vis)
        axes[row][0].set_title(f"HOG + SVM ({Path(img_path).stem})")
        axes[row][0].axis("off")

        axes[row][1].imshow(yolo_vis)
        axes[row][1].set_title(f"YOLOv8n ({Path(img_path).stem})")
        axes[row][1].axis("off")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_detection_examples.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_timing_distribution(results: dict):
    hog_times  = results["HOG_SVM"].get("times_ms", [])
    yolo_times = results["YOLOv8n"].get("times_ms", [])

    if not hog_times and not yolo_times:
        print("  No timing arrays available, skipping timing figure")
        return

    fig, ax = plt.subplots(figsize=(5, 3.8))
    data   = [t for t in [hog_times, yolo_times] if t]
    labels = [l for l, t in [("HOG + SVM", hog_times), ("YOLOv8n", yolo_times)] if t]
    colors = [HOG_COLOR if l == "HOG + SVM" else YOLO_COLOR for l in labels]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Inference time (ms / image)")
    ax.set_title("Inference Time Distribution (CPU, n=100)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    out = os.path.join(FIGURES_DIR, "07_timing_distribution.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=[])
    args = parser.parse_args()

    print("\nGenerating thesis figures ...\n")
    results = load_results()

    fig_map_comparison(results)
    fig_per_class_ap(results)
    fig_speed_accuracy(results)
    fig_comparison_table(results)
    fig_pr_curves(results)
    fig_timing_distribution(results)

    if args.samples:
        hog_path  = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "hog_svm_classifiers.pkl"
        )
        yolo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "yolov8n_voc2007", "weights", "best.pt"
        )
        fig_detection_examples(args.samples, hog_path, yolo_path)
    else:
        print("  (Pass --samples img1.jpg img2.jpg for detection example figure)")

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

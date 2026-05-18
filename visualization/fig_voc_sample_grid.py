"""
Figure: 4x5 grid of representative VOC 2007 test images with GT bounding boxes.
Output: figures/ch5_fig02_voc_sample_grid.{pdf,png}
"""
import sys
import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VOC_ROOT_TEST, CLASSES, FIGURES_DIR
from data.voc_loader import load_all_annotations

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
})

CLASS_COLORS = {
    "aeroplane":   (0.20, 0.51, 0.69),
    "bicycle":     (0.87, 0.52, 0.32),
    "bird":        (0.20, 0.63, 0.17),
    "boat":        (0.84, 0.15, 0.16),
    "bottle":      (0.58, 0.40, 0.74),
    "bus":         (0.55, 0.34, 0.29),
    "car":         (0.89, 0.47, 0.76),
    "cat":         (0.50, 0.50, 0.50),
    "chair":       (0.74, 0.74, 0.13),
    "cow":         (0.09, 0.75, 0.81),
    "diningtable": (0.20, 0.51, 0.69),
    "dog":         (0.87, 0.52, 0.32),
    "horse":       (0.20, 0.63, 0.17),
    "motorbike":   (0.84, 0.15, 0.16),
    "person":      (0.58, 0.40, 0.74),
    "pottedplant": (0.55, 0.34, 0.29),
    "sheep":       (0.89, 0.47, 0.76),
    "sofa":        (0.50, 0.50, 0.50),
    "train":       (0.74, 0.74, 0.13),
    "tvmonitor":   (0.09, 0.75, 0.81),
}

SEED = 49
random.seed(SEED)

# Pick one representative image per class
print("Loading VOC 2007 test annotations ...")
annotations = load_all_annotations(VOC_ROOT_TEST, "test", CLASSES)

class_to_anns = {c: [] for c in CLASSES}
for ann in annotations:
    for cls in CLASSES:
        objs = [o for o in ann["objects"] if o["name"] == cls and not o["difficult"]]
        if objs:
            class_to_anns[cls].append((ann, objs))

chosen = {}
for cls in CLASSES:
    candidates = class_to_anns[cls]
    if not candidates:
        print(f"  WARNING: no non-difficult instances for {cls}")
        continue
    clean = [(a, o) for a, o in candidates if len(a["objects"]) <= 3]
    pool  = clean if clean else candidates
    random.shuffle(pool)
    chosen[cls] = pool[0]

print(f"  Selected images for {len(chosen)}/20 classes")

# Draw grid
NROWS, NCOLS = 4, 5
THUMB = 512

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 2.8, NROWS * 2.8))
fig.suptitle(
    "Pascal VOC 2007 Test Set: One Representative Image per Class\n"
    "(ground-truth bounding boxes shown)",
    fontsize=12, fontweight="bold", y=1.01
)

for idx, cls in enumerate(CLASSES):
    ax = axes[idx // NCOLS][idx % NCOLS]

    if cls not in chosen:
        ax.axis("off")
        ax.set_title(cls, fontsize=9)
        continue

    ann, target_objs = chosen[cls]
    img_bgr = cv2.imread(ann["image_path"])
    if img_bgr is None:
        ax.axis("off")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Centre crop on target objects
    xs = [o["bbox"][0] for o in target_objs] + [o["bbox"][2] for o in target_objs]
    ys = [o["bbox"][1] for o in target_objs] + [o["bbox"][3] for o in target_objs]
    cx = int((min(xs) + max(xs)) / 2)
    cy = int((min(ys) + max(ys)) / 2)
    H, W = img_rgb.shape[:2]
    half = THUMB // 2
    x1   = max(0, min(cx - half, W - THUMB))
    y1   = max(0, min(cy - half, H - THUMB))
    x2   = x1 + THUMB
    y2   = y1 + THUMB
    crop = img_rgb[y1:y2, x1:x2]

    ax.imshow(crop)
    ax.axis("off")

    color = CLASS_COLORS[cls]
    for obj in ann["objects"]:
        bx1, by1, bx2, by2 = obj["bbox"]
        bx1c, by1c = bx1 - x1, by1 - y1
        bx2c, by2c = bx2 - x1, by2 - y1
        if bx2c <= 0 or by2c <= 0 or bx1c >= THUMB or by1c >= THUMB:
            continue
        rect = mpatches.FancyBboxPatch(
            (bx1c, by1c), bx2c - bx1c, by2c - by1c,
            boxstyle="square,pad=0",
            linewidth=1.6, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(bx1c + 2, by1c - 3, obj["name"],
                fontsize=6, color="white", fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.15", alpha=0.85))

    ax.set_title(cls, fontsize=9, pad=3)

plt.tight_layout()

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
pdf_path = os.path.join(FIGURES_DIR, "ch5_fig02_voc_sample_grid.pdf")
png_path = os.path.join(FIGURES_DIR, "ch5_fig02_voc_sample_grid.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")

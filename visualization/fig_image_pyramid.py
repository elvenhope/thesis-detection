"""
Figure: Image pyramid illustration showing 3 scales with sliding window overlay.
Output: figures/ch5_fig10_image_pyramid.{pdf,png}
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VOC_ROOT_TEST, FIGURES_DIR, SW_SCALES, HOG_WIN_SIZE, SW_STEP_SIZE
from data.voc_loader import get_image_ids

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.2,
})

SCALE_COLORS = {0.75: "#E74C3C", 1.0: "#2980B9", 1.25: "#27AE60"}
WIN_W, WIN_H = HOG_WIN_SIZE

# Pick a representative image
random.seed(42)
ids = get_image_ids(VOC_ROOT_TEST, "test")
random.shuffle(ids)

chosen_img = None
for img_id in ids:
    img_path = os.path.join(VOC_ROOT_TEST, "JPEGImages", f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    if 300 < w < 600 and h < w:
        chosen_img = (img_id, img_path, img)
        break

if chosen_img is None:
    for img_id in ids[:50]:
        img_path = os.path.join(VOC_ROOT_TEST, "JPEGImages", f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is not None:
            chosen_img = (img_id, img_path, img)
            break

img_id, img_path, orig_bgr = chosen_img
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = orig_rgb.shape[:2]
print(f"Using image: {img_id}  ({orig_w}x{orig_h})")

# Build scaled images
scaled = {}
for s in SW_SCALES:
    nw, nh = int(orig_w * s), int(orig_h * s)
    scaled[s] = cv2.resize(orig_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def sample_windows(img_h, img_w, win_w, win_h, step, n=3):
    """Return n evenly-spaced window positions."""
    xs = list(range(0, img_w - win_w, step))
    ys = list(range(0, img_h - win_h, step))
    if not xs or not ys:
        return []
    idxs = np.linspace(0, len(xs)-1, n, dtype=int)
    return [(xs[i], ys[min(i, len(ys)-1)]) for i in idxs]


# Layout
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, height_ratios=[2.2, 0.9], hspace=0.45, wspace=0.10)

SCALE_ORDER = [0.75, 1.0, 1.25]
LABELS      = ["Scale 0.75x\n(reduced)", "Scale 1.0x\n(original)", "Scale 1.25x\n(enlarged)"]

for col, (s, lbl) in enumerate(zip(SCALE_ORDER, LABELS)):
    ax  = fig.add_subplot(gs[0, col])
    img = scaled[s]
    h, w = img.shape[:2]
    color = SCALE_COLORS[s]

    ax.imshow(img)
    ax.set_title(lbl, fontsize=11, fontweight="bold", color=color, pad=6)

    wins = sample_windows(h, w, WIN_W, WIN_H, SW_STEP_SIZE, n=3)
    for i, (wx, wy) in enumerate(wins):
        alpha = 0.85 if i == 1 else 0.45
        lw    = 2.0  if i == 1 else 1.0
        rect = mpatches.Rectangle(
            (wx, wy), WIN_W, WIN_H,
            linewidth=lw, edgecolor=color,
            facecolor=color, alpha=0.15 if i == 1 else 0.05, zorder=3
        )
        ax.add_patch(rect)
        rect2 = mpatches.Rectangle(
            (wx, wy), WIN_W, WIN_H,
            linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha, zorder=4
        )
        ax.add_patch(rect2)

    ax.set_xlabel(f"{w} x {h} px", fontsize=9.5, color="#555555")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.2)

# Bottom panel: pyramid diagram
ax_pyr = fig.add_subplot(gs[1, :])
ax_pyr.set_xlim(0, 10); ax_pyr.set_ylim(0, 3.2)
ax_pyr.axis("off")

levels = [
    (1.25, 1.0, 2.2, 1.8, "#27AE60"),
    (1.0,  2.0, 1.4, 6.0, "#2980B9"),
    (0.75, 3.0, 0.6, 4.0, "#E74C3C"),
]
rect_heights = [0.7, 0.7, 0.7]

for i, (scale, x, y, w, color) in enumerate(levels):
    rh = rect_heights[i]
    rect = mpatches.FancyBboxPatch(
        (x, y), w, rh, boxstyle="square,pad=0",
        facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.8
    )
    ax_pyr.add_patch(rect)
    ax_pyr.text(x - 0.1, y + rh/2, f"{scale}x",
                ha="right", va="center", fontsize=10, fontweight="bold", color=color)
    s_h, s_w = int(orig_h * scale), int(orig_w * scale)
    ax_pyr.text(x + w/2, y + rh/2, f"{s_w} x {s_h} px",
                ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    steps_x = max(0, (s_w - WIN_W) // SW_STEP_SIZE + 1)
    steps_y = max(0, (s_h - WIN_H) // SW_STEP_SIZE + 1)
    ax_pyr.text(x + w + 0.1, y + rh/2, f"~{steps_x * steps_y:,} windows",
                ha="left", va="center", fontsize=8.5, color="#666666")

ax_pyr.annotate("", xy=(5, 2.2 + 0.7 + 0.15), xytext=(5, 0.6 - 0.15),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.4))
ax_pyr.text(5.2, 1.65, "Scale\nrange",
            ha="left", va="center", fontsize=8.5, color="#888888", style="italic")

ax_pyr.set_title(
    "Image pyramid: same image rescaled so the sliding window\n"
    "detects objects of different sizes",
    fontsize=10, color="#333333", pad=4
)

legend_patches = [
    mpatches.Patch(facecolor=SCALE_COLORS[s], alpha=0.75,
                   label=f"Scale {s}x  ({int(orig_w*s)}x{int(orig_h*s)} px)")
    for s in SCALE_ORDER
]
legend_patches.append(
    mpatches.Patch(facecolor="#555555", alpha=0.5,
                   label=f"Sliding window: {WIN_W}x{WIN_H} px, step={SW_STEP_SIZE} px")
)
fig.legend(handles=legend_patches,
           loc="lower center", ncol=4, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
pdf_path = os.path.join(FIGURES_DIR, "ch5_fig10_image_pyramid.pdf")
png_path = os.path.join(FIGURES_DIR, "ch5_fig10_image_pyramid.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")

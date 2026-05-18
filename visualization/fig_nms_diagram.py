"""
Figure: NMS diagram, before (4 overlapping boxes) vs after (1 surviving box).
Uses a real VOC 2007 image as background.
Output: figures/ch5_fig11_nms_diagram.{pdf,png}
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
from config import VOC_ROOT_TEST, FIGURES_DIR, NMS_THRESHOLD
from data.voc_loader import get_image_ids, parse_annotation

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})

# Pick a clean image with one prominent object
random.seed(7)
ids = get_image_ids(VOC_ROOT_TEST, "test")
random.shuffle(ids)

chosen = None
for img_id in ids:
    ann  = parse_annotation(VOC_ROOT_TEST, img_id)
    objs = [o for o in ann["objects"] if not o["difficult"]]
    if len(objs) == 1:
        img = cv2.imread(ann["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        obj = objs[0]
        bx1, by1, bx2, by2 = obj["bbox"]
        bw, bh = bx2 - bx1, by2 - by1
        if bw > 60 and bh > 60 and w > h:
            chosen = (ann, obj, img)
            break

if chosen is None:
    for img_id in ids[:30]:
        ann = parse_annotation(VOC_ROOT_TEST, img_id)
        img = cv2.imread(ann["image_path"])
        if img is not None:
            chosen = (ann, ann["objects"][0], img)
            break

ann, main_obj, bgr = chosen
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
bx1, by1, bx2, by2 = main_obj["bbox"]
cls_name = main_obj["name"]
print(f"Using image: {ann['image_id']}  class: {cls_name}")

# Generate overlapping candidate boxes around the GT box
bw = bx2 - bx1
bh = by2 - by1
cx = (bx1 + bx2) / 2
cy = (by1 + by2) / 2


def jitter_box(cx, cy, bw, bh, dx_frac, dy_frac, dw_frac, dh_frac):
    ncx = cx + dx_frac * bw
    ncy = cy + dy_frac * bh
    nw  = bw * (1 + dw_frac)
    nh  = bh * (1 + dh_frac)
    return [int(ncx - nw/2), int(ncy - nh/2),
            int(ncx + nw/2), int(ncy + nh/2)]


candidates = [
    (jitter_box(cx, cy, bw, bh,  0.00,  0.00,  0.00,  0.00), 0.91, "#E74C3C"),
    (jitter_box(cx, cy, bw, bh,  0.12, -0.08,  0.10, -0.05), 0.78, "#E67E22"),
    (jitter_box(cx, cy, bw, bh, -0.10,  0.10, -0.08,  0.12), 0.65, "#F39C12"),
    (jitter_box(cx, cy, bw, bh,  0.18,  0.12, -0.12,  0.08), 0.52, "#9B59B6"),
]


def iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    union = aa + ab - inter
    return inter/union if union > 0 else 0.0


def draw_box(ax, box, score, color, linewidth, alpha_fill=0.08):
    x1, y1, x2, y2 = box
    rect_fill = mpatches.FancyBboxPatch(
        (x1, y1), x2-x1, y2-y1, boxstyle="square,pad=0",
        facecolor=color, alpha=alpha_fill, edgecolor="none"
    )
    ax.add_patch(rect_fill)
    rect_edge = mpatches.FancyBboxPatch(
        (x1, y1), x2-x1, y2-y1, boxstyle="square,pad=0",
        facecolor="none", edgecolor=color, linewidth=linewidth
    )
    ax.add_patch(rect_edge)
    ax.text(x1 + 4, y1 - 6, f"{score:.2f}",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(facecolor=color, edgecolor="none",
                      boxstyle="round,pad=0.25", alpha=0.92),
            zorder=5)


# Draw
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(13, 5.5))

# Before NMS
ax_before.imshow(rgb)
ax_before.set_title("Before NMS", fontsize=13, fontweight="bold", color="#C0392B", pad=10)
ax_before.set_xticks([]); ax_before.set_yticks([])

for box, score, color in candidates:
    draw_box(ax_before, box, score, color, linewidth=2.0)

b0, b1 = candidates[0][0], candidates[1][0]
iou_val = iou(b0, b1)
mid_x = (b0[2] + b1[0]) / 2
mid_y = (b0[1] + b1[3]) / 2
ax_before.annotate(
    f"IoU = {iou_val:.2f}",
    xy=(mid_x, mid_y),
    xytext=(mid_x + bw*0.6, mid_y - bh*0.3),
    fontsize=9, color="#7D3C98", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=1.2),
    bbox=dict(facecolor="white", edgecolor="#7D3C98",
              boxstyle="round,pad=0.3", alpha=0.9),
    zorder=6
)

ax_before.text(0.02, 0.04,
               f"NMS threshold: IoU >= {NMS_THRESHOLD}  ->  suppress",
               transform=ax_before.transAxes, fontsize=8.5, color="#555555", style="italic",
               bbox=dict(facecolor="white", alpha=0.8,
                         edgecolor="#cccccc", boxstyle="round,pad=0.3"))

# After NMS
ax_after.imshow(rgb)
ax_after.set_title("After NMS", fontsize=13, fontweight="bold", color="#27AE60", pad=10)
ax_after.set_xticks([]); ax_after.set_yticks([])

winner = candidates[0]
draw_box(ax_after, winner[0], winner[1], winner[2], linewidth=3.0, alpha_fill=0.15)

wx1, wy1, wx2, wy2 = winner[0]
ax_after.text(wx1 + 4, wy2 + 16,
              f"Kept: {cls_name}  (score {winner[1]:.2f})",
              fontsize=10, fontweight="bold", color="white",
              bbox=dict(facecolor=winner[2], edgecolor="none",
                        boxstyle="round,pad=0.35", alpha=0.92),
              zorder=5)

for box, score, color in candidates[1:]:
    x1, y1, x2, y2 = box
    rect_fade = mpatches.FancyBboxPatch(
        (x1, y1), x2-x1, y2-y1, boxstyle="square,pad=0",
        facecolor="none", edgecolor=color,
        linewidth=1.2, alpha=0.25, linestyle="--"
    )
    ax_after.add_patch(rect_fade)
    ax_after.plot([x1, x2], [y1, y2], color=color, lw=1.0, alpha=0.22)
    ax_after.plot([x1, x2], [y2, y1], color=color, lw=1.0, alpha=0.22)

ax_after.text(0.02, 0.04,
              f"{len(candidates)-1} boxes suppressed  (IoU >= {NMS_THRESHOLD})",
              transform=ax_after.transAxes, fontsize=8.5, color="#555555", style="italic",
              bbox=dict(facecolor="white", alpha=0.8,
                        edgecolor="#cccccc", boxstyle="round,pad=0.3"))

legend_items = [
    mpatches.Patch(facecolor=c[2], alpha=0.8,
                   label=f"Candidate box  score={c[1]:.2f}"
                         + ("  [kept]" if i == 0 else "  [suppressed]"))
    for i, c in enumerate(candidates)
]
fig.legend(handles=legend_items, loc="lower center", ncol=2,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
pdf_path = os.path.join(FIGURES_DIR, "ch5_fig11_nms_diagram.pdf")
png_path = os.path.join(FIGURES_DIR, "ch5_fig11_nms_diagram.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")

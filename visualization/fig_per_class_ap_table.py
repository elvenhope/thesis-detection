"""
Figure: Per-class AP heatmap table, all 5 models x 20 VOC classes + mAP.
Output: figures/ch5_fig06_per_class_ap_table.{pdf,png}
"""
import sys
import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, OUTPUT_DIR, CLASSES

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

# Load data
with open(os.path.join(OUTPUT_DIR, "combined_results.json")) as f:
    results = json.load(f)

MODELS = [
    ("Faster_RCNN", "Faster R-CNN"),
    ("YOLOv8n",     "YOLOv8n"),
    ("SSD300",      "SSD300"),
    ("YOLOv5n",     "YOLOv5n"),
    ("HOG_SVM",     "HOG + SVM"),
]

col_labels = [label for _, label in MODELS]

row_labels = [c.replace("diningtable", "din.table")
               .replace("pottedplant", "pot.plant")
              for c in CLASSES] + ["mAP"]

data = []
for c in CLASSES:
    row = [results[key]["per_class_ap"].get(c, 0.0) * 100 for key, _ in MODELS]
    data.append(row)
mAP_row = [results[key]["mAP"] * 100 for key, _ in MODELS]
data.append(mAP_row)
data = np.array(data)

# Draw table
n_rows, n_cols = data.shape

fig, ax = plt.subplots(figsize=(8, 16))
ax.axis("off")

cmap_cells = plt.cm.RdYlGn

def ap_color(val, vmin=0, vmax=100):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap_cells(norm(val))

HEADER_BG    = "#2C3E50"
MAP_ROW_BG   = "#1A252F"
COL_LABEL_BG = "#ECF0F1"
COL_W        = 1.0 / (n_cols + 1.5)

table_width = 0.08 + n_cols * COL_W
TABLE_LEFT  = (1.0 - table_width) / 2

ax.text(0.5, 0.96,
        "Per-Class AP@0.5 (%), All Models, Pascal VOC 2007 Test Set",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=12, fontweight="bold", color="#1A2A3A")

def draw_cell(ax, x, y, w, h, text, bg, fg="black",
              fontsize=8, bold=False, border_color="#CCCCCC", lw=0.5):
    rect = plt.Rectangle((x, y), w, h,
                          facecolor=bg, edgecolor=border_color,
                          linewidth=lw, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=fontsize, color=fg,
            fontweight="bold" if bold else "normal", zorder=3)

TABLE_TOP = 0.88
LABEL_H   = 0.06
DATA_H    = (TABLE_TOP - LABEL_H) / n_rows

# Header: "Class" cell
draw_cell(ax, TABLE_LEFT, TABLE_TOP, 0.08, LABEL_H,
          "Class", HEADER_BG, "white", fontsize=9, bold=True,
          border_color="#ffffff", lw=0.8)

# Header: model name cells
for j, col in enumerate(col_labels):
    x  = TABLE_LEFT + 0.08 + j * COL_W
    bg = MAP_ROW_BG if j == n_cols - 1 else HEADER_BG
    rect = plt.Rectangle((x, TABLE_TOP), COL_W, LABEL_H,
                          facecolor=bg, edgecolor="white",
                          linewidth=0.8, transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x + COL_W/2, TABLE_TOP + LABEL_H/2, col,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color="white", fontweight="bold", zorder=3)

# Data rows
for i, (row_label, row_data) in enumerate(zip(row_labels, data)):
    y = TABLE_TOP - (i+1) * DATA_H

    draw_cell(ax, TABLE_LEFT, y, 0.08, DATA_H,
              row_label, COL_LABEL_BG, "#1A2A3A",
              fontsize=7.5, bold=False, border_color="#BBBBBB", lw=0.6)

    for j, val in enumerate(row_data):
        x  = TABLE_LEFT + 0.08 + j * COL_W
        bg = ap_color(val)
        luminance = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
        fg = "white" if luminance < 0.45 else "#1A1A1A"

        is_map_row = (i == n_rows - 1)
        fs     = 7.5 if not is_map_row else 8.5
        bw     = 1.4 if is_map_row else 0.5
        border = "#E67E22" if is_map_row else "#CCCCCC"

        rect = plt.Rectangle((x, y), COL_W, DATA_H,
                              facecolor=bg, edgecolor=border,
                              linewidth=bw, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + COL_W/2, y + DATA_H/2, f"{val:.1f}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=fs, color=fg,
                fontweight="bold" if is_map_row else "normal", zorder=3)

# Colour scale legend
cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
sm = plt.cm.ScalarMappable(cmap=cmap_cells, norm=mcolors.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("AP@0.5 (%)", fontsize=8)
cbar.ax.tick_params(labelsize=7.5)

ax.text(0.5, 0.005,
        "Source: own experiment. Evaluation protocol: Everingham et al. (2010).",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=7.5, color="#888888", style="italic")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
pdf_path = os.path.join(FIGURES_DIR, "ch5_fig06_per_class_ap_table.pdf")
png_path = os.path.join(FIGURES_DIR, "ch5_fig06_per_class_ap_table.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")

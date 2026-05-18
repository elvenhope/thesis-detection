"""
Figure: VOC 2007 test set, annotated instance count per class.
Output: figures/ch5_fig01_voc_class_distribution.{pdf,png}
"""
import sys
import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VOC_ROOT_TEST, CLASSES, FIGURES_DIR
from data.voc_loader import load_all_annotations

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

BAR_COLOR       = "#4C72B0"
HIGHLIGHT_COLOR = "#C0392B"
HARD_CLASSES    = {"bottle", "chair", "pottedplant"}

# Count instances
print("Loading VOC 2007 test annotations ...")
annotations = load_all_annotations(VOC_ROOT_TEST, "test", CLASSES)

counter = Counter()
for ann in annotations:
    for obj in ann["objects"]:
        if not obj["difficult"]:
            counter[obj["name"]] += 1

counts_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
class_names   = [item[0] for item in counts_sorted]
counts        = [item[1] for item in counts_sorted]
colors        = [HIGHLIGHT_COLOR if c in HARD_CLASSES else BAR_COLOR for c in class_names]

total_instances = sum(counts)
print(f"  Total non-difficult instances: {total_instances}")
for name, cnt in counts_sorted:
    print(f"    {name:<15s} {cnt:5d}")

# Draw
fig, ax = plt.subplots(figsize=(9, 6))

y_pos = np.arange(len(class_names))
bars  = ax.barh(y_pos, counts, color=colors,
                edgecolor="white", linewidth=0.6,
                height=0.7, zorder=3)

for bar, val in zip(bars, counts):
    ax.text(bar.get_width() + 18,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left",
            fontsize=8, color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(class_names)
ax.invert_yaxis()

ax.set_xlabel("Number of annotated instances")
ax.set_title(
    f"Pascal VOC 2007 Test Set: Annotated Object Instances per Class\n"
    f"(20 classes, {len(annotations):,} images, {total_instances:,} non-difficult instances)",
    pad=10
)

ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)

legend_elements = [
    Patch(facecolor=BAR_COLOR,       label="Standard class"),
    Patch(facecolor=HIGHLIGHT_COLOR, label="Low-AP class (bottle, chair, pottedplant)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.8)

plt.tight_layout()

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
pdf_path = os.path.join(FIGURES_DIR, "ch5_fig01_voc_class_distribution.pdf")
png_path = os.path.join(FIGURES_DIR, "ch5_fig01_voc_class_distribution.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
plt.close(fig)
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")

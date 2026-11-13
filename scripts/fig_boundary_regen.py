import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data: (R, raw/DFT, C/DFT, color, label) -- all from Table 6 local anchors
points = [
    (0.88, 0.04, 0.92, "#1f77b4", "Al"),
    (1.28, 0.17, 1.45, "#2ca02c", "Cu"),
    (3.36, 0.04, 3.40, "#d62728", "Ni"),
    (0.00, 0.33, 0.33, "#ff7f0e", "Ag"),
    (1.23, 0.67, 1.90, "#9467bd", "Au"),
]

fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

# Safe zone: R in [0.8, 1.2], raw/DFT < 0.1
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((0.8, 0), 0.4, 0.1, alpha=0.15, color="#2ca02c", zorder=0))
ax.axhline(0.1, color="gray", linestyle=":", linewidth=1, zorder=1)
ax.axvline(0.8, color="gray", linestyle=":", linewidth=1, zorder=1)
ax.axvline(1.2, color="gray", linestyle=":", linewidth=1, zorder=1)

for r, raw, cdf, color, name in points:
    ax.scatter(r, raw, s=200, color=color, edgecolor="black", linewidth=0.8, zorder=5)
    if name == "Ni":
        ax.annotate(f"{name}\n(C/DFT={cdf:.2f})", xy=(r, raw), xytext=(r-0.15, raw),
                    fontsize=9, va="center", ha="right")
    elif name == "Au":
        ax.annotate(f"{name}\n(C/DFT={cdf:.2f})", xy=(r, raw), xytext=(r-0.13, raw),
                    fontsize=9, va="center", ha="right")
    elif name == "Al":
        ax.annotate(f"{name}\n(C/DFT={cdf:.2f})", xy=(r, raw), xytext=(r, raw+0.075),
                    fontsize=9, va="bottom", ha="center")
    elif name == "Ag":
        ax.annotate(f"{name}\n(C/DFT={cdf:.2f})", xy=(r, raw), xytext=(r+0.15, raw+0.02),
                    fontsize=9, va="bottom", ha="left")
    else:
        ax.annotate(f"{name}\n(C/DFT={cdf:.2f})", xy=(r, raw), xytext=(r+0.15, raw+0.03),
                    fontsize=9, va="bottom")

ax.set_xlim(-0.2, 4.5)
ax.set_ylim(0, 0.75)
ax.set_xlabel(r"Geometric ratio: $R = 2\varepsilon_{\mathrm{DFT}}/(\mathrm{ISF}_{\mathrm{DFT}}\cdot A)$", fontsize=12)
ax.set_ylabel(r"Raw MACE ISF / DFT ISF", fontsize=12)
ax.set_title("Two-dimensional transferability boundary", fontsize=12)

ax.text(4.3, 0.73, "safe zone:\nR∈[0.8,1.2]\nraw/DFT<0.1", fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.9))

plt.tight_layout()
plt.savefig(r"D:\20261029铝合金论文\figures\fig8_boundary_map.png", dpi=200, bbox_inches="tight")
plt.savefig(r"D:\20261029铝合金论文\figures\fig8_boundary_map.pdf", dpi=300, bbox_inches="tight")
print("Saved boundary map")

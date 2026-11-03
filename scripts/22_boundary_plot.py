# -*- coding: utf-8 -*-
"""2D boundary criterion: R vs raw/DFT"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (metal, R=penalty/ISF_DFT, raw/DFT, corrected/DFT ratio)
data = [
    ("Al", 143.5/163.5, 6.6/163.5, 0.90, "#2166ac"),
    ("Cu", 41.9/39.5,   5.6/39.5,  1.23, "#4dac26"),
    ("Ni", 560.8/154.0, 6.0/154.0, 3.82, "#b2182b"),
    ("Ag", 0.0,          5.7/20.0,  0.29, "#d08c2b"),
    ("Au", 40.1/39.0,   20.8/39.0, 1.60, "#762a83"),
]

fig, ax = plt.subplots(figsize=(6, 4.8))

# Safe zone: R in [0.8, 1.2], raw/DFT < 0.1
# Derived from Al C/DFT <= 10% and raw/DFT <= 0.1
ax.add_patch(plt.Rectangle((0.8, 0), 0.4, 0.1, alpha=0.15, color='green', label="Safe zone"))

for name, x, y, corr_ratio, c in data:
    ax.scatter(x, y, s=150, c=c, zorder=5, edgecolors='black', linewidth=0.8)
    ax.annotate(f"{name}\n(C/DFT={corr_ratio:.2f})", (x, y),
                textcoords="offset points", xytext=(12, 5), fontsize=9)

ax.axvline(0.8, ls=':', color='gray', alpha=0.5)
ax.axvline(1.2, ls=':', color='gray', alpha=0.5)
ax.axhline(0.1, ls=':', color='gray', alpha=0.5)

ax.set_xlabel(r"Geometric ratio: $R = 2\epsilon_{\mathrm{DFT}}/(\mathrm{ISF}_{\mathrm{DFT}}\cdot A)$", fontsize=11)
ax.set_ylabel(r"Raw MACE ISF / DFT ISF", fontsize=11)
ax.set_xlim(-0.1, 4.5)
ax.set_ylim(-0.02, 0.6)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_title("Two-dimensional boundary: $R$ near 1 AND raw/DFT near 0", fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(r"D:\20261029铝合金论文\figures\fig8_boundary_map.pdf", dpi=300)
plt.savefig(r"D:\20261029铝合金论文\figures\fig8_boundary_map.png", dpi=200)
print("Saved 2D boundary plot")

# -*- coding: utf-8 -*-
"""fig6：Phase 2A 缺陷能量学三势对比 vs 文献锚点。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/defect_audit.json").read_text(encoding="utf-8"))

models = [r["model"] for r in d["rows"] if r["type"] == "isf_usf"]
isf = [r["gamma_isf_mJ_m2"] for r in d["rows"] if r["type"] == "isf_usf"]
usf = [r["gamma_usf_mJ_m2"] for r in d["rows"] if r["type"] == "isf_usf"]
gint = [r["gamma_mJ_m2"] for r in d["rows"] if r["type"] == "interface_100"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
colors = ["#4C72B0", "#DD8452", "#55A868"]
labels = ["MACE-MP-0 small", "MACE-MP-0 medium", "MACE-MP-0 large"]

# ISF
ax = axes[0]
ax.bar(labels, isf, color=colors, alpha=0.85)
ax.axhline(164, color="k", ls="--", lw=1.2)
ax.fill_between([-0.5, 2.5], 161, 166, color="k", alpha=0.08)
ax.text(2.45, 168, "DFT 161-166", ha="right", fontsize=9)
ax.set_title("Al (111) γ_ISF（mJ/m²）", fontsize=11)
ax.set_ylim(-40, 200)

# USF
ax = axes[1]
ax.bar(labels, usf, color=colors, alpha=0.85)
ax.axhline(224, color="k", ls="--", lw=1.2)
ax.text(2.45, 228, "DFT 224", ha="right", fontsize=9)
ax.set_title("Al (111) γ_USF（mJ/m²）", fontsize=11)
ax.set_ylim(0, 260)

# Interface
ax = axes[2]
ax.bar(labels, gint, color=colors, alpha=0.85)
ax.axhline(192, color="k", ls="--", lw=1.2)
ax.axhline(32.5, color="gray", ls=":", lw=1.2)
ax.text(2.45, 198, "DFT 192 (Asta)", ha="right", fontsize=9)
ax.text(2.45, 40, "EAM 32.5", ha="right", fontsize=9, color="gray")
ax.set_title("Al3Sc/Al (100) γ_int（mJ/m²）", fontsize=11)
ax.set_ylim(-60, 240)

for ax in axes:
    for lab in ax.get_xticklabels():
        lab.set_fontsize(8)
    ax.tick_params(axis="y", labelsize=9)
fig.suptitle("Phase 2A：缺陷能量学三势审计 vs 文献锚点（MACE-MP-0）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "figures/fig6_defect_energies.png", dpi=150)
print("saved figures/fig6_defect_energies.png")

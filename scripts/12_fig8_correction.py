# -*- coding: utf-8 -*-
"""fig8：Phase 2C 修正路径——(a) ISF 堆垛序罚项修正 (b) USF 弛豫坍缩 (c) 界面常数移位。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/phase2c_correction.json").read_text(encoding="utf-8"))

models = [r["model"] for r in d["rows"]]
labels = ["small", "medium", "large"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) ISF 修正
ax = axes[0]
raw = [r["isf_layer_removal_MACE"] for r in d["rows"]]
corr = [r["isf_layer_removal_corrected"] for r in d["rows"]]
x = range(3)
ax.bar([i - 0.19 for i in x], raw, width=0.38, color=colors, alpha=0.55, label="MACE (层移除)")
ax.bar([i + 0.19 for i in x], corr, width=0.38, color=colors, alpha=0.95, label="+堆垛序罚项")
ax.fill_between([-0.5, 2.5], 161, 166, color="k", alpha=0.10)
ax.text(2.45, 168, "DFT 161-166", ha="right", fontsize=9)
for i, v in enumerate(corr):
    ax.text(i + 0.19, v + 4, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("γ_ISF：堆垛序罚项修正", fontsize=11)
ax.set_ylim(-40, 200); ax.legend(fontsize=8)

# (b) USF 坍缩
ax = axes[1]
sp = [r["usf_single_point"] for r in d["rows"]]
rx = [r["usf_relaxed"] for r in d["rows"]]
ax.bar([i - 0.19 for i in x], sp, width=0.38, color=colors, alpha=0.55, label="单点（刚性）")
ax.bar([i + 0.19 for i in x], rx, width=0.38, color=colors, alpha=0.95, label="位置弛豫后")
ax.axhline(224, color="k", ls="--", lw=1.2)
ax.text(2.45, 228, "DFT 224", ha="right", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("γ_USF：位置弛豫坍缩", fontsize=11)
ax.set_ylim(0, 280); ax.legend(fontsize=8)

# (c) 界面常数移位
ax = axes[2]
gm = [r["interface_MACE"] for r in d["rows"]]
ax.bar([i - 0.19 for i in x], gm, width=0.38, color=colors, alpha=0.55, label="MACE")
ax.bar([i + 0.19 for i in x], [192] * 3, width=0.38, color=colors, alpha=0.95, label="+常数移位 δ")
ax.axhline(192, color="k", ls="--", lw=1.2)
ax.text(2.45, 196, "DFT 192", ha="right", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Al3Sc/Al(100) 界面：常数移位", fontsize=11)
ax.set_ylim(-40, 240); ax.legend(fontsize=8)

fig.suptitle("Phase 2C：域外区修正路径（缺陷能量学 δ 延伸，mJ/m²）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "figures/fig8_correction_path.png", dpi=150)
print("saved figures/fig8_correction_path.png")

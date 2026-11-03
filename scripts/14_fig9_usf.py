# -*- coding: utf-8 -*-
"""fig9：Phase 3 USF 弛豫协议——(a) 三协议 γ 对比 (b) 上半块横向质心位移。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/usf_protocol.json").read_text(encoding="utf-8"))

labels = ["small", "medium", "large"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# (a) 三协议 γ
ax = axes[0]
none = [r["gamma_none_mJ_m2"] for r in d["rows"]]
zonly = [r["gamma_zonly_mJ_m2"] for r in d["rows"]]
full = [r["gamma_full_mJ_m2"] for r in d["rows"]]
x = range(3)
w = 0.27
ax.bar([i - w for i in x], none, width=w, color=colors, alpha=0.55, label="不弛豫（单点）")
ax.bar([i for i in x], zonly, width=w, color=colors, alpha=0.8, label="仅垂直弛豫")
ax.bar([i + w for i in x], full, width=w, color=colors, alpha=1.0, label="全位置弛豫")
ax.axhline(224, color="k", ls="--", lw=1.2)
ax.text(2.42, 228, "DFT USF 224", ha="right", fontsize=9)
for i in x:
    ax.text(i - w, none[i] + 4, f"{none[i]:.0f}", ha="center", fontsize=8)
    ax.text(i, zonly[i] + 4, f"{zonly[i]:.0f}", ha="center", fontsize=8)
    ax.text(i + w, full[i] + 4, f"{full[i]:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("γ_USF：三协议对照（mJ/m²）", fontsize=11)
ax.set_ylim(0, 280); ax.legend(fontsize=8)

# (b) 横向质心位移
ax = axes[1]
sz = [r["slide_xy_zonly_A"] for r in d["rows"]]
sf = [r["slide_xy_full_A"] for r in d["rows"]]
ax.bar([i - 0.19 for i in x], sz, width=0.38, color=colors, alpha=0.55, label="z-only 弛豫")
ax.bar([i + 0.19 for i in x], sf, width=0.38, color=colors, alpha=0.95, label="全弛豫")
ax.axhline(0.827, color="k", ls="--", lw=1.2)
ax.text(2.42, 0.86, "T_STEP/2 = 0.827（滑回完美堆垛应有位移）", ha="right", fontsize=8)
for i in x:
    ax.text(i + 0.19, sf[i] + 0.005, f"{sf[i]:.3f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("上半块横向质心位移（Å）", fontsize=11)
ax.set_ylim(0, 1.0); ax.legend(fontsize=8)

fig.suptitle("Phase 3：USF 弛豫协议验证——坍缩机制修正（MACE-MP-0, CPU/float64）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "figures/fig9_usf_protocol.png", dpi=150)
print("saved figures/fig9_usf_protocol.png")

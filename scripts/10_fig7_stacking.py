# -*- coding: utf-8 -*-
"""fig7：Phase 2B 堆垛盲视机制诊断——(a) hcp/fcc 多形体判别 (b) γ 曲面粗扫。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/phase2b_stacking.json").read_text(encoding="utf-8"))

models = [r["model"] for r in d["polytype_discrimination"]]
dE = [r["dE_hcp_minus_fcc_eV_atom"] * 1000 for r in d["polytype_discrimination"]]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# (a) 多形体判别（meV/atom）
ax = axes[0]
bars = ax.bar(["small", "medium", "large"], dE, color=colors, alpha=0.85, width=0.55)
ax.axhline(31.5, color="k", ls="--", lw=1.2)
ax.text(2.42, 34, "DFT 31.5 meV/atom\n(2H-3C)", ha="right", va="bottom", fontsize=9)
for b, v in zip(bars, dE):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:+.1f}", ha="center", fontsize=9)
ax.set_title("ΔE(hcp−fcc) 多形体判别（meV/atom）", fontsize=11)
ax.set_ylim(-6, 44)
ax.set_ylabel("meV / atom")

# (b) γ 曲面粗扫
ax = axes[1]
x = [k / 6 for k in range(7)]
for row, c in zip(d["gamma_surface"], colors):
    ax.plot(x, row["gamma_mJ_m2"], "o-", color=c, label=row["model"], lw=1.8, ms=5)
ax.plot([0.5], [224], "k*", ms=12, label="DFT USF 224")
ax.plot([1.0], [164], "kD", ms=7, label="DFT ISF 161-166")
ax.axvspan(0.92, 1.08, color="k", alpha=0.06)
ax.set_xticks(x)
ax.set_xticklabels(["0", "T/6", "T/3", "T/2", "2T/3", "5T/6", "T"])
ax.set_title("Al (111) γ 曲面粗扫（单点，mJ/m²）", fontsize=11)
ax.set_ylabel("γ (mJ/m²)")
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Phase 2B：层错失效机制诊断——MACE 堆垛序盲视（MACE-MP-0, 单点）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "figures/fig7_stacking_blindness.png", dpi=150)
print("saved figures/fig7_stacking_blindness.png")

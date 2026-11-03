# -*- coding: utf-8 -*-
"""
AM1 · 锚点常数校准（anchor-based per-phase δ calibration）
==========================================================
前置条件：results/audit_relaxed.json（或回退 audit_raw.json）+ results/dft_anchors.json。

概念（诚实命名）：
- 本脚本做的是「每相常数偏移 δ_phase」校准：对已有 DFT/实验锚点的相，
  δ = E_anchor − E_f(MACE)，修正后 E_f^corr = E_f + δ。
- 这是训练式 δ-learning（用 MLIP−DFT 的力/能量差训练修正模型，需 DFT 训练集）
  的**单点锚定版本**，作为 Phase 3 训练式 δ-learning 的基线；未锚定相不做常数
  修正，仅标注「未校准」。

输出：
  - results/delta_corrections.json   逐相×逐势 δ 与修正后形成能
  - results/校准表.md                独立交付表
  - figures/fig5_calibrated.png      修正前后 vs 锚点
  - 追加一节到 results/审计报告.md
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# 形成能按 COMPOUNDS 的原子配比与元素参考相计算，与 03 一致
COMPOUNDS = {
    "Al3Sc_L12": (4, {"Al": 3, "Sc": 1}),
    "Al3Zr_L12": (4, {"Al": 3, "Zr": 1}),
    "Al3ScZr_L12": (8, {"Al": 6, "Sc": 1, "Zr": 1}),
    "Al2Cu_theta": (12, {"Al": 4, "Cu": 8}),
    "Al2Cu_theta_prime": (6, {"Al": 4, "Cu": 2}),
    "Mg2Si": (12, {"Mg": 8, "Si": 4}),
}
ELEM_REFS = {"Al": "Al_fcc", "Cu": "Cu_fcc", "Si": "Si_dia",
             "Mg": "Mg_hcp", "Sc": "Sc_hcp", "Zr": "Zr_hcp"}


def load_E(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    E, E_tot = {}, {}
    for r in data["rows"]:
        per = r.get("E_per_atom_relaxed_eV", r.get("E_per_atom_eV"))
        tot = r.get("E_tot_relaxed_eV", r.get("E_tot_eV"))
        E.setdefault(r["structure"], {})[r["model"]] = per
        E_tot.setdefault(r["structure"], {})[r["model"]] = tot
    return E, E_tot, data["models"]


def main():
    rel_path = RESULT_DIR / "audit_relaxed.json"
    if rel_path.exists():
        E, E_tot, models = load_E(rel_path)
    else:
        E, E_tot, models = load_E(RESULT_DIR / "audit_raw.json")

    anchors = json.loads((RESULT_DIR / "dft_anchors.json").read_text(encoding="utf-8"))["anchors"]

    # 计算各相 MACE 形成能（弛豫优先）
    form = {}
    for phase, (n, comp) in COMPOUNDS.items():
        form[phase] = {}
        for m in models:
            tot = E_tot[phase][m]
            ref = sum(cnt * E[ELEM_REFS[el]][m] for el, cnt in comp.items())
            form[phase][m] = (tot - ref) / n

    # δ 校准
    delta, corrected = {}, {}
    anchored, unanchored = [], []
    for phase in COMPOUNDS:
        a = anchors.get(phase, {})
        if "formation_eV_per_atom" in a:
            anchored.append(phase)
            anchor_val = a["formation_eV_per_atom"]
            delta[phase] = {}
            corrected[phase] = {}
            for m in models:
                d = anchor_val - form[phase][m]
                delta[phase][m] = d
                corrected[phase][m] = form[phase][m] + d
        else:
            unanchored.append(phase)

    mean_delta = {p: float(np.mean(list(delta[p].values()))) for p in anchored}

    # ---------- 图：修正前后 vs 锚点 ----------
    x = np.arange(len(anchored))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, m in enumerate(models):
        vals_before = [form[p][m] for p in anchored]
        vals_after = [corrected[p][m] for p in anchored]
        ax.bar(x + (i - 1) * w * 1.2, vals_before, w, label=m + " (raw)", alpha=0.55)
        ax.bar(x + (i - 1) * w * 1.2 + w, vals_after, w, label=m + " (calibrated)", alpha=1.0)
    ax.scatter([v + 1.1 for v in x], [anchors[p]["formation_eV_per_atom"] for p in anchored],
               marker="D", s=70, color="black", label="DFT anchor", zorder=5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x + 0.5 * w)
    ax.set_xticklabels(anchored, rotation=15, fontsize=9)
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.set_title("锚点常数校准：raw vs calibrated vs DFT anchor")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_calibrated.png", dpi=150)
    plt.close(fig)

    # ---------- 输出 JSON ----------
    out = {
        "method": "anchor-based per-phase constant delta (单点锚定版，非训练式 δ-learning)",
        "models": models,
        "delta_eV_per_atom": {p: {m: delta[p][m] for m in models} for p in anchored},
        "mean_delta_eV_per_atom": mean_delta,
        "corrected_formation_eV_per_atom": {p: {m: corrected[p][m] for m in models}
                                            for p in anchored},
        "unanchored_phases": unanchored,
        "notes": "未锚定相（Al3ScZr_L12、Al2Cu_theta_prime）不做常数修正，需 DFT 锚点后校准。",
    }
    (RESULT_DIR / "delta_corrections.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 校准表.md ----------
    lines = ["# AM1 · 锚点常数校准表（δ, eV/atom）\n",
             "> 方法：对已有 DFT/实验锚点的相取 δ = E_anchor − E_f(MACE)；修正后 E_f^corr = E_f + δ。\n",
             "> 说明：单点锚定版本，是训练式 δ-learning（需 DFT 力/能量训练集）的 Phase 3 基线。\n",
             "| 相 | " + " | ".join(models) + " | 均值 δ | 锚点值 | 来源 |",
             "|---|---" + "|---|" * (len(models)) + "|---|---|"]
    for p in anchored:
        vals = " | ".join(f"{delta[p][m]:+.4f}" for m in models)
        a = anchors[p]
        lines.append(f"| {p} | {vals} | {mean_delta[p]:+.4f} | {a['formation_eV_per_atom']:+.3f} | "
                     f"{a['kind']} |")
    lines.append("")
    lines.append("**修正后形成能（eV/atom）**：")
    lines.append("| 相 | " + " | ".join(models) + " |")
    lines.append("|---|---" + "|---|" * len(models))
    for p in anchored:
        vals = " | ".join(f"{corrected[p][m]:+.4f}" for m in models)
        lines.append(f"| {p} | {vals} |")
    lines.append("")
    lines.append(f"**未锚定相（不校准）**：{', '.join(unanchored)} —— 需 DFT 单点后补 δ。")
    (RESULT_DIR / "校准表.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- 追加到审计报告 ----------
    report_path = RESULT_DIR / "审计报告.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        sec = ["\n## 10. 锚点常数校准（Phase 1b，δ-learning 前置基线）\n",
               "| 相 | " + " | ".join(models) + " | 均值 δ | 锚点值 | 来源 |",
               "|---|---" + "|---|" * (len(models)) + "|---|---|"]
        for p in anchored:
            vals = " | ".join(f"{delta[p][m]:+.4f}" for m in models)
            a = anchors[p]
            sec.append(f"| {p} | {vals} | {mean_delta[p]:+.4f} | "
                       f"{a['formation_eV_per_atom']:+.3f} | {a['kind']} |")
        sec.append("")
        sec.append(f"未锚定相（{', '.join(unanchored)}）不做常数修正；"
                   "校准为单点锚定版本，训练式 δ-learning 需 DFT 力/能量训练集（Phase 3）。\n")
        report += "\n".join(sec)
        report_path.write_text(report, encoding="utf-8")

    print("[校准] 已写入 results/delta_corrections.json、校准表.md、审计报告.md §10、fig5")
    for p in anchored:
        print(f"  {p:16s} δ_mean={mean_delta[p]:+.4f} eV/atom"
              f"  (raw→corr: {form[p][models[0]]:+.4f}→{corrected[p][models[0]]:+.4f})")


if __name__ == "__main__":
    main()

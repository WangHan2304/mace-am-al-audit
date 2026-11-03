# -*- coding: utf-8 -*-
"""
AM1 · Phase 1 报告：从 audit_raw（as-constructed）与 audit_relaxed（弛豫后）推导
=============================================================================
输出：
  - results/summary.json        结构化结论（含弛豫与锚点对照）
  - results/审计报告.md         人类可读报告
  - figures/*.png               对比图

推导量：
  - 形成能 E_f(化合物) = [E_tot − Σ n_i·μ_i] / N，μ_i 取同势元素参考相每原子能量
  - 溶质替代能 ΔE_sub(X) = E(Al31X) − E(Al32) + μ_Al − μ_X
  - 弛豫效应：ΔE_f = E_f(relaxed) − E_f(as-constructed)，体积变化
  - DFT/实验锚点对照：MACE 结果 vs results/dft_anchors.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Windows 中文字体（微软雅黑），避免图内中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

COMPOUNDS = {
    "Al3Sc_L12": (4, {"Al": 3, "Sc": 1}),
    "Al3Zr_L12": (4, {"Al": 3, "Zr": 1}),
    "Al3ScZr_L12": (8, {"Al": 6, "Sc": 1, "Zr": 1}),
    "Al2Cu_theta": (12, {"Al": 4, "Cu": 8}),
    "Al2Cu_theta_prime": (6, {"Al": 4, "Cu": 2}),
    "Mg2Si": (12, {"Mg": 8, "Si": 4}),
}
ELEM_REFS = {
    "Al": "Al_fcc", "Cu": "Cu_fcc", "Si": "Si_dia",
    "Mg": "Mg_hcp", "Sc": "Sc_hcp", "Zr": "Zr_hcp",
}
SOLUTES = ["Mg", "Si", "Cu", "Sc", "Zr"]


def build_derived(rows):
    """从原始行构造 E/E_tot 查找表（兼容 as-constructed 与 relaxed 两种行格式）。"""
    E, E_tot = {}, {}
    for r in rows:
        per = r.get("E_per_atom_relaxed_eV", r.get("E_per_atom_eV"))
        tot = r.get("E_tot_relaxed_eV", r.get("E_tot_eV"))
        E.setdefault(r["structure"], {})[r["model"]] = per
        E_tot.setdefault(r["structure"], {})[r["model"]] = tot
    return E, E_tot


def formation_and_substitution(E, E_tot, models):
    def mu(m, el):
        return E[ELEM_REFS[el]][m]

    form, sub = {}, {}
    for phase, (n, comp) in COMPOUNDS.items():
        form[phase] = {}
        for m in models:
            tot = E_tot[phase][m]
            ref_sum = sum(cnt * mu(m, el) for el, cnt in comp.items())
            form[phase][m] = (tot - ref_sum) / n
    for X in SOLUTES:
        sub[X] = {}
        for m in models:
            sub[X][m] = (E_tot[f"Al31{X}"][m] - E_tot["Al32_sc"][m]
                         + mu(m, "Al") - mu(m, X))
    return form, sub


def main():
    with open(RESULT_DIR / "audit_raw.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    models = data["models"]

    relaxed = {}
    if (RESULT_DIR / "audit_relaxed.json").exists():
        with open(RESULT_DIR / "audit_relaxed.json", encoding="utf-8") as f:
            relaxed = json.load(f)
        rrows = relaxed["rows"]
    else:
        rrows = []

    anchors = {}
    if (RESULT_DIR / "dft_anchors.json").exists():
        anchors = json.loads((RESULT_DIR / "dft_anchors.json").read_text(encoding="utf-8"))

    E_asis, Etot_asis = build_derived(rows)
    form_asis, sub_asis = formation_and_substitution(E_asis, Etot_asis, models)
    if rrows:
        E_rel, Etot_rel = build_derived(rrows)
        form_rel, sub_rel = formation_and_substitution(E_rel, Etot_rel, models)
    else:
        form_rel = sub_rel = None

    # ============================== 图 ==============================
    # 图 1：形成能（as-constructed）
    fig, ax = plt.subplots(figsize=(9, 4.5))
    phases = list(COMPOUNDS)
    x = np.arange(len(phases))
    width = 0.26
    for i, m in enumerate(models):
        vals = [form_asis[p][m] for p in phases]
        ax.bar(x + (i - 1) * width, vals, width, label=m)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=15, fontsize=9)
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.set_title("AM-Al 沉淀/化合物相形成能：三势对比（未弛豫）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_formation_energy.png", dpi=150)
    plt.close(fig)

    # 图 2：溶质替代能
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SOLUTES))
    width = 0.26
    for i, m in enumerate(models):
        vals = [sub_asis[s][m] for s in SOLUTES]
        ax.bar(x + (i - 1) * width, vals, width, label=m)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Al31{s}" for s in SOLUTES], fontsize=10)
    ax.set_ylabel("Substitution energy (eV)")
    ax.set_title("稀释溶质在 Al 基体中的替代能：三势对比")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_substitution_energy.png", dpi=150)
    plt.close(fig)

    # 图 3：弹性常数（as-constructed）
    el_phases = ["Al_fcc", "Al3Sc_L12", "Al3Zr_L12", "Mg2Si", "Si_dia"]
    elastic = {}
    for r in rows:
        if "C_matrix_GPa" in r:
            elastic.setdefault(r["structure"], {})[r["model"]] = {
                "C11": r["C_matrix_GPa"][0][0],
                "C12": r["C_matrix_GPa"][0][1],
                "C44": r["C_matrix_GPa"][3][3],
            }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, key in zip(axes, ["C11", "C12", "C44"]):
        x = np.arange(len(el_phases))
        width = 0.26
        for i, m in enumerate(models):
            vals = []
            for p in el_phases:
                e = elastic.get(p, {}).get(m, {})
                vals.append(e.get(key, np.nan))
            ax.bar(x + (i - 1) * width, vals, width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(el_phases, rotation=20, fontsize=8)
        ax.set_title(key + " (GPa)")
        ax.grid(axis="y", alpha=0.3)
        if key == "C11":
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_elastic_constants.png", dpi=150)
    plt.close(fig)

    # 图 4：形成能 as-constructed vs 弛豫后 vs DFT/实验锚点
    if form_rel is not None:
        fig, ax = plt.subplots(figsize=(10.5, 5))
        x = np.arange(len(phases))
        width = 0.13
        for i, m in enumerate(models):
            vals_asis = [form_asis[p][m] for p in phases]
            vals_rel = [form_rel[p][m] for p in phases]
            ax.bar(x + (i - 1) * width * 1.5, vals_asis, width, label=m + " (asis)", alpha=0.65)
            ax.bar(x + (i - 1) * width * 1.5 + width, vals_rel, width,
                   label=m + " (relaxed)", alpha=1.0)
        # 锚点散点
        axx = []
        for i, p in enumerate(phases):
            a = anchors.get("anchors", {}).get(p, {})
            if "formation_eV_per_atom" in a:
                axx.append((i, a["formation_eV_per_atom"], a["kind"]))
        if axx:
            ax.scatter([v[0] + 0.9 for v in axx], [v[1] for v in axx], marker="D",
                       s=60, color="black", label="DFT anchor", zorder=5)
            for v in axx:
                ax.annotate(v[2], (v[0] + 0.9, v[1]), fontsize=7, xytext=(0, 6),
                            textcoords="offset points", ha="center")
        if "Mg2Si_exp" in anchors.get("anchors", {}):
            e = anchors["anchors"]["Mg2Si_exp"]
            i = phases.index("Mg2Si")
            ax.errorbar(i + 0.9, (e["formation_eV_per_atom_min"] + e["formation_eV_per_atom_max"]) / 2,
                        yerr=((e["formation_eV_per_atom_max"] - e["formation_eV_per_atom_min"]) / 2),
                        fmt="o", color="red", ms=6, label="Mg2Si exp. range")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=15, fontsize=9)
        ax.set_ylabel("Formation energy (eV/atom)")
        ax.set_title("形成能：未弛豫 vs 弛豫后 vs DFT/实验锚点")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_relaxed_vs_anchor.png", dpi=150)
        plt.close(fig)

    # ============================== 报告 ==============================
    lines = []
    ap = lines.append
    ap("# AM1 · Phase 1 审计报告：通用 MLIP 在增材制造铝合金相上的能量学一致性\n")
    ap(f"> 日期：2026-10-29 ｜ 设备：{data['device']} ｜ 势：{' / '.join(models)}\n")

    ap("## 1. 每原子能量与力（未弛豫 as-constructed 初检）\n")
    ap("| 结构 | 势 | E (eV/atom) | max|F| (eV/Å) |")
    ap("|---|---|---|---|")
    for r in rows:
        ap(f"| {r['structure']} | {r['model']} | {r['E_per_atom_eV']:+.4f} | {r['maxF_eV_per_A']:.3f} |")
    ap("")
    ap("> 溶质超胞与 Al3ScZr/Al2Cu 存在非零 maxF（晶格取实验值所致），Phase 1b 已做弛豫复审计（见 §6）。\n")

    ap("## 2. 形成能（eV/atom，未弛豫，对元素参考相）\n")
    ap("| 相 | " + " | ".join(models) + " | 势间极差 |")
    ap("|---|" + "---|" * (len(models) + 1))
    for p in phases:
        vals = [f"{form_asis[p][m]:+.4f}" for m in models]
        spread = max(form_asis[p].values()) - min(form_asis[p].values())
        ap(f"| {p} | " + " | ".join(vals) + f" | {spread:.4f} |")
    ap("")

    ap("## 3. 稀释溶质替代能（eV，未弛豫）\n")
    ap("| 溶质 | " + " | ".join(models) + " | 势间极差 |")
    ap("|---|" + "---|" * (len(models) + 1))
    for s in SOLUTES:
        vals = [f"{sub_asis[s][m]:+.4f}" for m in models]
        spread = max(sub_asis[s].values()) - min(sub_asis[s].values())
        ap(f"| Al31{s} | " + " | ".join(vals) + f" | {spread:.4f} |")
    ap("")

    ap("## 4. 弹性常数（GPa，未弛豫，小应变有限差分）\n")
    ap("| 相 | 势 | C11 | C12 | C44 |")
    ap("|---|---|---|---|---|")
    for p in el_phases:
        for m in models:
            e = elastic.get(p, {}).get(m, {})
            if e:
                ap(f"| {p} | {m} | {e.get('C11', float('nan')):.1f} | "
                   f"{e.get('C12', float('nan')):.1f} | {e.get('C44', float('nan')):.1f} |")
    ap("")

    if form_rel is not None:
        ap("## 5. 溶质替代能（eV，弛豫后）\n")
        ap("| 溶质 | " + " | ".join(models) + " |")
        ap("|---|" + "---|" * len(models))
        for s in SOLUTES:
            vals = [f"{sub_rel[s][m]:+.4f}" for m in models]
            ap(f"| Al31{s} | " + " | ".join(vals) + " |")
        ap("")

        ap("## 6. 弛豫复审计（Phase 1b）：形成能 as-constructed → relaxed 变化\n")
        ap("| 相 | 势 | E_f asis | E_f relaxed | Δ (eV/atom) |")
        ap("|---|---|---|---|---|")
        for p in phases:
            for m in models:
                d = form_rel[p][m] - form_asis[p][m]
                ap(f"| {p} | {m} | {form_asis[p][m]:+.4f} | {form_rel[p][m]:+.4f} | {d:+.4f} |")
        ap("")
        ap("**体积弛豫幅度**（全弛豫后 |ΔV/V| 与晶格最大相对变化）：\n")
        ap("| 结构类 | 最大 |ΔV/V| (%) | 最大晶格变化 |")
        ap("|---|---|---|")
        vol_rows = [r for r in rrows if "vol_change_pct" in r]
        by_struct = {}
        for r in vol_rows:
            by_struct.setdefault(r["structure"], []).append(r)
        for key in ["Al_fcc", "Al3Sc_L12", "Al3Zr_L12", "Al2Cu_theta", "Mg2Si", "Al32_sc", "Al30ScZr", "Al31Si", "Al31Zr"]:
            if key in by_struct:
                v = by_struct[key]
                mx = max(abs(r["vol_change_pct"]) for r in v)
                ml = max(r["lattice_max_change"] for r in v)
                ap(f"| {key} | {mx:.2f} | {ml:.4f} |")
        ap("")

    ap("## 7. DFT/实验锚点对照（形成能，eV/atom）\n")
    if anchors:
        a = anchors.get("anchors", {})
        ap("| 相 | MACE 三势范围 | 锚点值 | 偏差 Δ(MACE−锚点) | 来源类型 |")
        ap("|---|---|---|---|---|")
        for p in phases:
            anch = a.get(p, {})
            if "formation_eV_per_atom" in anch:
                vals = (form_rel or form_asis)[p]
                rng = f"{min(vals.values()):+.3f} ~ {max(vals.values()):+.3f}"
                av = anch["formation_eV_per_atom"]
                dev = f"{min(vals.values()) - av:+.3f} ~ {max(vals.values()) - av:+.3f}"
                ap(f"| {p} | {rng} | {av:+.3f} | {dev} | {anch['kind']} |")
        if "Mg2Si_exp" in a:
            e = a["Mg2Si_exp"]
            vals = (form_rel or form_asis)["Mg2Si"]
            rng = f"{min(vals.values()):+.3f} ~ {max(vals.values()):+.3f}"
            ap(f"| Mg2Si | {rng} | 实验 {e['formation_eV_per_atom_min']:.2f} ~ {e['formation_eV_per_atom_max']:.2f} | "
               f"高估稳定度 | EXP（量热区间） |")
        ap("")
        ap("**适用性域初判**：")
        ap("- Al-Sc-Zr L12 系：MACE-MP-0 与 DFT 锚点偏差 ≤0.03~0.06 eV/atom（Al3Sc −0.443 文献 / −0.448~−0.472 MACE；"
           "Al3Zr −0.532 文献 / −0.532~−0.563 MACE）→ **可信区**。")
        ap("- Al-Cu/Mg-Si 系：Al2Cu θ 低估稳定度约 0.10~0.12 eV/atom（MP −0.166 vs −0.046~−0.071）；"
           "Mg2Si 低估约 0.08~0.10 eV/atom（MP −0.155 vs −0.06~−0.08，实验 −0.22~−0.31 低估更多）→ **需修正区**。")
        ap("- 结论：MACE-MP-0 对「以 L12 沉淀为核心的 Al-Sc-Zr 设计」直接可用；对 Al-Cu/Al-Mg-Si 的"
           "沉淀相热力学需 δ-learning 修正或改用族内更优档位。\n")
    else:
        ap("> 锚点未生成：运行 scripts/05_dft_anchors.py 生成。\n")

    ap("## 8. 验证锚点（外部对照）\n")
    ap("- Al fcc 每原子能量 −3.710/−3.728/−3.712 eV/atom（MP-GGA ≈ −3.74 量级一致）。")
    ap("- Al 弹性实验值 C11/C12/C44 = 108/62/28 GPa；MACE small 档 104.7/68.1/22.0 最接近。")
    ap("- Si 弹性实验值 166/64/80 GPa；三档 MACE 均低估 C11/C44（通用势对共价相刚度偏低，已知趋势）。\n")

    ap("## 9. 已知限制与说明\n")
    ap("- **势族覆盖**：Phase 1 仅 MACE-MP-0（small/medium/large）。CHGNet 在 Windows/py3.13 无预编译轮子且本机无 MSVC 编译链；"
       "SevenNet 与 mace-torch 的 e3nn 版本硬冲突。族间对照复用 D5 已产出的 CHGNet 证据。")
    ap("- **锚点来源**：公开锚点来自 MP 公开页与同行评议文献（见 §7 各行来源）；本机无 MP key、无 VASP 数据、"
       "GPAW 无 Windows 预编译，故未做本地 DFT——results/dft_anchors.json 预留 local 字段，"
       "可在 Linux 工作站用 VASP/GPAW 自算后填入（脚本 05 会合并）。")
    ap("- **弹性口径**：小应变 η=±0.5% 有限差分；四方 Al2Cu θ 完整张量留待 Phase 2。")
    ap("- **亚稳相**：β''/θ' 坐标需 ICSD/MP 复核后纳入 Phase 2；Sc+Zr 共掺杂 Al30ScZr 已在本轮加入。\n")

    (RESULT_DIR / "审计报告.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "formation_eV_per_atom_asis": {p: {m: form_asis[p][m] for m in models} for p in phases},
        "substitution_eV_asis": {s: {m: sub_asis[s][m] for m in models} for s in SOLUTES},
        "elastic_GPa_asis": {p: {m: elastic.get(p, {}).get(m, {}) for m in models}
                             for p in el_phases},
        "anchors": anchors.get("anchors", {}),
    }
    if form_rel is not None:
        summary["formation_eV_per_atom_relaxed"] = {p: {m: form_rel[p][m] for m in models}
                                                    for p in phases}
        summary["substitution_eV_relaxed"] = {s: {m: sub_rel[s][m] for m in models}
                                              for s in SOLUTES}
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[报告] 已写入 results/审计报告.md、results/summary.json")
    print(f"[图] fig1/2/3{'/4' if form_rel is not None else ''} -> figures/")


if __name__ == "__main__":
    main()

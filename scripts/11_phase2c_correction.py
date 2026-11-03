# -*- coding: utf-8 -*-
"""Phase 2C：域外区修正路径（缺陷能量学 δ 延伸，纯分析，无新 MACE 计算）

修正模型：
  1) ISF：堆垛序罚项 —— 层错局部含 1 个 hcp 型近邻层对（2 个原子处于 hcp 型环境），
     罚项 = 2·ε，ε = ΔE(hcp−fcc) 取 DFT 锚点 31.5 meV/atom（由体相多形体测试推导，
     不拟合 ISF 本身）→ γ_ISF_corr = γ_MACE + 2ε/A。
  2) USF：单点值已≈DFT（峰对），病理在位置弛豫坍缩 → 量化 Δγ_relax = γ_sp − γ_relaxed。
  3) 界面：非堆垛机制（共格界面无 hcp 层对）→ 单锚点常数移位 δ_int = γ_DFT − γ_MACE
     （与 Phase 1b 的 δ 同型，单点锚定版，诚实标注不可迁移）。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

EV2J = 1.602176634e-19
AREA = 7.10                          # (111) 1x1 原胞面积, A^2
EPS_DFT_meV = 31.5                   # ΔE(hcp-fcc) DFT 锚点 (2H-3C, Denteneer & van Haeringen 1991)
GAMMA_ISF_DFT = [161.0, 166.0]
GAMMA_USF_DFT = 224.0
GAMMA_INT_DFT = 192.0

penalty_mJ = 2 * EPS_DFT_meV * 1e-3 * EV2J / (AREA * 1e-20) * 1e3   # 2ε/A -> mJ/m2

d_2a = json.loads((RESULT_DIR / "defect_audit.json").read_text(encoding="utf-8"))
d_2b = json.loads((RESULT_DIR / "phase2b_stacking.json").read_text(encoding="utf-8"))

isf_rows = {r["model"]: r for r in d_2a["rows"] if r["type"] == "isf_usf"}
gsf_rows = {r["model"]: r for r in d_2b["gamma_surface"]}
poly = {r["model"]: r for r in d_2b["polytype_discrimination"]}
int_rows = {r["model"]: r for r in d_2a["rows"] if r["type"] == "interface_100"}

models = list(isf_rows)
rows = []
for m in models:
    g_isf_rm = isf_rows[m]["gamma_isf_mJ_m2"]          # Phase 2A 层移除法（弛豫）
    g_isf_rigid = gsf_rows[m]["gamma_mJ_m2"][6]        # Phase 2B 刚性平移 s=T（单点）
    g_usf_sp = gsf_rows[m]["gamma_mJ_m2"][3]           # Phase 2B 单点 USF
    g_usf_rx = isf_rows[m]["gamma_usf_mJ_m2"]          # Phase 2A 弛豫 USF
    g_int = int_rows[m]["gamma_mJ_m2"]
    dE_hcpfcc_mev = poly[m]["dE_hcp_minus_fcc_eV_atom"] * 1e3

    isf_rm_corr = g_isf_rm + penalty_mJ
    isf_rigid_corr = g_isf_rigid + penalty_mJ
    rows.append({
        "model": m,
        "dE_hcp_minus_fcc_meV_atom_MACE": dE_hcpfcc_mev,
        "epsilon_DFT_meV_atom": EPS_DFT_meV,
        "stacking_penalty_mJ_m2": round(penalty_mJ, 1),
        "isf_layer_removal_MACE": round(g_isf_rm, 1),
        "isf_layer_removal_corrected": round(isf_rm_corr, 1),
        "isf_layer_removal_err_vs_anchor": round(isf_rm_corr - 163.5, 1),
        "isf_rigid_shift_MACE": round(g_isf_rigid, 1),
        "isf_rigid_shift_corrected": round(isf_rigid_corr, 1),
        "usf_single_point": round(g_usf_sp, 1),
        "usf_relaxed": round(g_usf_rx, 1),
        "usf_relaxation_collapse_mJ_m2": round(g_usf_sp - g_usf_rx, 1),
        "interface_MACE": round(g_int, 1),
        "interface_constant_shift_delta_mJ_m2": round(GAMMA_INT_DFT - g_int, 1),
        "interface_corrected": round(GAMMA_INT_DFT, 1),
    })

out = {
    "phase": "2C 域外区修正路径（缺陷能量学 δ 延伸）",
    "method": [
        "ISF: 堆垛序罚项 2ε/A, ε=31.5 meV/atom (DFT 体相多形体锚点, 未拟合 ISF); 层错局部 1 个 hcp 型近邻层对 → 2 原子受罚",
        "USF: 不施加能量修正——单点值已≈DFT, 病理在位置弛豫坍缩; 量化 Δγ = γ(单点)-γ(弛豫)",
        "界面: 共格界面无 hcp 层对, 堆垛罚项不适用; 单锚点常数移位 (与 Phase 1b δ 同型, 诚实标注不可迁移, 需 DFT 训练集)",
    ],
    "anchors": {
        "ISF_DFT_mJ_m2": GAMMA_ISF_DFT,
        "USF_DFT_mJ_m2": GAMMA_USF_DFT,
        "interface_DFT_mJ_m2": GAMMA_INT_DFT,
        "epsilon_hcp_minus_fcc_DFT_meV_atom": EPS_DFT_meV,
    },
    "area_A2": AREA,
    "rows": rows,
}
(RESULT_DIR / "phase2c_correction.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

for r in rows:
    print(f"  {r['model']:20s} ISF修正 {r['isf_layer_removal_corrected']:5.1f} "
          f"(err {r['isf_layer_removal_err_vs_anchor']:+5.1f})  "
          f"USF坍缩 {r['usf_relaxation_collapse_mJ_m2']:5.1f}  "
          f"界面δ {r['interface_constant_shift_delta_mJ_m2']:+6.1f}")
print("saved results/phase2c_correction.json")

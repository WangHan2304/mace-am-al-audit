# -*- coding: utf-8 -*-
"""Phase 2B：层错失效机制诊断（MACE 密排堆垛序盲视假说）
1) hcp-Al vs fcc-Al 多形体能量判别（单点，理想几何）
2) γ 曲面粗扫：12 层(111) 1x1 胞，位移 s = k*T_STEP/6 (k=0..6)，上半平移、双等效层错 /2A、单点能量
3) 输出 results/phase2b_stacking.json
"""
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

A_AL = 4.05                       # fcc 实验晶格常数
D111 = A_AL / np.sqrt(3)
P1 = np.array([A_AL / 2, -A_AL / 2, 0.0])
P2 = np.array([A_AL / 2, 0.0, -A_AL / 2])
N_HAT = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
T_STEP = (P1 + P2) / 3
AREA = float(np.linalg.norm(np.cross(P1, P2)))   # 1x1 (111) 面内原胞面积 A^2
EV2J = 1.602176634e-19

MODELS = [
    ("MACE-MP-0 small", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    ("MACE-MP-0 medium", "2023-12-03-mace-128-L1_epoch-199.model"),
    ("MACE-MP-0 large", "MACE_MPtrj_2022.9.model"),
]


def build_fcc():
    a = A_AL
    pos = np.array([[0, 0, 0], [a / 2, a / 2, 0], [a / 2, 0, a / 2], [0, a / 2, a / 2]])
    return Atoms("Al4", positions=pos, cell=[a, a, a], pbc=True)


def build_hcp():
    from ase.build import bulk
    a = A_AL / np.sqrt(2)
    c = a * np.sqrt(8 / 3)
    at = bulk("Al", crystalstructure="hcp", a=a, c=c, cubic=False)
    return at


def build_111_stack(nlayers):
    pos = np.array([((k % 3) * T_STEP) + k * D111 * N_HAT for k in range(nlayers)])
    return Atoms("Al" * nlayers, positions=pos, cell=[P1, P2, nlayers * D111 * N_HAT], pbc=True)


def shift_upper(cell, s):
    pos = cell.get_positions()
    proj = pos @ N_HAT
    mid = (proj.min() + proj.max()) / 2
    pos[proj > mid] += s
    cell.set_positions(pos)
    cell.wrap()
    return cell


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    # 本阶段负载极小（21 次 12 原子单点）；CPU 规避本机 CUDA 间歇性崩溃(-1073741819)，
    # float64 下能量与设备无关。
    device = "cpu"
    print(f"[设备] {device}")
    calcs = [(name, MACECalculator(model_paths=str(MODEL_DIR / f), device=device,
                                   default_dtype="float64")) for name, f in MODELS]

    fcc = build_fcc()
    hcp = build_hcp()
    for a in (fcc, hcp):
        d = neighbor_list("d", a, cutoff=3.8)
        d = np.sort(d[d > 0.01])
        print(f"  {a.get_chemical_formula()}: 最近邻 {d[:2].round(3)}")
        assert np.allclose(d[:2], 2.864, atol=0.02), f"{a.get_chemical_formula()} 最近邻异常 {d[:2]}"

    # ---------- 1) 多形体判别 ----------
    poly_rows = []
    for mname, calc in calcs:
        e_fcc = fcc.copy(); e_fcc.calc = calc
        e_hcp = hcp.copy(); e_hcp.calc = calc
        Ef, Eh = e_fcc.get_potential_energy(), e_hcp.get_potential_energy()
        dE = (Eh / len(hcp)) - (Ef / len(fcc))
        poly_rows.append({"model": mname, "E_fcc_eV_atom": Ef / 4,
                          "E_hcp_eV_atom": Eh / 2, "dE_hcp_minus_fcc_eV_atom": dE})
        print(f"  {mname:20s} E_fcc={Ef/4:+.4f}  E_hcp={Eh/2:+.4f}  dE(hcp-fcc)={dE:+.4f} eV/atom")

    # ---------- 2) γ 曲面粗扫 ----------
    perfect = build_111_stack(12)
    gsf_rows = []
    shifts = [k * T_STEP / 6 for k in range(7)]   # 0, T/6, T/3, T/2, 2T/3, 5T/6, T
    for mname, calc in calcs:
        E0 = None
        gs = []
        for k, s in enumerate(shifts):
            a = shift_upper(perfect.copy(), s)
            a.calc = calc
            E = a.get_potential_energy()
            if k == 0:
                E0 = E
            gam = (E - E0) * EV2J / (AREA * 1e-20) * 1e3 / 2   # 双等效层错 /2A
            gs.append(float(gam))
            print(f"  {mname:20s} k={k} s={np.linalg.norm(s):.3f} A  gamma={gam:7.1f} mJ/m2")
        gsf_rows.append({"model": mname, "shifts_frac": list(np.arange(7) / 6),
                         "gamma_mJ_m2": gs})

    out = {
        "phase": "2B 层错失效机制诊断",
        "anchors": {
            "hcp_minus_fcc_DFT_eV_atom": {"value": 0.0315,
                "source": "DFT APW 1991 (Denteneer & van Haeringen, 2H-3C 多形体差 = 31.5 meV/atom)"},
            "Al_ISF_DFT": {"value": 164, "range": [161, 166],
                "source": "DFT (Hammer 1992 / Wright 1992 / Sun & Kaxiras 1997)"},
            "Al_USF_DFT": {"value": 224, "source": "DFT (cond-mat/9903440)"},
        },
        "polytype_discrimination": poly_rows,
        "gamma_surface": gsf_rows,
        "construction": [
            "hcp: 理想 c/a=1.633 (a=A_AL/sqrt2), fcc: 实验 a=4.05; 单点理想几何",
            "γ曲面: 12层(111) 1x1原胞, 位移 s=k*T_STEP/6 (k=0..6), 上半平移, 中/边界双等效层错 /2A, 单点能量(不弛豫, 诊断曲线形状)",
            "A = 7.10 A^2; γ = (E(s)-E(0))·1.602e-19 / (A·1e-20) / 2 × 1e3 (mJ/m2)",
        ],
        "device": device,
    }
    (RESULT_DIR / "phase2b_stacking.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved results/phase2b_stacking.json")


if __name__ == "__main__":
    main()

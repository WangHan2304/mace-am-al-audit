# -*- coding: utf-8 -*-
"""Phase 3：USF 弛豫协议对照实验（坍缩机制验证）

假设（Phase 2C §13.3）：MACE 的 USF 位置弛豫坍缩 = 上半块在面内横向滑回完美堆垛
（因为堆垛序对 MACE 无能量代价，力场把刚性位移的 USF 构型"拉回" fcc）。

三协议 × 三势：
  1) none    — 不弛豫（单点）：γ_sp 应 ≈ DFT 224（峰对）
  2) zonly   — 仅垂直(z)弛豫：横向被 FixXY 约束 → 应基本保留峰
  3) full    — 全位置弛豫：若假设成立 → 上半块横向滑回，γ 坍缩

同时量化上半块横向质心位移 |ΔCOM_xy|，与 T_STEP/2=0.827 A 对比。
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

A_AL = 4.05
D111 = A_AL / np.sqrt(3)
P1 = np.array([A_AL / 2, -A_AL / 2, 0.0])
P2 = np.array([A_AL / 2, 0.0, -A_AL / 2])
N_HAT = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
T_STEP = (P1 + P2) / 3
AREA = float(np.linalg.norm(np.cross(P1, P2)))
EV2J = 1.602176634e-19

MODELS = [
    ("MACE-MP-0 small", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    ("MACE-MP-0 medium", "2023-12-03-mace-128-L1_epoch-199.model"),
    ("MACE-MP-0 large", "MACE_MPtrj_2022.9.model"),
]


def build_111_stack(nlayers):
    pos = np.array([((k % 3) * T_STEP) + k * D111 * N_HAT for k in range(nlayers)])
    return Atoms("Al" * nlayers, positions=pos, cell=[P1, P2, nlayers * D111 * N_HAT], pbc=True)


def make_usf(perfect):
    cell = perfect.copy()
    pos = cell.get_positions()
    proj = pos @ N_HAT
    mid = (proj.min() + proj.max()) / 2
    upper = proj > mid
    pos[upper] += 0.5 * T_STEP
    cell.set_positions(pos)
    cell.wrap()
    return cell, upper


def relax(a, calc, zonly=False, fmax=0.05, steps=200):
    a.calc = calc
    if zonly:
        a.set_constraint(FixCartesian(list(range(len(a))), mask=(1, 1, 0)))  # 冻结 x,y，仅 z 可动
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a.get_potential_energy(), opt.nsteps


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    device = "cpu"   # 规避本机 CUDA 间歇崩溃；float64 能量与设备无关
    print(f"[设备] {device}")
    calcs = [(name, MACECalculator(model_paths=str(MODEL_DIR / f), device=device,
                                   default_dtype="float64")) for name, f in MODELS]

    perfect = build_111_stack(12)
    d = neighbor_list("d", perfect, cutoff=3.8)
    d = np.sort(d[d > 0.01])
    assert np.allclose(d[:2], 2.864, atol=0.02), f"完美胞最近邻异常 {d[:2]}"
    print(f"完美胞最近邻 {d[:2]} 面积 {AREA:.2f} A2")

    usf, upper = make_usf(perfect)
    upper_xy0 = usf.get_positions()[upper][:, :2].mean(axis=0)

    rows = []
    for mname, calc in calcs:
        out = {"model": mname}
        # --- none: 单点 ---
        a = usf.copy(); a.calc = calc
        e_usf_none = a.get_potential_energy()
        a2 = perfect.copy(); a2.calc = calc
        e_perf_none = a2.get_potential_energy()
        out["gamma_none_mJ_m2"] = (e_usf_none - e_perf_none) * EV2J / (AREA * 1e-20) * 1e3 / 2

        # --- zonly: 仅垂直弛豫 ---
        a = usf.copy()
        e_usf_z, n_z = relax(a, calc, zonly=True)
        b = perfect.copy()
        e_perf_z, _ = relax(b, calc, zonly=True)
        out["gamma_zonly_mJ_m2"] = (e_usf_z - e_perf_z) * EV2J / (AREA * 1e-20) * 1e3 / 2
        out["zonly_steps"] = n_z
        pos_z = a.get_positions()
        out["slide_xy_zonly_A"] = float(np.linalg.norm(pos_z[upper][:, :2].mean(axis=0) - upper_xy0))

        # --- full: 全位置弛豫 ---
        a = usf.copy()
        e_usf_full, n_f = relax(a, calc, zonly=False)
        out["gamma_full_mJ_m2"] = (e_usf_full - e_perf_none) * EV2J / (AREA * 1e-20) * 1e3 / 2
        out["full_steps"] = n_f
        pos_f = a.get_positions()
        out["slide_xy_full_A"] = float(np.linalg.norm(pos_f[upper][:, :2].mean(axis=0) - upper_xy0))

        out["slide_expected_T_half_A"] = float(np.linalg.norm(T_STEP) / 2)
        rows.append(out)
        print(f"  {mname:20s} none={out['gamma_none_mJ_m2']:6.1f}  "
              f"zonly={out['gamma_zonly_mJ_m2']:6.1f}  full={out['gamma_full_mJ_m2']:6.1f}  "
              f"slide(full)={out['slide_xy_full_A']:.3f} A")

    out = {
        "phase": "3 USF 弛豫协议对照实验",
        "hypothesis": "坍缩 = 上半块横向滑回完美堆垛（堆垛序对 MACE 无能量代价）",
        "protocols": [
            "none: 单点刚性位移, 期望≈DFT 224 (峰对)",
            "zonly: FixXY 冻结面内 x,y, 仅垂直弛豫 → 若峰保留则证明横向滑动是坍缩主因",
            "full: 全位置弛豫 → 若坍缩则横向滑回",
        ],
        "anchor_USF_DFT_mJ_m2": 224.0,
        "construction": "12层(111) 1x1原胞, 上半平移 T_STEP/2, 双等效层错 /2A, "
                        "BFGS fmax=0.05, 胞固定; CPU(float64)",
        "rows": rows,
    }
    (RESULT_DIR / "usf_protocol.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print("saved results/usf_protocol.json")


if __name__ == "__main__":
    main()

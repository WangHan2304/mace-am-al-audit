# -*- coding: utf-8 -*-
"""
AM1 · Phase 1b 弛豫复审计
=========================
对结构库×三档 MACE 做两阶段弛豫（固定胞位置弛豫 → 全弛豫），
在弛豫几何上重算能量/力/弹性，输出 results/audit_relaxed.json+csv。

意义：消除"晶格取文献实验值"带来的系统偏差（as-constructed 几何非势的平衡位形），
得到每个势的"自洽平衡"结果，与 02 的 as-constructed 结果对比可量化晶格约束偏差。
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.filters import ExpCellFilter  # ASE≥3.23 已从 ase.constraints 迁移

ROOT = Path(__file__).resolve().parents[1]
STRUCT_DIR = ROOT / "structures"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
EV2GPA = 160.21766208
STRAIN_ETA = 0.005

MODEL_MAPPING = {
    "MACE-MP-0 small": "2023-12-10-mace-128-L0_energy_epoch-249.model",
    "MACE-MP-0 medium": "2023-12-03-mace-128-L1_epoch-199.model",
    "MACE-MP-0 large": "MACE_MPtrj_2022.9.model",
}
ELASTIC_CUBIC = {"Al_fcc", "Si_dia", "Mg2Si", "Al3Sc_L12", "Al3Zr_L12"}
ELASTIC_TETRA = {"Al2Cu_theta", "Al2Cu_theta_prime"}


def load_calculators(device):
    from mace.calculators import MACECalculator
    out = []
    for tag, fname in MODEL_MAPPING.items():
        calc = MACECalculator(model_paths=str(MODEL_DIR / fname), device=device,
                              default_dtype="float64")
        out.append((tag, calc))
        print(f"[模型] {tag} 就绪")
    return out


def detect_stress_scale(atoms, calc):
    defo = atoms.copy()
    cell = np.array(atoms.cell)
    e = np.array([[STRAIN_ETA, 0, 0], [0, 0, 0], [0, 0, 0]])
    defo.set_cell(np.dot((np.eye(3) + e), cell), scale_atoms=True)
    defo.calc = calc
    raw6 = defo.get_stress(voigt=True)
    if abs(raw6[0]) > 0.05:
        return 1.0
    return EV2GPA


def elastic_tensor(atoms, calc, scale):
    modes = [
        (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0), (0, 1, 1, 0, 0, 0),
    ]
    X, Y = [], []
    for m in modes:
        for sgn in (+1.0, -1.0):
            e = np.array([
                [m[0], m[5] / 2, m[4] / 2],
                [m[5] / 2, m[1], m[3] / 2],
                [m[4] / 2, m[3] / 2, m[2]],
            ]) * (sgn * STRAIN_ETA)
            cell = np.array(atoms.cell)
            defo = atoms.copy()
            defo.set_cell(np.dot((np.eye(3) + e), cell), scale_atoms=True)
            defo.calc = calc
            stress_gpa = np.array(defo.get_stress(voigt=True)) * scale
            strain_vec = np.array([e[0, 0], e[1, 1], e[2, 2],
                                   2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])
            X.append(strain_vec)
            Y.append(stress_gpa)
    C, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return C


def relax_structure(atoms, calc, fmax_pos=0.02, fmax_cell=0.05):
    """两阶段弛豫：固定胞位置弛豫 → 全弛豫。返回 (relaxed_atoms, nsteps, ncell)."""
    a = atoms.copy()
    a.calc = calc
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax_pos, steps=200)
    n1 = opt.nsteps
    ecf = ExpCellFilter(a)
    opt2 = BFGS(ecf, logfile=None)
    opt2.run(fmax=fmax_cell, steps=200)
    n2 = opt2.nsteps
    return a, n1, n2


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    with open(STRUCT_DIR / "phases.json", encoding="utf-8") as f:
        meta = json.load(f)
    atoms_all = {k: read(STRUCT_DIR / f"{k}.cif", format="cif") for k in meta}

    models = load_calculators(device)
    rows = []
    t0 = time.time()
    for key, atoms0 in atoms_all.items():
        phase = meta[key]["name"]
        for model_name, calc in models:
            scale = detect_stress_scale(atoms0, calc)
            t1 = time.time()
            rel, n1, n2 = relax_structure(atoms0, calc)
            E = rel.get_potential_energy()
            F = rel.get_forces()
            maxf = float(np.abs(F).max())
            vol0 = atoms0.get_volume()
            vol1 = rel.get_volume()
            cell0 = np.array(atoms0.cell)
            cell1 = np.array(rel.cell)
            # 单胞参数变化幅度（相对，按晶格矢量长度）
            lat_change = float(np.max(np.abs(np.diag(cell1) - np.diag(cell0)) / np.diag(cell0)))
            row = {
                "structure": key, "phase": phase,
                "formula": meta[key]["formula"], "natoms": len(rel),
                "model": model_name,
                "E_tot_relaxed_eV": float(E),
                "E_per_atom_relaxed_eV": float(E / len(rel)),
                "maxF_relaxed_eV_per_A": maxf,
                "relax_steps_pos": n1, "relax_steps_cell": n2,
                "vol_change_pct": float((vol1 - vol0) / vol0 * 100),
                "lattice_max_change": lat_change,
            }
            if key in ELASTIC_CUBIC or key in ELASTIC_TETRA:
                try:
                    C = elastic_tensor(rel, calc, scale)
                    row["C11_GPa"] = float(C[0, 0])
                    row["C12_GPa"] = float(C[0, 1])
                    row["C44_GPa"] = float(C[3, 3])
                except Exception as e:  # noqa: BLE001
                    row["elastic_error"] = str(e)
            print(f"  {key:14s} {model_name:20s} E={row['E_per_atom_relaxed_eV']:+.4f} eV/atom "
                  f"dV={row['vol_change_pct']:+.2f}% steps=({n1},{n2}) {time.time()-t1:.0f}s")
            rows.append(row)
    print(f"[完成] {len(rows)} 行，耗时 {time.time()-t0:.0f}s")

    with open(RESULT_DIR / "audit_relaxed.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "meta": meta,
                   "models": [m[0] for m in models], "device": device},
                  f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "audit_relaxed.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["structure", "phase", "formula", "natoms", "model",
                "E_tot_relaxed_eV", "E_per_atom_relaxed_eV", "maxF_relaxed_eV_per_A",
                "relax_steps_pos", "relax_steps_cell", "vol_change_pct",
                "lattice_max_change", "C11_GPa", "C12_GPa", "C44_GPa"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

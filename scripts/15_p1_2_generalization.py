# -*- coding: utf-8 -*-
"""P1.2 ISF 堆垛序惩罚泛化验证

(1) Cu 体系（非 Al fcc）：hcp/fcc 能量差 + ISF 原始值 + 用 Cu 自身 ε 修正
(2) Al slab 厚度收敛：8 层 vs 12 层，两种构造（层移除 vs 刚性平移）
(3) 检验 2ε/A 是否只是 Al 特例
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

EV2J = 1.602176634e-19

MODELS = [
    ("MACE-MP-0 small", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    ("MACE-MP-0 medium", "2023-12-03-mace-128-L1_epoch-199.model"),
    ("MACE-MP-0 large", "MACE_MPtrj_2022.9.model"),
]

# Al and Cu lattice constants (experimental)
A_AL = 4.05
A_CU = 3.615


def fcc_to_hcp_energy(calc, symbol, a_fcc):
    """Compute ΔE(hcp - fcc) in meV/atom. hcp a = fcc_a/√2, c = 1.633a."""
    fcc = bulk(symbol, "fcc", a=a_fcc)
    a_hcp = a_fcc / np.sqrt(2.0)
    hcp = bulk(symbol, "hcp", a=a_hcp, c=a_hcp * 1.633)
    fcc.calc = calc
    hcp.calc = calc
    e_fcc = fcc.get_potential_energy() / len(fcc)
    e_hcp = hcp.get_potential_energy() / len(hcp)
    return (e_hcp - e_fcc) * 1000  # meV/atom


def build_111_stack(a, nlayers):
    """Build (111) fcc slab with given lattice constant and nlayers."""
    d111 = a / np.sqrt(3)
    p1 = np.array([a / 2, -a / 2, 0.0])
    p2 = np.array([a / 2, 0.0, -a / 2])
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    t_step = (p1 + p2) / 3.0
    pos = np.array([((k % 3) * t_step) + k * d111 * n_hat for k in range(nlayers)])
    cell = Atoms("Al" * nlayers, positions=pos, cell=[p1, p2, nlayers * d111 * n_hat], pbc=True)
    return cell, p1, p2, n_hat, t_step, d111


def make_isf_layer_removal(cell, n_hat, d111):
    """ISF by layer removal: remove middle layer, shift upper block down by d111, shrink cell."""
    pos = cell.get_positions().copy()
    proj = pos @ n_hat
    sorted_idx = np.argsort(proj)
    n = len(pos)
    mid = n // 2
    keep = np.ones(n, dtype=bool)
    keep[sorted_idx[mid]] = False
    # Upper block shifts down by one d111 to close gap
    upper = proj > (proj[sorted_idx[mid]] + 0.1)
    pos[upper] -= d111 * n_hat
    # Shrink cell c-vector by one d111 (periodic boundary closes)
    old_cell = cell.cell.copy()
    new_cell_matrix = old_cell.copy()
    new_cell_matrix[2] = old_cell[2] - d111 * n_hat
    symbols = np.array(cell.get_chemical_symbols())[keep]
    new_cell = Atoms(symbols, positions=pos[keep], cell=new_cell_matrix, pbc=True)
    return new_cell


def make_isf_rigid_shift(cell, n_hat, t_step):
    """ISF by rigid shift: upper half shifted by full T_STEP."""
    pos = cell.get_positions().copy()
    proj = pos @ n_hat
    mid = (proj.min() + proj.max()) / 2
    upper = proj > mid
    pos[upper] += t_step
    new_cell = Atoms(cell.get_chemical_symbols(), positions=pos,
                     cell=cell.cell, pbc=True)
    new_cell.wrap()
    return new_cell, upper


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    calcs = [(name, MACECalculator(model_paths=str(MODEL_DIR / f), device="cpu",
                                   default_dtype="float64")) for name, f in MODELS]

    results = {}

    # ========== Part 1: Cu hcp/fcc + ISF ==========
    print("=== Cu hcp/fcc energy difference ===")
    cu_hcpfcc = {}
    for mname, calc in calcs:
        de = fcc_to_hcp_energy(calc, "Cu", A_CU)
        cu_hcpfcc[mname] = de
        print(f"  {mname:20s} ΔE(hcp-fcc) = {de:+.2f} meV/atom")
    # DFT reference for Cu: ~+35 meV/atom (2H-3C, similar to Al)
    results["cu_hcp_fcc_meV_atom"] = cu_hcpfcc
    results["cu_dft_hcp_fcc_meV_atom"] = 35.0  # approximate DFT reference

    # Cu ISF (layer removal, 12 layers)
    print("\n=== Cu ISF (layer removal, 12 layers) ===")
    cu_results = {}
    for mname, calc in calcs:
        perfect, p1, p2, n_hat, t_step, d111 = build_111_stack(A_CU, 12)
        area = float(np.linalg.norm(np.cross(p1, p2)))
        # Verify NN
        d = neighbor_list("d", perfect, cutoff=3.3)
        d = np.sort(d[d > 0.01])
        nn = d[0]
        # Perfect energy
        perfect.calc = calc
        e_perf = perfect.get_potential_energy()
        # ISF layer removal
        isf = make_isf_layer_removal(perfect, n_hat, d111)
        isf.calc = calc
        e_isf = isf.get_potential_energy()
        # γ = (E_isf - E_perf_bulk)/2A (11 atoms ISF vs 12 atoms perf -> normalize per atom)
        # Actually: perfect has 12 atoms, ISF has 11. Need to compare per-atom energy difference.
        # γ = (E_isf - (n_isf/n_perf)*E_perf) / (2*A)
        n_perf = len(perfect)
        n_isf = len(isf)
        gamma_isf_raw = (e_isf - (n_isf / n_perf) * e_perf) * EV2J / (area * 1e-20) * 1e3 / 2

        # Corrected with DFT anchor epsilon (not MACE value!)
        eps_cu_dft = 35.0 / 1000  # eV/atom, DFT anchor for Cu hcp/fcc
        penalty = 2 * eps_cu_dft * EV2J / (area * 1e-20) * 1e3
        gamma_isf_corr = gamma_isf_raw + penalty

        cu_results[mname] = {
            "nn_A": round(nn, 3),
            "area_A2": round(area, 2),
            "gamma_isf_raw_mJ_m2": round(gamma_isf_raw, 1),
            "epsilon_cu_MACE_meV_atom": round(cu_hcpfcc[mname], 2),
            "epsilon_cu_DFT_meV_atom": 35.0,
            "penalty_mJ_m2": round(penalty, 1),
            "gamma_isf_corrected_mJ_m2": round(gamma_isf_corr, 1),
        }
        print(f"  {mname:20s} ISF raw={gamma_isf_raw:.1f} corr={gamma_isf_corr:.1f} "
              f"(ε_Cu={cu_hcpfcc[mname]:+.1f} meV, penalty={penalty:.1f})")

    results["cu_isf"] = cu_results
    results["cu_isf_dft_mJ_m2"] = {"range": [40, 45], "note": "Cu ISF ~40-45 mJ/m2 (literature)"}

    # ========== Part 2: Al slab thickness convergence (8 vs 12 layers) ==========
    print("\n=== Al ISF slab thickness convergence ===")
    al_thickness = {}
    for nlayers in [8, 12]:
        for mname, calc in calcs:
            perfect, p1, p2, n_hat, t_step, d111 = build_111_stack(A_AL, nlayers)
            area = float(np.linalg.norm(np.cross(p1, p2)))
            perfect.calc = calc
            e_perf = perfect.get_potential_energy()

            # Layer removal ISF
            isf_lr = make_isf_layer_removal(perfect, n_hat, d111)
            isf_lr.calc = calc
            e_isf_lr = isf_lr.get_potential_energy()
            n_perf = len(perfect)
            n_isf_lr = len(isf_lr)
            gamma_lr = (e_isf_lr - (n_isf_lr / n_perf) * e_perf) * EV2J / (area * 1e-20) * 1e3 / 2

            # Rigid shift ISF
            isf_rs, _ = make_isf_rigid_shift(perfect, n_hat, t_step)
            isf_rs.calc = calc
            e_isf_rs = isf_rs.get_potential_energy()
            gamma_rs = (e_isf_rs - e_perf) * EV2J / (area * 1e-20) * 1e3 / 2

            key = f"{mname}_{nlayers}layer"
            al_thickness[key] = {
                "nlayers": nlayers,
                "gamma_layermoval_raw": round(gamma_lr, 1),
                "gamma_rigiddrift_raw": round(gamma_rs, 1),
                "area_A2": round(area, 2),
            }
            print(f"  {mname:20s} {nlayers}层  层移除={gamma_lr:7.1f}  刚性平移={gamma_rs:7.1f}")

    results["al_thickness_convergence"] = al_thickness

    # ========== Save ==========
    out = {
        "phase": "P1.2 ISF 惩罚泛化验证",
        "cu_hcp_fcc": results.get("cu_hcp_fcc_meV_atom"),
        "cu_dft_hcp_fcc_meV_atom": 35.0,
        "cu_isf": results.get("cu_isf"),
        "cu_isf_dft": {"range": [40, 45], "note": "Cu ISF ~40-45 mJ/m2 (literature)"},
        "al_thickness": results.get("al_thickness_convergence"),
        "al_dft_isf_mJ_m2": [161, 166],
        "note": "Cu 体系用自身 ε_Cu 修正，检验 2ε/A 是否 Al 特例",
    }
    (RESULT_DIR / "p1_2_generalization.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved results/p1_2_generalization.json")


if __name__ == "__main__":
    main()

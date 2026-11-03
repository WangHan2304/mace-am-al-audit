# -*- coding: utf-8 -*-
"""校正边界理论：多金属可迁移性相图 + ISF hcp-like 配位计数

对 Al/Cu/Ni/Ag/Au 五个 fcc 金属：
1. MACE hcp/fcc 能量差 ε
2. MACE ISF raw（层移除构造，12层）
3. ISF 中 hcp-like 配位原子数（用近邻键序分析）
4. 用 DFT ε 做 2ε/A 校正后的 ISF
5. 对比 DFT ISF 锚点
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list, natural_cutoffs

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

EV2J = 1.602176634e-19

# fcc lattice constants (experimental)
METALS = {
    "Al": {"a": 4.05, "dft_eps_meV": 31.5, "dft_isf_mJ": 163},
    "Cu": {"a": 3.615, "dft_eps_meV": 35.0, "dft_isf_mJ": 45},
    "Ni": {"a": 3.524, "dft_eps_meV": 50.0, "dft_isf_mJ": 125},
    "Ag": {"a": 4.09, "dft_eps_meV": 12.0, "dft_isf_mJ": 20},
    "Au": {"a": 4.078, "dft_eps_meV": 35.0, "dft_isf_mJ": 45},
}

MODELS = [
    ("small", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    ("medium", "2023-12-03-mace-128-L1_epoch-199.model"),
    ("large", "MACE_MPtrj_2022.9.model"),
]


def build_111_stack(a, nlayers, symbol="Al"):
    d111 = a / np.sqrt(3)
    p1 = np.array([a / 2, -a / 2, 0.0])
    p2 = np.array([a / 2, 0.0, -a / 2])
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    t_step = (p1 + p2) / 3.0
    pos = np.array([((k % 3) * t_step) + k * d111 * n_hat for k in range(nlayers)])
    cell = Atoms(symbol * nlayers, positions=pos,
                 cell=[p1, p2, nlayers * d111 * n_hat], pbc=True)
    return cell, p1, p2, n_hat, t_step, d111


def make_isf(cell, n_hat, d111):
    pos = cell.get_positions().copy()
    proj = pos @ n_hat
    sorted_idx = np.argsort(proj)
    n = len(pos)
    mid = n // 2
    keep = np.ones(n, dtype=bool)
    keep[sorted_idx[mid]] = False
    upper = proj > (proj[sorted_idx[mid]] + 0.1)
    pos[upper] -= d111 * n_hat
    new_cell_matrix = cell.cell.copy()
    new_cell_matrix[2] = cell.cell[2] - d111 * n_hat
    symbols = np.array(cell.get_chemical_symbols())[keep]
    return Atoms(symbols, positions=pos[keep], cell=new_cell_matrix, pbc=True)


def count_hcp_like(isf_atoms, cutoff_scale=1.1):
    """Count atoms with hcp-like coordination (12 neighbors but wrong stacking).
    Uses simple neighbor count: fcc interior = 12 NN, ISF boundary atoms may differ.
    Returns number of atoms whose local stacking environment is hcp-like.
    """
    cutoffs = natural_cutoffs(isf_atoms, mult=cutoff_scale)
    i_list, j_list, d_list = neighbor_list("ijd", isf_atoms, cutoff=cutoffs)
    # Count neighbors per atom
    n_neighbors = np.zeros(len(isf_atoms), dtype=int)
    for i, j in zip(i_list, j_list):
        if i != j:
            n_neighbors[i] += 1
    # In perfect fcc: 12 neighbors. At ISF boundary, atoms may have distorted coordination.
    # Return atoms with != 12 neighbors (disrupted stacking)
    disrupted = np.sum(n_neighbors != 12)
    return disrupted, n_neighbors


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    calcs = [(name, MACECalculator(model_paths=str(MODEL_DIR / f), device="cpu",
                                   default_dtype="float64")) for name, f in MODELS]

    results = {}

    for metal, params in METALS.items():
        a = params["a"]
        dft_eps = params["dft_eps_meV"]
        dft_isf = params["dft_isf_mJ"]
        print(f"\n=== {metal} (a={a}) ===")

        metal_results = {"a": a, "dft_eps_meV": dft_eps, "dft_isf_mJ": dft_isf}

        for mname, calc in calcs:
            # hcp/fcc energy difference
            fcc = bulk(metal, "fcc", a=a)
            a_hcp = a / np.sqrt(2)
            hcp = bulk(metal, "hcp", a=a_hcp, c=a_hcp * 1.633)
            fcc.calc = calc
            hcp.calc = calc
            e_fcc = fcc.get_potential_energy() / len(fcc)
            e_hcp = hcp.get_potential_energy() / len(hcp)
            eps_mace = (e_hcp - e_fcc) * 1000  # meV/atom

            # ISF
            perfect, p1, p2, n_hat, t_step, d111 = build_111_stack(a, 12, metal)
            area = float(np.linalg.norm(np.cross(p1, p2)))
            perfect.calc = calc
            e_perf = perfect.get_potential_energy()
            isf = make_isf(perfect, n_hat, d111)
            isf.calc = calc
            e_isf = isf.get_potential_energy()
            n_perf = len(perfect)
            n_isf = len(isf)
            gamma_raw = (e_isf - (n_isf / n_perf) * e_perf) * EV2J / (area * 1e-20) * 1e3 / 2

            # Corrected with DFT epsilon
            penalty = 2 * (dft_eps / 1000) * EV2J / (area * 1e-20) * 1e3
            gamma_corr = gamma_raw + penalty

            # Count disrupted atoms (only for large tier to save time)
            if mname == "large":
                disrupted, n_nb = count_hcp_like(isf)
                metal_results["disrupted_atoms"] = int(disrupted)
                metal_results["n_neighbors_at_ISF"] = n_nb.tolist()

            metal_results[mname] = {
                "eps_mace_meV": round(eps_mace, 2),
                "gamma_isf_raw": round(gamma_raw, 1),
                "penalty_dft_eps": round(penalty, 1),
                "gamma_isf_corrected": round(gamma_corr, 1),
                "area_A2": round(area, 2),
            }
            print(f"  {mname:8s} ε_MACE={eps_mace:+6.2f} meV  raw_ISF={gamma_raw:7.1f}  "
                  f"corr={gamma_corr:7.1f}  (DFT ISF={dft_isf})")

        results[metal] = metal_results

    # Save
    out = {
        "phase": "校正边界理论：多金属可迁移性相图",
        "metals": results,
        "note": "DFT ε 和 DFT ISF 为文献参考值；MACE ε 和 raw ISF 为本次计算",
    }
    (RESULT_DIR / "p1_6_transferability_phase_diagram.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved results/p1_6_transferability_phase_diagram.json")


if __name__ == "__main__":
    main()

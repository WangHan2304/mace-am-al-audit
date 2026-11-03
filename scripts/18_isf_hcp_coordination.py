# -*- coding: utf-8 -*-
"""ISF 中 hcp-like 配位层计数：解释 factor=2 的微观来源

对 Al 和 Cu 的 12 层 ISF slab：
1. 按 z 投影分层
2. 检查每层的面内偏移序列是 ABC(fcc) 还是 AB(hcp)
3. 统计有多少层/多少原子是 hcp-like
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METALS = {
    "Al": 4.05,
    "Cu": 3.615,
}


def build_111_stack(a, nlayers, symbol):
    d111 = a / np.sqrt(3)
    p1 = np.array([a / 2, -a / 2, 0.0])
    p2 = np.array([a / 2, 0.0, -a / 2])
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    t_step = (p1 + p2) / 3.0
    pos = np.array([((k % 3) * t_step) + k * d111 * n_hat for k in range(nlayers)])
    from ase import Atoms
    cell = Atoms(symbol * nlayers, positions=pos,
                 cell=[p1, p2, nlayers * d111 * n_hat], pbc=True)
    return cell, p1, p2, n_hat, d111


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
    from ase import Atoms
    return Atoms(symbols, positions=pos[keep], cell=new_cell_matrix, pbc=True)


def analyze_stacking(isf_atoms, n_hat, p1, p2, d111):
    """Project positions onto (p1,p2) plane and classify each layer's stacking type."""
    pos = isf_atoms.get_positions()
    proj_z = pos @ n_hat
    order = np.argsort(proj_z)
    layers = []
    current_layer = [order[0]]
    for i in order[1:]:
        if abs(proj_z[i] - proj_z[current_layer[0]]) < d111 * 0.4:
            current_layer.append(i)
        else:
            layers.append(current_layer)
            current_layer = [i]
    layers.append(current_layer)

    # For each layer, get center-of-mass projected onto p1,p2 basis
    # Compute projection coefficients: pos = c1*p1 + c2*p2 + c3*n_hat
    # Solve using all three vectors
    basis = np.array([p1, p2, n_hat]).T  # 3x3
    inv = np.linalg.inv(basis)

    layer_offsets = []
    for layer in layers:
        layer_pos = pos[layer]
        com = np.mean(layer_pos, axis=0)
        coeffs = inv @ com
        layer_offsets.append(coeffs[:2])  # fractional in p1,p2 plane

    layer_offsets = np.array(layer_offsets)
    n_layers = len(layers)

    # In fcc ABC stacking, consecutive layers shift by (1/3, 1/3) in fractional coords
    # The ISF introduces one mistake: after the fault, sequence shifts by one step
    # Compare each layer's offset to expected fcc (i*1/3, i*1/3) mod 1
    t_frac = np.array([1/3, 1/3])
    deviations = []
    for i in range(n_layers):
        expected = (i * t_frac) % 1.0
        diff = layer_offsets[i] - expected
        diff = diff - np.round(diff)
        deviations.append(np.linalg.norm(diff))

    hcp_layers = [i for i, d in enumerate(deviations) if d > 0.15]
    n_hcp_atoms = sum(len(layers[i]) for i in hcp_layers)

    return {
        "n_layers": n_layers,
        "n_atoms_total": len(isf_atoms),
        "hcp_layer_indices": hcp_layers,
        "n_hcp_layers": len(hcp_layers),
        "n_hcp_atoms": n_hcp_atoms,
        "layer_offsets_frac": [[round(o[0],3), round(o[1],3)] for o in layer_offsets],
        "deviations": [round(d, 3) for d in deviations],
    }


def main():
    results = {}
    for metal, a in METALS.items():
        perfect, p1, p2, n_hat, d111 = build_111_stack(a, 12, metal)
        isf = make_isf(perfect, n_hat, d111)
        analysis = analyze_stacking(isf, n_hat, p1, p2, d111)
        results[metal] = analysis
        print(f"\n=== {metal} (a={a}) ===")
        print(f"  Layers: {analysis['n_layers']}, atoms: {analysis['n_atoms_total']}")
        print(f"  hcp-like layers: {analysis['hcp_layer_indices']} ({analysis['n_hcp_layers']})")
        print(f"  hcp-like atoms: {analysis['n_hcp_atoms']}")
        print(f"  deviations from fcc sequence: {analysis['deviations']}")

    out = {"analysis": "ISF hcp-like layer count", "results": results}
    Path(ROOT / "results" / "p1_7_isf_hcp_coordination.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved results/p1_7_isf_hcp_coordination.json")


if __name__ == "__main__":
    main()

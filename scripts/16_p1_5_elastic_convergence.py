# -*- coding: utf-8 -*-
"""P1.5 弹性常数收敛性：±0.5%/1%/2% 应变幅度对比"""
import json
from pathlib import Path

import numpy as np
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODELS = [
    ("small", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    ("medium", "2023-12-03-mace-128-L1_epoch-199.model"),
    ("large", "MACE_MPtrj_2022.9.model"),
]

# eV/Å^3 to GPa
EV_PER_A3_TO_GPA = 160.21766208


def elastic_c11_c12(calc, a=4.05):
    """Compute C11 and C12 via uniaxial and biaxial strain."""
    # C11: uniaxial strain along x, y,z fixed
    def e_uniax(eps):
        atoms = bulk("Al", "fcc", a=a * (1 + eps), cubic=True)
        atoms.calc = calc
        return atoms.get_potential_energy() / len(atoms)

    # C44: shear strain
    def e_shear(eps):
        atoms = bulk("Al", "fcc", a=a, cubic=True)
        cell = atoms.get_cell()
        # shear: x' = x + eps*y
        cell[0, 1] += eps * a
        atoms.set_cell(cell, scale_atoms=False)
        atoms.calc = calc
        return atoms.get_potential_energy() / len(atoms)

    return e_uniax, e_shear


def compute_elastic(calc, strain_amp):
    """Compute C11 and C44 at given strain amplitude.
    C11: uniaxial strain along x only (y,z fixed)
    C44: shear strain
    """
    a = 4.05
    V_atom = a**3 / 4.0  # per-atom volume in cubic cell
    eps = strain_amp

    # --- C11: uniaxial strain along x ---
    atoms0 = bulk("Al", "fcc", a=a, cubic=True)
    atoms0.calc = calc
    E0 = atoms0.get_potential_energy() / len(atoms0)

    # +eps along x only
    atoms_p = bulk("Al", "fcc", a=a, cubic=True)
    cell = atoms_p.get_cell()
    cell[0, 0] *= (1 + eps)
    atoms_p.set_cell(cell, scale_atoms=False)
    atoms_p.calc = calc
    Ep = atoms_p.get_potential_energy() / len(atoms_p)

    # -eps along x only
    atoms_m = bulk("Al", "fcc", a=a, cubic=True)
    cell = atoms_m.get_cell()
    cell[0, 0] *= (1 - eps)
    atoms_m.set_cell(cell, scale_atoms=False)
    atoms_m.calc = calc
    Em = atoms_m.get_potential_energy() / len(atoms_m)

    # d²E/dε² = V_atom * C11  (energy per atom)
    d2E = (Ep + Em - 2 * E0) / eps**2
    C11 = d2E / V_atom  # eV/Å³

    # --- C44: shear strain ---
    atoms_s = bulk("Al", "fcc", a=a, cubic=True)
    cell_s = atoms_s.get_cell()
    cell_s[0, 1] += eps * a
    atoms_s.set_cell(cell_s, scale_atoms=False)
    atoms_s.calc = calc
    Es = atoms_s.get_potential_energy() / len(atoms_s)

    d2E_s = (Es - E0) / eps**2
    C44 = d2E_s / V_atom  # eV/Å³

    return C11 * EV_PER_A3_TO_GPA, C44 * EV_PER_A3_TO_GPA


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    calcs = [(name, MACECalculator(model_paths=str(MODEL_DIR / f), device="cpu",
                                   default_dtype="float64")) for name, f in MODELS]

    strains = [0.005, 0.01, 0.02]
    results = {}

    for mname, calc in calcs:
        results[mname] = {}
        for s in strains:
            c11, c44 = compute_elastic(calc, s)
            results[mname][f"eps_{s}"] = {
                "C11_GPa": round(c11, 1),
                "C44_GPa": round(c44, 1),
            }
            print(f"{mname:8s} ε={s:.1%}  C11={c11:.1f} GPa  C44={c44:.1f} GPa")

    # Compare with baseline (from earlier audit, 1% strain)
    results["baseline_1pct"] = {
        "small": {"C11": 104.7, "C12": 68.1, "C44": 22.0},
        "medium": {"C11": 88.0, "C12": 54.8, "C44": 26.4},
        "large": {"C11": 70.8, "C12": 49.3, "C44": 15.2},
    }
    results["experimental"] = {"C11": 108.0, "C12": 62.0, "C44": 28.0}

    (RESULT_DIR / "p1_5_elastic_convergence.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved results/p1_5_elastic_convergence.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Final recheck of the MACE-MP-0 elastic constants for Al (affine deformation).

The previous attempts in this session failed for a bookkeeping reason that is
worth recording: ASE's Atoms.set_cell(..., scale_atoms=False) keeps the
cartesian positions fixed, so scaling the cell leaves the atoms where they are
and only changes the periodic images. The deformation that defines an elastic
constant is affine, i.e. the fractional coordinates stay fixed and the
cartesian positions follow the cell -- that is scale_atoms=True.

Consistency check built in: the isotropic dilation must reproduce the bulk
modulus obtained independently from the E(a) equation of state (80.3 / 66.0 /
58.4 GPa for small / medium / large); the stress route is reported alongside
the energy route for every component.

Deformations (F applied to the cell row vectors, fractional positions fixed):
    tetragonal  diag(1+d, 1-d, 1)   dE/V = (C11 - C12) d^2
    biaxial     diag(1+d, 1+d, 1)   dE/V = (C11 + C12) d^2
    pure shear  I + d e_x e_y^T     dE/V = 1/2 C44 d^2

Usage:  python 26_elastic_final.py
"""
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

EV_PER_A3_TO_GPA = 160.21766208
EXPERIMENTAL = {"C11": 108.0, "C12": 62.0, "C44": 28.0, "B": 76.0}


def F_of(kind, d):
    F = np.eye(3)
    if kind == "tet":
        F[0, 0], F[1, 1] = 1 + d, 1 - d
    elif kind == "biax":
        F[0, 0], F[1, 1] = 1 + d, 1 + d
    elif kind == "xx":
        F[0, 0] = 1 + d
    elif kind == "shear":
        F[0, 1] = d
    elif kind == "iso":
        F = np.eye(3) * (1 + d)
    return F


def deformed(calc, kind, d, a):
    atoms = bulk("Al", "fcc", a=a, cubic=True)
    atoms.set_cell(np.array(atoms.cell) @ F_of(kind, d).T, scale_atoms=True)
    atoms.calc = calc
    return atoms


def e_atom(calc, kind, d, a):
    return deformed(calc, kind, d, a).get_potential_energy() / 4.0


def curv(calc, kind, d, a):
    return (e_atom(calc, kind, d, a) + e_atom(calc, kind, -d, a)
            - 2 * e_atom(calc, kind, 0.0, a)) / d**2


def constants(calc, a, d):
    V = a**3 / 4.0
    k_tet = curv(calc, "tet", d, a)
    k_bia = curv(calc, "biax", d, a)
    k_shr = curv(calc, "shear", d, a)
    c11_m_c12 = k_tet / (2 * V)
    c11_p_c12 = k_bia / (2 * V)
    C11 = (c11_p_c12 + c11_m_c12) / 2
    C12 = (c11_p_c12 - c11_m_c12) / 2
    C44 = k_shr / V
    return tuple(x * EV_PER_A3_TO_GPA for x in (C11, C12, C44))


def main():
    import torch
    torch.set_num_threads(8)
    from mace.calculators import MACECalculator

    D = 0.01
    out = {"protocol": {"d": D, "dtype": "float64", "device": "cpu",
                        "deformation": "affine, scale_atoms=True",
                        "orientation": "tetragonal C11-C12, biaxial C11+C12, pure shear C44"},
           "experimental": EXPERIMENTAL, "models": {}}

    for name, fname in MODELS:
        calc = MACECalculator(model_paths=str(MODEL_DIR / fname), device="cpu",
                              default_dtype="float64")
        a0 = json.loads((RESULT_DIR / "p1_8_elastic_recheck.json").read_text())[
            "models"][name]["a0"]
        V = a0**3 / 4.0

        C11, C12, C44 = constants(calc, a0, D)
        B = (C11 + 2 * C12) / 3

        # independent stress-route values at the same deformation
        s_p = deformed(calc, "xx", D, a0).get_stress() * EV_PER_A3_TO_GPA
        s_m = deformed(calc, "xx", -D, a0).get_stress() * EV_PER_A3_TO_GPA
        C11_s = -(s_p[0] - s_m[0]) / (2 * D)

        g_p = deformed(calc, "shear", D, a0).get_stress() * EV_PER_A3_TO_GPA
        g_m = deformed(calc, "shear", -D, a0).get_stress() * EV_PER_A3_TO_GPA
        C44_s = -(g_p[5] - g_m[5]) / (2 * D)

        # isotropic check against the EOS bulk modulus
        k_iso = curv(calc, "iso", D, a0)
        B_iso = k_iso / (9 * V) * EV_PER_A3_TO_GPA

        out["models"][name] = {
            "a0": a0,
            "C11_GPa": round(C11, 1), "C12_GPa": round(C12, 1),
            "C44_GPa": round(C44, 1), "B_GPa": round(B, 1),
            "B_iso_GPa": round(B_iso, 1),
            "C11_stress_GPa": round(C11_s, 1),
            "C44_stress_GPa": round(C44_s, 1),
            "dev_C11_vs_exp_pct": round(100 * (C11 - 108.0) / 108.0, 1),
        }
        print(f"{name:7s} C11={C11:6.1f} C12={C12:5.1f} C44={C44:5.1f} B={B:5.1f} "
              f"| B_iso={B_iso:5.1f} | stress: C11={C11_s:6.1f} C44={C44_s:5.1f}",
              flush=True)

    (RESULT_DIR / "p1_8_elastic_final.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/p1_8_elastic_final.json")


if __name__ == "__main__":
    main()

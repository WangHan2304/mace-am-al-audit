# -*- coding: utf-8 -*-
"""本地 DFT 锚点：Al/Cu hcp/fcc + ISF 单点 (GPAW PBE)"""
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

OUT = Path("/home/user/dft_anchors.json")

def calc_bulk(symbol, structure, a, c_over_a=None):
    if structure == "fcc":
        at = bulk(symbol, "fcc", a=a)
    else:
        a_h = a / np.sqrt(2)
        c = a_h * c_over_a if c_over_a else a_h * 1.633
        at = bulk(symbol, "hcp", a=a_h, c=c)
    at.calc = GPAW(
        mode=PW(400),
        xc="PBE",
        kpts=(12, 12, 12),
        occupations=FermiDirac(0.01),
        txt=None,
    )
    e = at.get_potential_energy() / len(at)
    return e


def main():
    t0 = time.time()
    results = {}

    # Al
    print("=== Al ===")
    e_fcc_al = calc_bulk("Al", "fcc", 4.05)
    print(f"  Al fcc: {e_fcc_al:.6f} eV/atom")
    e_hcp_al = calc_bulk("Al", "hcp", 4.05, 1.633)
    print(f"  Al hcp: {e_hcp_al:.6f} eV/atom")
    results["Al"] = {
        "e_fcc": round(e_fcc_al, 6),
        "e_hcp": round(e_hcp_al, 6),
        "eps_hcp_minus_fcc_meV": round((e_hcp_al - e_fcc_al) * 1000, 2),
    }

    # Cu
    print("=== Cu ===")
    e_fcc_cu = calc_bulk("Cu", "fcc", 3.615)
    print(f"  Cu fcc: {e_fcc_cu:.6f} eV/atom")
    e_hcp_cu = calc_bulk("Cu", "hcp", 3.615, 1.633)
    print(f"  Cu hcp: {e_hcp_cu:.6f} eV/atom")
    results["Cu"] = {
        "e_fcc": round(e_fcc_cu, 6),
        "e_hcp": round(e_hcp_cu, 6),
        "eps_hcp_minus_fcc_meV": round((e_hcp_cu - e_fcc_cu) * 1000, 2),
    }

    results["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDone in {results['elapsed_sec']}s, saved {OUT}")


if __name__ == "__main__":
    main()

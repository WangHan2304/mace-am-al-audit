# -*- coding: utf-8 -*-
"""本地 DFT：Ni/Ag/Au hcp/fcc + Al ISF slab"""
import json, time
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

t0 = time.time()
results = {}

def calc_bulk(symbol, structure, a, c_over_a=1.633):
    if structure == "fcc":
        at = bulk(symbol, "fcc", a=a)
    else:
        a_h = a / np.sqrt(2)
        c = a_h * c_over_a
        at = bulk(symbol, "hcp", a=a_h, c=c)
    at.calc = GPAW(mode=PW(400), xc="PBE", kpts=(12,12,12),
                   occupations=FermiDirac(0.01), txt=None)
    return at.get_potential_energy() / len(at)

# Lattice constants (experimental, Angstrom)
lat = {"Ni": 3.52, "Ag": 4.09, "Au": 4.078, "Al": 4.05, "Cu": 3.615}

for sym in ["Ni", "Ag", "Au"]:
    print(f"=== {sym} ===")
    ef = calc_bulk(sym, "fcc", lat[sym])
    eh = calc_bulk(sym, "hcp", lat[sym])
    eps = (eh - ef) * 1000
    results[sym] = {"e_fcc": round(ef,6), "e_hcp": round(eh,6),
                    "eps_meV": round(eps, 2)}
    print(f"  fcc={ef:.4f}  hcp={eh:.4f}  eps={eps:.2f} meV/atom")

results["elapsed"] = round(time.time()-t0, 1)
print(json.dumps(results, indent=2))
with open("/home/cl/dft_bulk_extra.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Done in {results['elapsed']}s")

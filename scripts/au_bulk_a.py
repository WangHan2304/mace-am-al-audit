"""PBE equilibrium lattice constant for fcc Au, same PAW/PW setup as the slab.

The slab cell is held fixed with the bottom two layers constrained, so any
mismatch between the chosen a and the PBE equilibrium value shows up as a
residual in-plane stress that FIRE cannot remove (it only relaxes the layers).
The experimental 4.078 A is ~2 % below the PBE value for Au and left fmax at
0.60 eV/A after the first step, so the slab must be built at the computed a.

Usage:  python au_bulk_a.py
"""
import json
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW

ECUT = 400
SYM = {'point_group': False}
KPTS = (12, 12, 12)
A_GUESS = [4.08, 4.12, 4.16, 4.20]


def energy(a):
    at = bulk('Au', 'fcc', a=a, cubic=False)
    calc = GPAW(mode=PW(ECUT), xc='PBE', kpts=KPTS, symmetry=SYM,
                txt=f'au_bulk_{a:.2f}.txt')
    at.calc = calc
    return at.get_potential_energy()


if __name__ == '__main__':
    rows = []
    for a in A_GUESS:
        e = energy(a)
        rows.append((a, e))
        print(f'a={a:.2f}  E={e:.6f} eV/atom', flush=True)

    aa = np.array([r[0] for r in rows])
    ee = np.array([r[1] for r in rows])
    c2, c1, c0 = np.polyfit(aa, ee, 2)
    a_min = -c1 / (2 * c2)
    print(f'--- quadratic fit: a0 = {a_min:.4f} A ---')
    with open('au_bulk_a.json', 'w') as fh:
        json.dump(dict(points=rows, a0=float(a_min), ecut=ECUT,
                       kpts=list(KPTS), xc='PBE'), fh, indent=2)

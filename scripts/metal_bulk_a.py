"""PBE equilibrium lattice constant for an fcc metal, same PAW/PW setup as the slab.

Generalises au_bulk_a.py to Al / Ni / Ag so that the three new slab families
use a computed a0 rather than an experimental one: the slab cell is held fixed
with the bottom two layers constrained, so a mismatch shows up as residual
in-plane stress that FIRE cannot remove.

Usage:  python metal_bulk_a.py Al
        python metal_bulk_a.py Ni
        python metal_bulk_a.py Ag
"""
import json
import sys
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW

ECUT = 400
SYM = {'point_group': False}
KPTS = (12, 12, 12)

GUESS = {
    'Al': [3.98, 4.02, 4.06, 4.10],
    'Ni': [3.44, 3.48, 3.52, 3.56],
    'Ag': [4.02, 4.06, 4.10, 4.14],
}


def energy(metal, a):
    at = bulk(metal, 'fcc', a=a, cubic=False)
    calc = GPAW(mode=PW(ECUT), xc='PBE', kpts=KPTS, symmetry=SYM,
                txt=f'{metal.lower()}_bulk_{a:.2f}.txt')
    at.calc = calc
    return at.get_potential_energy()


if __name__ == '__main__':
    metal = sys.argv[1]
    rows = []
    for a in GUESS[metal]:
        e = energy(metal, a)
        rows.append((a, e))
        print(f'{metal} a={a:.2f}  E={e:.6f} eV/atom', flush=True)

    aa = np.array([r[0] for r in rows])
    ee = np.array([r[1] for r in rows])
    c2, c1, c0 = np.polyfit(aa, ee, 2)
    a_min = -c1 / (2 * c2)
    print(f'--- quadratic fit: {metal} a0 = {a_min:.4f} A ---')
    with open(f'{metal.lower()}_bulk_a.json', 'w') as fh:
        json.dump(dict(metal=metal, points=rows, a0=float(a_min), ecut=ECUT,
                       kpts=list(KPTS), xc='PBE'), fh, indent=2)

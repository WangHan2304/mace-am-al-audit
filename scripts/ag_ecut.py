"""Plane-wave cutoff convergence test for the Ag bulk cell.

The 400 eV scan returned a0 = 4.149 A, which is 1.5% above the experimental
4.085 A -- far outside the usual PBE window (4.06-4.09). Ag carries 4d
electrons, so the suspicion is that 400 eV is not converged for this element
even though it is adequate for Al, Cu, Ni and Au.

Two lattice constants are evaluated at 400 and 500 eV. If the higher cutoff
lowers the energy and shrinks the residual stress at a = 4.09, the 400 eV
equilibrium is an artefact of the basis, not a property of the metal.

Usage:  python ag_ecut.py
"""
import json
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW

SYM = {'point_group': False}
KPTS = (12, 12, 12)
ROWS = []

for ecut in (400, 500):
    for a in (4.09, 4.15):
        at = bulk('Ag', 'fcc', a=a, cubic=False)
        calc = GPAW(mode=PW(ecut), xc='PBE', kpts=KPTS, symmetry=SYM,
                    txt=f'ag_ecut{ecut}_{a:.2f}.txt')
        at.calc = calc
        e = at.get_potential_energy()
        s = at.get_stress(voigt=True)
        ROWS.append(dict(ecut=ecut, a=a, E=e, p_xx=float(s[0]), p_trace=float(s[:3].sum())))
        print(f'ecut={ecut}  a={a:.2f}  E={e:.6f}  tr(sigma)={s[:3].sum():+.4f} eV/A3',
              flush=True)

with open('ag_ecut.json', 'w') as fh:
    json.dump(ROWS, fh, indent=2)
print('saved ag_ecut.json')

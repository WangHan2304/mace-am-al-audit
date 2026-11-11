"""Au ISF, PW basis, relaxed, same protocol as the Cu slab family.

Identical setup to cu_isf_relax2.py so that the two metals are directly
comparable: 1x1 (111), one fault plane, slip (a1+a2)/3, 15 A vacuum, PW 400 eV,
PBE, default Fermi-Dirac smearing, bottom two layers fixed, point-group symmetry
off (time reversal kept), FIRE with maxstep 0.05, fmax 0.05 eV/A.

Lattice constant 4.1761 A is the PBE equilibrium value computed with the same
PAW/PW setup (au_bulk_a.py, quadratic fit over 4.08-4.20 A). The experimental
4.078 A is 2.4 % below it; with the cell held fixed and the bottom two layers
constrained, that mismatch leaves a residual in-plane stress that FIRE cannot
remove, and fmax stalls near 0.6 eV/A. Slip length is a/sqrt(6) = 1.705 A, cell
area 15.103 A^2.

Usage:  python au_isf_relax2.py 8
        python au_isf_relax2.py all
"""
import json
import sys
import time
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.optimize import FIRE, BFGS
from gpaw import GPAW, PW

A_BULK = 4.1761
VACUUM = 15.0
ECUT = 400
KPTS = (8, 8, 1)
CONV = 16021.8
SYM = {'point_group': False}


def run_case(nlayers, fmax=0.05, maxsteps=80):
    t0 = time.time()
    sp = fcc111('Au', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Au', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - A_BULK / np.sqrt(6)) < 1e-6
    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    zs = np.sort(sp.positions[:, 2])
    sp.set_constraint(FixAtoms(mask=sp.positions[:, 2] < zs[1] + 1e-3))
    zs2 = np.sort(sf.positions[:, 2])
    sf.set_constraint(FixAtoms(mask=sf.positions[:, 2] < zs2[1] + 1e-3))

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)
    tag = f'{nlayers}L'

    res = {}
    for label, slab in (('perfect', sp), ('faulted', sf)):
        calc = GPAW(mode=PW(ECUT), xc='PBE', kpts=KPTS, symmetry=SYM,
                    txt=f'au2_{label}_{tag}.txt')
        slab.calc = calc
        opt = FIRE(slab, logfile=f'au2_{label}_{tag}.opt', maxstep=0.05)
        opt.run(fmax=fmax, steps=maxsteps)
        res[label] = dict(
            E_0K=slab.get_potential_energy(),
            E_free=slab.get_potential_energy(force_consistent=True),
            steps=len(open(f'au2_{label}_{tag}.opt').readlines()) - 1,
        )

    dE_0K = res['faulted']['E_0K'] - res['perfect']['E_0K']
    dE_free = res['faulted']['E_free'] - res['perfect']['E_free']
    out = dict(metal='Au', nlayers=nlayers, natoms=len(sp), ecut=ECUT,
               kpts=list(KPTS), symmetry='point_group off',
               a_bulk=A_BULK, slip_A=float(np.linalg.norm(shift)), area=area,
               E_perfect=res['perfect'], E_faulted=res['faulted'],
               dE_0K=dE_0K, dE_free=dE_free,
               ISF_mJm2=dE_0K / area * CONV,
               ISF_free_mJm2=dE_free / area * CONV,
               seconds=time.time() - t0)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    layers = [8, 10, 12] if arg == 'all' else [int(arg)]
    out = [run_case(n) for n in layers]
    name = f'au_isf_relax2_{arg}.json'
    with open(name, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY (0 K convention) ---')
    for r in out:
        print(f'{r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(dE = {r["dE_0K"]*1000:.2f} meV, free-energy conv '
              f'{r["ISF_free_mJm2"]:.1f}, {r["seconds"]:.0f} s)')
    with open(f'au_isf_relax2_{arg}.done', 'w') as fh:
        fh.write('ok\n')

"""fcc (111) intrinsic stacking fault, PW basis, relaxed -- generalised slab driver.

Identical protocol to au_isf_relax2.py / cu_isf_relax2.py so that every metal
family is directly comparable: 1x1 (111) cell, one fault plane, slip (a1+a2)/3
with |b| = a/sqrt(6), 15 A vacuum, PW 400 eV, PBE, bottom two layers fixed,
point-group symmetry off, FIRE with maxstep 0.05 and fmax 0.05 eV/A.

The lattice constant is read from <metal>_bulk_a.json (metal_bulk_a.py), i.e.
the PBE equilibrium value in the same PAW/PW setup, never the experimental one,
because the slab cell is held fixed and FIRE only relaxes the layers.

The script never reads a pre-existing .opt, so a killed run can simply be restarted.

Usage:  python metal_isf.py Al 8
        python metal_isf.py Ni all
        python metal_isf.py Ag 12
"""
import json
import os
import sys
import time
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from gpaw import GPAW, PW

VACUUM = 15.0
ECUT = 400
KPTS = (8, 8, 1)
CONV = 16021.8
SYM = {'point_group': False}


def load_a(metal):
    path = f'{metal.lower()}_bulk_a.json'
    with open(path) as fh:
        return json.load(fh)['a0']


def run_case(metal, a_bulk, nlayers, fmax=0.05, maxsteps=80):
    t0 = time.time()
    sp = fcc111(metal, size=(1, 1, nlayers), a=a_bulk, vacuum=VACUUM)
    sf = fcc111(metal, size=(1, 1, nlayers), a=a_bulk, vacuum=VACUUM)

    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - a_bulk / np.sqrt(6)) < 1e-6
    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    zs = np.sort(sp.positions[:, 2])
    sp.set_constraint(FixAtoms(mask=sp.positions[:, 2] < zs[1] + 1e-3))
    zs2 = np.sort(sf.positions[:, 2])
    sf.set_constraint(FixAtoms(mask=sf.positions[:, 2] < zs2[1] + 1e-3))

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)
    tag = f'{nlayers}L'
    pre = metal.lower()

    res = {}
    gk = dict(mode=PW(ECUT), xc='PBE', kpts=KPTS, symmetry=SYM)
    if metal == 'Ni':
        for at in (sp, sf):
            at.set_initial_magnetic_moments([1.0] * len(at))
        gk['spinpol'] = True
    for label, slab in (('perfect', sp), ('faulted', sf)):
        calc = GPAW(**gk, txt=f'{pre}2_{label}_{tag}.txt')
        slab.calc = calc
        opt = FIRE(slab, logfile=f'{pre}2_{label}_{tag}.opt', maxstep=0.05)
        opt.run(fmax=fmax, steps=maxsteps)
        res[label] = dict(
            E_0K=slab.get_potential_energy(),
            E_free=slab.get_potential_energy(force_consistent=True),
            steps=len(open(f'{pre}2_{label}_{tag}.opt').readlines()) - 1,
        )

    dE_0K = res['faulted']['E_0K'] - res['perfect']['E_0K']
    dE_free = res['faulted']['E_free'] - res['perfect']['E_free']
    out = dict(metal=metal, nlayers=nlayers, natoms=len(sp), ecut=ECUT,
               kpts=list(KPTS), symmetry='point_group off',
               a_bulk=a_bulk, slip_A=float(np.linalg.norm(shift)), area=area,
               E_perfect=res['perfect'], E_faulted=res['faulted'],
               dE_0K=dE_0K, dE_free=dE_free,
               ISF_mJm2=dE_0K / area * CONV,
               ISF_free_mJm2=dE_free / area * CONV,
               seconds=time.time() - t0)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    metal = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else 'all'
    layers = [8, 10, 12] if arg == 'all' else [int(arg)]
    a_bulk = load_a(metal)
    print(f'# {metal}: a0 = {a_bulk:.4f} A, layers {layers}', flush=True)
    out = [run_case(metal, a_bulk, n) for n in layers]
    name = f'{metal.lower()}_isf_relax2_{arg}.json'
    with open(name, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY (0 K convention) ---')
    for r in out:
        print(f'{metal} {r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(dE = {r["dE_0K"]*1000:.2f} meV, free-energy conv '
              f'{r["ISF_free_mJm2"]:.1f}, {r["seconds"]:.0f} s)')
    with open(f'{metal.lower()}_isf_relax2_{arg}.done', 'w') as fh:
        fh.write('ok\n')

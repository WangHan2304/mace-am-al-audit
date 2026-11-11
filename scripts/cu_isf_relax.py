"""Cu ISF, PW basis with ionic relaxation and thickness convergence.

Why this exists: the unrelaxed PW run gives 57.1 mJ/m^2 at 6 layers against a
34--45 mJ/m^2 literature band. An intrinsic fault relaxes the layer spacing
across the fault plane, which lowers the energy appreciably; a frozen-ideal
slab therefore overestimates. The LCAO/dzp run (18.6) is not used further: with
dE = 6.6 meV a dzp basis on a 3d metal absorbs too much of the fault energy, and
the PW value at the same geometry is three times larger, so the two cannot both
be right and PW is the one with the variational guarantee.

Protocol: 1x1 (111) cell, one fault plane, slip (a1+a2)/3, 15 A vacuum,
kpts 8x8x1, PW 400 eV, PBE, Fermi-Dirac smearing (default). The bottom two
layers are fixed to represent the bulk constraint; all layers above relax.
Force criterion 0.05 eV/A.

Usage:  python cu_isf_relax.py 6
        python cu_isf_relax.py all
"""
import json
import sys
import time
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW, PW

A_BULK = 3.615
VACUUM = 15.0
ECUT = 400
KPTS = (8, 8, 1)
CONV = 16021.8


def run_case(nlayers, fmax=0.05, maxsteps=60):
    t0 = time.time()
    sp = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - A_BULK / np.sqrt(6)) < 1e-6
    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    # freeze the two lowest layers in both slabs (same constraint, symmetric)
    zs = np.sort(sp.positions[:, 2])
    zcut = zs[1] + 1e-3
    sp.set_constraint(FixAtoms(mask=sp.positions[:, 2] < zcut))
    zs2 = np.sort(sf.positions[:, 2])
    sf.set_constraint(FixAtoms(mask=sf.positions[:, 2] < zs2[1] + 1e-3))

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)
    tag = f'{nlayers}L'

    out = {}
    for label, slab in (('perfect', sp), ('faulted', sf)):
        calc = GPAW(mode=PW(ECUT), xc='PBE', kpts=KPTS,
                    txt=f'curel_{label}_{tag}.txt')
        slab.calc = calc
        opt = BFGS(slab, logfile=f'curel_{label}_{tag}.opt')
        nsteps = opt.run(fmax=fmax, steps=maxsteps)
        out[label] = slab.get_potential_energy()
        out[f'{label}_steps'] = int(nsteps)

    isf = (out['faulted'] - out['perfect']) / area * CONV
    res = dict(metal='Cu', nlayers=nlayers, natoms=len(sp), ecut=ECUT,
               kpts=list(KPTS), slip_A=float(np.linalg.norm(shift)),
               E_perfect=out['perfect'], E_faulted=out['faulted'],
               dE=out['faulted'] - out['perfect'], area=area,
               relax_steps=[out['perfect_steps'], out['faulted_steps']],
               ISF_mJm2=isf, seconds=time.time() - t0)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    layers = [6, 8, 10] if arg == 'all' else [int(arg)]
    out = [run_case(n) for n in layers]
    name = f'cu_isf_relax_{arg}.json'
    with open(name, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY ---')
    for r in out:
        print(f'{r["nlayers"]} layers relaxed: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(dE = {r["dE"]*1000:.2f} meV, steps {r["relax_steps"]}, '
              f'{r["seconds"]:.0f} s)')
    with open(f'cu_isf_relax_{arg}.done', 'w') as fh:
        fh.write('ok\n')

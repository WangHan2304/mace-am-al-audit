"""Cu ISF, PW basis, relaxed, with k-point symmetry held fixed across the pair.

Two fixes over cu_isf_relax.py:

1. k-point consistency. The perfect slab and the faulted slab do not share a
   point group, so GPAW reduces their 8x8x1 grids to different irreducible sets
   and the two energies are sampled differently. At dE ~ 20 meV that is a
   several-mJ/m^2 artifact. Point-group symmetry is switched off here; time
   reversal is kept, so both slabs use the same k set.

2. Energy convention. ASE's get_potential_energy() returns the T->0 extrapolated
   value while the BFGS optimiser works with the Mermin free energy, which differ
   by tens of meV. Both are reported; the fault energy is quoted from the
   extrapolated (0 K) values, consistent with how the Al anchors were taken.

Protocol: 1x1 (111), one fault plane, slip (a1+a2)/3, 15 A vacuum, PW 400 eV,
PBE, default Fermi-Dirac smearing, bottom two layers fixed, fmax 0.05 eV/A.

Usage:  python cu_isf_relax2.py 6
        python cu_isf_relax2.py all
"""
import json
import sys
import time
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.optimize import FIRE, BFGS
from gpaw import GPAW, PW

A_BULK = 3.615
VACUUM = 15.0
ECUT = 400
KPTS = (8, 8, 1)
CONV = 16021.8
SYM = {'point_group': False}


def run_case(nlayers, fmax=0.05, maxsteps=80):
    t0 = time.time()
    sp = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

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
                    txt=f'cu2_{label}_{tag}.txt')
        slab.calc = calc
        opt = FIRE(slab, logfile=f'cu2_{label}_{tag}.opt', maxstep=0.05)
        opt.run(fmax=fmax, steps=maxsteps)
        # both conventions, from the same converged calculator state
        res[label] = dict(
            E_0K=slab.get_potential_energy(),
            E_free=slab.get_potential_energy(force_consistent=True),
            steps=len(open(f'cu2_{label}_{tag}.opt').readlines()) - 1,
        )

    dE_0K = res['faulted']['E_0K'] - res['perfect']['E_0K']
    dE_free = res['faulted']['E_free'] - res['perfect']['E_free']
    out = dict(metal='Cu', nlayers=nlayers, natoms=len(sp), ecut=ECUT,
               kpts=list(KPTS), symmetry='point_group off',
               slip_A=float(np.linalg.norm(shift)), area=area,
               E_perfect=res['perfect'], E_faulted=res['faulted'],
               dE_0K=dE_0K, dE_free=dE_free,
               ISF_mJm2=dE_0K / area * CONV,
               ISF_free_mJm2=dE_free / area * CONV,
               seconds=time.time() - t0)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    layers = [6, 8, 10] if arg == 'all' else [int(arg)]
    out = [run_case(n) for n in layers]
    name = f'cu_isf_relax2_{arg}.json'
    with open(name, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY (0 K convention) ---')
    for r in out:
        print(f'{r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(dE = {r["dE_0K"]*1000:.2f} meV, free-energy conv '
              f'{r["ISF_free_mJm2"]:.1f}, {r["seconds"]:.0f} s)')
    with open(f'cu_isf_relax2_{arg}.done', 'w') as fh:
        fh.write('ok\n')

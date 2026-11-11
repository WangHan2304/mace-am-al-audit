"""Cu intrinsic stacking fault: corrected construction.

Three defects in the earlier cu_isf_lcao.py, in order of severity:

  1. SLIP VECTOR (the reason the old run gave ~300 mJ/m^2, six to nine times the
     literature band). The old script translated the upper half by
         [a/sqrt(3)/3, a/sqrt(3)/3, 0]      ->  |b| = 0.696 A along 45 deg
     which is neither the right length nor the right direction. The standard ISF
     translation is (a1 + a2)/3 with |b| = a/sqrt(6) = 1.476 A along <11-20>.
     This is the same bug that was found and fixed in the Al scripts; it was
     never propagated to Cu.

  2. The 10 A vacuum is raised to 15 A.

  3. The area formula and the factor-of-two were already corrected here
     (area = |a1||a2|sin60 = 5.658 A^2, gamma = dE/A with no factor 2).

Nothing else is changed relative to the Al protocol: full relaxation of all
layers, LCAO dzp, PBE, kpts 8x8x1, 1x1 (111) cell carrying exactly one fault
plane.

Usage:  python cu_isf_v2.py 6        # single thickness
        python cu_isf_v2.py all      # 6, 8, 10 in sequence
"""
import json
import sys
import time
import numpy as np
from ase.build import fcc111
from gpaw import GPAW

A_BULK = 3.615
VACUUM = 15.0
KPTS = (8, 8, 1)
CONV = 16021.8  # eV/A^2 -> mJ/m^2


def run_case(nlayers):
    t0 = time.time()
    sp = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - A_BULK / np.sqrt(6)) < 1e-6, np.linalg.norm(shift)

    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)

    tag = f'{nlayers}L'
    cp = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=KPTS, txt=f'cu_perfect_{tag}.txt')
    sp.calc = cp
    ep = sp.get_potential_energy()

    cf = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=KPTS, txt=f'cu_faulted_{tag}.txt')
    sf.calc = cf
    ef = sf.get_potential_energy()

    isf = (ef - ep) / area * CONV
    res = dict(metal='Cu', nlayers=nlayers, natoms=len(sp),
               slip_A=float(np.linalg.norm(shift)), E_perfect=ep, E_faulted=ef,
               dE=ef - ep, area=area, ISF_mJm2=isf, seconds=time.time() - t0)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    layers = [6, 8, 10] if arg == 'all' else [int(arg)]
    out = []
    for n in layers:
        out.append(run_case(n))
    with open(f'cu_isf_v2_{arg}.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY ---')
    for r in out:
        print(f'{r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(slip {r["slip_A"]:.4f} A, area {r["area"]:.4f} A2, '
              f'{r["seconds"]:.0f} s)')
    with open(f'cu_isf_v2_{arg}.done', 'w') as fh:
        fh.write('ok\n')

"""Al ISF slab: CORRECT intrinsic stacking-fault construction.

Two bugs in the earlier al_isf_lcao.py are fixed:
  1. cell area: |a1|*|a2|*sin(60) = 7.0675 A^2, not cell[0,0]*cell[1,1]*sin(60) = 6.1206.
  2. slip vector: the standard ISF translation is (a1+a2)/3 (magnitude a/sqrt(6) = 1.649 A,
     along <11-20>), not (a/sqrt3/3, a/sqrt3/3) (magnitude 1.099 A at 45 deg, which is
     neither the right length nor the right direction).

Thickness: 6, 8, 10 layers. Vacuum 15 A, kpts 8x8x1, LCAO dzp, PBE.
"""
import json
import time
import numpy as np
from ase.build import fcc111
from gpaw import GPAW

A_BULK = 4.04
VACUUM = 15.0
KPTS = (8, 8, 1)
CONV = 16021.8  # eV/A^2 -> mJ/m^2


def run_case(nlayers):
    t0 = time.time()
    sp = fcc111('Al', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Al', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    # standard ISF slip = (a1 + a2)/3, built from the actual cell vectors
    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - A_BULK / np.sqrt(6)) < 1e-6, np.linalg.norm(shift)

    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)

    tag = f'{nlayers}L'
    cp = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=KPTS, txt=f'v2_perfect_{tag}.txt')
    sp.calc = cp
    ep = sp.get_potential_energy()

    cf = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=KPTS, txt=f'v2_faulted_{tag}.txt')
    sf.calc = cf
    ef = sf.get_potential_energy()

    # gamma = dE/A: a 1x1 (111) slab carries ONE fault plane of in-plane area A.
    isf = (ef - ep) / area * CONV
    res = dict(nlayers=nlayers, natoms=len(sp), slip_A=float(np.linalg.norm(shift)),
               E_perfect=ep, E_faulted=ef, dE=ef - ep, area=area,
               ISF_mJm2=isf, seconds=time.time() - t0)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    out = [run_case(n) for n in (6, 8, 10)]
    with open('al_isf_v2.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY ---')
    for r in out:
        print(f'{r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(slip {r["slip_A"]:.4f} A, area {r["area"]:.4f} A2)')
    with open('al_isf_v2.done', 'w') as fh:
        fh.write('ok\n')

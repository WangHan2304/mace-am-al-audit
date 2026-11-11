"""Al ISF slab convergence: thickness (6/8/10 layers) + vacuum 15 A + correct (111) area.

Fixes the area bug in al_isf_lcao.py, which used cell[0,0]*cell[1,1]*sin60
(2.857 x 2.474) instead of |a1|*|a2|*sin60 (2.857 x 2.857), overestimating ISF by ~13%.
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


def run_case(nlayers, kpts=KPTS):
    t0 = time.time()
    sp = fcc111('Al', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Al', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += [A_BULK / np.sqrt(3) / 3, A_BULK / np.sqrt(3) / 3, 0.0]

    a1 = np.linalg.norm(sp.cell[0])
    a2 = np.linalg.norm(sp.cell[1])
    area = a1 * a2 * np.sin(np.pi / 3)

    tag = f'{nlayers}L_k{kpts[0]}'
    cp = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=kpts, txt=f'scf_perfect_{tag}.txt')
    sp.calc = cp
    ep = sp.get_potential_energy()

    cf = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=kpts, txt=f'scf_faulted_{tag}.txt')
    sf.calc = cf
    ef = sf.get_potential_energy()

    # gamma = dE/A: a 1x1 (111) slab carries ONE fault plane of in-plane area A.
    isf = (ef - ep) / area * CONV
    res = dict(nlayers=nlayers, kpts=list(kpts), vacuum=VACUUM, natoms=len(sp),
               E_perfect=ep, E_faulted=ef, dE=ef - ep, area=area, ISF_mJm2=isf,
               seconds=time.time() - t0)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    out = []
    for n in (6, 8, 10):
        out.append(run_case(n))
    with open('al_isf_conv.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY ---')
    for r in out:
        print(f'{r["nlayers"]} layers: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 (area {r["area"]:.4f} A2, {r["natoms"]} atoms)')
    with open('al_isf_conv.done', 'w') as fh:
        fh.write('ok\n')

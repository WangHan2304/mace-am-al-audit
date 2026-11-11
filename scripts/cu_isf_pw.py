"""Cu ISF with the plane-wave basis -- basis-set control for the LCAO run.

The corrected LCAO/dzp construction gives 18.6 mJ/m^2 at 6 layers, about half
the 34--45 mJ/m^2 literature band, with dE = 6.6 meV only. At that magnitude a
dzp basis on a 3d metal can absorb a large fraction of the fault energy, so the
same geometry is repeated in PW mode.

Everything except the basis is held fixed so the two runs differ in one
variable: 15 A vacuum, kpts 8x8x1, one fault plane, slip (a1+a2)/3.

Usage:  python cu_isf_pw.py 6 [ecut] [kpts]
"""
import json
import sys
import time
import numpy as np
from ase.build import fcc111
from gpaw import GPAW, PW

A_BULK = 3.615
VACUUM = 15.0
CONV = 16021.8


def run_case(nlayers, ecut=400, kpts=(8, 8, 1)):
    t0 = time.time()
    sp = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)
    sf = fcc111('Cu', size=(1, 1, nlayers), a=A_BULK, vacuum=VACUUM)

    shift = (np.array(sp.cell[0]) + np.array(sp.cell[1])) / 3.0
    assert abs(np.linalg.norm(shift) - A_BULK / np.sqrt(6)) < 1e-6

    top = sf.positions[:, 2] > np.median(sf.positions[:, 2])
    sf.positions[top] += shift

    area = np.linalg.norm(sp.cell[0]) * np.linalg.norm(sp.cell[1]) * np.sin(np.pi / 3)

    tag = f'{nlayers}L_e{ecut}_k{kpts[0]}'
    cp = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts, txt=f'cupw_perfect_{tag}.txt')
    sp.calc = cp
    ep = sp.get_potential_energy()

    cf = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts, txt=f'cupw_faulted_{tag}.txt')
    sf.calc = cf
    ef = sf.get_potential_energy()

    isf = (ef - ep) / area * CONV
    res = dict(metal='Cu', basis='pw', ecut=ecut, nlayers=nlayers, natoms=len(sp),
               kpts=list(kpts), slip_A=float(np.linalg.norm(shift)),
               E_perfect=ep, E_faulted=ef, dE=ef - ep, area=area,
               ISF_mJm2=isf, seconds=time.time() - t0)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ecut = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = [run_case(n, ecut, (k, k, 1))]
    name = f'cu_isf_pw_{n}L_e{ecut}_k{k}.json'
    with open(name, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('--- SUMMARY ---')
    for r in out:
        print(f'{r["nlayers"]} layers PW{ecut} k{k}: ISF = {r["ISF_mJm2"]:.1f} mJ/m2 '
              f'(dE = {r["dE"]*1000:.2f} meV, {r["seconds"]:.0f} s)')
    with open(f'cu_isf_pw_{n}L_e{ecut}_k{k}.done', 'w') as fh:
        fh.write('ok\n')

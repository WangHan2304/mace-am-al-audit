# -*- coding: utf-8 -*-
"""CHGNet version of the USF protocol comparison (Phase 3 replication).

Same construction as 13_usf_protocol.py: 12-layer (111) 1x1 cell, upper block
shifted by half a Shockley partial, double-equivalent fault divided by 2A.
Three protocols:
    none   single point, no relaxation
    zonly  in-plane x,y frozen, vertical relaxation only
    full   all positions free -- the collapse test

This exists to answer a reviewer question that the MACE-only table cannot:
whether the collapse behaviour is a property of the MACE-MP-0 checkpoints or
of the test itself. A second, architecturally different MLIP is the control.

Run inside the chgnet conda env:
    ~/miniconda3/envs/chgnet/bin/python 27_chgnet_usf.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS

HERE = Path(__file__).resolve().parent
OUT = HERE / "chgnet_usf_protocol.json"

A_AL = 4.05
D111 = A_AL / np.sqrt(3)
P1 = np.array([A_AL / 2, -A_AL / 2, 0.0])
P2 = np.array([A_AL / 2, 0.0, -A_AL / 2])
N_HAT = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
T_STEP = (P1 + P2) / 3
AREA = float(np.linalg.norm(np.cross(P1, P2)))
EV2J = 1.602176634e-19
ANCHOR_DFT = 224.0


def build_111_stack(nlayers):
    pos = np.array([((k % 3) * T_STEP) + k * D111 * N_HAT for k in range(nlayers)])
    return Atoms("Al" * nlayers, positions=pos,
                 cell=[P1, P2, nlayers * D111 * N_HAT], pbc=True)


def make_usf(perfect):
    cell = perfect.copy()
    pos = cell.get_positions()
    proj = pos @ N_HAT
    mid = (proj.min() + proj.max()) / 2
    upper = proj > mid
    pos[upper] += 0.5 * T_STEP
    cell.set_positions(pos)
    cell.wrap()
    return cell, upper


def relax(a, calc, zonly=False, fmax=0.05, steps=300):
    a.calc = calc
    if zonly:
        a.set_constraint(FixCartesian(list(range(len(a))), mask=(1, 1, 0)))
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a.get_potential_energy(), opt.nsteps


def gamma(de, area=AREA):
    return de * EV2J / (area * 1e-20) * 1e3 / 2


def main():
    import torch
    torch.set_num_threads(4)
    from chgnet.model.dynamics import CHGNetCalculator

    calc = CHGNetCalculator(use_device="cpu")
    name = "CHGNet"

    perfect = build_111_stack(12)
    d = neighbor_list("d", perfect, cutoff=3.8)
    d = np.sort(d[d > 0.01])
    print(f"nearest neighbour {d[:2]}  area {AREA:.2f} A2", flush=True)

    usf, upper = make_usf(perfect)
    upper_xy0 = usf.get_positions()[upper][:, :2].mean(axis=0)

    out = {"model": name, "anchor_USF_DFT_mJ_m2": ANCHOR_DFT,
           "construction": "12 layers (111) 1x1, upper block shifted by half a "
                           "Shockley partial, double-equivalent fault /2A, "
                           "BFGS fmax 0.05, fixed cell, CPU"}

    a = usf.copy()
    a.calc = calc
    e_usf_none = a.get_potential_energy()
    b = perfect.copy()
    b.calc = calc
    e_perf_none = b.get_potential_energy()
    out["gamma_none_mJ_m2"] = round(gamma(e_usf_none - e_perf_none), 2)

    a = usf.copy()
    e_usf_z, n_z = relax(a, calc, zonly=True)
    b = perfect.copy()
    e_perf_z, _ = relax(b, calc, zonly=True)
    out["gamma_zonly_mJ_m2"] = round(gamma(e_usf_z - e_perf_z), 2)
    out["zonly_steps"] = n_z
    out["slide_xy_zonly_A"] = round(float(np.linalg.norm(
        a.get_positions()[upper][:, :2].mean(axis=0) - upper_xy0)), 4)

    a = usf.copy()
    e_usf_full, n_f = relax(a, calc, zonly=False)
    out["gamma_full_mJ_m2"] = round(gamma(e_usf_full - e_perf_none), 2)
    out["full_steps"] = n_f
    out["slide_xy_full_A"] = round(float(np.linalg.norm(
        a.get_positions()[upper][:, :2].mean(axis=0) - upper_xy0)), 4)
    out["slide_expected_T_half_A"] = round(float(np.linalg.norm(T_STEP) / 2), 4)

    OUT.write_text(json.dumps([out], indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    sys.exit(main())

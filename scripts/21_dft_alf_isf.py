# -*- coding: utf-8 -*-
"""Al ISF slab DFT (GPAW PBE): 2x2x12 slab, layer-removal ISF"""
import json, time, numpy as np
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

t0 = time.time()
a = 4.05
d111 = a / np.sqrt(3)

# Build fcc(111) slab using bulk + surface creation
from ase.build import surface
bulk_al = bulk("Al", "fcc", a=a)
# 12-layer (111) slab, 2x2 in-plane
slab_p = surface(bulk_al, (1,1,1), 12, vacuum=12.0)
slab_p = slab_p.repeat((2,2,1))
# Center the slab
slab_p.center(vacuum=12.0, axis=2)
n = len(slab_p)
print(f"Perfect slab: {n} atoms", flush=True)

# ISF: remove layer 6, shift upper half down by d111
pos = slab_p.get_positions()
z = pos[:,2]
# Identify layers
z_unique = np.sort(np.unique(np.round(z, 2)))
print(f"Layers: {len(z_unique)} z-positions", flush=True)
# Find split point (between layer 5 and 6, 0-indexed)
mid_z = (z_unique[5] + z_unique[6]) / 2
mask_upper = z > mid_z

# ISF slab = same atoms, upper half shifted down by d111
slab_i = slab_p.copy()
pos_i = slab_i.get_positions()
pos_i[mask_upper, 2] -= d111
slab_i.set_positions(pos_i)
slab_i.center(vacuum=12.0, axis=2)

# Cell area
cell = slab_p.get_cell()
A = np.linalg.norm(np.cross(cell[0], cell[1]))
print(f"Area = {A:.3f} A^2", flush=True)

# Calc
kw = dict(mode=PW(400), xc="PBE", kpts=(8,8,1),
          occupations=FermiDirac(0.01), txt=None)

print("Running perfect slab...", flush=True)
slab_p.calc = GPAW(**kw)
e_p = slab_p.get_potential_energy()
print(f"  E_perfect = {e_p:.4f} eV ({time.time()-t0:.0f}s)", flush=True)

print("Running ISF slab...", flush=True)
slab_i.calc = GPAW(**kw)
e_i = slab_i.get_potential_energy()
print(f"  E_ISF = {e_i:.4f} eV ({time.time()-t0:.0f}s)", flush=True)

# ISF energy: periodic cell has 2 faults
gamma_isf = (e_i - e_p) / (2 * A) * 16.0218  # mJ/m2
print(f"ISF = {gamma_isf:.1f} mJ/m2", flush=True)

# USF: shift upper half by t_step in-plane
# t_step = a*sqrt(6)/6 = a/sqrt(6)
t_step = a / np.sqrt(6)
# Direction: [1,-1,0] normalized
shift = np.array([t_step/np.sqrt(2), -t_step/np.sqrt(2), 0])
slab_u = slab_p.copy()
pos_u = slab_u.get_positions()
pos_u[mask_upper] += shift
slab_u.set_positions(pos_u)

print("Running USF slab...", flush=True)
slab_u.calc = GPAW(**kw)
e_u = slab_u.get_potential_energy()
gamma_usf = (e_u - e_p) / (2 * A) * 16.0218
print(f"USF = {gamma_usf:.1f} mJ/m2 ({time.time()-t0:.0f}s)", flush=True)

r = {"gamma_isf": round(gamma_isf,1), "gamma_usf": round(gamma_usf,1),
     "n_atoms": n, "area_A2": round(A,3), "elapsed": round(time.time()-t0,1)}
json.dump(r, open("/home/cl/dft_alf_isf.json","w"), indent=2)
print(json.dumps(r, indent=2))

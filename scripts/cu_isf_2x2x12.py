import numpy as np
from ase.build import fcc111
from gpaw import GPAW
import json

a_bulk = 3.615  # Cu fcc
n_layers = 12
slab_perfect = fcc111('Cu', size=(2,2,n_layers), a=a_bulk, vacuum=15.0)
slab_faulted = fcc111('Cu', size=(2,2,n_layers), a=a_bulk, vacuum=15.0)

# Shift top half by Shockley partial: a/6<112> in (111) plane
# On fcc111: shift = (a/(2*sqrt(3)), a/(2*sqrt(3)), 0) for top half
mid_z = np.median(slab_faulted.positions[:,2])
top_mask = slab_faulted.positions[:,2] > mid_z
# Shockley partial Burgers vector on (111): a/6<112> projected in-plane
# magnitude = a/sqrt(6), direction along <110> in-plane
shift_x = a_bulk / (2*np.sqrt(3))  # ~1.04 A
shift_y = a_bulk / (2*np.sqrt(3))
slab_faulted.positions[top_mask] += [shift_x, shift_y, 0]

nat = len(slab_perfect)
print(f'Perfect: {nat} atoms, size 2x2x{n_layers}')

calc = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=(4,4,1),
            txt='cu2x12_perfect.txt', parallel={'band': 1})
slab_perfect.calc = calc
E_perfect = slab_perfect.get_potential_energy()
print(f'E_perfect = {E_perfect:.4f} eV')

calc2 = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=(4,4,1),
             txt='cu2x12_faulted.txt', parallel={'band': 1})
slab_faulted.calc = calc2
E_faulted = slab_faulted.get_potential_energy()
print(f'E_faulted = {E_faulted:.4f} eV')

# Surface area of 2x2 cell
area = slab_perfect.cell[0,0] * slab_perfect.cell[1,1] * np.sin(np.pi/3)
isf = (E_faulted - E_perfect) / (2 * area) * 16021.8
print(f'ISF = {isf:.1f} mJ/m2, area = {area:.3f} A2')

with open('cu_isf_result.json','w') as f:
    json.dump({'E_perfect': E_perfect, 'E_faulted': E_faulted,
               'area': area, 'ISF_mJm2': isf, 'n_layers': n_layers,
               'cell': '2x2x12'}, f, indent=2)
print('Done.')

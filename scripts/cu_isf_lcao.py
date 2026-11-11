import numpy as np
from ase.build import fcc111
from gpaw import GPAW
import json

a_bulk = 3.615  # Cu fcc lattice constant
slab_perfect = fcc111('Cu', size=(1,1,6), a=a_bulk, vacuum=10.0)
slab_faulted = fcc111('Cu', size=(1,1,6), a=a_bulk, vacuum=10.0)
top_mask = slab_faulted.positions[:,2] > np.median(slab_faulted.positions[:,2])
slab_faulted.positions[top_mask] += [a_bulk/np.sqrt(3)/3, a_bulk/np.sqrt(3)/3, 0]

print(f'Perfect: {len(slab_perfect)} atoms')
calc = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=(8,8,1), txt='cu_perfect_lcao.txt')
slab_perfect.calc = calc
E_perfect = slab_perfect.get_potential_energy()
print(f'E_perfect = {E_perfect:.4f} eV')

calc2 = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts=(8,8,1), txt='cu_faulted_lcao.txt')
slab_faulted.calc = calc2
E_faulted = slab_faulted.get_potential_energy()
print(f'E_faulted = {E_faulted:.4f} eV')

area = np.linalg.norm(slab_perfect.cell[0]) * np.linalg.norm(slab_perfect.cell[1]) * np.sin(np.pi/3)
# gamma = dE/A: one fault plane of in-plane area A
isf = (E_faulted - E_perfect) / area * 16021.8
print(f'ISF = {isf:.1f} mJ/m2, area = {area:.3f} A2')

with open('cu_isf_result.json','w') as f:
    json.dump({'E_perfect': E_perfect, 'E_faulted': E_faulted, 'area': area, 'ISF_mJm2': isf}, f, indent=2)
print('Done.')

import numpy as np
from ase.build import fcc111
s = fcc111('Al', size=(1, 1, 6), a=4.04, vacuum=10.0)
print('cell:')
print(s.cell)
print('natoms', len(s))
print('cell00,cell11', s.cell[0, 0], s.cell[1, 1])
a1 = np.linalg.norm(s.cell[0])
a2 = np.linalg.norm(s.cell[1])
print('a1,a2', a1, a2)
print('area_correct', a1 * a2 * np.sin(np.pi / 3))
print('area_script', s.cell[0, 0] * s.cell[1, 1] * np.sin(np.pi / 3))
print('cz', s.cell[2, 2])
print('layer_spacing', 4.04 / np.sqrt(3))

# MACE-MP-0 Applicability Domain Audit for AM Aluminum Alloys

Reproducibility package for:

> **Applicability Domain of Foundation Machine-Learned Interatomic Potentials in Additively Manufacturable Aluminum Alloys: A Cross-Scale Energy Audit and a Boundary-Calibrated Stacking-Fault Correction**
> Han Wang, Zhejiang Lab. Corresponding: wanghan2304@zhejianglab.org

## Overview

Audit of MACE-MP-0 (small/medium/large) across 19 Al-containing structures:
bulk formation energies, solute substitution, elastic constants, defect energetics (ISF/USF/coherent interface), and a five-metal fcc boundary scan (Al/Cu/Ni/Ag/Au).

Key result: MACE-MP-0 has systematic stacking-order blindness (hcp/fcc near-degenerate for Al vs DFT +31.8 meV/atom). A single-parameter bulk-anchored penalty recovers Al ISF to 7–9% (medium/large). Transferability boundary is two-dimensional: geometric ratio R ≈ 1 AND raw MACE ISF ≈ 0.

## Repository structure

```
├── structures/     # 19 Al-containing POSCAR/CIF files
├── scripts/        # Audit pipeline Python scripts
├── results/        # Raw energy tables (JSON)
├── figures/        # Paper figures (PDF + PNG)
└── data/           # Supplementary data tables
```

## Requirements

- Python 3.10+
- mace-torch, ASE, pymatgen, numpy, matplotlib
- GPAW (for local DFT anchors)

## Quick start

```bash
pip install mace-torch ase pymatgen
python scripts/01_build_structures.py
python scripts/02_run_audit.py
python scripts/fig_boundary_regen.py   # regenerates the two-dimensional boundary map
```

## Local DFT anchors

The five-metal transferability boundary does not rely on literature stacking-fault energies: Cu, Au, Al, Ag and Ni (111) intrinsic stacking faults are recomputed here with GPAW plane waves (400 eV, 8x8x1 k-points, point-group symmetry off, bottom two layers fixed, FIRE to fmax 0.05 eV/A).

| Metal | a0 (A) | Layers | ISF_DFT (mJ/m2) | R | C/DFT |
|---|---|---|---|---|---|
| Al | 4.0442 | 12 | 162.4 | 0.88 | 0.92 |
| Cu | 3.6150 | 10 | 32.8 | 1.28 | 1.45 |
| Ni | 3.5292 | 12 | 165.7 | 3.36 | 3.40 |
| Ag | 4.1489 | 10 | 17.3 | ≈0 | 0.33 |
| Au | 4.1761 | 12 | 31.0 | 1.23 | 1.90 |

Ni is the only magnetic metal and is treated spin-polarized; its thickness series (171.8, 160.3, 165.7 mJ/m2 at 8, 10, 12 layers) is not monotonic, and the 12-layer value is adopted with the non-monotonicity recorded rather than averaged. Raw slab outputs are in `results/*_slab.log`. Elastic constants are cross-checked by `scripts/26_elastic_final.py`, and the USF collapse mechanism by `scripts/27_chgnet_usf.py`.

## Citation

```bibtex
@article{wang2026mlipal,
  title={Applicability Domain of Foundation MLIPs in AM Aluminum Alloys},
  author={Wang, Han},
  journal={npj Computational Materials},
  year={2026}
}
```

## License

MIT

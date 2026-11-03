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
python scripts/01_audit_bulk.py
python scripts/22_boundary_plot.py
```

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

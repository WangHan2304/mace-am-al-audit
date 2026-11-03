# -*- coding: utf-8 -*-
"""
AM1 · Phase 1 结构库：增材制造铝合金相关相/沉淀/溶质结构
========================================================
方向 AM1（通用 MLIP 在 AM 铝合金亚稳相/沉淀相上的误差审计）Phase 1 输入。

构建原则：
1. 所有晶格常数取文献实验值（标注来源说明），空间群用标准原型位点，
   保证结构"教科书级可复核"，不依赖私有数据库。
2. Phase 1 只含体相结构（单胞/超胞）与稀释溶质超胞；界面/表面/层错
   留待 Phase 2（需 slab 构造与更多 DFT 锚点）。
3. 每种结构输出 CIF（structures/*.cif）并登记 phases.json 元数据。

覆盖对象（对应 AM Al 体系）：
- 基体/元素参考：Al(fcc)、Si(金刚石)、Mg(hcp)、Sc(hcp)、Zr(hcp)、Cu(fcc)
- 沉淀相原型：Al3Sc(L12)、Al3Zr(L12)、Al3(Sc,Zr)(L12, 2x1x1)、Al2Cu θ(C16)、Mg2Si(反萤石)
- 稀释溶质超胞：Al32 与 Al31X (X=Mg,Si,Cu,Sc,Zr)，2×2×2 fcc
"""

import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
STRUCT_DIR = ROOT / "structures"
STRUCT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- helpers
def fcc_symbols_and_scaled_positions():
    """FCC 晶胞：4 个原子，分数坐标。"""
    frac = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    return frac


def l12_positions():
    """L12（Cu3Au 型）4 原子：A 在角点 (0,0,0)，B 在面心。"""
    return np.array([
        [0.0, 0.0, 0.0],          # 角点 = 溶质原子位点（Sc/Zr）
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])


def build_fcc(a, symbol, name):
    atoms = Atoms(symbols=[symbol] * 4,
                  scaled_positions=fcc_symbols_and_scaled_positions(),
                  cell=[a, a, a], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"fcc a={a:.4f} A"
    return atoms


def build_hcp(a, c, symbol, name):
    frac = np.array([
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.5],
    ])
    atoms = Atoms(symbols=[symbol] * 2,
                  scaled_positions=frac,
                  cell=[[a, 0, 0],
                        [-a / 2, a * np.sqrt(3) / 2, 0],
                        [0, 0, c]],
                  pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"hcp a={a:.4f} c={c:.4f} A"
    return atoms


def build_diamond(a, symbol, name):
    atoms = Atoms(symbols=[symbol] * 8,
                  scaled_positions=np.array([
                      [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
                      [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
                  ]),
                  cell=[a, a, a], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"diamond a={a:.4f} A"
    return atoms


def build_l12(a, corner_symbol, face_symbol, name):
    """L12 有序结构：corner_symbol 在角点，face_symbol 在面心。"""
    symbols = [corner_symbol] + [face_symbol] * 3
    atoms = Atoms(symbols=symbols,
                  scaled_positions=l12_positions(),
                  cell=[a, a, a], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"L12 a={a:.4f} A"
    return atoms


def build_l12_mixed(a, corner_symbols, face_symbol, name):
    """L12 2×1×1 超胞，角点两个位置分别放两种溶质（如 Sc / Zr）。"""
    atoms = build_l12(a, corner_symbols[0], face_symbol, name + "_t")
    atoms = atoms.repeat((2, 1, 1))
    # repeat 后角点在 (0,0,0) 与 (1,0,0) 平移
    # 直接把第二个角点原子换成第二种溶质
    syms = list(atoms.get_chemical_symbols())
    # 找角点：位置最接近 (0,0,0) 与 (a,0,0) 的原子
    pos = atoms.get_positions()
    corner_idx = sorted(range(len(atoms)),
                        key=lambda i: (pos[i][0] - 0) ** 2 + pos[i][1] ** 2 + pos[i][2] ** 2)[0]
    # 角点原子是唯一与 (0,0,0) 重合的
    syms[corner_idx] = corner_symbols[1]
    atoms.set_chemical_symbols(syms)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"L12(2x1x1) a={a:.4f} A, corners={corner_symbols}"
    return atoms


def build_al2cu_theta(a, c, x, name):
    """Al2Cu θ (C16, I4/mcm #140)：Cu 4a (0,0,1/4)，Al 8h (x, 1/2+x, 0)，x≈0.1581。
    晶格 a=6.066, c=4.877（实验值；AFLOW 原型 A2B_tI12_140_h_a，ICSD 198177）。
    近邻核验：Cu-Cu=c/2=2.44 Å，Cu-Al≈2.6 Å。"""
    cu = np.array([
        [0.0, 0.0, 0.25], [0.0, 0.0, 0.75],
        [0.5, 0.5, 0.75], [0.5, 0.5, 0.25],
    ])
    al_fracs = [
        (x, 0.5 + x, 0.0), (0.5 - x, x, 0.0),
        (-x, 0.5 - x, 0.0), (0.5 + x, -x, 0.0),
    ]
    al = []
    for (u, v, w) in al_fracs:
        al.append([u, v, w])
        al.append([u + 0.5, v + 0.5, w + 0.5])
    al = np.array(al) % 1.0
    frac = np.vstack([cu, al])
    symbols = ["Cu"] * 4 + ["Al"] * 8
    atoms = Atoms(symbols=symbols, scaled_positions=frac,
                  cell=[[a, 0, 0], [0, a, 0], [0, 0, c]], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"C16(I4/mcm) a={a:.4f} c={c:.4f} x(Al)={x:.4f}"
    return atoms


def build_al2cu_theta_prime(a, c, name):
    """Al2Cu θ′ (I4/mmm #139)：Al 4d (0,1/2,1/4)，Cu 2a (0,0,0)。
    晶格 a=4.04, c=5.80 Å（MDPI metals-09-01037「Crystal Structures of Al2Cu Revisited」，
    与 Semantic Scholar 独立表格交叉核验一致）。6 原子/胞，Al:Cu=4:2。"""
    cu = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
    ])
    al = np.array([
        [0.0, 0.5, 0.25], [0.5, 0.0, 0.25],
        [0.0, 0.5, 0.75], [0.5, 0.0, 0.75],
    ])
    frac = np.vstack([cu, al])
    symbols = ["Cu"] * 2 + ["Al"] * 4
    atoms = Atoms(symbols=symbols, scaled_positions=frac,
                  cell=[[a, 0, 0], [0, a, 0], [0, 0, c]], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"I4/mmm a={a:.4f} c={c:.4f} (Al 4d, Cu 2a)"
    return atoms


def build_mg2si(a, name):
    """Mg2Si 反萤石型 (Fm-3m)：Si 4a，Mg 8c。a=6.351（实验值）。"""
    si = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    ])
    mg = []
    for (u, v, w) in [(0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)]:
        mg.append([u, v, w])
    mg = np.array(mg)
    mg2 = np.vstack([mg, mg + np.array([0.5, 0.5, 0.5])]) % 1.0
    frac = np.vstack([si, mg2])
    atoms = Atoms(symbols=["Si"] * 4 + ["Mg"] * 8,
                  scaled_positions=frac,
                  cell=[a, a, a], pbc=True)
    atoms.info["name"] = name
    atoms.info["lattice"] = f"anti-fluorite a={a:.4f} A"
    return atoms


def build_solute_supercell(a, symbol, x_solute, name):
    """2×2×2 fcc 超胞（32 原子），替换 1 个 Al 为溶质 X。"""
    base = build_fcc(a, symbol, name + "_base")
    sc = base.repeat((2, 2, 2))
    syms = list(sc.get_chemical_symbols())
    syms[0] = x_solute  # 把第 0 个原子换成溶质（等效位点）
    sc.set_chemical_symbols(syms)
    sc.info["name"] = name
    sc.info["lattice"] = f"fcc 2x2x2 supercell, 1 solute {x_solute} (dilute)"
    return sc


# ---------------------------------------------------------------- library
def make_library():
    lib = {}

    # --- 元素参考相（形成能/替代能的化学势参考）---
    lib["Al_fcc"] = build_fcc(4.05, "Al", "Al_fcc")
    lib["Cu_fcc"] = build_fcc(3.615, "Cu", "Cu_fcc")
    lib["Si_dia"] = build_diamond(5.431, "Si", "Si_dia")
    lib["Mg_hcp"] = build_hcp(3.209, 5.210, "Mg", "Mg_hcp")
    lib["Sc_hcp"] = build_hcp(3.309, 5.273, "Sc", "Sc_hcp")
    lib["Zr_hcp"] = build_hcp(3.232, 5.147, "Zr", "Zr_hcp")

    # --- AM 铝沉淀相原型 ---
    # Al3Sc：L12，a=4.103（实验）
    lib["Al3Sc_L12"] = build_l12(4.103, "Sc", "Al", "Al3Sc_L12")
    # Al3Zr：L12 亚稳相（Al-Sc-Zr 体系经典强化相；平衡相为 D023），a=4.08（实验/文献）
    lib["Al3Zr_L12"] = build_l12(4.080, "Zr", "Al", "Al3Zr_L12")
    # Al3(Sc0.5Zr0.5)：L12 2x1x1，角点 Sc/Zr 各一
    lib["Al3ScZr_L12"] = build_l12_mixed(4.10, ["Sc", "Zr"], "Al", "Al3ScZr_L12")
    # Al2Cu θ (C16)：a=6.066, c=4.877, x(Al)=0.1581（Eshelman 1978 / ICSD 198177）
    lib["Al2Cu_theta"] = build_al2cu_theta(6.066, 4.877, 0.1581, "Al2Cu_theta")
    # Al2Cu θ′ (I4/mmm)：a=4.04, c=5.80（MDPI metals-09-01037，双源核验）——Al-Cu 系主要硬化亚稳相
    lib["Al2Cu_theta_prime"] = build_al2cu_theta_prime(4.04, 5.80, "Al2Cu_theta_prime")
    # Mg2Si 反萤石：a=6.351（实验）
    lib["Mg2Si"] = build_mg2si(6.351, "Mg2Si")

    # --- 稀释溶质超胞（Al31X，2×2×2）---
    lib["Al32_sc"] = build_fcc(4.05, "Al", "Al32_sc").repeat((2, 2, 2))
    lib["Al32_sc"].info["name"] = "Al32_sc"
    lib["Al32_sc"].info["lattice"] = "fcc 2x2x2 supercell (reference)"
    for x in ["Mg", "Si", "Cu", "Sc", "Zr"]:
        lib[f"Al31{x}"] = build_solute_supercell(4.05, "Al", x, f"Al31{x}")

    # --- 双溶质共掺杂超胞：Al30ScZr（Scalmalloy 型 Sc+Zr 共掺杂，溶质尽量远离）---
    lib["Al30ScZr"] = build_solute_supercell(4.05, "Al", "Sc", "Al30ScZr")
    pos = lib["Al30ScZr"].get_positions()
    # 找最接近 (a,0,0) 平移的原子（即相邻晶胞原点位点），换成 Zr，使 Sc/Zr 相距≈a
    target = np.array([4.05, 0.0, 0.0])
    idx = min(range(len(lib["Al30ScZr"])), key=lambda i: np.sum((pos[i] - target) ** 2))
    syms = list(lib["Al30ScZr"].get_chemical_symbols())
    syms[idx] = "Zr"
    lib["Al30ScZr"].set_chemical_symbols(syms)
    lib["Al30ScZr"].info["name"] = "Al30ScZr"
    lib["Al30ScZr"].info["lattice"] = "fcc 2x2x2 supercell, co-doped Sc+Zr (dilute)"

    return lib


def main():
    lib = make_library()
    meta = {}
    for key, atoms in lib.items():
        cif = STRUCT_DIR / f"{key}.cif"
        write(cif, atoms, format="cif")
        comp = atoms.get_chemical_formula()
        counts = {}
        for s in atoms.get_chemical_symbols():
            counts[s] = counts.get(s, 0) + 1
        meta[key] = {
            "name": atoms.info.get("name", key),
            "formula": comp,
            "natoms": len(atoms),
            "composition": counts,
            "lattice": atoms.info.get("lattice", ""),
            "cif": str(cif.relative_to(ROOT)),
        }
    with open(ROOT / "structures" / "phases.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    n_total = sum(m["natoms"] for m in meta.values())
    print(f"结构库构建完成：{len(meta)} 个结构，共 {n_total} 原子")
    for k, m in meta.items():
        print(f"  {k:14s} {m['formula']:8s} n={m['natoms']:3d}  {m['lattice']}")


if __name__ == "__main__":
    main()

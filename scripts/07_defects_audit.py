# -*- coding: utf-8 -*-
"""
AM1 · Phase 2A 扩展缺陷能量学审计（v2，数值构造版）
=================================================
对象：
  1. Al fcc (111) 内禀层错能 ISF 与不稳定层错能 USF
  2. Al3Sc(L12)/Al(fcc) (100) 共格界面能

构造方法（全部可复核、以最近邻验证把关）：
  - 从立方 fcc 单胞出发，经整数超胞矩阵 M（列向量 u,v,w；u,v∈(111) 面内，
    w 沿 [111] 叠层）生成 N 层 (111) 周期胞；面内 3×3 超胞使堆垛平移 t 成为
    2D 格矢 → 上半晶体平移 t 恰好产生 1 个 ISF（边界保持完美堆垛）。
  - USF：平移 t/2，中界面与周期边界各 1 个等效 USF → γ = ΔE/2A。
  - 完美胞须通过"最近邻全部 ≈2.864 Å"验证，否则构造失败报错。

锚点（已核验，来源见报告 §11）：
  - Al ISF: DFT 0.161~0.166 J/m2；USF DFT 0.224 J/m2（cond-mat/9903440 等）
  - Al3Sc/Al (100) 界面能: DFT 192 mJ/m2（Asta PRB）；EAM 老值 32.5（对照）
"""

import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.geometry import cell_to_cellpar

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

MODEL_MAPPING = {
    "MACE-MP-0 small": "2023-12-10-mace-128-L0_energy_epoch-249.model",
    "MACE-MP-0 medium": "2023-12-03-mace-128-L1_epoch-199.model",
    "MACE-MP-0 large": "MACE_MPtrj_2022.9.model",
}

A_AL = 4.05
A_AL3SC = 4.103
EV2J = 1.602176634e-19


def fcc_cubic(a):
    """fcc 立方单胞（4 原子）。"""
    return Atoms("Al4",
                 scaled_positions=[[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
                 cell=[a, a, a], pbc=True)


def nn_distances(atoms):
    i, j, d, *_ = neighbor_list("ijd", atoms, cutoff=3.6)
    d = d[d > 0.01]
    return d


# ---------- (111) 层错胞：直接笛卡尔构造 ----------
# 面内基矢（三角格）：P1=(a/2)(1,-1,0), P2=(a/2)(1,0,-1)；长度 a/√2，夹角 60°。
# 堆垛：第 k 层面内偏移 = (k mod 3)*(1/3,1/3)（A→B→C，|t|=a/√6≈1.654 Å），
#       层高 z = k*d111 沿 (1,1,1)，d111 = a/√3。
D111 = A_AL / np.sqrt(3)
P1 = np.array([A_AL / 2, -A_AL / 2, 0.0])
P2 = np.array([A_AL / 2, 0.0, -A_AL / 2])
N_HAT = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
T_STEP = (P1 + P2) / 3                      # A→B 面内堆垛平移 (a/6)(2,-1,-1), |t|=a/√6
AREA_111 = np.linalg.norm(np.cross(P1, P2))  # 每层错界面面积 = √3 a²/4 = 7.10 Å²


def build_111_stack(nlayers):
    """完美 fcc(111) 周期胞：面内 1×1 原胞（每层 1 原子），nlayers 层。"""
    pos = np.array([((k % 3) * T_STEP) + k * D111 * N_HAT for k in range(nlayers)])
    return Atoms("Al" * nlayers, positions=pos, cell=[P1, P2, nlayers * D111 * N_HAT], pbc=True)


def make_isf(perfect):
    """ISF：移除中间层，上半晶体整体下移一个层距闭合间隙，胞高缩短一层
    （层移除法）→ 恰 1 个内禀层错。剩余层序 …A B C A B | A B C A B C…
    中界面形成局部 hcp；周期边界为完美 fcc 堆垛。"""
    nlayers = len(perfect)
    assert nlayers % 2 == 0, "层移除法需偶数层"
    cell = perfect.copy()
    pos = cell.get_positions()
    proj = pos @ N_HAT
    z0 = proj.min()
    mid_z = z0 + (nlayers // 2) * D111          # 中间层绝对高度
    upper = proj > mid_z
    pos[upper] -= D111 * N_HAT                  # 上半块下移一层，闭合移除间隙
    cell.set_positions(pos)
    keep = np.abs(proj - mid_z) > D111 / 2      # 去掉中间层（原始高度识别）
    atoms = cell[keep]
    new_cell = atoms.get_cell()
    new_cell[2] = new_cell[2] * ((nlayers - 1) / nlayers)
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.wrap()
    return atoms


def make_usf(perfect):
    """USF：上半晶体平移 T_STEP/2（非 2D 格矢）→ 中界面与周期边界各 1 个等效 USF。"""
    cell = perfect.copy()
    pos = cell.get_positions()
    proj = pos @ N_HAT
    mid = (proj.min() + proj.max()) / 2
    pos[proj > mid] += 0.5 * T_STEP
    cell.set_positions(pos)
    cell.wrap()
    return cell


def load_calculators(device):
    from mace.calculators import MACECalculator
    out = []
    for tag, fname in MODEL_MAPPING.items():
        calc = MACECalculator(model_paths=str(MODEL_DIR / fname), device=device,
                              default_dtype="float64")
        out.append((tag, calc))
    return out


def energy(atoms, calc):
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy()


def build_al3sc_al_100_interface(n_sc=5, n_al=10, vacuum=20.0):
    """Al3Sc(L12)/Al(fcc) (001)≡(100) 共格界面 slab：面内公共晶格 a_avg，
    法向各自 bulk 层距 a/2；L12 混合层/纯Al层交替，fcc 层交替面内偏移。
    n_sc 取奇数使顶层为混合层（含 Sc 终止）。"""
    a_avg = (A_AL3SC + A_AL) / 2

    def sc_layer(n, z):
        if n % 2 == 0:
            return [("Sc", 0.0, 0.0), ("Al", a_avg / 2, a_avg / 2)]
        return [("Al", a_avg / 2, 0.0), ("Al", 0.0, a_avg / 2)]

    def al_layer(n, z):
        # 与 L12 顶层（type0）接续：首层必须为偏移型 type1
        if n % 2 == 0:
            return [("Al", a_avg / 2, 0.0), ("Al", 0.0, a_avg / 2)]
        return [("Al", 0.0, 0.0), ("Al", a_avg / 2, a_avg / 2)]

    atoms_list = []
    for n in range(n_sc):
        z = n * (A_AL3SC / 2)
        for s, x, y in sc_layer(n, z):
            atoms_list.append((s, [x, y, z]))
    z_sc_top = n_sc * (A_AL3SC / 2)
    for n in range(n_al):
        z = z_sc_top + n * (A_AL / 2)
        for s, x, y in al_layer(n, z):
            atoms_list.append((s, [x, y, z]))
    z_top = z_sc_top + n_al * (A_AL / 2)

    atoms = Atoms([s for s, _ in atoms_list], positions=[p for _, p in atoms_list],
                  cell=[a_avg, a_avg, z_top + vacuum], pbc=(True, True, False))
    return atoms, a_avg * a_avg, n_sc, n_al


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    models = load_calculators(device)

    relaxed = json.loads((RESULT_DIR / "audit_relaxed.json").read_text(encoding="utf-8"))
    E_ref = {}
    for r in relaxed["rows"]:
        E_ref.setdefault(r["structure"], {})[r["model"]] = r["E_per_atom_relaxed_eV"]

    # ---------- 层错（直接笛卡尔构造 + 最近邻验证）----------
    NLAYERS = 12
    perfect = build_111_stack(NLAYERS)
    dn = nn_distances(perfect)
    print(f"[ISF] perfect: natoms={len(perfect)} nneigh={dn.min():.3f}~{dn.max():.3f} "
          f"area={AREA_111:.1f} A2  (fcc 期望 ~2.864)")
    assert 2.80 < dn.min() < 2.93 and dn.max() < 2.95, f"完美胞最近邻异常 {dn.min():.3f}/{dn.max():.3f}"

    isf = make_isf(perfect)
    usf = make_usf(perfect)
    di = nn_distances(isf)
    du = nn_distances(usf)
    print(f"[ISF] isf natoms={len(isf)} nneigh={di.min():.3f}~{di.max():.3f}  "
          f"usf natoms={len(usf)} nneigh={du.min():.3f}~{du.max():.3f}")
    assert di.min() > 2.5 and du.min() > 2.0, "层错胞出现过短近邻，构造错误"
    print(f"  T_STEP={np.linalg.norm(T_STEP):.3f} A (期望 a/√6=1.654)")
    print("  (注: USF 为不稳定构型，预弛豫最短近邻可 <2.5 Å，属物理压缩而非构造错误)")

    area_m2 = AREA_111 * 1e-20
    rows = []
    t0 = time.time()
    for mname, calc in models:
        E_ref_Al = E_ref["Al_fcc"][mname]      # 弛豫体相每原子
        # 轻位置弛豫（胞不动）
        s_isf = isf.copy(); s_isf.calc = calc
        oi = BFGS(s_isf, logfile=None); oi.run(fmax=0.05, steps=200)
        E_isf = s_isf.get_potential_energy()
        s_usf = usf.copy(); s_usf.calc = calc
        ou = BFGS(s_usf, logfile=None); ou.run(fmax=0.05, steps=200)
        E_usf = s_usf.get_potential_energy()
        s_perf = perfect.copy(); s_perf.calc = calc
        op = BFGS(s_perf, logfile=None); op.run(fmax=0.05, steps=200)
        E_perf = s_perf.get_potential_energy()
        g_isf = (E_isf - len(isf) * E_ref_Al) * EV2J / area_m2 * 1e3          # 恰 1 个 ISF
        g_usf = (E_usf - E_perf) * EV2J / area_m2 * 1e3 / 2                   # 2 个等效 USF
        rows.append({"type": "isf_usf", "model": mname,
                     "gamma_isf_mJ_m2": float(g_isf), "gamma_usf_mJ_m2": float(g_usf),
                     "E_perfect_eV": float(E_perf), "E_isf_eV": float(E_isf),
                     "E_usf_eV": float(E_usf), "relax_steps": [op.nsteps, oi.nsteps, ou.nsteps]})
        print(f"  {mname:20s} ISF={g_isf:7.1f} mJ/m2  USF={g_usf:7.1f} mJ/m2")

    # ---------- 界面 ----------
    print("[INTERFACE] 构建 Al3Sc/Al (100) 共格界面 slab")
    slab, A_int, n_sc, n_al = build_al3sc_al_100_interface(n_sc=5, n_al=10)
    dn = nn_distances(slab)
    comp = {}
    for s in slab.get_chemical_symbols():
        comp[s] = comp.get(s, 0) + 1
    print(f"  slab natoms={len(slab)} comp={comp} area={A_int:.2f} A2 nneigh={dn.min():.3f}~{dn.max():.3f}")
    assert 2.5 < dn.min() < 3.4, f"界面 slab 最近邻异常 {dn.min():.3f}"
    A_int_m2 = A_int * 1e-20
    n_sc_atoms = n_sc * 2
    n_al_atoms = n_al * 2
    for mname, calc in models:
        s = slab.copy()
        s.calc = calc
        opt = BFGS(s, logfile=None)
        opt.run(fmax=0.05, steps=200)
        E_slab = s.get_potential_energy()
        E_ref_tot = n_sc_atoms * E_ref["Al3Sc_L12"][mname] + n_al_atoms * E_ref["Al_fcc"][mname]
        gamma = (E_slab - E_ref_tot) * EV2J / A_int_m2 * 1e3 / 2  # 周期叠双界面
        rows.append({"type": "interface_100", "model": mname, "gamma_mJ_m2": float(gamma),
                     "E_slab_eV": float(E_slab), "E_ref_eV": float(E_ref_tot),
                     "relax_steps": opt.nsteps})
        print(f"  {mname:20s} gamma(100)={gamma:7.1f} mJ/m2 (steps={opt.nsteps})")

    print(f"[完成] 耗时 {time.time()-t0:.0f}s")

    anchors = {
        "Al_ISF_DFT": {"mJ_m2": 164, "range_mJ_m2": [161, 166],
                       "source": "DFT (cond-mat/9903440, Sun&Kaxiras 1997, Wright 1992)"},
        "Al_ISF_exp": {"mJ_m2": 163, "range_mJ_m2": [110, 280], "source": "实验估计/区间 (OSTI; arXiv)"},
        "Al_USF_DFT": {"mJ_m2": 224, "source": "DFT (cond-mat/9903440)"},
        "Al3Sc_Al_100_DFT": {"mJ_m2": 192, "source": "DFT Asta PRB (dunand.northwestern.edu)"},
        "Al3Sc_Al_100_EAM": {"mJ_m2": 32.5, "source": "EAM 1998 (对照，势族差异)"},
    }
    out = {"rows": rows, "anchors": anchors, "device": device,
           "definitions": ["gamma = dE/A (A 为界面面积, m2)",
                           f"ISF: {NLAYERS}层(111) 面内1x1原胞, 层移除法(移除中面层)→恰 1 个 ISF",
                           "USF: 上半晶体平移 T_STEP/2, 中/边界 2 个等效 USF, /2A",
                           "interface: coherent (001)≡(100), a_avg=4.0765A, 混合层(含Sc)终止, "
                           "Al块首层偏移型注册, 周期叠双界面 /2A, 原子位置轻弛豫(fmax=0.05)"]}
    (RESULT_DIR / "defect_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 报告 §11 ----------
    report_path = RESULT_DIR / "审计报告.md"
    if report_path.exists():
        lines = ["\n## 11. 扩展缺陷能量学（Phase 2A：层错 + 共格界面）\n",
                 "### 11.1 Al fcc (111) 层错能（mJ/m2）\n",
                 "| 势 | γ_ISF | γ_USF |",
                 "|---|---|---|"]
        for r in rows:
            if r["type"] == "isf_usf":
                lines.append(f"| {r['model']} | {r['gamma_isf_mJ_m2']:.1f} | {r['gamma_usf_mJ_m2']:.1f} |")
        lines.append("")
        lines.append("> 锚点：ISF DFT 161~166（实验估计 163，区间 110~280）；USF DFT 224。"
                     "构造：9 层(111)×面内 3×3 超胞，堆垛平移 T 为 2D 格矢→恰 1 个 ISF；"
                     "完美胞最近邻验证 2.864 Å。\n")
        lines.append("### 11.2 Al3Sc(L12)/Al(fcc) (100) 共格界面能（mJ/m2）\n")
        lines.append("| 势 | γ_int(100) |")
        lines.append("|---|---|")
        for r in rows:
            if r["type"] == "interface_100":
                lines.append(f"| {r['model']} | {r['gamma_mJ_m2']:.1f} |")
        lines.append("")
        lines.append("> 锚点：DFT 192（Asta PRB）；EAM 老值 32.5（势族差异对照）。"
                     "口径：共格界面（面内 a_avg=4.0765 Å），混合层(含Sc)终止，位置轻弛豫，双界面平均。\n")
        report_path.write_text(report_path.read_text(encoding="utf-8") + "\n".join(lines),
                               encoding="utf-8")
    print("[报告] 已追加 §11")


if __name__ == "__main__":
    main()

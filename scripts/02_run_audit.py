# -*- coding: utf-8 -*-
"""
AM1 · Phase 1 审计执行：MACE-MP-0 (small/medium/large) 对 AM 铝相结构库的能量/力/弹性审计
============================================================================================
读取 structures/phases.json 结构库，对每个势计算：
  - 总能量 E_tot / 每原子能量 E_per_atom
  - 最大原子力 max|F|（结构是否在该势下近平衡的初检）
  - 立方/四方相的小应变弹性张量（有限差分 + 最小二乘，Voigt）
输出 results/audit_raw.json + results/audit_raw.csv（逐结构×逐势原始行）。

模型下载容错：
  - MACE：HF 官方 → hf-mirror 镜像 → 本地 models/ 目录
  - CHGNet：官方 GitHub release → 镜像 → 本地 models/ 目录
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRUCT_DIR = ROOT / "structures"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

EV2GPA = 160.21766208
STRAIN_ETA = 0.005

# 每个势的应力单位换算系数：检测后写入，全部结构复用
STRESS_SCALE = {}


def detect_stress_scale(atoms, calc):
    """探测应力单位。ASE 约定为 eV/Å³（MACE 默认）；若数值量级已是 GPa
    （如部分 CHGNet 版本），则不换算。返回换算系数（×GPa）。"""
    defo = atoms.copy()
    cell = np.array(atoms.cell)
    e = np.array([[STRAIN_ETA, 0, 0], [0, 0, 0], [0, 0, 0]])
    defo.set_cell(np.dot((np.eye(3) + e), cell), scale_atoms=True)
    defo.calc = calc
    raw6 = defo.get_stress(voigt=True)
    val = raw6[0]
    if abs(val) > 0.05:   # 已是 GPa 量级（≈C11·η≈0.5）
        return 1.0, "GPa"
    return EV2GPA, "eV/A3"

# ------------------------------------------------------------------ models
def download_file(url, dest, mirrors=(), timeout=120):
    import ssl
    import urllib.request

    # 本机为 TLS 拦截代理环境，Python 默认证书校验不认根证书；
    # 权重为公开文件，使用 unverified 上下文下载（无凭据风险）。
    ctx = ssl._create_unverified_context()
    urls = [url] + list(mirrors)
    last_err = None
    for u in urls:
        try:
            print(f"  下载 {u}")
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  失败: {e}")
    print(f"  !! 全部源失败: {last_err}")
    return False


def load_mace_calculator(model_tag, device):
    """加载 MACE-MP-0（small/medium/large）。优先本地 models/，其次走下载。
    权重官方源：GitHub releases（ACEsuit/mace-mp），带 ghproxy 镜像回退。"""
    mapping = {
        "MACE-MP-0 small": "2023-12-10-mace-128-L0_energy_epoch-249.model",
        "MACE-MP-0 medium": "2023-12-03-mace-128-L1_epoch-199.model",
        "MACE-MP-0 large": "MACE_MPtrj_2022.9.model",
    }
    fname = mapping[model_tag]
    local = MODEL_DIR / fname
    base = f"https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/{fname}"
    if not local.exists():
        ok = download_file(
            base, local,
            mirrors=(
                f"https://ghproxy.net/{base}",
                f"https://mirror.ghproxy.com/{base}",
                f"https://hf-mirror.com/ACEsuit/mace-mp-0/resolve/main/{fname}",
            ))
        if not ok:
            raise RuntimeError(f"无法下载 MACE 模型 {model_tag}")
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(local), device=device,
                          default_dtype="float64")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ------------------------------------------------------------------ helpers
def deformed_cells(atoms, eta):
    """生成应变模式 → 变形原子簇。模式含 6 单轴 + 3 双轴，各 ±eta。"""
    modes = [
        (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0), (0, 1, 1, 0, 0, 0),
    ]
    out = []
    for m in modes:
        for sgn in (+1.0, -1.0):
            e = np.array([
                [m[0], m[5] / 2, m[4] / 2],
                [m[5] / 2, m[1], m[3] / 2],
                [m[4] / 2, m[3] / 2, m[2]],
            ]) * (sgn * eta)
            cell = np.array(atoms.cell)
            new = np.dot((np.eye(3) + e), cell)
            a = atoms.copy()
            a.set_cell(new, scale_atoms=True)
            out.append((m, sgn, e, a))
    return out


def elastic_tensor(atoms, calc, log, scale):
    """小应变有限差分求弹性张量（Voigt 6×6，GPa）。"""
    X, Y = [], []
    for mode, sgn, e, defo in deformed_cells(atoms, STRAIN_ETA):
        defo.calc = calc
        raw = defo.get_stress(voigt=True)
        stress_gpa = np.array(raw) * scale
        # 应变 Voigt 向量（与 ASE stress 顺序一致：xx,yy,zz,yz,xz,xy；剪应变用工程应变）
        strain_vec = np.array([e[0, 0], e[1, 1], e[2, 2],
                               2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])
        X.append(strain_vec)
        Y.append(stress_gpa)
    X = np.array(X)
    Y = np.array(Y)
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return C, (X, Y)


def cubic_moduli(C):
    C11, C12, C44 = C[0, 0], C[0, 1], C[3, 3]
    KV = (C11 + 2 * C12) / 3.0
    GV = (C11 - C12 + 3 * C44) / 5.0
    GR = 5.0 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    GH = (GV + GR) / 2.0
    zener = 2 * C44 / (C11 - C12)
    return {"C11": float(C11), "C12": float(C12), "C44": float(C44),
            "B_Voigt": float(KV),
            "G_Hill": float(GH), "Zener_A": float(zener)}


def run_audit():
    with open(STRUCT_DIR / "phases.json", encoding="utf-8") as f:
        meta = json.load(f)
    from ase.io import read

    atoms_all = {key: read(STRUCT_DIR / f"{key}.cif", format="cif")
                 for key in meta}

    device = pick_device()
    print(f"[设备] {device}")
    calc_mace_s = load_mace_calculator("MACE-MP-0 small", device)
    print("[模型] MACE-MP-0 small 就绪")
    calc_mace_m = load_mace_calculator("MACE-MP-0 medium", device)
    print("[模型] MACE-MP-0 medium 就绪")
    calc_mace_l = load_mace_calculator("MACE-MP-0 large", device)
    print("[模型] MACE-MP-0 large 就绪")

    models = [
        ("MACE-MP-0 small", calc_mace_s),
        ("MACE-MP-0 medium", calc_mace_m),
        ("MACE-MP-0 large", calc_mace_l),
    ]

    ELASTIC_CUBIC = {"Al_fcc", "Si_dia", "Mg2Si", "Al3Sc_L12", "Al3Zr_L12"}
    ELASTIC_TETRA = {"Al2Cu_theta", "Al2Cu_theta_prime"}

    rows = []
    t0 = time.time()
    # 先对每个势探测应力单位（Al fcc 单轴应变量级判别）
    al_fcc = atoms_all["Al_fcc"]
    for model_name, calc in models:
        scale, unit = detect_stress_scale(al_fcc, calc)
        STRESS_SCALE[model_name] = scale
        print(f"[单位] {model_name}: {unit} (scale={scale:.3f})")

    for key, atoms in atoms_all.items():
        phase = meta[key]["name"]
        for model_name, calc in models:
            atoms.calc = calc
            log = {"stress_unit": "GPa" if STRESS_SCALE[model_name] == 1.0 else "eV/A3"}
            t1 = time.time()
            E = atoms.get_potential_energy()
            F = atoms.get_forces()
            maxf = float(np.abs(F).max())
            row = {
                "structure": key,
                "phase": phase,
                "formula": meta[key]["formula"],
                "natoms": len(atoms),
                "model": model_name,
                "E_tot_eV": float(E),
                "E_per_atom_eV": float(E / len(atoms)),
                "maxF_eV_per_A": maxf,
            }
            if key in ELASTIC_CUBIC or key in ELASTIC_TETRA:
                try:
                    C, (X, Y) = elastic_tensor(atoms, calc, log, STRESS_SCALE[model_name])
                    row["C_matrix_GPa"] = C.tolist()
                    row["stress_unit_detected"] = log["stress_unit"]
                    if key in ELASTIC_CUBIC:
                        row["moduli"] = cubic_moduli(C)
                except Exception as e:  # noqa: BLE001
                    row["elastic_error"] = str(e)
            print(f"  {key:14s} {model_name:20s} E={row['E_per_atom_eV']:+.4f} eV/atom "
                  f"maxF={maxf:.4f} eV/A {time.time()-t1:.1f}s")
            rows.append(row)
    print(f"[完成] {len(rows)} 行，耗时 {time.time()-t0:.0f}s")

    with open(RESULT_DIR / "audit_raw.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "meta": meta,
                   "models": [m[0] for m in models],
                   "device": device}, f, ensure_ascii=False, indent=2)

    import csv
    with open(RESULT_DIR / "audit_raw.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["structure", "phase", "formula", "natoms", "model",
                "E_tot_eV", "E_per_atom_eV", "maxF_eV_per_A"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    run_audit()

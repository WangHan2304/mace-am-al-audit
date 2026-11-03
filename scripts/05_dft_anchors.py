# -*- coding: utf-8 -*-
"""
AM1 · DFT/实验锚点表
====================
生成 results/dft_anchors.json：各相的形成能参考值（eV/atom）。

来源分级（每个值带 source + kind）：
  - MP   ：Materials Project（GGA-PBE）公开页面值
  - LIT  ：同行评议文献第一性原理值
  - EXP  ：实验量热值
  - LOCAL：用户在本机/工作站用 VASP/GPAW 自算后填入（模板预留）

用法：
  1. 本脚本首次运行生成模板（含已核验的公开锚点）；
  2. 若用户在 Linux 工作站自跑 DFT，把结果按同结构 key 填入
     results/dft_anchors.json 的 "local" 字段，脚本会合并输出。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

# 已核验公开锚点（来源与日期见 report 引用；值全部来自检索结果原文）
ANCHORS = {
    # phase_key: {"formation_eV_per_atom": float, "source": str, "kind": "MP"/"LIT"/"EXP", "url": str}
    "Al_fcc": {"formation_eV_per_atom": 0.0, "source": "元素参考态", "kind": "REF"},
    "Al3Sc_L12": {"formation_eV_per_atom": -0.443,
                  "source": "PMC6539934 PBE 第一性原理", "kind": "LIT",
                  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6539934/"},
    "Al3Zr_L12": {"formation_eV_per_atom": -0.532,
                  "source": "PMC5951520 PBE 第一性原理", "kind": "LIT",
                  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5951520/"},
    "Al2Cu_theta": {"formation_eV_per_atom": -0.166,
                    "source": "Materials Project mp-998 (GGA-PBE)", "kind": "MP",
                    "url": "https://legacy.materialsproject.org/materials/mp-998/"},
    "Mg2Si": {"formation_eV_per_atom": -0.155,
              "source": "Materials Project mp-1367 (GGA-PBE)", "kind": "MP",
              "url": "https://legacy.materialsproject.org/materials/mp-1367/"},
    "Mg2Si_exp": {"formation_eV_per_atom_min": -0.31, "formation_eV_per_atom_max": -0.22,
                  "source": "实验量热区间（文献汇总, arXiv 1309.7246）", "kind": "EXP",
                  "url": "https://arxiv.org/pdf/1309.7246v1"},
    "Al3ScZr_L12_note": {"note": "Al24Sc6Zr2 (Zr 6.25 at%) ΔHf=-0.456 eV/atom 可作定性对照",
                         "source": "PMC6539934", "kind": "LIT",
                         "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6539934/"},
}


def main():
    out = {"anchors": ANCHORS,
           "local": {},   # 预留：本地 DFT 自算结果（VASP/GPAW），见 README
           "notes": "kind 分级：MP=Materials Project GGA-PBE 公开值；LIT=同行评议文献；EXP=实验；"
                    "LOCAL=工作站 DFT 自算待填。"}
    path = RESULT_DIR / "dft_anchors.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("local"):
            out["local"] = existing["local"]
            print(f"[合并] 已保留 local 字段 {len(existing['local'])} 条")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写入] {path}")
    for k, v in ANCHORS.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()

# Round 6 结论简报

**Branch:** `cursor/femtools-remaining-d551`  
**pytest at Round-6 close:** 183 passed / 3 skipped; ruff clean; 128 top-level exports (112 names).  
Round-5 close was 109/3. Round 6 landed every frozen R5+ API except pyvista extras (still R5+) and the explicit N/A binaries/DAQ.

## 各代理结论

| ID | 结果 |
|----|------|
| R6-F1 | PRODUCT_MAP/SOTA/ARCHITECTURE 先标 R6-wip；无未解析 `__all__`。编排器在内核落地后升为 R6 并补 13 个稳定 lazy export |
| R6-F2 | `read_inp`/`write_inp`/`read_k` 文本子集；HEX8/QUAD4/BEAM2 可 `assemble_km`；未知关键字聚合警告 |
| R6-F3 | 既有 8 examples 仍 PASS；算法笔记 UQ/shape/SSI-DATA/INP-K；新 example 因当时内核未 import 未加 |
| R6-F4 | plotly 可选后端；`read-mesh` 按后缀懒加载；修复非方 MAC、`update` 回溯泄漏、`--mode-index` 越界 |
| R6-O1 | 按节点转动标架：倾斜平板 7→6 个刚体模态；轴对齐 K 位相同；HEX8 98.6% 不变。EAS-9 ≡ Wilson–Taylor，未猜 EAS-30 |
| R6-O2 | `modal_strain_energy`/`modal_kinetic_energy`；修复奇异伪逆、模态谐波丢阻尼、1-D C 广播、DC NaN、MBA 映射 |
| R6-O3 | `nmd`/`macx`（实模态 MACX≡MAC 逐位）；修复 `candidate_dofs` exclude 与 `select_exciters` 坐标折叠 |
| R6-O4 | UQ 一阶协方差 + 必选 seed MC；两杆拱 f1 +304%；SSI-DATA 噪声 SDOF 0.12% / MAC≈1 |
| R6-G1 | CDB ETBLOCK/COMPACT/RMORE/BEAM3 与 H1/H2=γ² 回归 |
| R6-G2 | 边界探针：原 7/7；新探针在内核缺席时 skip，落地后应变绿 |

## 仍 N/A / R5+

- N/A：NI DAQ；Nastran OP2 / ANSYS RST / Abaqus ODB 二进制
- R5+：pyvista 可视化 extras
- SOTA.md §10：HEX8 畸变软化、`bbar` 薄弯过软（Wilson–Taylor 已是 EAS-9，畸变需 EAS-30）

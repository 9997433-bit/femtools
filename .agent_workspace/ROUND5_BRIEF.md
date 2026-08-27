# Round 5 结论简报

**Branch:** `cursor/femtools-remaining-d551`  
**pytest:** 109 passed / 3 skipped / 0 failed; `ruff check src/femtools tests examples` clean.

Round 5 是缺陷猎取 + 文档/example 对齐轮，不是新能力冻结轮。十个代理全部完成。

## 各代理结论

| ID | 结果 |
|----|------|
| R5-F1 | 30 个 R4 名从 provisional 升为稳定 lazy export；`__all__` 115；PRODUCT_MAP 18 行 R4-wip→R4 |
| R5-F2 | `pch.py` 无再现缺陷；`cdb.py` 六个已复现缺陷全部修复（ETBLOCK `a` 描述符、EBLOCK COMPACT、R 档案形、续行、BEAM3 实常数、RMORE 对齐） |
| R5-F3 | 8/8 examples PASS；Rubin 走 `free_interface_assembly`；SSI `order=20, n_modes=3` |
| R5-F4 | CLI `report-mac`/`reduce`/`estimate-frf` 冒烟通过；SEREP compare 与 `--noverlap` 对齐 |
| R5-O1 | `solve_static(enforced=)` 从自由集抽出被约束 DOF；倾斜平板钻孔机构改为运行时警告（合同：按节点转动标架） |
| R5-O2 | Nigam–Jennings 刚体/近零频、`q0` 长度、Craig–Bampton interior 补集、`lower_residual=0` 的 `0*inf` 四处修复；Rubin 0.028% 不变 |
| R5-O3 | 七处输入层缺陷（lumped mass 动态展开、Hungarian 配对、mask 等）；核数值 1e-15 级未动 |
| R5-O4 | `mount_k` 三种数值输入、MPE 别名、更新参数选择等五处修复；H1/H2=γ² 8.9e-16；10% E 收回 6.5e-10 |
| R5-G1 | Round 4 核测试去掉 `importorskip`；缺模块改为收集失败 |
| R5-G2 | `scripts/probe_boundaries.py` 7/7 PASS（含 Guyan、H1） |

## 数值基线（合并后）

- 金标：轴向杆离散精确值；BEAM2 悬臂弯曲 vs EB ≤2.5e-6；HEX8 Wilson–Taylor 悬臂尖端 **98.6%**
- Guyan 凝聚 6e-16；IRS 频率 4.5e-8；SEREP 从传感器恢复 ~1e-15
- 完整谱 eigh 残差 5.7e-15
- Rubin 40 质量链前 8 阶 **0.028%**
- CDB BEAM3 悬臂频率相对 Euler–Bernoulli ≤3%（F2 端到端）

## 仍 N/A / R5+（交 Round 6）

- N/A：NI DAQ；Nastran OP2 / ANSYS RST / Abaqus ODB 二进制
- R5+：`optimization.shape`、`updating.uq`、Abaqus INP / LS-DYNA K 文本、SSI-DATA、扩展 MAC 指标、plotly/pyvista extras
- 合同缺口：倾斜平板钻孔机构（需按节点转动标架，O1 已警告未根治）
- HEX8 畸变软化、`bbar` 薄弯过软：SOTA.md §10，非本轮必做

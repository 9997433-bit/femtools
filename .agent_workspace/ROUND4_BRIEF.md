# Round 4 结论简报

**Branch:** `cursor/femtools-remaining-d551`  
**pytest:** 65 passed / 3 skipped / 0 failed; ruff clean.

## 已落地（相对 R3 的 R3+ 清单）

| 能力 | 结果 |
|------|------|
| Guyan / IRS / SEREP | Guyan 静力精确 6e-16；IRS 频率偏差 1.35e-3→4.5e-8；SEREP 从 8 传感器恢复未测 DOF ~1e-15 |
| 完整谱 `eigh` | SPD 全谱残差 3.5e-8→5.7e-15；FRF ~9e-14 |
| 复模态 QZ | 2-DOF 非比例阻尼 5e-16；Fan–Lin–Van Dooren 缩放 |
| Rubin/MacNeal CMS | 40 质量链对切 Rubin 前 8 阶 **0.028%**；完整基 6.7e-13 |
| PSD 随机响应 | SDOF RMS vs 闭式 1.4e-8 |
| FMAC / 激振点 / Procrustes | 共线传感器 underdetermined 已标记 |
| H1/H2 | `H1/H2=γ²` 到 8.9e-16 |
| SSI-COV | 4 模态 2% 噪声频率误差 3.2e-4，MAC 0.99995 |
| FRF 修正 | 干净 log\|H\| 精确收回；5% 噪声 0.228% |
| Punch / CDB | 文本子集；CDB→solve→pch 1e-7 |
| CLI | `report-mac`, `reduce`, `estimate-frf` |

## Round 5 重点

1. 新 examples（guyan_serep / cms_rubin / h1_ssi）对着真实内核实跑
2. PRODUCT_MAP R4-wip → R4 done；补齐顶层 lazy export（cms_free 等）
3. `ComplexModalResult.modes_complex` 别名已由编排器补上
4. 仍 N/A：DAQ、OP2/RST/ODB 二进制；R5+：shape、Abaqus INP、UQ

# Round 1 结论简报

**Status:** Round 1 merged to `cursor/femtools-sota-d551` (PR #1).  
**Test baseline after glue fix:** 17 passed, 3 skipped (`tests/perf`, `FEMTOOLS_PERF=1`), 0 failed.

## 已实现功能

| 模块 | 能力 | 来源 |
|------|------|------|
| `core` | FEModel/Node/Element/Material/Property/SPC/sets/units/coords/results/validation | [R1-F2](bc-6f6a7d5a-4380-5cfe-a001-7d77103402de) |
| `io` | UNV 151/164/15/2411/2412/82/55/58, BDF GRID/C* /MAT1/P*/SPC/FORCE, `.ftproj` | R1-F2 |
| `fea` | BAR2/BEAM2/TRUSS2D/TRIA3/QUAD4/TET4/HEX8/MASS/SPRING/DAMPER, sparse K/M/C, SPC 消元, eigsh, 质量归一化 | [R1-O1](bc-7d8a934b-c46a-5d1d-8e19-b5c14d6ae811) |
| `dynamics` | modal/direct FRF, harmonic ODS, residual vectors, Craig–Bampton, MBA/SDM, FBA, time history | [R1-O2](bc-6dd05296-3d45-599f-b7ec-8291574c8e85) |
| `correlation`/`pretest` | MAC/CoMAC/POC, Hungarian pairing, FRAC/CSAC, EFI, NKE, mass loading | [R1-O3](bc-baec8f19-fb81-586d-821b-6c1679afa9b6) |
| `updating`/`optimization`/`mpe`/`rbpe` | WLS/Bayesian 修正, 力识别, SLSQP size, SIMP, LHS, p-LSCF/FDD/LSCE, 刚体惯量 | [R1-O4](bc-8f7b5aec-1341-5854-8dd2-52632a5b7028) |
| `script`/`cli`/`gui`/`viz` | FSL 命令语言, typer CLI, FastAPI+stdlib GUI, matplotlib | [R1-F4](bc-d249eb7e-dd53-5e8e-b6af-fa20736d4ad1) |
| docs/CI | 架构、产品映射、SOTA 文献、算法公式、验收表、GitHub Actions | [R1-F1](bc-1607c7fa-f177-5b9f-bea9-77289bf37d34) [R1-F3](bc-44f88846-aa6d-5e5c-98c8-ff1ec5447d96) |
| tests/bench | 金标+MAC+EFI+更新+IO+CLI；边界探针 | [R1-G1](bc-524d5100-ec79-5667-ad27-9d8a42716737) [R1-G2](bc-dffdb5c6-76a4-514c-b9d2-5c3f736455f7) |

## 数值基线（代理自测 + 合并后 pytest）

- 轴向杆频率 rel err ~1e-16；悬臂梁前三阶相对 EB 理论 ≤0.025%
- ΦᵀMΦ−I max ~1e-15
- 2-DOF 全模态 FRF modal vs direct L2 ~1e-14
- EFI 玩具 off-diag MAC 0.095 < 0.15
- E 偏 10% 的更新恢复到 ~1e-7%（无噪声）
- pytest 默认套件已绿（perf 跳过）

## 遗留缺陷

1. **HEX8 弯曲剪切闭锁**：单层厚度仅约 66% 参考挠度（O1）
2. **UNV 不携带材料/属性卡**；BDF 丢 TET10/HEX20 中节点
3. **CLI `--damping` 仅 modal zeta**；`plot_mode` 只画平动；GUI 无上传
4. **`femtools.__init__` 未再导出公共 API**
5. **examples/** 尚未在合并树上实跑验收
6. **CI** 仍对 pytest exit-5 宽容（应关闭）
7. **core.errors 异常层次** 文档有、代码可能未齐
8. **FEA Protocol vs 真实 FEModel**：属性字段别名（Iy/IY、extra bag）需端到端钉死
9. **模态截断 FRF**：fmax 必须是保留模态最高频，否则 5% 带内容差失真（O2）
10. **DAQ 硬件** 明确不做；仅 synthetic

## 性能瓶颈

- 装配曾是 O(nelem×nnodes)，O1 已改为索引；80×80 板模态约 1.9s（本环境）
- EFI/MAC 已向量化；大规模（>1e5 DOF）未做并行/外存
- 直接 FRF 逐频分解未做 Pade/频段复用

## Round 2 攻坚重点

1. 端到端打通 core↔fea↔cli↔examples，公开 API 从 `femtools` 顶层导出
2. HEX8 抗闭锁（incompatible modes 或选择积分）
3. examples + ACCEPTANCE 表在合并树上全部可跑
4. 扩展测试：cantilever golden、截断 FRF 5% 带、IO 双参数序、更新有噪声
5. 关掉 CI exit-5；补 mypy/coverage 可选
6. GUI 预载模型与 CLI 阻尼类型
7. 文档 PRODUCT_MAP 按合并现实重打 R1/R2 标签

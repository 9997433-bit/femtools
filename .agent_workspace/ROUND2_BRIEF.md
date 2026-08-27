# Round 2 结论简报

**Merged to** `cursor/femtools-sota-d551`.  
**Baseline:** pytest 17 passed / 3 skipped / 0 failed; `ruff check .` clean after I001 autofix.

## 演进对比（相对 Round 1）

| 项 | Round 1 | Round 2 |
|----|---------|---------|
| 公开 API | 仅子包导入 | `femtools` 顶层 PEP 562 惰性导出 75 names |
| HEX8 弯曲 | 单层 ~64–66% 参考挠度 | Wilson/Taylor 非协调模态，单层 **98.6%**；自由-自由仍 6 个刚体模态 |
| `solve_modes` | Rayleigh–Ritz 会吞弹性模态 | 只增广新方向；与 dense 参考 ≤2.3e-11 |
| FRF 截断验收 | 误用父模型 fmax → 13% | `retained_band`；20 模态 **rel L2 3.0%** < 5% |
| 模型修正 | 与 example/CLI 签名不一致 | 描述符/`measured`/`analytic` 接通；10% E 恢复 7.9e-8% |
| examples | 4/5 对草稿 API 失败 | **5/5 PASS**（悬臂 2.55e-4，FRF 1.4%，EFI MAC 0.05，E 恢复 1.8e-10） |
| core 属性 | 大小写别名不稳 | `IY`/`I1` 等规范化 + `attrs` bag |
| UNV 材料 | 丢失 | 私有 dataset 30000 JSON 回写（第三方仍不可见） |
| CLI/GUI | `.ftproj` 把 Project 交给求解器 | 统一 loader；阻尼 spec；GUI `/api/load` |
| 相关分析 | 0-based vs 1-based DOF 断裂 | `DOFMap` 识别 fea 映射；平动切片 |
| CI | pytest exit-5 放行 | 严格 pytest；非阻塞 mypy |
| 边界探针 | 模块缺失则 skip | **5 PASS / 0 WARN / 0 FAIL** |

## 潜在边界风险

1. HEX8 非协调模态在畸变（skew→0.4）时挠度比 0.98→0.36，仍优于闭锁单元但需网格质量约束。
2. `bbar` 薄弯曲过软（比 2.13），仅适用于近不可压厚件。
3. 截断 FRF：低于约 20 模态或 zeta=0.02 时 5% 带可能达不到（合约是 20 模态陈述）。
4. UNV 材料靠私有 dataset，第三方读取器会忽略。
5. CLI `pretest` 仍可能给旋转/SPC DOF 打分（O3 已提供 `translational_dofs`）。
6. ACCEPTANCE 仍有 12 个未实测用例。
7. mypy 约 65 条 informational；`import femtools` 惰性导出对静态检查依赖 TYPE_CHECKING。

## SOTA 验收差距（Round 3 冲刺）

1. 文档 `docs/algorithms/fea.md` 仍写 HEX8 单层过刚 — 需按 98.6% 改写。
2. CLI pretest 改用平动切片。
3. 补金标：HEX8 patch/bending、`import femtools` 冒烟、截断 FRF 正式测试（G1 已加强但需确认合并后仍绿）。
4. `direct_frf` 可选接受 FEModel 以去掉 example 样板。
5. 交叉核验：examples + pytest + probe 全绿；CI 与 ruff 保持干净。
6. DAQ 硬件仍 N/A。Guyan/IRS/SEREP、OP2 驱动、SSI 标为后续。

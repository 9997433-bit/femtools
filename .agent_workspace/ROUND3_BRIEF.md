# Round 3 结论简报（最终）

**Merged to** `cursor/femtools-sota-d551`.  
**Acceptance:** pytest **21 passed / 3 skipped / 0 failed**; `ruff check .` clean; examples **5/5 PASS**; probes **5 PASS**.

## 本轮收敛

| 项 | 结果 |
|----|------|
| HEX8 文档 | 单层弯曲 ~98.6%；畸变归因改为梯形 Jacobian 变化，非平行四边形剪切 |
| HEX8 健壮性 | 单元内 det J 变号 / 近零体积拒绝；折叠单元不再静默给出“健康”刚度 |
| `direct_frf` | 可接受 FEModel；`(K,M)` 签名不变；截断 3.91e-2、全模态矩阵路径 ~6e-14 |
| CLI pretest | 只用平动自由 DOF（11Z/11Y…），不再把 RY/RZ 排到前十 |
| 相关分析 | 布尔 mask 不再被当成 0/1 下标；`"RZ"`/`"-Z"` 整词解析；`achieved_fraction` 与 include 一致 |
| 力识别 | 方阵不再被 GCV 正则化成近零力；L-curve 角点修复 |
| UNV | 非 SI 模型 roundtrip 保留单位，不再把 E=210000 MPa 当成 Pa |
| 公开 API | 惰性导出含 `core.errors`（84 names）；`test_public_api` + HEX8 verification 测试 |
| PRODUCT_MAP | R2 已实现 vs R3+ 未实现拆开，避免把 Guyan/OP2/SSI 标成已完成 |

## 仍超出本仓库范围 / 已知限制

- DAQ 硬件、Nastran OP2 / ANSYS RST 驱动、Guyan/IRS/SEREP、free-interface CMS、SSI
- HEX8 梯形畸变仍变软；`bbar` 薄弯曲过软
- 全谱 `solve_modes` 在大条件数下不如 `eigh`（完整基 FRF 到 ~1e-10 而非 1e-14）
- UNV 材料仅私有 dataset 30000；第三方读取器忽略
- mypy informational 债务；ACCEPTANCE 仍有未单测覆盖的条目

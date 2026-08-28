MODEL_SLUG: claude-fable-5-thinking-xhigh

# AUDIT-R10-GAPS — Round 10 之后的差距审计（Cycle D Round 11 输入）

审计对象:`main` @ `6fb7b4e`(Round 10 关闭后)。只读审计,不实现、不开 PR。

## 0. 实测证据(本树,2026-08-28)

| 探测 | 结果 |
|---|---|
| `_EXPORTS` 数量 | **149**(与 `bec1045` 提交说明一致);149 个顶层懒导出全部可解析,无一失败 |
| Round-10 冻结名 | `tet10`、`recover_spr`、`read_pch_stress`、`era`、`expanded_mac`、`residual_flexibility` 模块级 + 顶层懒导入全部成功 |
| R1–R9 抽查 | 26 个 PRODUCT_MAP 符号(`solve_modes`、`apply_mpc`、`mapped_mac`、`static_stress_response`、三个文本 driver、`NastranPunchDriver().read_static` 等)全部可用 |
| 单元注册表 | `BAR2, BEAM2, DAMPER, HEX8, MASS, QUAD4, SPRING, TET4, TET10, TRIA3, TRUSS2D`;TET10 带 `CTETRA10/TETRA10/C3D10/TET10N` 别名 |
| 完整 pytest | **590 passed / 3 skipped**(与 ACCEPTANCE Round-10 状态块声称完全一致) |
| `scripts/probe_boundaries.py` | **39/39 pass** |
| `expanded_mac` 数值复核 | 正交基自检:diag 误差 4.4e-16,off-diag 4.4e-16;`identity_error` 诊断字段存在 |
| HEX20 | BDF(`CHEXA` 20 节点)、CDB(SOLID95/186)、K 文件均确认仍降为 HEX8 + 聚合警告;TET10 三种格式(BDF/CDB/INP/K)均一等公民保留 |
| GUI / script / CLI | `/api/spr`、`/api/stress` 端点存在;`RECOVER SPR / DUMP PSD / LOAD PSD / ERA / UPDATE STATIC` 脚本动词存在;CLI 实际 22 个 typer 子命令 |

结论先行:**Round 10 的六个冻结名全部落地、经测试、已提升为稳定导出;没有"声称已做但代码缺失"的行。剩余问题集中在 (1) PRODUCT_MAP 一处高可见度的陈旧段落,(2) 一批 ACCEPTANCE 测试欠账,(3) 少量真实的下一轮能力缺口(HEX20、UNV 2414、MPE 指示函数等)。**

---

## A. N/A by design(设计上不做,替代品齐备,文档一致)

| 项 | 替代品(已验证存在) | 文档一致性 |
|---|---|---|
| NI DAQ 硬件采集 | `dynamics.synthetic.synthetic_frf / synthetic_time_response`、`mpe.synthetic`(带种子噪声),§6/§9.5 的 MPE/OMA 验证全部消费它 | PRODUCT_MAP §8 N/A 行 + SOTA §10 一致 ✓ |
| Nastran OP2 / ANSYS RST / Abaqus ODB 二进制 | punch 文本全家桶(`read_pch` 模态、`read_pch_static` 静力、R10 `read_pch_stress` 应力)+ CDB/INP/K 文本翻译器 + `SolverDriver` 协议和三个文本 driver;`.op2/.rst/.odb` 路径抛 `SolverError` 并点名 N/A(有 stub 测试钉住) | PRODUCT_MAP §9 N/A 行 + SOTA §10/§14 + ARCHITECTURE §10 一致 ✓ |
| CAD 内核 / 网格生成 | UNV/BDF(以及 CDB/INP/K)网格导入 | PRODUCT_MAP §1 N/A 行一致 ✓ |
| License server | 无需替代 | REMAINING.md「Explicitly still N/A」一致 ✓ |

**A 桶无需任何动作。**

## B. 已合并代码的已知 caveat(能力在,质量差距,SOTA §10/§14 已如实记录)

1. **HEX8 畸变敏感 / EAS-30 未实现**——非缺失产品行,是**数值 caveat**(SOTA §10):skew→0.4 时单层弯曲比 0.986→~0.36;Simo–Rifai 假设应变是已知升级,Round 10 简报明确禁做。维持 caveat 分类。
2. **`bbar` 变体薄弯过软**(~2.13×),仅适用近不可压厚件——已记录。
3. **截断 FRF 5% 是 20 模态陈述**——<20 模态需残差补偿;R10 的 `residual_flexibility` 正是公开补偿通道(实测 4 模态 1.90e-2→1.31e-3,14.5×),契约陈述本身不变。caveat 保留但已有官方出路。
4. **UNV 数据集 30000 第三方不可见**(femtools 私有 JSON;BDF/`.ftproj` 是无损通道)——设计使然;若做 D 桶的 UNV 2414,结果类数据可部分走公开通道。
5. **HEX20 中侧节点仍丢弃**(BDF/CDB/K 三处、聚合警告;TET10 半已在 R10 关闭)——同时也是 D 桶头号候选。
6. **CTETRA 部分中侧节点集(5–9 节点)仍降 TET4**——已文档化,合理。
7. **TET10 的 SPR 是诚实的最小二乘补丁拟合,非形式化超收敛**(二次四面体的超收敛点是 4 个 Gauss 点,不是形心)——`recover.py` 文档串 + fea.md §15 + SOTA §14 三处一致地披露了这一偏差,符合简报"二选一并文档化"的许可。
8. **mypy 非阻断**——内部模块数十条 informational findings,CI 步骤保持 non-blocking。

**B 桶全部有据可查、文档如实,无需重分类。**

## C. 文档/代码错位(本次审计的实质发现)

按严重度排序:

| # | 位置 | 问题 | 严重度 |
|---|---|---|---|
| C1 | `docs/PRODUCT_MAP.md` 横切段(约 210–225 行) | **仍写"143 stable names",并断言 Round-10 冻结名"not in `_EXPORTS`/`__all__` yet: the dict stays at 143 until the parent glue confirms... and promotes them"。实际 `bec1045` 已提升到 149,六个名字全在。**同文件的状态图例(R10 = stable lazy export)与之自相矛盾 | **高**(公开门面文档自相矛盾) |
| C2 | `docs/PRODUCT_MAP.md` 图例第 22–23 行 | "The transitional **R10** tag is retired" —— `R10-wip → R10` 全局替换时把这句话里的 "R10-wip" 也替换了,现在字面上说"R10 标签已退役",与整个文件都在用 R10 矛盾。应为 "The transitional **R10-wip** tag is retired"(与 R4-wip/R7-wip/R9-wip 句式对齐,那些句子未受损) | 中(用户提示的 over-replace,确认存在,但只此一处) |
| C3 | `docs/ACCEPTANCE.md` 第 373 行 | 残句 "status block until the parent measures the merged tree — constructions in §12." —— R10 收尾编辑第 371–372 行后遗留的半句,悬空无主语 | 中(阅读困惑) |
| C4 | `docs/PRODUCT_MAP.md` 图例第 35 行 | **R5+** 图例行仍在,但全表已无任何 R5+ 行(Round 6 已全部落地)——陈旧图例 | 低 |
| C5 | `docs/PRODUCT_MAP.md` §1 CLI 行 | CLI 行(R1 行 16 个 + R10 行 5 个 = 21 个)漏列实际存在的 `gui` 子命令(实测 typer app 共 22 个命令:solve-modes, read-mesh, write-mesh, recover-stress, recover-spr, plot-stress, mac, expanded-mac, report-mac, frf, dump-frf, load-frf, dump-psd, load-psd, reduce, estimate-frf, era, update, update-static, pretest, script, **gui**) | 低 |
| C6 | 四处历史元素清单 | PRODUCT_MAP 第 72 行(元素库 R1 行)、第 86 行(recovery R7 行)、SOTA §11 第 322 行、fea.md §10 第 448 行都停在 "…HEX8, TET4",没有"TET10 是下方 R10 行"的交叉引用(其余行都有这种 "…is the Round-N row below" 惯例)。**不算错误**——R10 行/§14 各自正确声明 TET10 已被 recover 覆盖,且代码 `_tet10_recovery` 确在派发表里——只是历史段落缺交叉引用 | 低(软性陈旧) |

**排查过、确认无错位的项**(用户点名的猎物清单里的其余项):
- **`expanded_mac` identity vs AutoMAC off-diagonal**:PRODUCT_MAP 第 123 行、ACCEPTANCE §12.4、correlation.md §6 三处一致且精确("unweighted off-diagonal = 参考模态集自身的 AutoMAC 0.240,属于模态集而非扩展缺陷;`weights=M` 下 identity_error 4.4e-16"),与代码行为数值吻合。✓
- **SOTA 是否还说 TET4-only recovery**:没有断言性的 "TET4-only";只有 C6 那种历史清单,§14 已正面声明 TET10 recovery。✓
- **CLI vs PRODUCT_MAP**:除 C5 的 `gui` 外,21 个命令一一对上;R10 五个新命令(dump-psd/load-psd/era/recover-spr/expanded-mac)行为与 lazy-fail exit 3 约定一致。✓
- **`__all__`/`_EXPORTS` vs CONTRACT_API**:CONTRACT_API 是 R1 冻结合同,所有签名仍满足(超集);149 名全部解析,`test_public_api.py` 在测。✓
- **R5+ 行是否有实际已落地未改标的**:无——表中已无 R5+ 行(只剩 C4 的图例残留)。✓
- **REMAINING.md / ROUND10_BRIEF.md / PROGRESS.md**:均已更新为 "Round 10 landed / closed, `_EXPORTS` 149"。✓
- **ACCEPTANCE Round-10 状态块**:声称的 590/3、15/15 examples、39/39 probes 全部实测复现。✓

## D. 范围内缺失能力(Cycle D Round 11–12 的真实下一步)

逐项裁决用户给出的候选(接受/拒绝 + 理由):

| 候选 | 裁决 | 理由与冻结建议 |
|---|---|---|
| **HEX20** | **接受(头号)** | TET10 已趟平二次实体单元的全部管线(注册表、四点求积、recover、verification.PATCH_TYPES、BDF/CDB/INP/K 保留、SPR)。HEX20 补上即可**彻底关闭** SOTA §10 中侧节点丢弃 caveat。公开教材 Serendipity 20 节点 + 3×3×3(或 14 点)求积。冻结:`from femtools.fea.elements import hex20`;门:常应变 patch ≤1e-12、自由-自由单元 6 RBM、`CHEXA` 20 节点 / SOLID186 / C3D20 往返保留(替换现有 warn+drop) |
| **UNV 2414** | **接受** | R10 简报已点名"optional if cheap"但未落地;公开 UFF 数据集(节点/单元分析数据),是把位移/应力结果送给第三方工具的公开通道,顺带缓解数据集 30000 的私有性 caveat(结果侧)。冻结:扩展 `io.unv.read_unv` / `write_unv`(无新顶层名);门:节点位移 + 单元应力记录写→读 bit-exact 往返,未知记录保持容忍跳过 |
| **MPE 指示函数(PolyMAX 已有,缺的是 MIF)** | **接受** | `poly_lscf`/LSCE/SSI/ERA/FDD/稳定图都在,但**没有 CMIF/MMIF**(实测 grep 无)——FEMtools 级 MPE 的标配预处理(定阶、重根探测;Shih et al. 1988,Allemang–Brown 综述,公开文献)。冻结:`from femtools.mpe.mif import cmif`;门:2-DOF 合成 FRF 的 CMIF 峰值定位共振 ±1 谱线,重根处第二条奇异值曲线抬升 |
| **模态复杂度指标 MPC/MPD** | **接受** | 复模态验证标配(Pappa–Elliott–Schenk 1993 一致模式指标家族,公开文献),现完全缺失;与 `macx`/`nmd`(R6 已有)互补。冻结:`from femtools.mpe.common import modal_phase_collinearity`(+ `mean_phase_deviation` 同文件);门:实模态 MPC=1/MPD=0 至 1e-12,种子随机复向量 MPC 显著低 |
| **更多 punch 块** | **接受(小件)** | `$STRESSES` 已做;`$ELEMENT FORCES` / `$ELEMENT STRAINS`(公开 80 列 punch 文本)是同一解析骨架的低成本延伸。冻结:`from femtools.io.pch import read_pch_force`(strain 可并入或同批);门:stub 文本解析 + 与现有块互相容忍跳过;**仍无 OP2** |
| **ODS 动画** | **接受(R12 打磨位)** | `harmonic_response`(数据)与 `plot_stress`/`plot_mesh3d`(静态图)都在,缺把模态/ODS 摇起来的展示层。matplotlib `FuncAnimation`,pyvista 仍然可选。冻结:`from femtools.viz.plots import animate_mode`;门:N 帧写出、相位单调、`import femtools.viz` 不需要 pyvista |
| **QUAD8/TRIA6** | **暂拒(R12 再议)** | 二次壳要配套弯曲公式(现 QUAD4 是 MITC4;Serendipity 8 节点做纯膜没价值,做壳需 MITC8 类公式,风险和体量都高);测试相关性场景中壳网格 QUAD4/TRIA3 已服务,实体细化走 TET10/HEX20 更顺。HEX20 先行 |
| **EAS-30** | **拒绝(维持 §10 数值 caveat)** | 按本审计约束分类为 numerical caveat 而非缺失产品行;两轮简报明令禁做;HEX8 非畸变网格 98.6% 黄金在。若未来某轮想关闭畸变 caveat,Simo–Rifai 是公开修法,但那是质量升级,不进 Round 11 冻结清单 |
| **PolyMAX 本体 / MAC-X** | 无需动作 | `poly_lscf`(R1)、`macx`(R6)均已在,确认无缺 |

## E. Acceptance 欠账(内核在、缺测试/样例——测试债,不是缺产品)

ACCEPTANCE 主表未勾选行逐一核对(是否已被别名覆盖):

| 行 | 内核 | 现有覆盖 | 判定 |
|---|---|---|---|
| 1b 杆离散色散 | `assemble_km`+`solve_modes`(R1) | 无(test_golden_fea 只有 1a+case 2) | 纯测试债 |
| 1c 杆网格收敛 | 同上 | 无 | 纯测试债 |
| 3b 刚度正交性 | `solve_modes`(R1) | 未单独断言(质量归一 3a 在测) | 纯测试债(一行断言的事) |
| 5 悬臂有效质量 | `effective_mass`(R1) | **零数值测试**(仅 import 检查) | 纯测试债 |
| 7a SDOF FRF 闭式 | `modal_frf`(R1) | test_frf 只有 7b 截断带 | 纯测试债 |
| 9 自由-自由梁 | R1 | 6-RBM 已在壳(round6_o1)、HEX8 块(hex8_verification)、TET10(round10_o1)多处钉死;**梁**件 + 自由-自由 EB 弹性根未测 | 大半被别名覆盖,补梁件即可 |
| 10 CB 精确性 | `craig_bampton`(R1) | round5_dynamics 只有参数校验;`cms_rubin.py` 有 CB 基线 2.93e-5(8 保留模态,非全内模态) | "全内模态→1e-10" 恒等式未钉,补一测 |
| 11 静力端点 | `solve_static`(R1) | **已被别名覆盖**:`examples/update_static.py` 实测 tip vs FL³/3EI = 2.3e-13;test_round8_o4 BAR2 轴向 | 可直接勾选 + 指针,或补两行断言 |
| 12 Newmark 周期误差 | `time_history`(R1) | round5_dynamics 有 Newmark vs 精确递推 ≤1e-3(相关但非周期延长界) | 半覆盖,补周期界断言 |
| 13 力识别 | `identify_harmonic_forces`(R1) | **零数值测试** | 纯测试债 |
| 15 RBPE 合成刚体 | `rigid_body_properties`(R1+R4) | **零数值测试**(整个 rbpe 包无一数值测试!) | E 桶最痛的一项 |
| 16 FDD 合成 2-DOF | `fdd`/`efdd`(R1) | **零数值测试** | 纯测试债 |
| SIMP 完整 MBB | `topology_simp`(R1) | 仅 smoke(60×20 准则未测,ACCEPTANCE 已如实标注) | 测试债(可保持标注) |

**E 桶结论**:13 项全部是测试/样例欠账,无一是缺失能力;其中 15(RBPE)、16(FDD)、13(力识别)、5(有效质量)四项是 R1 内核至今零数值验证,建议 Round 11 的 G1 类 agent 一次清掉,并同步勾选 ACCEPTANCE(11 可直接以现有实测数字勾选)。

---

## 建议下一轮冻结 3–6 个名字(Round 11)

1. `from femtools.fea.elements import hex20` —— 20 节点六面体;门:常应变 patch ≤1e-12 + 自由-自由 6 RBM + BDF/CDB/INP 20 节点卡保留往返(彻底关闭 §10 中侧丢弃 caveat;HEX8 98.6% 黄金不动)。
2. `io.unv` **数据集 2414**(扩展 `read_unv`/`write_unv`,不新增顶层名)—— 门:节点位移 + 单元应力记录 UNV 往返 bit-exact,未知记录容忍跳过;数据集 55/58/2412/30000 不回归。
3. `from femtools.mpe.mif import cmif` —— 门:2-DOF 合成 FRF 的 CMIF 峰在真频 ±1 谱线,`poly_lscf`/ERA 黄金不动。
4. `from femtools.mpe.common import modal_phase_collinearity`(同文件带 `mean_phase_deviation`)—— 门:实模态 MPC=1/MPD=0 至 1e-12,种子随机复向量 MPC 低。
5. `from femtools.io.pch import read_pch_force` —— punch `$ELEMENT FORCES`(strain 块可同批)文本解析,stub 测试,仍无 OP2。
6. `from femtools.viz.plots import animate_mode` —— 模态/ODS 动画,matplotlib 默认、pyvista 可选,`import femtools.viz` 不需要 pyvista(可顺延 R12)。

随行(非冻结名):C1–C3 三处文档修正(PRODUCT_MAP 149 段落、R10-wip 残句、ACCEPTANCE 373 行残句)+ E 桶测试欠账清扫(优先 15/16/13/5,顺手勾选 11)。

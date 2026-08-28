# Round 11 对齐修复（文档债 + ACCEPTANCE 可勾选项）

Base: `origin/main` @ `6fb7b4e` (Round 10 closed, `_EXPORTS` 149).
Audit: `.agent_workspace/reports/AUDIT-R10-GAPS.md` on `cursor/r11-audit-gaps-d551`.

Do **not** implement HEX20 / UNV 2414 / CMIF / EAS-30 / DAQ / OP2.
Do **not** regress goldens.

## Fix (C 桶 + case 11)

1. **PRODUCT_MAP 横切段** (~L210–225): 仍写 143 且声称 R10 名尚未进 `_EXPORTS`。改为 **149**，并写明 Round 10 已提升 `tet10`、`recover_spr`、`read_pch_stress`、`era`、`expanded_mac`、`residual_flexibility`。
2. **图例 L22–23**: 「The transitional **R10** tag is retired」→ **R10-wip**（与 R4-wip/R7-wip/R9-wip 句式一致）。**R10 行保留。**
3. **R5+ 图例行**: 表中已无 R5+ 行。删掉图例行，或注明「retired after Round 6」。
4. **ACCEPTANCE L373 残句**: 删除 `status block until the parent measures the merged tree — constructions in §12.`
5. **ACCEPTANCE case 11**: 改为 `[x]`，指针 `examples/update_static.py` 实测 tip vs $FL^3/3EI$ **2.3e-13**（已有，勿改数字除非你重测）。
6. **CLI 行**: 补上实际存在的 `gui` 子命令（共 22 个 typer 命令）。
7. **历史元素清单**（PRODUCT_MAP 元素库 R1 行、recovery R7 行、SOTA §11、`docs/algorithms/fea.md` 若停在 TET4）: 加「TET10 见 R10 / §14」交叉引用，不要假装 R1 就有 TET10。

## Optional if cheap (E 桶，内核已在)

给 **15 RBPE / 16 FDD / 13 力识别 / 5 有效质量 / 3b 刚度正交** 各加一个小数值测试（公开公式，确定性种子）。不要为了写测试去改内核数值。测不完就在报告里列出未做项。

## Report

`.agent_workspace/reports/R11-O5F-ALIGN.md` 首行:
`MODEL_SLUG: claude-opus-5-thinking-high-fast`

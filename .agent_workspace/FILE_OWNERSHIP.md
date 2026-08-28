# File Ownership — Round 8 (Cycle C)

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`.
Parent already added `RBE3` / `FEModel.add_rbe3` in `src/femtools/core/model.py` (and the stable top-level `RBE3` export) — **do not rewrite that table**; consume it.

Agents write `.agent_workspace/reports/R8-<ID>.md` (unique filenames). First line: `MODEL_SLUG: <slug>`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R8-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py` |
| R8-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `tests/test_round8_io.py` |
| R8-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R8-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R8-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**`, `tests/test_round8_o1.py` |
| R8-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R8-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R8-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**`, `tests/test_round8_o4.py` |
| R8-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` and except `tests/test_round8_*.py` |
| R8-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

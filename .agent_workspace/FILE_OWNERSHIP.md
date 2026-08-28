# File Ownership — Round 7 (Cycle C)

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`.
Parent already added `RBE2` / `FEModel.add_rbe2` in `src/femtools/core/model.py` — **do not rewrite that table**; consume it.

Agents write `.agent_workspace/reports/R7-<ID>.md` (unique filenames). First line: `MODEL_SLUG: <slug>`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R7-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py` |
| R7-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `tests/test_round7_io.py` |
| R7-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R7-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R7-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**`, `tests/test_round7_o1.py` |
| R7-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R7-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R7-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**`, `tests/test_round7_o4.py` |
| R7-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` and except `tests/test_round7_*.py` |
| R7-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

# File Ownership — Round 6

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`.

Agents write their report to `.agent_workspace/reports/R6-<ID>.md` (shared reports dir; unique filenames).

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R6-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py` |
| R6-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `tests/test_round6_io.py` |
| R6-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R6-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R6-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**`, `tests/test_round6_o1.py` |
| R6-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R6-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R6-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**`, `tests/test_round6_o4.py` |
| R6-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` and except `tests/test_round6_*.py` |
| R6-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

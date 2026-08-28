# File Ownership — Round 9 (Cycle C)

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`.
Do **not** rewrite `core.model.RBE2` / `RBE3` dataclasses.

Agents write `.agent_workspace/reports/R9-<ID>.md` (unique filenames). First line: `MODEL_SLUG: <slug>`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R9-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py`, `README.md`, `.gitignore` |
| R9-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `tests/test_round9_io.py` |
| R9-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R9-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R9-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**`, `tests/test_round9_o1.py` |
| R9-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R9-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R9-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**`, `tests/test_round9_o4.py` |
| R9-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` and except `tests/test_round9_*.py` |
| R9-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

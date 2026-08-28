# File Ownership — Round 10 (Cycle D)

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`.
Do **not** rewrite `core.model.RBE2` / `RBE3` dataclasses.
Parent seeded `ELEMENT_NODE_COUNTS["TET10"] = (10,)` and `_ELEMENT_NEEDS_PROPERTY`.

Agents write `.agent_workspace/reports/R10-<ID>.md` (unique filenames). First line: `MODEL_SLUG: <slug>`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R10-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py`, `README.md`, `.gitignore` |
| R10-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `tests/test_round10_io.py` |
| R10-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R10-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R10-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**`, `tests/test_round10_o1.py` |
| R10-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**`, `tests/test_round10_o2.py` |
| R10-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**`, `tests/test_round10_o3.py` |
| R10-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**`, `tests/test_round10_o4.py` |
| R10-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` and except `tests/test_round10_*.py` |
| R10-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

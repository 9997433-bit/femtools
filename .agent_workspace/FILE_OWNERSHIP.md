# File Ownership — Remaining cycle Round 4

Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND*_BRIEF.md`, `REMAINING.md`, `FINAL_REPORT.md`.

Each agent writes `.agent_workspace/reports/<ID>.md`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R4-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py` |
| R4-F2 | claude-fable-5-thinking-xhigh | `src/femtools/io/**`, `src/femtools/drivers/**`, `src/femtools/core/**` only if required for I/O types |
| R4-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R4-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R4-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**` |
| R4-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R4-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R4-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**` |
| R4-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` |
| R4-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

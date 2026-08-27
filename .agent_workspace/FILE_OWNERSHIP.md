# File Ownership — Round 3

Agents MUST only write exclusive paths. Parent-owned: `.agent_workspace/PROGRESS.md`, `FILE_OWNERSHIP.md`, `ORCHESTRATION.md`, `ROUND1_BRIEF.md`, `ROUND2_BRIEF.md`.

Each agent writes `.agent_workspace/reports/<ID>.md`.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R3-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `src/femtools/__init__.py`, `src/femtools/py.typed` |
| R3-F2 | claude-fable-5-thinking-xhigh | `src/femtools/core/**`, `src/femtools/io/**` |
| R3-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R3-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R3-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**` |
| R3-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R3-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R3-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**` |
| R3-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` |
| R3-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

# File Ownership — Round 1

Agents MUST only write files under their exclusive paths.
Shared-read: everything else. If a change is needed outside ownership, document it in the report; do not edit.

| ID | Model slug | Exclusive write paths |
|----|------------|----------------------|
| R1-F1 | claude-fable-5-thinking-xhigh | `docs/ARCHITECTURE.md`, `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `.github/**`, `pyproject.toml`, `LICENSE` |
| R1-F2 | claude-fable-5-thinking-xhigh | `src/femtools/core/**`, `src/femtools/io/**` |
| R1-F3 | claude-fable-5-thinking-xhigh | `docs/algorithms/**`, `docs/ACCEPTANCE.md`, `examples/**` |
| R1-F4 | claude-fable-5-thinking-xhigh | `src/femtools/script/**`, `src/femtools/cli.py`, `src/femtools/gui/**`, `src/femtools/viz/**` |
| R1-O1 | claude-opus-5-thinking-high-fast | `src/femtools/fea/**` |
| R1-O2 | claude-opus-5-thinking-high-fast | `src/femtools/dynamics/**` |
| R1-O3 | claude-opus-5-thinking-high-fast | `src/femtools/correlation/**`, `src/femtools/pretest/**` |
| R1-O4 | claude-opus-5-thinking-high-fast | `src/femtools/updating/**`, `src/femtools/optimization/**`, `src/femtools/mpe/**`, `src/femtools/rbpe/**` |
| R1-G1 | gpt-5.6-sol-xhigh-fast | `tests/**` except `tests/perf/**` |
| R1-G2 | gpt-5.6-sol-xhigh-fast | `benchmarks/**`, `tests/perf/**`, `scripts/**` |

Parent-owned (do not touch): `.agent_workspace/PROGRESS.md`, `.agent_workspace/FILE_OWNERSHIP.md`, `.agent_workspace/ORCHESTRATION.md`

Each agent MAY write exactly one report: `.agent_workspace/reports/<ID>.md`

Shared read-only contract: `docs/CONTRACT_API.md`, `src/femtools/__init__.py` (parent seed). F1 may extend `__init__.py` exports only if needed; others import from it.

`src/femtools/daq/` is deferred (hardware). Provide a synthetic generator under `src/femtools/dynamics/` or `examples/` instead.

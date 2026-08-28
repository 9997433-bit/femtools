# Orchestration

- Repo: `github.com/9997433-bit/femtools`
- Isolated branch: `cursor/femtools-cycle-c-d551` (Cycle C = Rounds 7–9). Previous cycles: `cursor/femtools-sota-d551` (R1–R3), `cursor/femtools-remaining-d551` (R4–R6), both merged to main.
- Goal: 1:1 functional equivalent of commercial FEMtools (DDS) — solver-independent CAE framework for structural dynamics, pretest, correlation, model updating, optimization, EMA/OMA, scripting, visualization.
- Constraint: original implementation. Do not copy proprietary source, binaries, or copyrighted manuals. Public algorithms (MAC, EFI, Craig-Bampton, SIMP, Friswell–Mottershead sensitivity updating, PolyMAX/p-LSCF class estimators, etc.) are in-scope.
- Stack: Python 3.11+, numpy, scipy, pydantic v2, typer, matplotlib. Optional: pyvista, fastapi, plotly.
- Quality bar: typed public API, pytest + golden analytical cases, ruff, mypy-friendly, deterministic seeds, documented numerical tolerances.

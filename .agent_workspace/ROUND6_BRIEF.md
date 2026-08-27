# Round 6 任务简报（剩余周期最后一轮）

Base branch: `cursor/femtools-remaining-d551` (PR #2 → `main`).  
pytest at Round-5 close: **109 passed / 3 skipped**. Do not regress.

This round lands the PRODUCT_MAP **R5+** rows that are still original-algorithm / public-text work. **Do not** implement NI DAQ hardware or commercial OP2/RST/ODB binary parsers.

Frozen extra APIs (also in `REMAINING.md`). Implement exactly these names; do not rename.

## R6-O1 — shell drilling frames (`fea/**`)

A free-free flat TRIA3/QUAD4 plate whose normal is **not** a global axis currently keeps a fictitious 7th rigid mechanism (warned at assemble). Contract: **per-node rotational frames** whose local 3-axis is the averaged shell normal, so drilling can be auto-constrained for arbitrary orientation.

Acceptance: `femtools.fea.verification.shell_drilling_orientation_gap` reports **6** rigid-body frequencies (~0) not 7; existing goldens unchanged (HEX8 98.6% tip, 6 RBM on a cube, MITC4 thin plate, patch tests, BEAM2 EB, `solve_static(enforced=)`).

Optional only if the drilling contract is already green: Simo–Rifai EAS for HEX8 distortion, without breaking 98.6% or 6 RBM.

## R6-O4 — UQ + shape + SSI-DATA

```python
from femtools.updating.uq import parameter_covariance, monte_carlo_update, UQResult
# First-order Cov(θ) from residual covariance and J (Friswell–Mottershead);
# MC with a required seed. Do not invent Bayesian samplers beyond this.

from femtools.optimization.shape import shape_optimize, ShapeResult
# Selected node xyz as design variables; frequency or compliance objective;
# SLSQP/trust-constr; a simple Laplacian / min-jacobian mesh-quality barrier.
# Literature: Haftka & Grandhi shape-optimization-with-FEM class methods.

from femtools.mpe.ssi import ssi_data
# Data-driven SSI / N4SID-class (Van Overschee & De Moor 1996): Hankel of
# past/future outputs, projection, then the same SVD + shift-invariance path
# as the existing ssi_cov. Return the same result type as ssi_cov.
```

Keep H1/H2=γ² and 10% E recovery invariants.

## R6-F2 — Abaqus INP + LS-DYNA K (text subsets)

```python
from femtools.io.inp import read_inp  # write_inp optional
# *NODE, *ELEMENT (C3D8/C3D4/S4/S3/B31/T3D2), *MATERIAL, *ELASTIC, *DENSITY,
# *SOLID SECTION / *SHELL SECTION / *BEAM GENERAL SECTION, *BOUNDARY, *NSET, *ELSET

from femtools.io.kfile import read_k
# *NODE, *ELEMENT_SOLID / *ELEMENT_SHELL / *ELEMENT_BEAM, *MAT_ELASTIC,
# *SECTION_SHELL / *SECTION_SOLID / *SECTION_BEAM, *BOUNDARY_SPC_NODE, *PART
```

Public card layouts only. Map into `FEModel`. Round-trip a tiny HEX8/QUAD4/BEAM deck through `assemble_km` without crashing. No OP2/RST/ODB.

## R6-O3 — extended MAC

```python
from femtools.correlation.mac import nmd, macx
# nmd: sqrt(1-MAC) (Allemang NMD). macx: extended MAC for complex modes
# using both φ and conj(φ). Do not change mac_matrix / fmac numerics on real modes.
```

## R6-O2 — dynamics diagnostics + bug hunt

```python
from femtools.dynamics.energy import modal_strain_energy, modal_kinetic_energy
# Per-mode (and optional per-element) MSE / MKE from assembled K, M and mass-normalised Φ.
```

Plus a reproduced-bug hunt in `dynamics/**` (do not touch cms_free numerics that hold 0.028% Rubin).

## R6-F4 — viz extras + CLI

Optional plotly backend behind `femtools.viz.plots` if plotly is importable (matplotlib remains default). CLI: `femtools read-mesh` accepting `.inp`/`.k` when F2 lands; if those modules are absent, keep existing commands green. Do not break `report-mac` / `reduce` / `estimate-frf`.

## R6-F1 — docs / lazy exports

PRODUCT_MAP: add R6 rows as **R6-wip** until the modules exist in *this* tree. **Do not** add `__all__` names for missing modules (CI resolves every export). If `read_inp` / `shape_optimize` / `ssi_data` / `nmd` / `parameter_covariance` / `modal_strain_energy` are already importable, promote them to stable `_EXPORTS`. Update SOTA.md with the public references (Van Overschee SSI-DATA, Friswell UQ, Haftka shape). Architecture driver section: mention INP/K text translators.

## R6-F3 — algorithms + examples

Notes under `docs/algorithms/` for UQ, shape, SSI-DATA, INP/K. Keep the existing 8 examples **PASS**. Add `examples/shape_plate.py` and/or `examples/ssi_data_oma.py` only if the kernels import; otherwise document the example as pending in ACCEPTANCE.md — do not leave a broken example in CI-run paths.

## R6-G1 — tests (existing files only)

Do **not** create `tests/test_round6_*.py` (cloud O1/O4/F2 own those). Add CDB regressions (ETBLOCK `a` format, COMPACT EBLOCK, RMORE slots 7–12, BEAM3 A/IZZ) into existing `tests/test_io_roundtrip.py` or `tests/test_round4_pch.py`. Pin `solve_static(enforced=)` and H1/H2 invariants if not already pinned outside `test_round6_*`.

## R6-G2 — probes / perf

Extend `scripts/probe_boundaries.py` with probes for `read_inp`/`read_k`/`nmd`/`ssi_data`/`parameter_covariance` that **skip** if the module is absent (probe script convention). Do not fail the remaining branch. No perf-test unskip unless numbers are recorded.

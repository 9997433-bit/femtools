# Round 7 任务简报（Cycle C 第一轮）

Base: `cursor/femtools-cycle-c-d551` from `main` (post R6, pytest 183/3).  
Do not regress goldens. No DAQ, no OP2/RST/ODB.

Frozen names: `.agent_workspace/REMAINING.md`.

## R7-O1 — stress recovery + RBE2 (cloud)

`recover_stress` / `recover_strain` / `StressResult` for BAR2, BEAM2, QUAD4, TRIA3, HEX8, TET4 at element centroid. Constant-strain patch test ≤1e-12.

`apply_rbe2` + `ConstraintTransform`. Consume **existing** `FEModel.rbe2` / `add_rbe2` (do not replace the dataclass). `assemble_km` must honor `model.rbe2` (or `mpc=`). Two nodes welded: free-free still 6 RBM; a rigid offset beam should carry moment.

Keep HEX8 98.6%, MITC4, drilling 6 RBM, `solve_static(enforced=)`.

Tests: `tests/test_round7_o1.py`.

## R7-F2 — write_cdb / write_k / INCLUDE / RBE2 / Nastran driver (cloud)

`write_cdb`, `write_k` round-trip Round-6 HEX8/QUAD4/BEAM2 decks through `assemble_km`.

`read_bdf`: follow `INCLUDE` (relative, depth≤8, cycle-safe); parse `RBE2` via `model.add_rbe2`. RBE3 optional.

`NastranPunchDriver` in `drivers/nastran.py`: implements `SolverDriver`; `write_input` uses `write_bdf` + SOL 103 punch request; `read_modal` uses `read_pch`; `run` raises `SolverError` if executable missing. **No Nastran binary in tests.** Export from `drivers/__init__.py`. Do not edit `femtools/__init__.py`.

Punch/CDB read: only fix reproduced bugs. No OP2.

Tests: `tests/test_round7_io.py`.

## R7-O4 — topometry + static displacement response (cloud)

`topometry_optimize` on an **existing** FEModel mesh (element thickness or density as design vars), min-compliance, volume/mean-thickness constraint, OC or SLSQP. Distinct from `topology_simp`'s built-in grid. A cantilever plate should drop compliance vs uniform start without inverted elements.

`static_displacement_response` for `update_model`. 10% E recovery invariant still holds.

Tests: `tests/test_round7_o4.py`. Do not edit `fea/**` or `io/**`.

## R7-O2 — base-accel PSD + CMS dump (local, no git)

`psd_response(..., base_accel=)` — SDOF closed-form RMS. Do not break force-PSD path or Miles number.

`dump_cms` / `load_cms` npz for Craig–Bampton (and Rubin if the result object has K,M,T). Bit-identical K/M after load. Do not change `cms_free` Rubin 0.028%.

## R7-O3 — nearest-node map + MAC contribution (local, no git)

`map_nearest_nodes`. Two translated copies of the same 8-node cube match 1–1 with distance = translation.

`mac_contribution` per DOF for one mode pair. Sum of contributions related to MAC documented. Do not change `mac_matrix` real-mode numerics.

## R7-F4 — pyvista optional + CLI/script (local, no git)

`plot_mesh3d` behind `import pyvista` (never required to import `femtools.viz`). matplotlib default.

CLI: `femtools recover-stress`, `femtools write-mesh` (cdb/k/inp by suffix) only if the kernels import; otherwise lazy fail like `read-mesh`.

Script: `SOLVE STATIC` and `SET name=value` if cheap. Do not break the existing 7 commands.

## R7-F1 — docs (local, no git)

PRODUCT_MAP rows **R7-wip** until symbols import on THIS tree. **Do not** add `__all__` names for missing modules (CI resolves every export). `RBE2` is already a stable export — leave it. Update SOTA with stress recovery / RBE2 / topometry public refs (Cook, Bathe, Sigmund topometry vs topology). Architecture: mention NastranPunchDriver as optional text driver, still no OP2.

## R7-F3 — notes + examples (local, no git)

Keep 8 examples PASS. Algorithm notes for stress recovery, RBE2 condensation, topometry, INCLUDE. New examples only if kernels import. Fix ACCEPTANCE.md R6 pending rows that are now implemented (21–28) to measured/pass where tests exist.

## R7-G1 — existing tests (local, no git)

Do **not** create `tests/test_round7_*.py`. Pin `FEModel.add_rbe2` validation (duplicate id, missing node, independent∈dependents). Pin H1/H2 and CDB regressions stay. No importorskip for missing R7 modules.

## R7-G2 — probes (local, no git)

Extend `scripts/probe_boundaries.py` for `recover_stress`, `write_cdb`, `write_k`, `map_nearest_nodes`, `topometry_optimize`, `NastranPunchDriver` — **skip** if absent. Existing probes stay green.

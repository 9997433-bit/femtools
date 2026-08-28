# Round 8 任务简报（Cycle C 第二轮）

Base: `cursor/femtools-cycle-c-d551` (Round 7 closed, pytest 295/3).
Do not regress goldens (HEX8 98.6%, Rubin 0.028%, H1/H2=γ², tilted-shell 6 RBM, 10% E recovery).
No DAQ, no OP2/RST/ODB.

Frozen names: `.agent_workspace/REMAINING.md` (Round 8 section).
File ownership: `.agent_workspace/FILE_OWNERSHIP.md`.

## Close-out

All ten Round-8 agents delivered. Kernels and tests are merged on
`cursor/femtools-cycle-c-d551`. Parent glue: top-level `_EXPORTS` (138 names),
PRODUCT_MAP R8-wip → R8, ACCEPTANCE Round-8 rows measured. Goldens unchanged
(HEX8 98.6%, Rubin 0.028%, H1/H2=γ², tilted-shell 6 RBM, 10% E recovery).
DAQ / OP2 / RST / ODB remain N/A.

The per-agent briefs below are historical.

## R8-O1 — RBE3 interpolation + nodal stress average (cloud)

`apply_rbe3` in `fea.mpc`. Public interpolation MPC (Cook / Zienkiewicz master–slave): one **dependent** node’s listed components are a **weighted average** of the independents (default equal weights, or `RBE3.weights`). Not a rigid weld — do not reuse RBE2 kinematics. No penalty springs.

`assemble_km` must honor `model.rbe3` together with `model.rbe2` (compose into one `ConstraintTransform`). `mpc=False` still disables all MPCs.

Gates:
- free–free: a mass at the RBE3 dependent node of a triangle of independent nodes still has **exactly 6** rigid-body modes.
- a force on the dependent node is distributed to the independents (virtual work / `G^T f`); equal weights → equal force shares on translations.
- existing RBE2 goldens stay bit-identical when `model.rbe3` is empty.

`average_nodal(stress: StressResult, model) -> StressResult` (or a small nodal result type) averages centroid stresses onto incident nodes (1/n_adj). A constant-stress patch stays exact at every node. **Not** Zienkiewicz–Zhu SPR.

Tests: `tests/test_round8_o1.py`. Keep HEX8 98.6%, MITC4, drilling 6 RBM, `solve_static(enforced=)`.

## R8-F2 — BDF RBE3 + Ansys/Abaqus text drivers (cloud)

`read_bdf`: parse `RBE3` via `model.add_rbe3` (public card layout: refgrid / refc / wt,c,g lists). `write_bdf` emits RBE3. Unknown/unsupported fields: one aggregated `UserWarning`. Do not invent copyrighted decks.

`AnsysCdbDriver` in `drivers/ansys.py`: `SolverDriver`; `write_input` = `write_cdb`; `is_available` via `shutil.which` (`ansys`/`mapdl` aliases ok); `run` raises `SolverError` if executable missing / nonzero / timeout. `read_modal` uses existing **text** readers (`.pch`/`.unv`); a `.rst` path raises `SolverError` naming RST as N/A. **No ANSYS binary in tests.** Stub shell executables like Round-7 Nastran tests.

`AbaqusInpDriver` in `drivers/abaqus.py`: `write_input` = `write_inp`; same availability/`SolverError` conventions; ODB is N/A. `read_modal` from `.unv`/`.pch` text only.

Export both from `drivers/__init__.py`. Do **not** edit `femtools/__init__.py`.

Tests: `tests/test_round8_io.py`. Keep `test_round7_io.py` green.

## R8-O4 — static-update convenience + stress residual (cloud)

`update_from_static(model, measured, ...)` wrapping `static_displacement_response` + `update_model`. Recover a 10% E error from static tip deflection to the same order as the modal path (~1e-9 relative).

Optional if cheap: a stress residual using `recover_stress` as an `update_model` response (constant-stress patch → parameter recovered). Do not break `parameter_covariance` / topometry / 10% modal E.

Tests: `tests/test_round8_o4.py`. Do not edit `fea/**` or `io/**`.

## R8-O2 — FRF dump/load (local, no git)

`dump_frf` / `load_frf` npz for `FRFResult` (H bit-identical after load, freq_hz bit-identical). Analogous to `dump_cms`. Do not change `modal_frf` numerics or Rubin 0.028%.

## R8-O3 — mapped-shape correlation (local, no git)

`mapped_mode_matrix(modes, fe_ids)` (or equivalent name in `correlation.dofmap`): pull FE mode rows at the node ids returned by `map_nearest_nodes`. Two translated copies of the same cube: mapped MAC diagonal = 1. Do not change `mac_matrix` real-mode numerics or `map_nearest_nodes` distances.

## R8-F4 — plot_stress + script/GUI/CLI (local, no git)

`plot_stress(model, stress, ...)` in `viz.plots`: color the mesh by von Mises (or a named component). matplotlib default; pyvista only if already imported by `plot_mesh3d` path. `import femtools.viz` must still not require pyvista.

CLI: `femtools plot-stress` lazy-fails like `recover-stress` if kernels missing.

Script: `RECOVER STRESS [NAME=..]` and `ADD RBE2` / `ADD RBE3` if cheap. Do not break the existing commands (incl. `SOLVE STATIC` / `SET`).

GUI: `/api/stress` (or equivalent) returns a small JSON table from `recover_stress` when a static result exists; 400 if the kernel is absent.

## R8-F1 — docs (local, no git)

PRODUCT_MAP rows **R8-wip** until symbols import on THIS tree. **Do not** add `__all__` names for missing modules (CI resolves every export). `RBE3` is already a stable export — leave it. Update SOTA with RBE3 interpolation (Cook/Zienkiewicz master–slave, **not** RBE2 kinematics) and nodal averaging vs ZZ-SPR. Architecture: AnsysCdbDriver / AbaqusInpDriver as optional **text** drivers; still no RST/ODB/OP2.

## R8-F3 — notes + examples (local, no git)

Keep the existing 8 examples PASS. Add kernel-backed examples only if the names import: `examples/rbe2_rigid.py`, `examples/topometry_plate.py`, `examples/recover_stress.py` (skip/`sys.exit(0)` if import fails is OK during this round, but prefer a real PASS when kernels are present). Algorithm notes for RBE3, nodal averaging, text drivers. ACCEPTANCE.md: Round-7 rows stay measured; add Round-8 pending rows.

## R8-G1 — existing tests (local, no git)

Do **not** create `tests/test_round8_*.py`. Pin `FEModel.add_rbe3` validation (duplicate id, missing node, dependent∈independents, bad weights, components outside 1..6). Pin RBE2 / H1/H2 / CDB regressions stay. No importorskip for missing R8 kernel modules.

## R8-G2 — probes (local, no git)

Extend `scripts/probe_boundaries.py` for `apply_rbe3`, `average_nodal`, `AnsysCdbDriver`, `AbaqusInpDriver`, `dump_frf`, `plot_stress`, `update_from_static`, `mapped_mode_matrix` — **skip** if absent. Existing probes stay green.

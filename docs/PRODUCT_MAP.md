# Product map — FEMtools capability ↔ femtools packages

Goal: a 1:1 *functional* equivalent of the commercial FEMtools product family (Dynamic Design
Solutions). This is an original implementation; no proprietary source, file formats
documentation, or manual text is used. Public-literature algorithm references: `docs/SOTA.md`.

**Status legend** (retagged at the start of Round 5: every Round-4 module is merged on
the integration branch, so the transitional *R4-wip* tag became the merged tag **R4**.
Rounds 1–3 are merged to main — Round 3 hardened docs/CI/packaging and added no
capability rows, so R1/R2 remain the merged-capability tags. Round 6 — the last round of
this remaining cycle — froze the exact API names for the leftover R5+ capabilities
(`.agent_workspace/ROUND6_BRIEF.md` / `REMAINING.md`); their rows carry the transitional
tag **R6-wip** until the module is importable from this tree, then become **R6**)

| Status | Meaning |
|---|---|
| **R1** | Merged in Round 1; importable from the integration branch and covered by tests |
| **R2** | Merged in Round 2; importable from the integration branch and covered by tests |
| **R4** | Merged in Round 4 (API frozen in `.agent_workspace/REMAINING.md`); importable from the integration branch, covered by tests, and a **stable** lazy export at the `femtools` top level (`src/femtools/__init__.py`) since Round 5 |
| **R6** | Merged in Round 6 (this remaining-cycle's last round; API frozen in `.agent_workspace/ROUND6_BRIEF.md`); importable from the integration branch, covered by tests, and a stable lazy export at the `femtools` top level |
| **R6-wip** | Round-6 work in progress: the API (exact names) is frozen, but the module is **not yet importable from this tree** — no top-level export exists yet (CI resolves every export); promoted to **R6** on merge |
| **R5+** | Not merged and not API-frozen for Round 6; direction fixed, API not yet frozen |
| **N/A** | Out of scope (hardware, licensing, closed binary dumps) with substitute noted |

Known numerical/functional distances between merged (R1/R2/R4) code and the state of the
art are tracked in `docs/SOTA.md` §10 "Merged-code gap" — a row tagged R1/R2/R4 means
*merged and tested*, not *defect-free*. Round 2 closed the two largest §10 items (HEX8
shear locking, UNV material/property cards); the residual caveats remain listed there.

## 1. FEMtools Framework

| Capability | femtools API | Status |
|---|---|---|
| FE/test relational database | `core.model.FEModel` (nodes, elements, materials, properties, SPCs, sets) | R1 |
| Node/element sets, selection | `core.sets.NodeSet`, `core.sets.ElementSet` | R1 |
| Coordinate systems (rect/cyl/sph) | `core.coords.CoordSys` (cartesian, cylindrical, spherical) | R1 |
| Unit system handling | `core.units.UnitSystem` (SI internal, convert at I/O) | R1 |
| Project persistence | `io.project.save_project` / `load_project` (`.ftproj` JSON+npz) | R1 |
| Scripting language (FEMtools Script analog) | `script.engine.ScriptEngine` — command interpreter over the Python API | R1 |
| Command-line surface | `cli` typer app: `solve-modes`, `mac`, `frf`, `update`, `pretest`, `script` | R1 |
| Python API | every package below; frozen contract re-exported from `femtools` top level | R1 |
| GUI shell | `gui` web shell (`gui.server` on stdlib http, FastAPI optional); model upload / preloaded examples via `/api/load` landed R2 | R1 |
| Mesh/geometry visualization | `viz.plots` (matplotlib) — `plot_mode` draws translations and rotation pseudo-vectors (rotations landed R2); an optional plotly backend is Round-6 work (R6-F4, only if plotly is importable — matplotlib remains the default); pyvista extras remain R5+ | R1 |
| Mesh generation / import CAD | not planned — import meshes via UNV/BDF | N/A |

## 2. FEMtools Dynamics

| Capability | femtools API | Status |
|---|---|---|
| Real normal modes (FEA) | `fea.eigen.solve_modes` → `ModalResult` (mass-normalized) | R1 |
| Static solution | `fea.static.solve_static` | R1 |
| Element library (BAR2, BEAM2, TRUSS2D, QUAD4, TRIA3, HEX8, TET4, MASS, SPRING, DAMPER) | `fea.elements` registry, `fea.assemble.assemble_km` — HEX8 uses Wilson–Taylor incompatible modes since R2 (98.6% reference tip deflection single-layer; residual caveats SOTA.md §10). Round-6 direction (R6-O1, not yet merged): per-node shell drilling frames for arbitrarily oriented flat shells, with a Simo–Rifai EAS distortion upgrade as an optional follow-on | R1 |
| Modal FRF synthesis (with damping models) | `dynamics.frf.modal_frf` (modal ζ, Rayleigh, structural η); truncation-aware `retained_band` helper and `dynamics.damping` specs landed R2 | R1 |
| Residual flexibility / residual vectors | `dynamics.residuals.residual_vectors`; upper/lower residual terms in `modal_frf` | R1 |
| Direct (full-order) FRF | `dynamics.frf.direct_frf` (dynamic stiffness inversion) | R1 |
| Harmonic forced response / ODS | `dynamics.harmonic.harmonic_response` | R1 |
| Transient response | `dynamics.time_domain.time_history` (Nigam–Jennings exact recurrence and Newmark, on modal coordinates) | R1 |
| Craig–Bampton CMS / superelements | `dynamics.craig_bampton.craig_bampton` | R1 |
| Modal-based assembly (coupling by modes) | `dynamics.mba.modal_based_assembly` | R1 |
| FRF-based assembly (substructure coupling on FRFs) | `dynamics.fba.frf_based_assembly` | R1 |
| Structural dynamics modification (SDM) | `dynamics.mba.structural_dynamic_modification` (mass/spring modifications on the modal model) | R1 |
| Free-interface CMS (MacNeal/Rubin) | `dynamics.cms_free.rubin` / `macneal` → `FreeCMSResult` (residual-flexibility free-interface CMS) | R4 |
| Complex modes (general viscous damping) | `fea.eigen.solve_complex_modes` → `ComplexModalResult` (state-space / quadratic eigenvalue) | R4 |
| Random/PSD response | `dynamics.random.psd_response` → `PSDResult` (modal PSD; rms and 1σ) | R4 |
| Modal strain / kinetic energy diagnostics | `dynamics.energy.modal_strain_energy` / `modal_kinetic_energy` — per-mode (optional per-element) MSE/MKE from assembled K, M and mass-normalized Φ | R6-wip |

## 3. FEMtools Pretest & Correlation

| Capability | femtools API | Status |
|---|---|---|
| Target mode selection / modal effective mass | `pretest.target_modes.effective_mass`, `select_target_modes` | R1 |
| Sensor placement — Effective Independence (EFI) | `pretest.efi.effective_independence` | R1 |
| Sensor elimination by MAC / kinetic energy ranking | `pretest.sensor.eliminate_by_mac`, `nodal_kinetic_energy` | R1 |
| Sensor mass-loading check | `pretest.mass_loading.mass_loading`, `sensor_mass_limit` | R1 |
| Candidate DOF selection (translational partition of free DOFs) | `pretest.candidates.translational_dofs`, `candidate_dofs` | R2 |
| Exciter placement (driving-point residues) | `pretest.exciter.driving_point_residues`, `select_exciters` | R4 |
| Test geometry definition & FE↔test node mapping | test geometry as `FEModel`; nearest-node mapping R1; typed DOF mapping `correlation.dofmap.DOFMap` (label parsing, 0/1-based recognition, fea kernel maps) R2; rigid-Procrustes geometry alignment `correlation.alignment.align_geometry` R4 | R1 |
| MAC / CoMAC / POC | `correlation.mac.mac_matrix`, `comac`, `poc` | R1 |
| Mode pairing | `correlation.pairing.pair_modes` (MAC-weighted assignment) | R1 |
| Mass-weighted cross-orthogonality | `correlation.orthogonality.cross_orthogonality` | R1 |
| FRF correlation (FRAC / CSAC / CSF) | `correlation.frf_corr.frac`, `csac`, `csf` | R1 |
| Shape expansion/reduction (Guyan, IRS, SEREP) | `fea.reduction.guyan` / `irs` / `serep` → `ReductionResult`; `correlation.expansion.expand_guyan` / `expand_serep` | R4 |
| ECOMAC / FDAC / modal scale factor | `correlation.mac.ecomac`, `modal_scale_factor`; `correlation.frf_corr.fdac` | R1 |
| FMAC and further extended metrics | `correlation.mac.fmac` (Round 4); NMD/MACX are the Round-6 row below | R4 |
| NMD / extended MAC for complex modes | `correlation.mac.nmd` (Allemang normalized modal difference), `correlation.mac.macx` (extended MAC using both φ and conj(φ)); `mac_matrix`/`fmac` numerics on real modes unchanged | R6-wip |
| Correlation report generation | `viz.report.mac_report_html` / `mac_report_text` / `save_mac_report` + `cli` `report-mac` | R4 |

## 4. FEMtools Model Updating

| Capability | femtools API | Status |
|---|---|---|
| Parameter/response sensitivity matrix | `updating.sensitivity.sensitivity_matrix` (semi-analytic eigen-sensitivities; finite-difference fallback) | R1 |
| WLS / Bayesian sensitivity updating (Friswell–Mottershead) | `updating.updater.update_model` → `UpdateResult` | R1 |
| Parameters: E, rho, thickness, spring k | `updating` parameter protocol | R1 |
| Responses: frequencies, MAC, FRF samples | `updating` residual definitions; `measured`/`analytic` reference wiring (`updating.reference`, `updating.responses`) landed R2 | R1 |
| Harmonic force identification | `updating.force_id.identify_harmonic_forces` | R1 |
| Regularization (Tikhonov, parameter weighting) | inside `update_model` options | R1 |
| Local (element-level) parameters, grouping | entity descriptors in `updating.parameters.as_parameters` target materials/properties/elements; region grouping via shared material/property | R2 |
| FRF-based updating (full-curve residuals) | `updating.frf_updating.update_from_frf` | R4 |
| Robust / uncertainty-quantified updating | `updating.uq.parameter_covariance` / `monte_carlo_update` → `UQResult` — first-order Cov(θ) from residual covariance and the sensitivity matrix (Friswell–Mottershead), seeded Monte Carlo | R6-wip |
| Automated parameter selection (subset selection) | `updating.selection.select_parameters` (EFS / column subset selection) | R4 |

## 5. FEMtools Optimization

| Capability | femtools API | Status |
|---|---|---|
| Size optimization (frequency/compliance constraints) | `optimization.size.size_optimize` (scipy SLSQP/trust-constr backend) | R1 |
| Topology optimization (SIMP + OC/MMA-style update, density filter) | `optimization.topology.topology_simp` | R1 |
| DOE — Latin hypercube, full factorial | `optimization.doe.latin_hypercube`, `full_factorial` | R1 |
| Response surface / surrogate models | `optimization.surrogate.fit_rsm`, `predict_rsm` (quadratic RSM) | R4 |
| Shape optimization | `optimization.shape.shape_optimize` → `ShapeResult` — selected node xyz as design variables, frequency/compliance objectives, SLSQP/trust-constr, mesh-quality barrier (Haftka–Grandhi class) | R6-wip |
| Multi-objective (Pareto) | `optimization.multi.pareto_weighted` (weighted sum; NSGA-lite optional) | R4 |

## 6. FEMtools MPE (Modal Parameter Extraction)

| Capability | femtools API | Status |
|---|---|---|
| Poly-reference LSCF (PolyMAX-class) | `mpe.p_lscf.poly_lscf` | R1 |
| LSCE (complex exponential) | `mpe.lsce.lsce` | R1 |
| Operational: FDD / EFDD | `mpe.fdd.fdd`, `efdd` | R1 |
| Stabilization diagram construction | `mpe.common.stabilization_diagram`, `StabilizationDiagram`, `select_physical_poles` | R1 |
| MIMO FRF estimation from time data (H1/H2, coherence) | `mpe.frf_estimation.estimate_h1`, `estimate_h2`, `coherence` | R4 |
| SSI (covariance-driven) | `mpe.ssi.ssi_cov` | R4 |
| SSI (data-driven, N4SID-class) | `mpe.ssi.ssi_data` — past/future output Hankel projection, then the `ssi_cov` SVD + shift-invariance path; same result type as `ssi_cov` (Van Overschee–De Moor 1996) | R6-wip |

## 7. FEMtools RBPE (Rigid Body Property Extraction)

| Capability | femtools API | Status |
|---|---|---|
| Mass, CoG, inertia tensor from low-frequency FRFs (mass-line) | `rbpe.rbfit.rigid_body_properties` | R1 |
| Inertia-restraint variant, mounting-stiffness correction | `rbpe.rbfit.rigid_body_properties(restraint=..., mount_k=...)` | R4 |

## 8. FEMtools DAQ

| Capability | femtools API | Status |
|---|---|---|
| Hardware acquisition (NI etc.) | not planned (hardware/licensing) — see SOTA.md §10 | N/A |
| Synthetic test-data generation (noisy FRFs, time responses) for MPE/OMA validation | `dynamics.synthetic.synthetic_frf` / `synthetic_time_response`; `mpe.synthetic` | R1 |

## 9. FEA interfaces

| Capability | femtools API | Status |
|---|---|---|
| Universal file (UNV) datasets 15/2411 (nodes), 2412 (elements), 82 (trace lines), 55 (shapes), 58 (FRFs/functions), 151/164 (header/units) | `io.unv.read_unv`, `write_unv` — materials/properties carried since R2 via private dataset 30000 (JSON; third-party readers skip it, SOTA.md §10) | R1 |
| Nastran BDF subset (GRID, C\*, MAT1, PSHELL/PBAR/P\*, SPC, FORCE) | `io.bdf.read_bdf`, `write_bdf` — TET10/HEX20 midside nodes dropped with aggregated warnings (SOTA.md §10) | R1 |
| Native project file | `io.project` (`.ftproj`) | R1 |
| Solver driver protocol (third-party results import) | `drivers.base.SolverDriver` (runtime-checkable PEP 544 Protocol — structural typing, no registration; `docs/ARCHITECTURE.md` §10) | R4 |
| Nastran punch results (`.pch` text: eigenvalues, mode shapes) | `io.pch.read_pch`, `write_pch` | R4 |
| ANSYS CDB mesh import (NBLOCK/EBLOCK subset) | `io.cdb.read_cdb` | R4 |
| Nastran OP2 / ANSYS RST / Abaqus ODB binary results | closed binary dumps — substitutes: the `.pch`/CDB text readers above plus the `SolverDriver` plug-in protocol | N/A |
| Abaqus INP / LS-DYNA K (text subsets) | `io.inp.read_inp`, `io.kfile.read_k` — public card layouts only, mapped into `FEModel`; no OP2/RST/ODB binaries (those stay N/A above) | R6-wip |

## Cross-cutting quality gates (all rounds)

* Typed public API (`py.typed` marker ships in the wheel since R2), pydantic-validated core.
* Frozen contract (`docs/CONTRACT_API.md`), the frozen Round-4 API
  (`.agent_workspace/REMAINING.md`), and the `femtools.core.errors` exception hierarchy
  re-exported lazily from `femtools` (`src/femtools/__init__.py`, 99 stable names +
  15 subpackages), so `import femtools` is cheap and cycle-free — no numpy/scipy/pydantic
  at import time. The transitional *provisional* export tier that carried the Round-4
  names while their modules were in flight was retired in Round 5: every Round-4 module
  is merged, and all names are stable `__all__` entries with `TYPE_CHECKING` re-exports.
* Golden analytical acceptance tests with the tolerance table of `docs/CONTRACT_API.md`.
* ruff + strict pytest (an empty collection fails CI since Round 2) on Python 3.11,
  plus a non-blocking mypy step (`.github/workflows/ci.yml`).
* Deterministic seeds for every stochastic routine.

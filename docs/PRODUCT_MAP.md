# Product map — FEMtools capability ↔ femtools packages

Goal: a 1:1 *functional* equivalent of the commercial FEMtools product family (Dynamic Design
Solutions). This is an original implementation; no proprietary source, file formats
documentation, or manual text is used. Public-literature algorithm references: `docs/SOTA.md`.

**Status legend**

| Status | Meaning |
|---|---|
| **R1** | In the frozen Round-1 contract (`docs/CONTRACT_API.md`); implemented this round |
| **R2** | Planned next round; API sketched below, subject to contract extension |
| **R3+** | Planned later; direction fixed, API not yet frozen |
| **N/A** | Out of scope (hardware, licensing) with substitute noted |

## 1. FEMtools Framework

| Capability | femtools API | Status |
|---|---|---|
| FE/test relational database | `core.model.FEModel` (nodes, elements, materials, properties, SPCs, sets) | R1 |
| Node/element sets, selection | `core.sets.NodeSet`, `core.sets.ElementSet` | R1 |
| Coordinate systems (rect/cyl/sph) | `core.coords.CoordSys` (rectangular R1; cyl/sph R2) | R1 |
| Unit system handling | `core.units.UnitSystem` (SI internal, convert at I/O) | R1 |
| Project persistence | `io.project.save_project` / `load_project` (`.ftproj` JSON+npz) | R1 |
| Scripting language (FEMtools Script analog) | `script.engine.ScriptEngine` — command interpreter over the Python API | R1 |
| Command-line surface | `cli` typer app: `solve-modes`, `mac`, `frf`, `update`, `pretest`, `script` | R1 |
| Python API | every package below; typed, documented | R1 |
| GUI shell | `gui` (matplotlib-embedded inspection; interactive shell R3+) | R2 |
| Mesh/geometry visualization | `viz` (matplotlib R1-level plots; pyvista/plotly optional extra) | R2 |
| Mesh generation / import CAD | not planned — import meshes via UNV/BDF | N/A |

## 2. FEMtools Dynamics

| Capability | femtools API | Status |
|---|---|---|
| Real normal modes (FEA) | `fea.eigen.solve_modes` → `ModalResult` (mass-normalized) | R1 |
| Static solution | `fea.static.solve_static` | R1 |
| Element library (BAR2, BEAM2, TRUSS2D, QUAD4, TRIA3, HEX8, TET4, MASS, SPRING, DAMPER) | `fea.elements` registry, `fea.assemble.assemble_km` | R1 |
| Modal FRF synthesis (with damping models) | `dynamics.frf.modal_frf` (modal ζ, Rayleigh, structural η) | R1 |
| Residual flexibility / residual vectors | `dynamics.residuals.residual_vectors`; upper/lower residual terms in `modal_frf` | R1 |
| Direct (full-order) FRF | `dynamics.frf.direct_frf` (dynamic stiffness inversion) | R1 |
| Harmonic forced response / ODS | `dynamics.harmonic.harmonic_response` | R1 |
| Transient response | `dynamics.time_domain.time_history` (modal superposition; Newmark R2) | R1 |
| Craig–Bampton CMS / superelements | `dynamics.craig_bampton.craig_bampton` | R1 |
| Modal-based assembly (coupling by modes) | `dynamics.mba.modal_based_assembly` | R1 |
| Free-interface CMS (MacNeal/Rubin) | `dynamics.cms_free` | R2 |
| Structural dynamics modification (SDM) | `dynamics.sdm` | R2 |
| Complex modes (general viscous damping) | `fea.eigen` state-space path | R2 |
| Random/PSD response | `dynamics.random` | R3+ |

## 3. FEMtools Pretest & Correlation

| Capability | femtools API | Status |
|---|---|---|
| Target mode selection / modal effective mass | `pretest.target_modes.effective_mass`, `select_target_modes` | R1 |
| Sensor placement — Effective Independence (EFI) | `pretest.efi.effective_independence` | R1 |
| Sensor elimination by MAC / kinetic energy ranking | `pretest.sensor.eliminate_by_mac`, `nodal_kinetic_energy` | R1 |
| Exciter placement (driving-point residues) | `pretest.exciter` | R2 |
| Test geometry definition & FE↔test node mapping | test geometry as `FEModel`; mapping in `correlation.pairing` (nearest-node R1; geometric alignment R2) | R1 |
| MAC / CoMAC / POC | `correlation.mac.mac_matrix`, `comac`, `poc` | R1 |
| Mode pairing | `correlation.pairing.pair_modes` (MAC-weighted assignment) | R1 |
| Mass-weighted cross-orthogonality | `correlation.orthogonality.cross_orthogonality` | R1 |
| FRF correlation (FRAC / CSAC / CSF) | `correlation.frf_corr.frac`, `csac`, `csf` | R1 |
| Shape expansion/reduction (Guyan, IRS, SEREP) | `fea.reduction` + `correlation.expansion` | R2 |
| ECOMAC, FMAC and extended metrics | `correlation.mac` extensions | R2 |
| Correlation report generation | `viz` + `cli` report commands | R2 |

## 4. FEMtools Model Updating

| Capability | femtools API | Status |
|---|---|---|
| Parameter/response sensitivity matrix | `updating.sensitivity.sensitivity_matrix` (semi-analytic eigen-sensitivities; finite-difference fallback) | R1 |
| WLS / Bayesian sensitivity updating (Friswell–Mottershead) | `updating.updater.update_model` → `UpdateResult` | R1 |
| Parameters: E, rho, thickness, spring k | `updating` parameter protocol | R1 |
| Responses: frequencies, MAC, FRF samples | `updating` residual definitions | R1 |
| Harmonic force identification | `updating.force_id.identify_harmonic_forces` | R1 |
| Regularization (Tikhonov, parameter weighting) | inside `update_model` options | R1 |
| Local (element-level) parameters, grouping | parameter protocol extension | R2 |
| FRF-based updating (full-curve residuals) | `updating.frf_updating` | R2 |
| Robust / uncertainty-quantified updating | `updating.uq` | R3+ |
| Automated parameter selection (subset selection) | `updating.selection` | R3+ |

## 5. FEMtools Optimization

| Capability | femtools API | Status |
|---|---|---|
| Size optimization (frequency/compliance constraints) | `optimization.size.size_optimize` (scipy SLSQP/trust-constr backend) | R1 |
| Topology optimization (SIMP + OC/MMA-style update, density filter) | `optimization.topology.topology_simp` | R1 |
| DOE — Latin hypercube, full factorial | `optimization.doe.latin_hypercube`, `full_factorial` | R1 |
| Response surface / surrogate models | `optimization.surrogate` | R2 |
| Shape optimization | `optimization.shape` | R3+ |
| Multi-objective (Pareto) | `optimization.multi` | R3+ |

## 6. FEMtools MPE (Modal Parameter Extraction)

| Capability | femtools API | Status |
|---|---|---|
| Poly-reference LSCF (PolyMAX-class) | `mpe.p_lscf.poly_lscf` | R1 |
| LSCE (complex exponential) | `mpe.lsce.lsce` | R1 |
| Operational: FDD / EFDD | `mpe.fdd.fdd`, `efdd` | R1 |
| Stabilization diagram construction | `mpe.stabilization` | R2 |
| MIMO FRF estimation from time data (H1/H2, coherence) | `mpe.frf_estimation` | R2 |
| SSI (covariance/data-driven) | `mpe.ssi` | R3+ |

## 7. FEMtools RBPE (Rigid Body Property Extraction)

| Capability | femtools API | Status |
|---|---|---|
| Mass, CoG, inertia tensor from low-frequency FRFs (mass-line) | `rbpe.rbfit.rigid_body_properties` | R1 |
| Inertia-restraint variant, mounting-stiffness correction | `rbpe.rbfit` options | R2 |

## 8. FEMtools DAQ

| Capability | femtools API | Status |
|---|---|---|
| Hardware acquisition (NI etc.) | not planned (hardware/licensing) | N/A |
| Synthetic test-data generation (noisy FRFs, shaped excitation) for MPE/OMA validation | `dynamics` synthetic generators + `examples/` | R1 |

## 9. FEA interfaces

| Capability | femtools API | Status |
|---|---|---|
| Universal file (UNV) datasets 15/2411 (nodes), 82 (trace lines), 55 (shapes), 58 (FRFs/functions), 151/164 (header/units) | `io.unv.read_unv`, `write_unv` | R1 |
| Nastran BDF subset (GRID, CQUAD4, CTRIA3, CBAR, CHEXA, MAT1, PSHELL, PBAR, SPC, FORCE) | `io.bdf.read_bdf`, `write_bdf` | R1 |
| Native project file | `io.project` (`.ftproj`) | R1 |
| Nastran OP2 results | `femtools.drivers` entry point (`SolverDriver` protocol, `docs/ARCHITECTURE.md` §10) | R2 |
| ANSYS cdb/rst | driver plug-in | R2+ |
| Abaqus inp/odb | driver plug-in | R2+ |
| LS-DYNA k | driver plug-in | R3 |

## Cross-cutting quality gates (all rounds)

* Typed public API (`py.typed` marker once subpackages land), pydantic-validated core.
* Golden analytical acceptance tests with the tolerance table of `docs/CONTRACT_API.md`.
* ruff + pytest in CI on Python 3.11 (`.github/workflows/ci.yml`); mypy gate planned R2.
* Deterministic seeds for every stochastic routine.

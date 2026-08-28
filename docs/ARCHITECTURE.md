# femtools Architecture

This document defines the layered architecture, data flow, numbering conventions, matrix
assembly strategy, result objects, error-handling policy, and extension points of `femtools`.
The public signatures frozen for Round 1 are in `docs/CONTRACT_API.md`; this document explains
how those signatures fit together and constrains how they must be implemented. The feature
scope per package is in `docs/PRODUCT_MAP.md`; the algorithmic references are in `docs/SOTA.md`.

## 1. Design principles

1. **Solver independence.** The in-memory database (`femtools.core`) is the single source of
   truth. External FE codes (Nastran, ANSYS, Abaqus, LS-DYNA) and test systems interact only
   through translators in `femtools.io` that read/write `FEModel`, `ModalResult`, `FRFResult`,
   and time/spectral records. No downstream package may parse a vendor file directly.
2. **Typed, validated data.** All persistent entities are pydantic v2 models with explicit
   units and referential integrity enforced at insertion time. Numeric payloads are numpy
   arrays with fixed dtypes: `float64` for real kernels, `complex128` for spectral kernels.
3. **Vectorized numerics.** Numeric kernels are numpy/scipy array programs. Python-level loops
   over nodes/elements are permitted only in assembly index generation and I/O, never in
   per-frequency or per-iteration inner loops.
4. **Determinism.** Given the same inputs, every function returns bit-identical results across
   runs on the same platform. Any stochastic algorithm (DOE sampling, perturbation restarts)
   takes an explicit `seed` and uses `numpy.random.Generator`; no global RNG state.
5. **Fail loudly, early, with diagnosis.** Invalid models are rejected at build time, not
   discovered as singular factorizations. Solver failures raise typed exceptions carrying the
   offending entity ids (see §7).

## 2. Layering

Dependencies point strictly downward. A layer may import from layers above it in this list,
never below it. `core` imports only numpy/scipy/pydantic.

```
Layer 0  core         FE/test relational database: nodes, elements, materials, properties,
                      SPCs, loads, sets, coordinate systems, units, test geometry, channels
Layer 1  io           UNV, Nastran BDF, project persistence (.ftproj), driver plug-ins
Layer 2  fea          element library, DOF management, sparse K/M/C assembly, static solver,
                      eigensolver (ARPACK), complex modes, reduction (Guyan/IRS/SEREP),
                      stress recovery + RBE2 constraint transform (Round 7); RBE3
                      interpolation transform + nodal stress averaging (Round 8)
Layer 3  dynamics     modal & direct FRF, harmonic response, time integration,
                      Craig–Bampton CMS, modal-based assembly, residual vectors
Layer 4  correlation  MAC/CoMAC/POC, cross-orthogonality, FRF correlation, mode pairing
         pretest      target-mode selection, effective mass, EFI, sensor elimination
Layer 5  updating     sensitivities, WLS/Bayesian parameter estimation, force identification
         optimization size optimization, SIMP topology, DOE
         mpe          p-LSCF, LSCE, FDD/EFDD estimators from measured FRF/spectral data
         rbpe         rigid-body property extraction from low-frequency FRFs
Layer 6  script       command interpreter (`ScriptEngine`) over the Python API
         cli          typer application (`femtools` entry point)
         gui / viz    plotting and interactive inspection (matplotlib; plotly/pyvista optional)
```

Rationale for the split at layer 4/5: correlation and pretest consume *pairs* of result
objects (FE vs. test) but never re-solve; updating/optimization own iteration loops that call
back into `fea`/`dynamics` through their public functions only.

## 3. Data flow

```
                 test lab                          FE authoring
                    │                                   │
        UNV 58/55 (FRFs, shapes)              BDF / UNV 2411 / API calls
                    │                                   │
                    ▼                                   ▼
              femtools.io  ─────────────────►  femtools.core.FEModel
                    │                                   │
                    │                            fea.assemble (K, M sparse)
                    │                                   │
                    │                     ┌─────────────┴─────────────┐
                    │                     ▼                           ▼
                    │              fea.static (u)             fea.eigen (ModalResult)
                    │                                                 │
                    ▼                                                 ▼
     mpe (p-LSCF / FDD / LSCE)  ────────►  correlation (MAC, pairing, orthogonality, FRAC)
     test ModalResult / FRFResult                     │
                                                      ▼
                                    updating (sensitivities → parameter estimates)
                                                      │ modified FEModel
                                                      ▼
                                    optimization (size / topology / DOE)
                                                      │
                                                      ▼
                                    femtools.io (UNV / BDF / .ftproj export)
```

Every arrow is a plain function call passing typed objects; there is no hidden global state.
A round trip (import → solve → correlate → update → export) must preserve entity ids so that
external post-processors can map results back to the source model.

## 4. Core database

`FEModel` is an in-memory relational store with integer-keyed dictionaries:

| Table | Key | Value | Referential constraints |
|---|---|---|---|
| `nodes` | node id | `Node(xyz: ndarray(3,), cs_ref, cd_ref)` | `cs_ref`/`cd_ref` in `coord_systems` |
| `elements` | element id | `Element(type, nodes: tuple[int,...], property_id)` | node ids exist; `property_id` exists; arity matches type |
| `materials` | material id | `Material(type, E, nu, rho, ...)` | — |
| `properties` | property id | `Property(type, material_id, A/I/t/k/...)` | `material_id` exists |
| `spcs` | — | `(node_id, mask: 6×bool)` | node id exists |
| `rbe2` | — | `RBE2(independent, dependents, components 1..6)` | unique id; node ids exist; independent ∉ dependents |
| `rbe3` | — | `RBE3(dependent, independents, components 1..6, weights)` | unique id; node ids exist; dependent ∉ independents; weights positive, one per independent |
| `sets` | name | `NodeSet` / `ElementSet` | member ids exist |
| `coord_systems` | cs id | `CoordSys(type, origin, rotation)` | — |
| `units` | — | `UnitSystem` (SI internal; converters at I/O boundary) | — |

Insertion (`add_node`, `add_element`, …) validates and raises `ModelError` on duplicate ids or
dangling references. Mutation of geometry/topology invalidates any cached assembly (caches are
keyed by a model revision counter incremented on every mutating call).

Test data lives in the same store: test geometry is an `FEModel` with only nodes/traces, and
measured results are the same `ModalResult`/`FRFResult` classes as FE results, so correlation
code never branches on data origin.

## 5. DOF numbering and constraint handling

* **Nodal DOF convention.** Every node carries 6 potential DOFs in its nodal displacement
  coordinate system, with fixed local indices:
  `0=UX, 1=UY, 2=UZ, 3=RX, 4=RY, 5=RZ`. Rotations are right-handed about the axes.
* **Full ordering.** Let `pos(n)` be the rank of node id `n` in ascending id order. The full
  (unconstrained) global index of `(n, d)` is `6*pos(n) + d`. This ordering is deterministic
  and independent of insertion order.
* **dof_map.** `AssemblyResult.dof_map` is the bijection `(node_id, local_dof) → active
  equation index` for the *active* set (see below), plus the inverse arrays needed to scatter
  solution vectors back to nodes. All solver outputs (`modes`, static `u`) are indexed by the
  active set; expansion to the full 6·n_nodes vector (zeros at constrained DOFs) is provided
  for export and animation.
* **Active set construction.** Starting from the full set, the assembler removes:
  1. DOFs constrained by SPCs (`mask=True` ⇒ fixed at zero; non-homogeneous SPCs via
     `solve_static(enforced=...)` since R2);
  2. DOFs with no stiffness and no mass contribution from any attached element
     (auto-SPC, e.g. rotational DOFs on a pure TRUSS/solid mesh, drilling rotation of
     flat shell patches). Auto-SPC decisions are recorded in `AssemblyResult` so users can
     audit them; silent zero-pivot factorization failures are not acceptable.
     Since Round 6, each shell node carries a right-handed triad whose third axis is
     the area-weighted average of attached element normals, so the fictitious drilling
     rotation of a flat patch is local component 5 at **any** orientation and can be
     auto-constrained. Axis-aligned models keep the identity frame (bit-identical K).
     See `docs/PRODUCT_MAP.md` (R6, shell drilling).
* **Constraint elimination.** SPCs are applied by row/column elimination (slicing the CSR
  matrix to the active set), not by penalty terms, so eigenvalues are not polluted by penalty
  artifacts. MPCs/RBEs are applied by null-space transformation `u = T q` with
  `K_r = Tᵀ K T` (and `M_r = Tᵀ M T`), keeping symmetry. Round 7 merged this as
  `fea.mpc.apply_rbe2` / `ConstraintTransform`, consuming the merged `FEModel.rbe2` table
  (`core.model.RBE2`, shared with the BDF `RBE2` card): the dependent node follows the
  independent one by the small-rotation rigid-body relation, and `assemble_km` honors
  `model.rbe2` (or an explicit `mpc=` transform). Round 8 (PRODUCT_MAP
  R8) adds the interpolation variant `fea.mpc.apply_rbe3` over the merged
  `FEModel.rbe3` table: the dependent node's listed components follow the *weighted
  average* of the independents (equal weights by default, or `RBE3.weights`) — an
  interpolation constraint, not the RBE2 rigid weld, with no penalty springs.
  `assemble_km` composes `model.rbe3` with `model.rbe2` into a single
  `ConstraintTransform`; `mpc=False` still disables all MPCs, and existing RBE2 results
  stay bit-identical when `model.rbe3` is empty. Round 9 (PRODUCT_MAP
  R9) freezes the composer itself as public contract: `fea.mpc.apply_mpc`, the
  function `assemble_km` already routes through, with `apply_rbe2` / `apply_rbe3`
  staying thin wrappers, no change to RBE2 kinematics or RBE3 weighted-average
  content, and `mpc=False` still disabling all MPCs.
  References: `docs/SOTA.md` §11–§13.

## 6. Sparse assembly

`fea.assemble.assemble_km(model) → AssemblyResult(K, M, dof_map, …)`:

1. **Element kernels.** Each element type provides `ke(coords, prop, mat) → (nd×nd)` and
   `me(...) → (nd×nd)` in the element/local frame plus its DOF signature (which of the 6 nodal
   DOFs it connects). Kernels are closed-form where available (BAR2, BEAM2 Euler–Bernoulli,
   TRUSS2D, MASS, SPRING, DAMPER) and Gauss-quadrature isoparametric otherwise
   (TRIA3, QUAD4, TET4, HEX8). Consistent mass is the default; lumped mass is an option.
2. **Local→global transformation.** `k_g = Tᵀ k_l T` with `T` block-diagonal in 3×3 direction
   cosine blocks derived from element geometry (and orientation vectors for beams).
3. **Triplet accumulation.** Assembly emits COO triplets `(rows, cols, vals)` per element type
   in vectorized batches (one call per type, arrays shaped `(n_elem, nd, nd)`), concatenates,
   and converts once: `coo_matrix(...).tocsr()`. Duplicate entries are summed by the COO→CSR
   conversion; no Python-level dict accumulation.
4. **Symmetry.** K and M are stored as full symmetric CSR (both triangles). Assembly asserts
   `max|A - Aᵀ| ≤ 1e-9 · max|A|` before returning; violation raises `AssemblyError` naming the
   element type responsible.
5. **Damping.** `C` is not assembled by default. Damping enters downstream as modal damping
   ratios, Rayleigh `C = αM + βK`, structural damping `(1+iη)K`, or discrete DAMPER elements
   (assembled on demand with the same machinery).

Complexity: O(Σ nd² ) memory for triplets, O(nnz log nnz) conversion; no dense n×n matrix is
ever formed by the framework (dense paths are allowed inside tests and for n < ~500 fallbacks).

## 7. Solvers

* **Static:** `solve_static` factors the active K with `scipy.sparse.linalg.splu` (or
  `spsolve` for single RHS) and returns the active-set displacement vector. A near-zero pivot
  triggers `SolverError` listing suspect DOFs (mapped back to node ids via `dof_map`).
* **Eigen:** `solve_modes(model, n_modes, shift)` solves `K φ = λ M φ` with ARPACK
  (`scipy.sparse.linalg.eigsh`) in shift-invert mode, `sigma = shift`. For free-free or
  partially constrained structures (semidefinite K), a small negative sigma (default
  `-(2π·0.1)²` when `shift=0`) keeps the factorization nonsingular while converging rigid-body
  modes; negative computed eigenvalues within tolerance are clamped to zero frequency.
  Dense `scipy.linalg.eigh` is the fallback when `n_modes ≥ n_active - 1` (ARPACK limit).
  Returned modes are **mass-normalized** (`Φᵀ M Φ = I` to 1e-8, enforced by post-scaling and
  verified), frequencies in Hz ascending. Eigenvalues are `ω² [rad²/s²]`.
* **Reduction (merged Round 4 — `fea.reduction`, `docs/PRODUCT_MAP.md` R4):** Guyan, IRS,
  SEREP as explicit transformation-matrix builders (`ReductionResult` carries `T` with
  `K_r = TᵀKT`, `M_r = TᵀMT`) reused by pretest (test-DOF reduction) and correlation
  (shape expansion via `correlation.expansion.expand_guyan` / `expand_serep`).
* **Stress recovery (Round 7 — `fea.recover`, `docs/PRODUCT_MAP.md` R7):**
  `recover_stress` / `recover_strain` evaluate element strain and stress from a displacement
  solution at the element centroid (or averaged Gauss points) through the same element
  kernels the assembler uses, returning a `StressResult`. Recovery is an fea-layer
  capability: it consumes `StaticResult` / `ModalResult` displacement fields and never
  re-solves. Linear elasticity only; Round 7 does no nodal smoothing. Round 8
  (PRODUCT_MAP R8) adds `fea.recover.average_nodal`: the centroid
  `StressResult` averaged onto incident nodes with `1/n_adj` weights — plain nodal
  averaging, deliberately not Zienkiewicz–Zhu SPR — exact at every node on a
  constant-stress patch.
  References: `docs/SOTA.md` §11 (Cook et al.; Bathe; Barlow points), §12 (averaging
  vs. SPR).

## 8. Result objects

All result objects are immutable after construction, carry provenance metadata
(`model_name`, `created`, solver options), serialize to `.npz` (arrays) + JSON sidecar
(metadata), and are the *only* currency between layers.

| Object | Producer | Payload | Invariants |
|---|---|---|---|
| `AssemblyResult` | `fea.assemble` | `K, M: csr_matrix`, `dof_map`, auto-SPC report | symmetric; active set consistent with model |
| `ModalResult` | `fea.eigen`, `mpe.*` | `freq_hz (m,)`, `eigenvalues (m,)`, `modes (n_active, m)`, `generalized_mass (m,)`, `dof_map` | ascending freq; mass-normalized when M known; complex modes allowed for test/MPE data |
| `FRFResult` | `dynamics.frf`, `io.unv` (dataset 58) | `H: complex128 (n_out, n_in, n_freq)`, `freq_hz`, input/output DOF descriptors, `kind ∈ {receptance, mobility, accelerance}` | frequency vector strictly increasing; unit-consistent with `kind` |
| `StaticResult` | `fea.static` | `u (n_active,)`, reaction forces at SPC DOFs | equilibrium residual reported |
| `UpdateResult` | `updating.updater` | parameter history, residual history, weighted covariance, convergence flag | monotone weighted residual or flagged non-convergence |

The `.npz` serialization promise is realized per result type as rounds land: Round 7
shipped `dynamics.superelement.dump_cms` / `load_cms` for CMS reduced models; Round 8
(PRODUCT_MAP R8) adds `dynamics.frf.dump_frf` / `load_frf` for
`FRFResult`, with `H` and `freq_hz` bit-identical after a load round trip. Round 9
(PRODUCT_MAP R9) extends the same pattern to
`dynamics.random.dump_psd` / `load_psd` for `PSDResult`, with the stored spectra and
`freq_hz` bit-identical after load and `psd_response` numerics untouched.

Shape/DOF compatibility between two results (e.g. FE vs. test in MAC) is established through
DOF descriptors `(node_id, local_dof)`, never through positional assumption; `pair_modes` and
geometry mapping utilities produce the common-DOF selection matrices.

## 9. Error handling

Exception hierarchy (all in `femtools.core.errors`, re-exported from `femtools.core` and —
lazily, like every contract symbol — from the `femtools` top level):

```
FemtoolsError(Exception)
├── ModelError(ValueError)       # integrity: duplicate id, dangling reference, bad arity
│   └── MeshError                # geometry/topology-specific integrity failures
├── UnitError(ValueError)        # unit-system conversion failures
├── FileFormatError(ValueError)  # unparseable UNV/BDF content; carries file/line/dataset/card
├── AssemblyError                # degenerate geometry (zero-length bar, negative Jacobian), asymmetry
├── SolverError                  # singular factorization, ARPACK breakdown; carries suspect DOFs
│   └── ConvergenceError         # iterative process exceeded max_iter / diverged; carries history
└── CompatibilityError           # mismatched DOF sets / frequency grids between result objects
```

`ModelError`, `UnitError` and `FileFormatError` also subclass `ValueError` so pre-hierarchy
`except ValueError` call sites keep working; io subclasses (`BdfError`, `ProjectError`)
re-parent onto `FileFormatError`.

Policy:

1. Validation happens at the earliest layer that has enough information: pydantic validators
   for field-level checks, `FEModel.add_*` for referential checks, assembler for geometric
   checks, solver for spectral checks.
2. Unsupported-but-recognized I/O content (e.g. an unimplemented BDF card) is skipped with a
   `UserWarning` subclass (`UnsupportedCardWarning`) listing card name and line; unrecognized
   content raises `FileFormatError`. Import never silently drops data without a warning.
3. Numerical guards: functions never return NaN/Inf silently; post-conditions (mass
   normalization, MAC ∈ [0,1] up to roundoff, Hermitian symmetry of spectral matrices) are
   asserted with the tolerances of `docs/CONTRACT_API.md`.
4. Messages must name entities by user-visible id (node 42, element 7, parameter "E:mat 1"),
   not by internal index.

## 10. Extension points

1. **Element registry.** `femtools.fea.elements` maintains a registry
   `{type_name → ElementFormulation}`; `available_elements()` lists it. New element types
   register a formulation object (kernel functions + DOF signature + arity) without touching
   the assembler.
2. **FE solver drivers.** Vendor interfaces beyond the built-in UNV/BDF translators implement
   the `drivers.base.SolverDriver` protocol (merged Round 4, `docs/PRODUCT_MAP.md` R4 —
   a runtime-checkable PEP 544 `Protocol`, so conformance is structural: any object with
   the right members qualifies, no registration or subclassing required):

   ```python
   @runtime_checkable
   class SolverDriver(Protocol):
       name: str                              # "nastran-punch", "ansys-cdb", ...
       def is_available(self) -> bool: ...    # executable/license probe, never raises
       def write_input(self, model: FEModel, workdir: str | Path) -> Path: ...
       def run(self, input_file: str | Path, timeout: float | None = None) -> Path: ...
       def read_modal(self, result_file: str | Path) -> ModalResult: ...
   ```

   femtools ships the contract and one optional concrete **text** driver,
   `NastranPunchDriver` (`docs/PRODUCT_MAP.md` R7): `write_input` emits the model through
   the public `write_bdf` translator plus a SOL 103 case control requesting punch output,
   `read_modal` parses the `.pch` text with `read_pch`, `is_available()` probes for a
   local executable (never raises), and `run()` raises `SolverError` when no installation
   is present. femtools never bundles, requires, or tests against the vendor binary —
   everything the driver touches is text, and no proprietary binary parser exists in the
   codebase. Adapters are built on the text translators of `femtools.io`, which since
   Round 4 include Nastran punch (`.pch`) results (`read_pch`/`write_pch`) and the
   ANSYS CDB NBLOCK/EBLOCK mesh subset (`read_cdb`) alongside the Round-1 BDF/UNV
   read/write.    Round 6 added Abaqus INP (`io.inp.read_inp`/`write_inp`) and LS-DYNA K
   (`io.kfile.read_k`) text-subset translators over publicly documented card layouts;
   Round 7 added the matching writers (`io.cdb.write_cdb`, `io.kfile.write_k`) and
   BDF `INCLUDE`/`RBE2` reading.
   Round 8 (`docs/PRODUCT_MAP.md` R8) extends the same pattern with two
   more optional **text** drivers: `drivers.ansys.AnsysCdbDriver` (`write_input` =
   `write_cdb`; `is_available()` probes `shutil.which` for `ansys`/`mapdl`-style
   aliases and never raises; `run()` raises `SolverError` on a missing executable,
   nonzero exit, or timeout; `read_modal` consumes the existing `.pch`/`.unv` text
   readers only — a `.rst` path raises `SolverError` naming RST as N/A) and
   `drivers.abaqus.AbaqusInpDriver` (`write_input` = `write_inp`, same availability and
   `SolverError` conventions, ODB is N/A, results from `.unv`/`.pch` text only). Tests
   never require an ANSYS or Abaqus install — they stub shell executables exactly like
   the Round-7 Nastran driver tests.
   Round 9 (`docs/PRODUCT_MAP.md` R9) thickens `NastranPunchDriver`
   with a **text** static path: `write_input(..., sol=101)` (or an equivalent explicit
   static method) emits a public SOL 101 case control requesting
   `DISPLACEMENT(PUNCH)=ALL`, and a driver `read_static` parses the punch
   `$DISPLACEMENTS` text (an `io.pch` sibling reader). The SOL 103 modal default is
   unchanged, the same `SolverError` conventions apply, and tests keep stubbing the
   executable.
   Closed binary result dumps (Nastran OP2, ANSYS rst, Abaqus odb) remain out
   of scope (N/A) — Rounds 7–9 change nothing there: the SOL 101 path is punch *text*,
   not OP2, so there is still no OP2, no RST, no ODB. The
   protocol stays the extension point for third-party binary drivers.
   Only a driver may depend on vendor
   formats; results always land as `ModalResult`/`FRFResult`, and a failed external
   run raises `SolverError`.
3. **Parameters for updating/optimization.** Updatable quantities implement a `Parameter`
   protocol (`get(model)`, `set(model, value)`, `bounds`); Round 1 ships E, rho, shell
   thickness, spring stiffness. New parameter types plug in without changes to the estimator.
4. **Script commands.** The `ScriptEngine` command table maps verb phrases to API calls; new
   commands register (verb, argspec, handler) tuples.

## 11. Concurrency and performance

* Heavy kernels (factorization, ARPACK, BLAS-3 products) release the GIL inside scipy; the
  framework itself is single-threaded and thread-compatible (no mutable module state).
* Per-frequency FRF loops are vectorized over the frequency axis; direct FRF factors the
  dynamic stiffness once per frequency but reuses symbolic factorization where the sparsity
  pattern is constant.
* DOE and multi-start updating parallelize at the process level (`concurrent.futures`) over
  independent model evaluations; results are reduced deterministically (ordered by sample
  index, not completion time).

## 12. Testing hooks

Every layer exposes deterministic golden cases (axial bar, Euler cantilever, free-free beam,
2-DOF spring-mass) used by `tests/` against the tolerance table in `docs/CONTRACT_API.md` and
the acceptance philosophy in `docs/SOTA.md` §9. CI (`.github/workflows/ci.yml`) runs, for
every push and pull request on Python 3.11: blocking ruff, an import smoke test, a public-API
resolution check over `femtools.__all__`, and strict pytest (an empty collection fails);
mypy runs as an additional non-blocking step.

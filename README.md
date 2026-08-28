# femtools

Solver-independent CAE framework for structural dynamics, experimental/FE correlation, finite-element model updating, and design optimization.

This is an original implementation of the capability set associated with the commercial FEMtools product family (Framework, Dynamics, Pretest & Correlation, Model Updating, Optimization, MPE, RBPE). It is not affiliated with Dynamic Design Solutions. Per-capability status is tracked in `docs/PRODUCT_MAP.md`; the published algorithm references are in `docs/SOTA.md`; layering and conventions are in `docs/ARCHITECTURE.md`.

## Install

```bash
pip install -e ".[dev]"
```

Optional extras: `viz` (plotly backend, pyvista 3-D view — matplotlib stays the default and never requires either) and `web` (FastAPI GUI backend; a stdlib fallback is built in).

## Quick start

```python
from femtools.core.model import FEModel
from femtools.fea.eigen import solve_modes
from femtools.fea.recover import recover_stress
from femtools.correlation.mac import mac_matrix

model = FEModel(name="cantilever")
# ... build mesh, materials, BCs (see examples/) ...
modal = solve_modes(model, n_modes=8)
mac = mac_matrix(modal.modes, modal.modes)
```

```bash
femtools solve-modes model.json --n-modes 10
femtools mac a.npz b.npz
femtools recover-stress model.json --load "12:2=-1000"
femtools script analysis.fsl
```

## Modules

| Module | Capability |
|--------|------------|
| `core` | FE/test relational database (nodes, elements, materials, properties, SPCs, loads), node/element sets, units, coordinate systems, RBE2/RBE3 constraint containers |
| `io` | Text translators only: UNV, Nastran BDF (incl. `INCLUDE`, RBE2/RBE3 cards), Nastran punch `.pch`, ANSYS CDB, Abaqus INP, LS-DYNA K subsets, `.ftproj` project files |
| `fea` | Element library (bars, beams, flat shells, solids), sparse assembly, statics (incl. enforced displacement), Lanczos/ARPACK real and complex eigen, Guyan/IRS/SEREP reduction, stress/strain recovery + nodal averaging, RBE2/RBE3 multipoint-constraint transforms |
| `dynamics` | Modal/direct FRF (+ npz dump/load), harmonic ODS, transient, random/PSD response (force and base-acceleration input), Craig–Bampton and free-interface CMS (+ superelement dump/load), modal-/FRF-based assembly, SDM, modal energy diagnostics, synthetic test data |
| `drivers` | `SolverDriver` protocol plus optional Nastran / ANSYS / Abaqus **text** drivers (BDF/CDB/INP in, punch/UNV text results out); closed binary results (OP2/RST/ODB) are out of scope by design |
| `pretest` | Target modes, modal effective mass, EFI, MAC/kinetic-energy sensor ranking, exciter placement, candidate DOFs, sensor mass-loading checks |
| `correlation` | MAC family (MAC, CoMAC, FMAC, NMD, MACX, per-DOF contributions), POC / cross-orthogonality, FRAC/CSAC/FDAC, mode pairing, geometry alignment + nearest-node DOF mapping, shape expansion |
| `updating` | Semi-analytic sensitivities, WLS/Bayesian updating, FRF-based and static-deflection updating, parameter subset selection, uncertainty (covariance / Monte Carlo), force identification |
| `optimization` | Size, SIMP topology, topometry (element-wise sizing on an existing mesh), shape, DOE, response-surface surrogates, weighted Pareto |
| `mpe` | p-LSCF (PolyMAX-class), LSCE, FDD/EFDD, SSI (covariance- and data-driven), H1/H2 FRF estimation + coherence, stabilization diagrams |
| `rbpe` | Rigid-body mass / CoG / inertia tensor from low-frequency FRFs |
| `script` | `ScriptEngine` command interpreter — `ADD` (nodes/elements/materials/loads/`RBE2`/`RBE3`), `SET`, `SOLVE MODES` / `SOLVE STATIC`, `RECOVER STRESS`, `MAC`, `SAVE`, `PRINT` |
| `cli` | `femtools` typer app — `solve-modes`, `read-mesh`, `write-mesh`, `recover-stress`, `plot-stress`, `mac`, `report-mac`, `frf`, `reduce`, `estimate-frf`, `update`, `pretest`, `script`, `gui` |
| `gui` / `viz` | Web GUI shell (stdlib http, FastAPI optional), matplotlib plotting (mesh, modes, FRFs, MAC, stress fields), optional plotly backend and pyvista 3-D view |

## Examples

Self-checking demo scripts live in `examples/` (cantilever modes, FRF synthesis, MAC,
EFI pretest, Guyan/SEREP, CMS, H1/SSI, RBE2, stress recovery, topometry, model
updating). Some write plot artifacts (e.g. `frf_synthesis.png`) into the working
directory; those are gitignored.

## Tests

```bash
pytest
```

## License

MIT

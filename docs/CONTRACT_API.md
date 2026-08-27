# Public API Contract (frozen for Round 1)

All implementations MUST satisfy these signatures. Tests are written against this contract.

Package: `femtools`

## Core database

```python
from femtools.core.model import FEModel, Node, Element, Material, Property, DOFSet
from femtools.core.sets import NodeSet, ElementSet
from femtools.core.units import UnitSystem
from femtools.core.coords import CoordSys

model = FEModel(name="beam")
n1 = model.add_node(id=1, xyz=(0.0, 0.0, 0.0))
n2 = model.add_node(id=2, xyz=(1.0, 0.0, 0.0))
mat = model.add_material(id=1, type="isotropic", E=210e9, nu=0.3, rho=7850.0)
prop = model.add_property(id=1, type="bar", material_id=1, A=1e-4)
el = model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
```

`FEModel` fields: `nodes: dict[int, Node]`, `elements: dict[int, Element]`, `materials`, `properties`, `spcs`, `sets`, `coord_systems`, `units`.
`Node.xyz` is `ndarray (3,)`. Element types Round 1: `BAR2`, `BEAM2`, `TRUSS2D`, `QUAD4`, `TRIA3`, `HEX8`, `TET4`, `MASS`, `SPRING`, `DAMPER`.

## I/O

```python
from femtools.io.unv import read_unv, write_unv  # datasets 15/2411, 82, 55, 58, 151/164
from femtools.io.bdf import read_bdf, write_bdf  # GRID, CQUAD4, CTRIA3, CBAR, CHEXA, MAT1, PSHELL, PBAR, SPC, FORCE
from femtools.io.project import save_project, load_project  # JSON/npz hybrid .ftproj
```

## FEA

```python
from femtools.fea.assemble import assemble_km, AssemblyResult  # K, M sparse csr, dof_map
from femtools.fea.static import solve_static  # -> ndarray ndof
from femtools.fea.eigen import solve_modes  # -> ModalResult(freq_hz, eigenvalues, modes, generalized_mass)
from femtools.fea.elements import available_elements
```

`solve_modes(model, n_modes=10, shift=0.0) -> ModalResult`
Modes mass-normalized: `phi.T @ M @ phi ≈ I`.
Frequencies in Hz, ascending, rigid-body modes allowed (freq ~ 0).

## Dynamics

```python
from femtools.dynamics.frf import modal_frf, direct_frf
from femtools.dynamics.harmonic import harmonic_response
from femtools.dynamics.mba import modal_based_assembly
from femtools.dynamics.craig_bampton import craig_bampton
from femtools.dynamics.time_domain import time_history
from femtools.dynamics.residuals import residual_vectors
```

`modal_frf(modal, inputs, outputs, freq_hz, damping) -> FRFResult` with complex array `(n_out, n_in, n_freq)`.
Damping: modal (per-mode zeta), Rayleigh (alpha, beta), structural (eta).

## Correlation

```python
from femtools.correlation.mac import mac_matrix, comac, poc
from femtools.correlation.pairing import pair_modes
from femtools.correlation.frf_corr import frac, csac, csf
from femtools.correlation.orthogonality import cross_orthogonality
```

MAC: `mac[i,j] = |phi_a[:,i].conj() @ phi_b[:,j]|^2 / (||phi_a_i||^2 ||phi_b_j||^2)`
Identical bases => diagonal ~ 1, off-diagonal ~ 0 (tol 1e-10 for orthonormal copies).

## Pretest

```python
from femtools.pretest.target_modes import effective_mass, select_target_modes
from femtools.pretest.efi import effective_independence
from femtools.pretest.sensor import eliminate_by_mac, nodal_kinetic_energy
```

EFI: Kammer Effective Independence. Returns ranked candidate DOF ids and EFI values.

## Updating

```python
from femtools.updating.sensitivity import sensitivity_matrix
from femtools.updating.updater import update_model, UpdateResult
from femtools.updating.force_id import identify_harmonic_forces
```

Sensitivity-based Bayesian / weighted least-squares (Friswell–Mottershead). Parameters: E, rho, thickness, spring_k. Responses: frequencies, MAC, FRF samples.

## Optimization

```python
from femtools.optimization.size import size_optimize
from femtools.optimization.topology import topology_simp
from femtools.optimization.doe import latin_hypercube, full_factorial
```

## MPE / RBPE

```python
from femtools.mpe.p_lscf import poly_lscf
from femtools.mpe.fdd import fdd, efdd
from femtools.mpe.lsce import lsce
from femtools.rbpe.rbfit import rigid_body_properties
```

## Script / CLI

```python
from femtools.script.engine import ScriptEngine
ScriptEngine().run("NEW PROJECT; ADD NODE 1 0 0 0")
```

CLI: `femtools --help` with subcommands `solve-modes`, `mac`, `frf`, `update`, `pretest`, `script`.

## Numerical tolerances (acceptance)

| Case | Metric | Tol |
|------|--------|-----|
| Axial bar 2-node, 1 mode vs analytical | rel freq | 1e-8 |
| Euler cantilever 10+ BEAM2, first 3 bending freq vs EB theory | rel | 2% |
| Mass-normalized modes | max\|Phi.T M Phi - I\| | 1e-8 |
| MAC(self) diagonal | 1 - min(diag) | 1e-12 |
| EFI on 2-mode 10-DOF toy | selected sensors keep MAC off-diag < 0.15 | — |
| Updating E on 2-param beam (10% error) | recover E within 2% | — |
| Modal FRF vs direct FRF (light damping, 20 modes) | rel L2 on 0.2–0.8 fmax | 5% |

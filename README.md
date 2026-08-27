# femtools

Solver-independent CAE framework for structural dynamics, experimental/FE correlation, finite-element model updating, and design optimization.

This is an original implementation of the capability set associated with the commercial FEMtools product family (Framework, Dynamics, Pretest & Correlation, Model Updating, Optimization, MPE, RBPE). It is not affiliated with Dynamic Design Solutions.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from femtools.core.model import FEModel
from femtools.fea.eigen import solve_modes
from femtools.correlation.mac import mac_matrix

model = FEModel(name="cantilever")
# ... build mesh, materials, BCs ...
modal = solve_modes(model, n_modes=8)
mac = mac_matrix(modal.modes, modal.modes)
```

```bash
femtools solve-modes model.json --n-modes 10
femtools mac a.npz b.npz
```

## Modules

| Module | Capability |
|--------|------------|
| `core` | FE/test relational database, mesh, materials, sets, units, coords |
| `io` | UNV, Nastran BDF, project files |
| `fea` | Element library, sparse assembly, statics, Lanczos/ARPACK eigen |
| `dynamics` | Modal/direct FRF, harmonic ODS, MBA, Craig–Bampton, time domain |
| `pretest` | Target modes, EFI, MAC elimination, kinetic energy sensor ranking |
| `correlation` | MAC, CoMAC, POC, FRAC/CSAC, orthogonality, pairing |
| `updating` | Sensitivity, Bayesian/WLS model updating, force identification |
| `optimization` | Size, SIMP topology, DOE |
| `mpe` | p-LSCF, FDD/EFDD, LSCE |
| `rbpe` | Rigid-body inertia from low-frequency FRFs |
| `script` / `cli` / `gui` | Automation, commands, visualization |

## Tests

```bash
pytest
```

## License

MIT

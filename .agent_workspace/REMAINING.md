# Remaining capabilities (post Round 3)

Continue the original 1:1 functional equivalent of FEMtools. **Do not** implement NI/DAQ hardware or proprietary closed binaries. Original algorithms from public literature only.

Rounds 4–5 froze and hardened the R3+ APIs below. Round 6 freezes the R5+ APIs at the bottom. Existing `docs/CONTRACT_API.md` still holds; do not break it.

## Round 4 APIs (merged, R5-hardened)

```python
from femtools.fea.reduction import guyan, irs, serep, ReductionResult
from femtools.fea.eigen import solve_complex_modes, ComplexModalResult
from femtools.dynamics.cms_free import rubin, macneal, FreeCMSResult
from femtools.dynamics.random import psd_response, PSDResult
from femtools.pretest.exciter import driving_point_residues, select_exciters
from femtools.correlation.expansion import expand_serep, expand_guyan
from femtools.correlation.mac import fmac
from femtools.correlation.alignment import align_geometry
from femtools.updating.frf_updating import update_from_frf
from femtools.updating.selection import select_parameters
from femtools.optimization.surrogate import fit_rsm, predict_rsm
from femtools.optimization.multi import pareto_weighted
from femtools.mpe.frf_estimation import estimate_h1, estimate_h2, coherence
from femtools.mpe.ssi import ssi_cov
from femtools.rbpe.rbfit import rigid_body_properties  # restraint=, mount_k=
from femtools.io.pch import read_pch, write_pch
from femtools.io.cdb import read_cdb
from femtools.drivers.base import SolverDriver
```

## Round 6 frozen APIs (R5+ → implement now)

```python
from femtools.updating.uq import parameter_covariance, monte_carlo_update, UQResult
from femtools.optimization.shape import shape_optimize, ShapeResult
from femtools.mpe.ssi import ssi_data
from femtools.io.inp import read_inp
from femtools.io.kfile import read_k
from femtools.correlation.mac import nmd, macx
from femtools.dynamics.energy import modal_strain_energy, modal_kinetic_energy
```

Shell drilling: `assemble_km` on an arbitrarily oriented flat shell must not retain a fictitious drilling mechanism (per-node rotational frames). Verified by `fea.verification.shell_drilling_orientation_gap`.

## Explicitly still N/A

NI DAQ, commercial OP2/RST/ODB binary dumps, license servers.

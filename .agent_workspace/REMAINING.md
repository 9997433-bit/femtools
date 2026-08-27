# Remaining capabilities (post Round 3)

Continue the original 1:1 functional equivalent of FEMtools. **Do not** implement NI/DAQ hardware or proprietary closed binaries. Original algorithms from public literature only.

Frozen extra APIs for this cycle (Round 4–6). Existing CONTRACT_API still holds; do not break it.

## fea (`src/femtools/fea/**`) — R4-O1

```python
from femtools.fea.reduction import guyan, irs, serep, ReductionResult
# guyan(K, master) -> T, Krr; slave recovered T_gs = -Kss^{-1} Ksm
# irs(K, M, master) -> Improved Reduced System (O'Callahan)
# serep(phi, master_rows) -> T so phi ≈ T @ phi[master]
from femtools.fea.eigen import solve_complex_modes, ComplexModalResult
# state-space / quadratic eigenvalue; freq_hz, zeta, modes_complex
# solve_modes: if n_modes >= n_free and Kff SPD, use scipy.linalg.eigh (not shift-invert)
```

HEX8 Simo–Rifai EAS is optional if it improves trapezoidal distortion without breaking 98.6% cantilever or 6 rigid modes.

## dynamics — R4-O2

```python
from femtools.dynamics.cms_free import rubin, macneal, FreeCMSResult
# free-interface residual-flexibility CMS
from femtools.dynamics.random import psd_response, PSDResult
# modal PSD: S_uu(w) from force PSD; return rms and 1-sigma
```

## pretest / correlation — R4-O3

```python
from femtools.pretest.exciter import driving_point_residues, select_exciters
from femtools.correlation.expansion import expand_serep, expand_guyan
from femtools.correlation.mac import fmac
from femtools.correlation.alignment import align_geometry  # rigid Procrustes
```

## updating / optimization / mpe / rbpe — R4-O4

```python
from femtools.updating.frf_updating import update_from_frf
from femtools.updating.selection import select_parameters  # EFS / column subset
from femtools.optimization.surrogate import fit_rsm, predict_rsm  # quadratic RSM
from femtools.optimization.multi import pareto_weighted  # weighted-sum + optional NSGA-lite
from femtools.mpe.frf_estimation import estimate_h1, estimate_h2, coherence
from femtools.mpe.ssi import ssi_cov
from femtools.rbpe.rbfit import rigid_body_properties  # add restraint= and mount_k=
```

## io / drivers — R4-F2

```python
from femtools.io.pch import read_pch, write_pch  # Nastran punch text (eigenvalues, modes)
from femtools.io.cdb import read_cdb  # ANSYS CDB NBLOCK/EBLOCK subset
from femtools.drivers.base import SolverDriver  # Protocol only; no OP2 binary parser required
```

## CLI / viz — R4-F4

`femtools report-mac`, `femtools reduce`, `femtools estimate-frf` if the kernels exist. Correlation HTML/text report via viz.

## docs — R4-F1 / R4-F3

Retag PRODUCT_MAP R3-done vs later. Algorithm notes. Examples for Guyan, H1, SSI, Rubin.

## Explicitly still N/A

NI DAQ, commercial OP2/RST/ODB binary dumps, license servers.

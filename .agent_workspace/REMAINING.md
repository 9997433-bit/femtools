# Remaining capabilities — Cycle C (Rounds 7–9)

Continue the original 1:1 functional equivalent of FEMtools. **Do not** implement NI/DAQ hardware or proprietary closed binaries (OP2/RST/ODB). Public algorithms and public text-card layouts only.

Round 7 freezes the APIs below. Existing `docs/CONTRACT_API.md` still holds; do not break it. Goldens (HEX8 98.6%, Rubin 0.028%, H1/H2=γ², 6 RBM on tilted shells) must stay.

**Round 7 landed** on `cursor/femtools-cycle-c-d551` (PR #3). The names below import and are
stable top-level exports.

## Round 8 frozen APIs

Round 8 thickens the 1:1 workflow on top of Round 7: interpolation MPCs, nodal stress
average, extra text drivers, FRF persistence, mapped-shape MAC, plot/script/GUI, and a
static-update convenience. Do not reopen N/A rows.

```python
from femtools.core.model import RBE3  # already on FEModel.rbe3 / add_rbe3

from femtools.fea.mpc import apply_rbe3
# Interpolation MPC: dependent node = weighted average of independents.
# assemble_km honors model.rbe3 (composed with model.rbe2). Free-free still 6 RBM.

from femtools.fea.recover import average_nodal
# Average centroid StressResult onto incident nodes (1/n_adj). Not ZZ-SPR.
# Constant-stress patch remains exact at every node.

from femtools.io.bdf import read_bdf, write_bdf  # parse + emit RBE3 via add_rbe3

from femtools.drivers.ansys import AnsysCdbDriver
from femtools.drivers.abaqus import AbaqusInpDriver
# SolverDriver text adapters. write_cdb / write_inp. run() raises SolverError if
# the executable is missing. read_modal from .pch/.unv text only. RST/ODB → SolverError.
# Tests must not require ANSYS or Abaqus.

from femtools.dynamics.frf import dump_frf, load_frf  # or dynamics.superelement analog
# npz dump/load of FRFResult; H and freq_hz bit-identical after load.

from femtools.correlation.dofmap import mapped_mode_matrix
# FE mode rows at map_nearest_nodes ids. Translated cube → MAC diagonal 1.

from femtools.viz.plots import plot_stress
# Color mesh by StressResult.von_mises (or a component). matplotlib default.

from femtools.updating.updater import update_from_static
# static_displacement_response + update_model. 10% E from tip deflection.
```

## Explicitly still N/A

NI DAQ, commercial OP2/RST/ODB binary dumps, license servers, CAD kernels.

## Round 7 frozen APIs

```python
from femtools.core.model import RBE2  # already on FEModel.rbe2 / add_rbe2

from femtools.fea.recover import recover_stress, recover_strain, StressResult
# Centroid (or averaged Gauss) stress/strain for BAR2, BEAM2, QUAD4, TRIA3, HEX8, TET4.
# Constant-strain patch test to 1e-12. No nonlinear/plasticity.

from femtools.fea.mpc import apply_rbe2, ConstraintTransform
# Build T from model.rbe2 (and/or explicit apply_rbe2). assemble_km(..., mpc=T)
# or assemble_km honors model.rbe2 by default. Free-free two-node rigid pair: 6 RBM.

from femtools.io.cdb import write_cdb
from femtools.io.kfile import write_k
# Round-trip the Round-6 HEX8/QUAD4/BEAM2 acceptance decks.

from femtools.io.bdf import read_bdf
# Follow INCLUDE (relative path, max depth 8, cycle-safe). Parse RBE2 (and RBE3 if cheap)
# into FEModel.add_rbe2. Do not invent INCLUDE from copyrighted decks.

from femtools.drivers.nastran import NastranPunchDriver
# Concrete SolverDriver: write_bdf + SOL 103 case control, read_pch.
# is_available() via shutil.which. run() raises SolverError if the executable is missing.
# Tests must not require a Nastran install.

from femtools.dynamics.random import psd_response  # new kw base_accel=
# Base-acceleration PSD. SDOF RMS vs closed form / Miles.

from femtools.dynamics.superelement import dump_cms, load_cms
# npz dump/load of CraigBamptonResult / FreeCMSResult (K, M, T, boundary ids).

from femtools.correlation.dofmap import map_nearest_nodes
# xyz_test (n,3) vs model/xyz_fe → (fe_ids, distances). PRODUCT_MAP claimed this since R1.

from femtools.correlation.mac import mac_contribution
# Per-DOF contribution to a single MAC pair (same inputs as mac_value).

from femtools.optimization.topometry import topometry_optimize, TopometryResult
# Element-wise thickness (or density) field on an existing mesh; OC or SLSQP;
# min-compliance 2-D plate first. Distinct from topology_simp (which builds its own grid).

from femtools.updating.responses import static_displacement_response
# Plug solve_static displacements into update_model as a response.
```

Optional plotly already exists. Round 7 viz: optional **pyvista** `plot_mesh3d` if pyvista imports, matplotlib remains default.

## Explicitly still N/A

NI DAQ, commercial OP2/RST/ODB binary dumps, license servers, CAD kernels.

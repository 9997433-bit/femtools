# FEA algorithms — elements, assembly, solvers

Spec for `femtools.fea` (owner: R1-O1). Frozen entry points per `docs/CONTRACT_API.md`:

```python
from femtools.fea.assemble import assemble_km, AssemblyResult
from femtools.fea.static import solve_static
from femtools.fea.eigen import solve_modes           # -> ModalResult
from femtools.fea.elements import available_elements
# Round-4 additions (REMAINING.md, owner R4-O1) — see §9:
from femtools.fea.reduction import guyan, irs, serep, ReductionResult
# Round-7 additions (REMAINING.md, owner R7-O1; landed and measured) — see §10–§11:
from femtools.fea.recover import recover_stress, recover_strain, StressResult
from femtools.fea.mpc import apply_rbe2, ConstraintTransform
# Round-8 additions (REMAINING.md, owner R8-O1; pending on this tree) — see §12–§13:
from femtools.fea.mpc import apply_rbe3
from femtools.fea.recover import average_nodal
# Round-10 additions (REMAINING.md, owner R10-O1; landed on this tree
# 2026-08-28) — see §14–§15:
from femtools.fea.elements import tet10   # registers etype "TET10"
from femtools.fea.recover import recover_spr
```

Notation: $K$, $M$ global stiffness/mass (`scipy.sparse.csr_matrix`, symmetric),
$\lambda_r = \omega_r^2$ eigenvalues, $\phi_r$ mode shapes, $E$ Young's modulus, $\nu$ Poisson,
$\rho$ density, $G = E / (2(1+\nu))$.

## 1. DOF convention and `dof_map`

Every node carries up to 6 DOFs, Nastran-style codes `1..6` = `(UX, UY, UZ, RX, RY, RZ)`.
A DOF is *active* if at least one attached element supplies stiffness or mass for it and it is
not eliminated by an SPC. `AssemblyResult.dof_map` is the bijection

```python
dof_map: dict[tuple[int, int], int]   # (node_id, dof_code 1..6) -> row index in K, M
```

over active DOFs only. All downstream modules (`dynamics`, `correlation`, `pretest`,
`updating`) address DOFs as `(node_id, dof_code)` tuples and translate through `dof_map`.
Recommended full signature (non-frozen fields may grow, never change):

```python
class AssemblyResult:
    K: csr_matrix          # (n_a, n_a) symmetric
    M: csr_matrix          # (n_a, n_a) symmetric positive semi-definite
    dof_map: dict[tuple[int, int], int]
    ndof: int              # n_a = number of active DOFs
    constrained: list[tuple[int, int]]  # DOFs removed by SPC elimination

def assemble_km(model: FEModel) -> AssemblyResult: ...
```

## 2. Element library

### 2.1 BAR2 / TRUSS2D — axial (and torsion) rod

Local axial DOFs $(u_1, u_2)$, length $L$, area $A$:

$$k_a = \frac{EA}{L}\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix},\qquad
  m_a = \frac{\rho A L}{6}\begin{bmatrix}2 & 1\\ 1 & 2\end{bmatrix}\ \text{(consistent)},\qquad
  m_a^{lump} = \frac{\rho A L}{2}\begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}.$$

Torsion (BAR2 with `J` set) is identical with $EA \to GJ$ and $\rho A \to \rho I_p$
($I_p$ polar area moment; for circular sections $I_p = J$, otherwise supply both).
3D transformation: only the direction cosines of the axis are needed;
$K = T^\top k_a T$ with $T = [\,\mathbf{e}_x^\top\ \ 0;\ 0\ \ \mathbf{e}_x^\top]$,
$\mathbf{e}_x = (x_2 - x_1)/L$. TRUSS2D is the same element restricted to the XY plane.

### 2.2 BEAM2 — Euler–Bernoulli beam (12-DOF)

Assumptions: plane sections remain plane and normal (no shear deformation), small strain,
St-Venant torsion, doubly-symmetric section (no bending–torsion coupling). Property fields
(`type="beam"`): `A`, `Iy`, `Iz`, `J`, optional `orientation` vector $\mathbf{v}$ (default:
global Z unless parallel to axis, then global Y — document the fallback in `available_elements`).

Local axes: $\mathbf{e}_x$ along the element, $\mathbf{e}_z = \mathbf{e}_x \times \mathbf{v} /
\lVert\cdot\rVert$, $\mathbf{e}_y = \mathbf{e}_z \times \mathbf{e}_x$; then $I_z$ bends in the
local x–y plane, $I_y$ in x–z. Local DOF order per node: $(u, v, w, \theta_x, \theta_y, \theta_z)$.

Transverse displacement interpolated with cubic Hermite functions of $\xi = x/L \in [0,1]$:

$$N_1 = 1 - 3\xi^2 + 2\xi^3,\quad N_2 = L(\xi - 2\xi^2 + \xi^3),\quad
  N_3 = 3\xi^2 - 2\xi^3,\quad N_4 = L(-\xi^2 + \xi^3).$$

Bending in x–y (DOFs $v_1, \theta_{z1}, v_2, \theta_{z2}$; $\theta_z = +v'$):

$$k_b = \frac{EI_z}{L^3}
\begin{bmatrix} 12 & 6L & -12 & 6L\\ 6L & 4L^2 & -6L & 2L^2\\ -12 & -6L & 12 & -6L\\ 6L & 2L^2 & -6L & 4L^2 \end{bmatrix},
\qquad
m_b = \frac{\rho A L}{420}
\begin{bmatrix} 156 & 22L & 54 & -13L\\ 22L & 4L^2 & 13L & -3L^2\\ 54 & 13L & 156 & -22L\\ -13L & -3L^2 & -22L & 4L^2 \end{bmatrix}.$$

Bending in x–z uses $EI_y$ and DOFs $(w_1, \theta_{y1}, w_2, \theta_{y2})$ where
$\theta_y = -w'$: **flip the sign of every term with an odd power of $L$** (rows/cols 2 and 4
off-diagonal blocks). Getting this sign flip wrong is the single most common BEAM2 bug; the
symptom is a cantilever whose x–z bending frequencies are wrong by ~2× while x–y is right.

Axial and torsion blocks as in §2.1. Optional consistent rotary-inertia add-on
$m_{rot} = \frac{\rho I}{30 L}\left[\begin{smallmatrix}36 & 3L & -36 & 3L\\ 3L & 4L^2 & -3L & -L^2\\ -36 & -3L & 36 & -3L\\ 3L & -L^2 & -3L & 4L^2\end{smallmatrix}\right]$
(off by default; EB acceptance cases assume translational consistent mass only).
Assemble the 12×12 local matrix, then $K = T^\top k\, T$ with $T = \mathrm{blkdiag}(R, R, R, R)$,
$R = [\mathbf{e}_x\ \mathbf{e}_y\ \mathbf{e}_z]^\top$.

Convergence: cubic Hermite is exact for static point/end loads; for eigenvalues the consistent
mass discretization converges from above at rate $O(h^4)$ per frequency — 10 elements put the
first three cantilever bending frequencies well inside 2 % (see `docs/ACCEPTANCE.md`).

### 2.3 TRIA3 — constant strain triangle (CST membrane)

Plane stress, thickness $t$ (property `type="shell"`, field `t`). With node coordinates
$(x_i, y_i)$, $x_{ij} = x_i - x_j$, $y_{ij} = y_i - y_j$, area
$A = \tfrac12 (x_{21} y_{31} - x_{31} y_{21})$:

$$B = \frac{1}{2A}\begin{bmatrix}
y_{23} & 0 & y_{31} & 0 & y_{12} & 0\\
0 & x_{32} & 0 & x_{13} & 0 & x_{21}\\
x_{32} & y_{23} & x_{13} & y_{31} & x_{21} & y_{12}
\end{bmatrix},\qquad
D = \frac{E}{1-\nu^2}\begin{bmatrix}1 & \nu & 0\\ \nu & 1 & 0\\ 0 & 0 & \tfrac{1-\nu}{2}\end{bmatrix}.$$

$B$ is constant, so exactly $K_e = t A\, B^\top D B$ (6×6). Consistent mass per translational
direction: $m = \frac{\rho t A}{12}\left[\begin{smallmatrix}2&1&1\\1&2&1\\1&1&2\end{smallmatrix}\right]$,
interleaved for $(u_1,v_1,u_2,v_2,u_3,v_3)$. CST is stiff (constant strain): expect slow $O(h)$
stress convergence; fine for mass/modal sanity meshes, not for stress acceptance.

### 2.4 QUAD4 — isoparametric bilinear membrane / flat shell

Bilinear shape functions on $(\xi,\eta) \in [-1,1]^2$:
$N_i = \tfrac14 (1 + \xi \xi_i)(1 + \eta \eta_i)$, $(\xi_i,\eta_i) \in \{\pm1\}^2$.
Jacobian $J = \partial(x,y)/\partial(\xi,\eta)$; Cartesian derivatives
$\partial N/\partial x = J^{-1} \partial N/\partial \xi$. Membrane stiffness by 2×2 Gauss
quadrature (points $\pm 1/\sqrt{3}$, weights 1):

$$K_e = t \sum_{g=1}^{4} B(\xi_g,\eta_g)^\top D\, B(\xi_g,\eta_g)\, \det J_g\, w_g .$$

Consistent mass $M_e = \rho t \sum_g N^\top N \det J_g w_g$ (8×8 membrane).

Flat-shell usage (out-of-plane): superpose a plate-bending stiffness (DKQ or Mindlin with
selective reduced integration on transverse shear) on DOFs $(w, \theta_x, \theta_y)$. The
drilling DOF $\theta_z$ has no physical stiffness in a flat element: assign a small artificial
stiffness $k_{\theta z} = \alpha \cdot \max(\mathrm{diag}\,K_e^{memb})$ with
$\alpha \approx 10^{-6}$ so the assembled $K$ is non-singular for coplanar shell patches, and
document that drilling results are non-physical. Round-1 acceptance only exercises the membrane
path.

Pitfalls: (i) full 2×2 integration shear-locks in bending-dominated thin geometry — use BEAM2 or
refine; (ii) 1-point reduced integration introduces two hourglass modes — if used, add hourglass
stabilization, otherwise keep 2×2; (iii) $\det J \le 0$ at any Gauss point means a re-entrant or
misnumbered element — raise with the element id; (iv) warped QUAD4 (non-planar nodes): project
onto the mean plane and warn beyond a warp tolerance (e.g. 10° normal deviation).

### 2.5 HEX8 — trilinear solid, incompatible modes by default

$N_i = \tfrac18 (1+\xi\xi_i)(1+\eta\eta_i)(1+\zeta\zeta_i)$, 2×2×2 Gauss, strain-displacement
$B$ is 6×24, isotropic $D$ from Lamé constants
$\lambda = E\nu/((1+\nu)(1-2\nu))$, $\mu = G$:

$$K_e = \sum_{g=1}^{8} B_g^\top D B_g \det J_g w_g,\qquad
  M_e = \rho \sum_g \bar N_g^\top \bar N_g \det J_g w_g \ (24\times 24).$$

The *plain* trilinear displacement element shear-locks in bending: on the slender verification
cantilever (10×1×1 mesh, i.e. a single element through the thickness) it recovers only ~64 % of
the Timoshenko tip deflection — noticeably too stiff, though not by an order of magnitude. The
implemented default is therefore the **incompatible modes** formulation of Wilson et al.
(a.k.a. Q6/QM6): the three quadratic bubbles $1-\xi^2$, $1-\eta^2$, $1-\zeta^2$ add nine
internal displacement DOFs that supply the linear bending strains the trilinear field cannot
represent, and are statically condensed out per element,
$K = K_{uu} - K_{ua} K_{aa}^{-1} K_{ua}^\top$. The bubble gradients are mapped with the
centre Jacobian $J_0$ scaled by $\det J_0 / \det J$ (Taylor–Beresford–Wilson correction), so
$\int_e B_{enh}\, dV = 0$ and the element still passes the constant-stress patch test on
distorted meshes (measured ~5e-16). With this default, **one HEX8 through the thickness
reaches ~98.6 % of the reference deflection** (0.9855 at 10×1×1, 0.96 already at 4×1×1) —
no ≥2-layer rule needed for well-shaped meshes. The condensation is a Schur complement of a
positive semi-definite matrix, so it cannot introduce a zero-energy mechanism: a free–free
block keeps exactly 6 rigid-body modes with the 7th eigenvalue cleanly separated (no hourglass
modes). The internal DOFs carry no mass; $M_e$ stays the consistent trilinear matrix above.
Cost: one extra 9×9 solve per element (~+13 % assembly time).

Formulation selection — `HEX8_FORMULATIONS = ("incompatible", "bbar", "full")`, settable on
the element/property record or assembly-wide via `assemble_km(model, options={"hex8": ...})`:

- `"incompatible"` (default): Wilson/Taylor element above; use it unless a note below applies.
- `"bbar"`: mean dilatation, numerically identical to selective reduced integration
  (volumetric term at the centroid, deviatoric 2×2×2). Cures volumetric locking as
  $\nu \to 0.5$ on bulky meshes, but does *not* cure shear locking and **over-softens thin
  bending** (deflection ratio 1.20 at 10×1×1, diverging with axial refinement) — reserve it
  for thick, nearly-incompressible parts.
- `"full"`: plain 2×2×2 displacement element, kept as the reference / patch-test baseline.

Pitfalls: incompatible modes still lose accuracy near incompressibility (deflection ratio
0.84 at $\nu = 0.499$ on the thin cantilever — prefer `"bbar"` there, on bulky meshes) and
are sensitive to element distortion (parallelogram-skewed cantilever drops 0.98 → 0.36 as
skew → 0.4; still better than `"full"` at every distortion level, but don't expect 90 %
accuracy from a bad mesh — enforce mesh quality or refine); combining `"bbar"` with the
internal modes is deliberately not offered (it is either rank-deficient or grossly over-soft
— see `elements/solid.py`); node ordering must follow the standard
bottom-face-CCW-then-top convention or $\det J < 0$. Reproducible builders behind every
number quoted here live in `femtools.fea.verification` (`hex8_bending_ratio`,
`hex8_patch_test_error`, `hex8_rigid_body_frequencies`) and are pinned by
`tests/test_hex8_verification.py`.

### 2.6 TET4, MASS, SPRING, DAMPER

TET4: constant-strain tetrahedron; volume
$V = \tfrac16 \lvert \det [x_2{-}x_1;\ x_3{-}x_1;\ x_4{-}x_1] \rvert$, constant $B$ (6×12),
$K_e = V B^\top D B$, consistent mass $\frac{\rho V}{20}(1 + \delta_{ij})$ pattern per direction.
Same locking caveats as CST, worse in 3D — meshing tool territory, not accuracy territory.

MASS: concentrated $m$ (scalar → 3 translations) or full symmetric 6×6 including inertia and
static-moment offsets — this is what `pretest` mass-loading studies attach. SPRING/DAMPER:
scalar $k$ / $c$ between two DOF codes (or DOF-to-ground), contributing
$k\left[\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\right]$ to $K$ (resp. damping table used
by `dynamics.direct_frf`; $C$ is assembled on demand, not part of `assemble_km`).

## 3. Sparse assembly

Triplet (COO) accumulation, one pass over elements:

```python
rows, cols, kv, mv = [], [], [], []       # preallocate to sum(ndof_e**2)
for el in model.elements.values():
    ke, me, gdofs = element_km(el)        # gdofs: local -> global active index, -1 if SPC'd
    ...append outer-product indices, skipping -1...
K = coo_matrix((kv, (rows, cols)), shape=(n, n)).tocsr()   # duplicates summed by tocsr()
K = 0.5 * (K + K.T)                       # kill round-off asymmetry before ARPACK
```

Complexity: time $O(\sum_e n_{dof,e}^2)$ to build triplets; `tocsr()` is
$O(nnz \log nnz)$-ish sort+sum; memory peak = triplet arrays (24 B/entry) — for HEX8 that is
576 entries/element/matrix. Never assemble dense; never insert into `lil_matrix` per entry
(quadratic behavior). Symmetrization above changes nothing analytically but guarantees
`eigsh` sees an exactly symmetric operator.

## 4. Boundary conditions — SPC elimination

`model.add_spc(node_id, mask)` fixes the masked DOF codes to zero (Round 1: homogeneous SPCs
only). We use **elimination**, not penalty or Lagrange:

1. Build the ordered active DOF list; constrained DOFs get no column.
2. Element scatter skips constrained local DOFs entirely (no post-hoc row/col deletion needed;
   if implemented as slicing instead, use `K[np.ix_(a, a)]` on CSR converted once to CSC —
   both are acceptable, slicing is simpler and $O(nnz)$).
3. Recovery embeds active solution into full vectors with zeros at SPCs.

Non-homogeneous SPCs (deferred): partition $u = (u_a, u_c)$, solve
$K_{aa} u_a = f_a - K_{ac} u_c$. Penalty methods are banned in Round 1: the penalty magnitude
biases eigenvalues and wrecks the $10^{-8}$ golden tolerances. MPC/RBE are out of scope Round 1.

## 5. Static solve

`solve_static(model, loads)` with `loads: dict[(node_id, dof_code), float]` → displacement
`ndarray (ndof,)` aligned with `dof_map`. Solve $K_{aa} u_a = f_a$ with
`scipy.sparse.linalg.spsolve` (SuperLU, COLAMD ordering) or `factorized()` when multiple RHS.
Complexity: LU fill-in dominates — roughly $O(n^{1.5})$ flops for 2D-topology meshes, $O(n^2)$
for 3D bricks. Detect mechanisms: SuperLU raising a singularity warning, or
$\lVert K u - f \rVert / \lVert f \rVert > 10^{-8}$ post-check → report the near-null-space DOF
(largest $|u|$) in the error.

## 6. Eigen solution — `solve_modes`

Generalized symmetric problem $K \phi = \lambda M \phi$, $\lambda = \omega^2$,
$f = \sqrt{\lambda} / 2\pi$ Hz.

```python
def solve_modes(model, n_modes: int = 10, shift: float = 0.0) -> ModalResult: ...

class ModalResult:
    freq_hz: np.ndarray          # (m,) ascending
    eigenvalues: np.ndarray      # (m,) rad^2/s^2, ascending
    modes: np.ndarray            # (ndof, m), mass-normalized
    generalized_mass: np.ndarray # (m,) == 1.0 after normalization (pre-norm values kept internally)
    dof_map: dict[tuple[int, int], int]
```

### 6.1 ARPACK shift-invert

`scipy.sparse.linalg.eigsh(K, k=n_modes, M=M, sigma=sigma, which="LM", mode="normal")` runs
Lanczos on $(K - \sigma M)^{-1} M$, whose largest magnitude eigenvalues map to the $\lambda$
nearest $\sigma$ — exactly what structural modes need (lowest cluster). Internally scipy
factorizes $K - \sigma M$ once (SuperLU) and each Lanczos step costs one sparse
triangular-solve pair, $O(nnz_{LU})$.

Shift selection (`shift` argument is in Hz; $\sigma = (2\pi f_{shift})^2$):

- Constrained structure, `shift=0.0`: $K$ is SPD, $\sigma = 0$ is safe and optimal for the
  lowest modes.
- Free–free / structures with rigid-body modes: $K$ is singular, $\sigma = 0$ makes the
  factorization fail or go garbage-quiet. Use a small **negative** shift,
  $\sigma = -(2\pi f_c)^2$ with $f_c \approx$ 0.1–1 % of the expected first elastic frequency
  (a robust default: $\sigma = -10^{-2}\,\mathrm{tr}(K)/\mathrm{tr}(M) / n$). Negative shifts
  can never collide with a physical eigenvalue ($\lambda \ge 0$).
- Mid-spectrum requests: set `shift` inside the band; ARPACK convergence degrades as the
  wanted set moves away from $\sigma$.

Pitfalls and required guards:

- **k bound**: ARPACK requires `k < n`. For `n_modes >= n - 1` or small systems
  (n ≲ 200) fall back to dense `scipy.linalg.eigh(K.toarray(), M.toarray())` — also the
  reference oracle in tests.
- **σ collides with an eigenvalue**: factorization of $K - \sigma M$ is near-singular; retry
  with $\sigma$ perturbed by a few percent.
- **Multiplicities**: repeated eigenvalues (symmetric structures) need subspace room — pass
  `ncv >= min(n, max(2*k + 1, 20))` and verify no mode is missed. Recommended verification:
  a Sturm/inertia count of $LDL^\top(K - \sigma_{hi} M)$ equals the number of computed
  eigenvalues below $\sigma_{hi}$ (optional, behind a `verify=` flag; dense fallback in tests
  covers it otherwise).
- **Round-off negatives**: rigid-body $\lambda$ come out as $\pm\varepsilon$; clamp
  $\lambda \in [-\tau, \tau] \to 0$ with $\tau = 10^{-6} \cdot \max\lambda$ before the square
  root, else `freq_hz` goes NaN.

### 6.2 Mass normalization and determinism

For each mode: $q_r = \phi_r^\top M \phi_r$ (generalized mass), then
$\phi_r \leftarrow \phi_r / \sqrt{q_r}$, giving $\Phi^\top M \Phi = I$ and
$\Phi^\top K \Phi = \Lambda$ (acceptance: $\max |\Phi^\top M \Phi - I| \le 10^{-8}$).
Sign convention for reproducibility: flip each $\phi_r$ so its largest-magnitude entry is
positive. ARPACK vectors for degenerate pairs span the right subspace but are an arbitrary
rotation within it — tests must compare via MAC/subspace, never entrywise (see
`docs/algorithms/correlation.md` §5).

## 7. Complexity summary

| Step | Cost |
|---|---|
| Element matrices | $O(n_e)$, constants: 4/8-point Gauss; HEX8 incompatible modes add one 9×9 condensation per element (~+13 %) |
| Triplet assembly + CSR | $O(\sum n_{dof,e}^2)$ + sort |
| SuperLU factorization | ~$O(n^{1.5})$ 2D-like, ~$O(n^2)$ 3D fill |
| ARPACK per iteration | $O(nnz_{LU})$ solve + $O(n \cdot ncv)$ orthogonalization |
| Dense fallback `eigh` | $O(n^3)$ — cap at n ≲ 2000 |

## 8. Verification hooks (consumed by `docs/ACCEPTANCE.md`)

- 2-node fixed–free bar, consistent mass: exact discrete $\omega = \sqrt{3E/\rho}/L$.
- N-element bar discrete dispersion closed form (ACCEPTANCE §1) — pins assembly + eigen to
  $10^{-8}$ without mesh-convergence excuses.
- Cantilever BEAM2 vs Euler–Bernoulli $\beta L$ roots — 2 % at 10 elements.
- Free–free: exactly 6 rigid-body modes at ~0 Hz, elastic modes match free–free roots.
- Patch tests: single QUAD4/HEX8 under uniform strain reproduces exact constant stress; HEX8
  additionally on the distorted 2×2×2 patch with one enclosed node (the Taylor-correction
  check, `verification.hex8_patch_test_error`).
- HEX8 anti-locking: one-through-thickness cantilever ≥ 0.95 of the Timoshenko tip deflection
  with the default formulation vs ~0.64 for `"full"`; free–free block has exactly 6 rigid
  modes (`verification.hex8_bending_ratio`, `verification.hex8_rigid_body_frequencies`,
  pinned by `tests/test_hex8_verification.py`).

## 9. Model reduction — Guyan, IRS, SEREP (Round 4, owner R4-O1)

Spec for `femtools.fea.reduction`, frozen in `.agent_workspace/REMAINING.md`.
Matrix-level API like `dynamics.craig_bampton`: inputs are the (typically free–free
partition) matrices plus **integer row indices**, not `(node_id, dof_code)` tuples — callers
translate through `dof_map`/`free_dof` exactly as `examples/guyan_serep.py` does.

```python
def guyan(K, master) -> ReductionResult                 # unpackable as (T, Krr)
def irs(K, M, master) -> ReductionResult                # O'Callahan IRS
def serep(phi, master_rows) -> ReductionResult          # T so phi ≈ T @ phi[master]
# serep must NOT return a bare ndarray: consumers that probe `.T` first
# (tests/test_round4_reduction.py::_transformation) would read its transpose.

class ReductionResult(NamedTuple):   # positions 0/1 are the "-> T, Krr" of REMAINING.md
    T: np.ndarray            # (n, n_m) recovery basis, rows in the ORIGINAL DOF ordering
    Krr: np.ndarray | None   # (n_m, n_m) = T'KT  (None for serep unless K supplied)
    Mrr: np.ndarray | None   # T'MT when M was an argument
```

Reduced coordinates are the physical displacements at `master`, in the order given —
consequently rows `T[master]` are exactly the identity (pinned by
`tests/test_round4_reduction.py`, which also accepts the reduced stiffness under
`K`/`Kr`/`Krr`/`K_reduced` or tuple position 1; a NamedTuple satisfies every consumer,
including plain tuple unpacking). `T` is dense; `K`/`M` may be sparse
(`T.T @ (K @ T)` stays cheap). The examples only rely on `.T` (or tuple position 0) —
keep at least that stable.

### 9.1 Guyan (static condensation)

Partition into masters $m$ and slaves $s$ (complement). Neglecting slave inertia in
$K u = \omega^2 M u$ gives $u_s = -K_{ss}^{-1} K_{sm} u_m \equiv T_{gs}\, u_m$, i.e.

$$T_G = \begin{bmatrix} I \\ -K_{ss}^{-1} K_{sm} \end{bmatrix},\qquad
K_{rr} = K_{mm} - K_{ms} K_{ss}^{-1} K_{sm} = T_G^\top K\, T_G
\quad\text{(Schur complement)},\qquad M_{rr} = T_G^\top M\, T_G.$$

Properties to pin: **statics are exact** — for loads applied only at masters,
$K_{rr}^{-1} f_m$ equals the full solution at the master rows to round-off (ACCEPTANCE
case 17a); eigenvalues are Rayleigh–Ritz **upper bounds** ($T_G$ spans a subspace);
accuracy of mode $r$ degrades like $\omega_r^2 / \omega_{s,1}^2$ where $\omega_{s,1}$ is
the first eigenvalue of the slave structure with masters clamped
($K_{ss}\phi = \lambda M_{ss}\phi$) — the classic validity rule is "use below
$\sim\!1/3\,f_{s,1}$". Implementation: one sparse factorization of $K_{ss}$ (CSC + `splu`),
$n_m$ solves; never invert. Pitfall: $K_{ss}$ is singular when clamping the masters does
not restrain the structure (free–free component with too few/coplanar masters) — raise
with the near-null-space DOF, don't regularize silently.

### 9.2 IRS — Improved Reduced System (O'Callahan 1989)

First-order inertia correction to Guyan: with $S = \mathrm{blkdiag}(0,\ K_{ss}^{-1})$ in
the $(m, s)$ partition,

$$T_{IRS} = T_G + S\, M\, T_G\, M_{rr}^{-1} K_{rr},$$

then $K_{red} = T_{IRS}^\top K T_{IRS}$, $M_{red} = T_{IRS}^\top M T_{IRS}$. The correction
term injects the $\omega^2$-dependent part of the exact condensation
$u_s = -(K_{ss} - \omega^2 M_{ss})^{-1}(K_{sm} - \omega^2 M_{sm})u_m$ evaluated with the
Guyan-reduced dynamics as the frequency estimate. Measured on the 10-element cantilever
demo (6 masters, 6 modes): mean relative frequency error drops 1.35e-2 (Guyan) →
**3.78e-4** (IRS); the acceptance check is the *mean* improvement, not per-mode, because
individual high modes may not improve monotonically. Reuse the $K_{ss}$ factorization from
§9.1 — the extra cost is one more slave-block solve set plus dense $O(n_m^3)$. Iterated
IRS (re-inserting $T_{IRS}$) converges toward SEREP accuracy but can diverge on
ill-conditioned $M_{rr}$; if offered, cap iterations and monitor $\lVert \Delta T \rVert$.

### 9.3 SEREP (System Equivalent Reduction/Expansion Process)

Given kept target shapes $\Phi$ ($n \times m$) and master rows $m$:

$$T = \Phi\, \Phi_m^{+},\qquad \Phi_m = \Phi[\text{master rows}, :].$$

With $n_m = m$ and $\Phi_m$ invertible the reduced model **reproduces the kept modes
exactly** (frequencies and shapes, ACCEPTANCE 17b: rel dev ≤ 1e-6, reconstruction
$\max|\Phi - T\Phi_m| \le$ 1e-8 relative); with $n_m > m$ it is the least-squares
projector and $T^\top M T$ has rank $m$ — the reduced eigenproblem is then singular, so
either warn or return the rank so callers use QZ. The same $T$ is the SEREP *expansion*
operator used by `correlation.expansion.expand_serep` (R4-O3). Pitfalls: master selection
governs $\mathrm{cond}(\Phi_m)$ — pick masters with EFI (`pretest.efi`), never co-linear
rows; scale/sign of $\Phi$ cancels in $T$; complex shapes need the conjugate-transpose
pseudo-inverse.

### 9.4 Complexity and hooks

| Kernel | Cost |
|---|---|
| `guyan` | 1 LU($K_{ss}$) + $n_m$ solves |
| `irs` | + 1 slave solve set + dense $O(n_m^3)$ |
| `serep` | SVD/pinv $O(n_m m^2)$ + $O(n\, m\, n_m)$ |

Verification: `examples/guyan_serep.py` (static exactness 1e-12 measured, upper-bound
property, IRS mean improvement, SEREP round-off reproduction), ACCEPTANCE case 17, and
`tests/test_round4_reduction.py` (R4-G1: 3-DOF closed-form Guyan basis, 5-DOF chain IRS
improving the first eigenvalue error by ≥ 10×, SEREP reconstruction at 1e-13).

## 10. Stress and strain recovery — `recover_stress` / `recover_strain` (Round 7, owner R7-O1)

Frozen entry points (REMAINING.md; landed on this tree, measured — ACCEPTANCE Round-7
status):

```python
from femtools.fea.recover import recover_stress, recover_strain, StressResult
# element-centroid stress/strain for BAR2, BEAM2, QUAD4, TRIA3, HEX8, TET4
# from a static displacement vector (solve_static) or one mode column.
# Linear elastic only — no nonlinearity, no plasticity, no failure criteria.
```

Recovery is *differentiate, then constitute*, evaluated per element at the centroid
(or as the average over the element's Gauss points — same value for the constant-strain
gates below): gather the element displacements $u_e$ from the global vector through
`dof_map`, rotate to the element local frame, evaluate the strain–displacement matrix at
the recovery point, and back-substitute the constitutive law:

$$\varepsilon = B(\xi_c)\, u_e^{loc}, \qquad \sigma = D\, \varepsilon,$$

with $D$ the same plane-stress / 3-D isotropic matrix the element stiffness was built
from (§2.3–§2.5) — recovery must share the element's $B$ and $D$ code paths, otherwise
the patch gate below tests the wrong thing. Per element type:

- **BAR2**: constant axial strain $\varepsilon_x = (u_{x2} - u_{x1})/L$ in the bar frame,
  $\sigma_x = E \varepsilon_x$ (with `J` set, the torsional moment $GJ\,\theta'_x$ is the
  analogous constant resultant). Exact everywhere, not just at the centroid.
- **BEAM2**: Euler–Bernoulli resultants from the local DOFs — axial force $N = EA\,u'_x$,
  torque $T = GJ\,\theta'_x$, and bending moments $M_y = EI_y\, w''$, $M_z = EI_z\, v''$
  from the second derivatives of the Hermite interpolation (§2.2) at $\xi = 1/2$. Since
  the Hermite curvature is linear in $\xi$, the centroid value is the element mean.
  Report the axial fiber stress as $\sigma_x = N/A \pm M c / I$ with $c$ the section
  extreme-fiber distance when the property carries it; the resultants are the primary
  output.
- **TRIA3 / TET4**: the strain field is constant (CST, §2.3) — the centroid value is the
  exact element strain; these two make the patch gate sharp.
- **QUAD4**: evaluate $B$ at the single point $\xi = \eta = 0$. The centroid is the
  bilinear element's superconvergent sampling point (Barlow points; for the in-plane
  shear of a distorted QUAD4 it is the *only* reliable point — corner extrapolation is a
  postprocessing choice deliberately out of scope). Flat-shell recovery reports the
  membrane strain of the mid-surface and the plate bending curvatures separately;
  combined fiber values $\sigma = \sigma_m \pm z\,\sigma_b$ are derived quantities.
- **HEX8**: evaluate the compatible $B$ at $\xi = \eta = \zeta = 0$. This is consistent
  with the incompatible-modes formulation of §2.5 *at the centroid specifically*: the
  Wilson/EAS enhancement shape functions are even ($1 - \xi^2$, …), so their strain
  contribution is odd and vanishes at the element center — no recovery of the condensed
  internal parameters is needed there, and under a constant-strain state the enhanced
  parameters are exactly zero anyway (that is the §2.5 patch-test property). Recovering
  anywhere *other* than the centroid would require re-solving the condensed
  $\alpha = -H^{-1} \Gamma u_e$ per element; out of Round-7 scope.

Frames and outputs: strains are computed in the element local frame; solid/membrane
components are rotated to global (Voigt transformation) before reporting, beam/bar
resultants stay in the element frame (they are meaningless elsewhere). `StressResult`
carries per-element components keyed by element id, plus derived von Mises
$\sigma_{vm} = \sqrt{\tfrac{1}{2}\left[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 +
(\sigma_3-\sigma_1)^2\right]}$ for continuum elements. `recover_strain` is the same walk
stopping before $D$.

Acceptance gate (`tests/test_round7_o1.py`, ACCEPTANCE Round-7 status): impose a linear
displacement field $u = A x$ on every supported element type — the recovered strain must
equal $\mathrm{sym}(A)$ and the stress $D\,\mathrm{sym}(A)$ to **1e-12** on every element
(constant-strain patch), including distorted meshes. References: Cook, Malkus, Plesha &
Witt, *Concepts and Applications of Finite Element Analysis* (4th ed.) ch. 3/6 (stress
computation, optimal sampling); Barlow, "Optimal stress locations in finite element
models", *IJNME* 10 (1976); Bathe, *Finite Element Procedures* §4.3.6.

## 11. RBE2 rigid constraints — condensation $u = Tq$ (Round 7, owner R7-O1)

Frozen entry points (REMAINING.md; landed on this tree, measured — ACCEPTANCE Round-7
status, demonstrated by `examples/rbe2_rigid.py`):

```python
from femtools.fea.mpc import apply_rbe2, ConstraintTransform
# T built from model.rbe2 (and/or an explicit list); assemble_km(..., mpc=T)
# or assemble_km honors model.rbe2 by default.
```

The data container is the **merged and stable** `core.model.RBE2`
(`id, independent, dependents, components (Nastran 1..6)`) via `FEModel.add_rbe2` —
shared with the BDF `RBE2` card (io.md §4.2); the Round-7 work is only the constraint
transform. Validation already pinned by the dataclass: components in 1..6, at least one
dependent, independent ∉ dependents, duplicate ids raise.

Kinematics — small-rotation rigid link. Dependent node $d$ at offset
$r = x_d - x_i$ from independent node $i$ moves as

$$u_d = u_i + \theta_i \times r, \qquad \theta_d = \theta_i
\quad\Longleftrightarrow\quad
\begin{bmatrix} u_d \\ \theta_d \end{bmatrix} =
\begin{bmatrix} I & -S(r) \\ 0 & I \end{bmatrix}
\begin{bmatrix} u_i \\ \theta_i \end{bmatrix},$$

with $S(r)$ the skew matrix of $r$ ($S(r)v = r \times v$; the sign comes from
$\theta \times r = -S(r)\,\theta$). Only the dependent components listed on the RBE2 are
constrained; unlisted dependent components remain free DOFs of their own.

Transform assembly: partition the global DOFs into the surviving set $q$ (everything not
listed as a dependent component) and the eliminated set. $T$ is the identity on surviving
DOFs; each eliminated row gets the matching row of the $6 \times 6$ block above, scattered
to the independent node's columns. Then

$$\hat K = T^\top K T, \qquad \hat M = T^\top M T, \qquad \hat f = T^\top f,$$

solve on $q$, recover $u = Tq$. This is exact **null-space (transformation) elimination**
— no penalty stiffness, no Lagrange saddle point — so symmetry and positive
semi-definiteness survive, no artificial stiffness ratio enters the conditioning ($T$
entries are lever arms, i.e. lengths), and rigid-body modes are preserved *exactly*: a
global RBM restricted to $q$ maps under $T$ onto the full RBM, hence the acceptance gate
that two free–free nodes welded by an RBE2 still show exactly **6** zero modes, and a
rigid offset correctly turns a tip force into force + moment at the independent node
($\hat f = T^\top f$ carries the $S(r)^\top$ couple) — the "rigid offset beam carries
moment" check.

Ordering vs SPC elimination (§4): build $T$ on the full DOF set first, then eliminate
SPCs in the reduced coordinates. An SPC on a *dependent* component is a conflict (it
would constrain the independent node implicitly) — raise, do not resolve silently. Also
raise on over-determination: a DOF dependent in two RBE2s, or chains where a dependent
node of one RBE2 is the independent node of another (topological resolution is a later
round if ever). Complexity: building $T$ is $O(n_{dep})$; the triple products are sparse
and dominated by `assemble_km` itself. References: Cook et al. ch. 13 (transformation
equations for constraints); Craig & Kurdila, *Fundamentals of Structural Dynamics* §14
(constraint reduction); the RBE2 layout is the public MSC/NX card.

## 12. RBE3 interpolation constraints — `apply_rbe3` (Round 8, owner R8-O1)

Frozen entry point (REMAINING.md; pending on this tree as of 2026-08-28):

```python
from femtools.fea.mpc import apply_rbe3
# Interpolation MPC: the dependent node's listed components are a weighted
# average of the independents. assemble_km honors model.rbe3, composed with
# model.rbe2 into one ConstraintTransform; mpc=False disables all MPCs.
```

The data container is the **merged and stable** `core.model.RBE3`
(`id, dependent, independents, components (default 1..3), independent_components
(default 1..3), weights (None = equal)`) via `FEModel.add_rbe3` — shared with the BDF
`RBE3` card (io.md §5). Do not replace the dataclass; validation (duplicate id, missing
nodes, dependent ∉ independents, components in 1..6, weight count) lives there.

Semantics — the exact opposite of §11 in *direction*, the same machinery in *method*. An
RBE2 makes many dependents follow one independent rigidly; an RBE3 makes **one dependent**
follow **many independents** as a weighted average. It is *not* a rigid weld: no relative
stiffening of the independent set occurs, no penalty springs enter, and the RBE2 lever-arm
kinematics of §11 must not be reused. With normalized weights
$\bar w_i = w_i / \sum_j w_j$ (equal by default), each listed dependent component $c$ is
eliminated by the master–slave row (Cook ch. 13; Zienkiewicz & Taylor master–slave
elimination — the same classical constraint-transformation both RBE kinds share)

$$u_{d,c} \;=\; \sum_i \bar w_i\, u_{i,c},$$

i.e. the dependent row of $G$ carries $\bar w_i$ in the columns of the matching
independent components and zero elsewhere. Everything of §11 then applies verbatim:
congruence $\hat K = G^\top K G$, $\hat M = G^\top M G$, virtual-work load mapping
$\hat f = G^\top f$, idempotent $G$, exact elimination (no penalty, no Lagrange
multipliers, conditioning untouched — the coefficients are dimensionless weights).

Composition with RBE2: `assemble_km` builds **one** `ConstraintTransform` from
`model.rbe2` *and* `model.rbe3` together — the row dictionaries merge before chain
resolution, so a DOF eliminated by both kinds is the same over-determination error as two
RBE2s claiming it, and RBE2→RBE3 chains resolve through the existing
`_resolve_chains` substitution (depth ≤ `MAX_CHAIN_DEPTH`, cycle-safe). `mpc=False`
still disables *all* multipoint constraints, and a model with `model.rbe3` empty must
leave the RBE2-only path **bit-identical** (the Round-7 goldens are regression gates).

Properties the acceptance gates pin (ACCEPTANCE Round-8 status; demonstrated by the
kernel-gated section of `examples/rbe2_rigid.py`):

- **6 rigid-body modes, free–free**: a mass RBE3-tied to a triangle of independent nodes
  keeps *exactly* 6 zero modes. This needs no geometric condition: any rigid motion of
  the independents maps under $G$ to a consistent dependent motion, so
  $\mathrm{null}(\hat K)$ keeps dimension 6. (Kinematic fidelity of the *inertia* placed
  at the dependent node is a separate matter: the translation average equals the true
  rigid-body motion of the **weighted centroid** of the independents, so a mass hung at
  any other point effectively sits at that centroid — a modeling approximation, never a
  mechanism.)
- **Load distribution by virtual work**: $\hat f = G^\top f$ sends a force $f_d$ on the
  dependent component to $\bar w_i f_d$ on each independent — equal weights give equal
  translational force shares. No moments appear for translation-only component lists
  (there are no lever arms in the rows).
- **RBE2 goldens bit-identical** when `model.rbe3` is empty.

Scope note vs the full Nastran card: the public MSC/NX RBE3 derives the reference-grid
motion from a weighted least-squares **rigid-body fit** of the independents (translations
*and* rotations from lever arms). The Round-8 frozen subset is the component-wise weighted
average above — it coincides with the LS fit's translation components when the reference
grid sits at the weighted centroid, and it is exactly what `core.model.RBE3` stores. The
lever-arm fit is a later round if ever; do not fake it with RBE2 kinematics. References:
Cook et al. ch. 13; Zienkiewicz & Taylor, *The Finite Element Method* (master–slave
constraint elimination); the RBE3 field layout is the public MSC/NX card.

## 13. Nodal stress averaging — `average_nodal` (Round 8, owner R8-O1)

Frozen entry point (REMAINING.md; pending on this tree as of 2026-08-28):

```python
from femtools.fea.recover import average_nodal
# average_nodal(stress: StressResult, model) -> StressResult (or a small
# nodal result type): centroid stresses averaged onto incident nodes, 1/n_adj.
```

The plain unweighted incidence average — for each node $n$ with $\mathrm{adj}(n)$ the set
of recovered elements containing it,

$$\sigma_n \;=\; \frac{1}{|\mathrm{adj}(n)|} \sum_{e \,\in\, \mathrm{adj}(n)} \sigma_e^{(c)},$$

component-wise on the Voigt tensor, where $\sigma_e^{(c)}$ is the §10 centroid value.
Rules that make the average meaningful:

- **Average in a common frame.** §10 element-frame components of different elements are
  not comparable; the average must run on the basic-frame tensors
  (`StressResult.stress_basic`) — or equivalently rotate first, average after. Beam/bar
  resultants (extras) stay per-element: an end moment has no nodal average.
- **Derived invariants are recomputed, never averaged.** Von Mises is a nonlinear
  functional of the tensor: $\sigma_{vm}(\bar\sigma) \ne \overline{\sigma_{vm}}$ in
  general. Average the six components, then evaluate von Mises / principals on the
  averaged tensor.
- **Skipped elements do not vote.** Elements absent from `StressResult`
  (`skipped`) contribute neither values nor incidence counts; a node touched only by
  skipped elements has no averaged value.

Acceptance gate: on a **constant-stress patch** (§10's patch construction, or the
uniform-tension bar of `examples/recover_stress.py`) every incident element reports the
same tensor, so the mean is exact at *every* node — machine-precision equality, no
tolerance slack needed. This is the sharp end: any frame mix-up or mis-normalization
breaks an exact identity rather than degrading an estimate.

Scope guard — **not Zienkiewicz–Zhu SPR**: no patch-wise polynomial fit over
superconvergent sampling points, no recovered-field error estimation
(Zienkiewicz & Zhu, *IJNME* 33 (1992), SPR), and no global $L_2$ smoothing mass-matrix
solve (Hinton & Campbell 1974). Those produce *better* nodal fields at real stress
gradients; the Round-8 contract is the classical incidence average FE post-processors
default to — $O(n_{elem})$ work, one pass over the connectivity, deterministic.
References: Cook et al. ch. 6 (stress averaging and smoothing); Zienkiewicz & Zhu 1992
(what this deliberately is not). As of Round 10 the SPR estimator exists as its own
frozen entry point, `recover_spr` (§15) — `average_nodal` keeps this contract
bit-identically.

## 14. TET10 — 10-node quadratic tetrahedron (Round 10, owner R10-O1)

Frozen entry point (REMAINING.md; landed on this tree 2026-08-28,
`tests/test_round10_o1.py` green — the parent had seeded
`core.model.ELEMENT_NODE_COUNTS["TET10"] = (10,)` and listed TET10 in
`_ELEMENT_NEEDS_PROPERTY`):

```python
from femtools.fea.elements import tet10   # registered as etype "TET10"
# aliases CTETRA10 / C3D10 acceptable if cheap; io maps them here (io.md §7)
```

Geometry and shape functions (Zienkiewicz & Taylor, *The Finite Element Method*,
Vol. 1, ch. 5 — the family tables; Bathe, *Finite Element Procedures* §5.3; Cook,
Malkus, Plesha & Witt ch. 3): 4 corner nodes plus 6 midside nodes, in the
Nastran-CTETRA/textbook order — midsides 5..10 on edges (1,2), (2,3), (3,1), (1,4),
(2,4), (3,4). In volume (barycentric) coordinates $L_1..L_4$,
$\sum_i L_i = 1$:

$$N_i = L_i (2 L_i - 1) \quad (\text{corners}), \qquad
N_{ij} = 4 L_i L_j \quad (\text{midsides on edge } i\!-\!j).$$

Isoparametric mapping $x = \sum N_a x_a$; when the midside nodes sit at the true edge
midpoints (the recommended, and the meshes the goldens build) the mapping is affine and
the Jacobian constant — curved edges are legal isoparametrically but push the
quadrature error up and are not exercised by any golden. Each node carries the 3
translational DOFs (30 per element); the strain interpolation is complete linear, so
the element **contains the constant-strain state exactly** — the patch test of case 29
holds to 1e-12 on distorted meshes, unlike anything that needs the §2.5 incompatible
modes to get there.

Quadrature: the stiffness integrand $B^\top D B \det J$ is quadratic (for the affine
mapping), so the standard **4-point degree-2 tet rule** (points at barycentric
$(\alpha, \beta, \beta, \beta)$, $\alpha = 0.5854102$, $\beta = 0.1381966$, weights
$V/4$) integrates it exactly. The consistent-mass integrand $N^\top N$ is quartic — use
a degree ≥ 4 rule (11-, 14- or 15-point; Zienkiewicz & Taylor's tables) or a
well-documented diagonal lumping (HRZ row-sum scaling); an under-integrated consistent
mass fails the $r^\top M r = \rho V$ identity of `examples/tet10_patch.py` and is the
first thing to check when it does.

Recovery (§10 conventions apply verbatim): `recover_stress` / `recover_strain`
evaluate the element's own $B$ at the **centroid** ($L_i = 1/4$) or as the average over
the 4 Gauss points — for TET10 the strain is linear inside the element, so the centroid
value *is* the element mean, and under a constant state both choices are exact. The
superconvergent points of the quadratic tet are the interior Gauss points (Barlow's
argument, one polynomial degree down); centroid reporting stays within the §10
contract. Add `"TET10"` to `verification.PATCH_TYPES` with a patch-mesh builder so the
parametrized constant-strain tests cover it.

Gates (ACCEPTANCE case 29, `tests/test_round10_o1.py`, `examples/tet10_patch.py`):
constant-strain patch ≤ 1e-12 for strain and stress; free–free single TET10 exactly
**6** rigid-body modes; HEX8 bending ratio still ≥ 0.98 and the tilted-shell 6-RBM
contract untouched (TET10 must not perturb the registry defaults — in particular the
HEX8 incompatible-modes default of §2.5 stays, and EAS-30 stays out of scope).
Pitfalls: node-order mistakes put a midside on the wrong edge and show up as a patch
failure, not a crash — check connectivity against the edge table above first; TET4's
§2.6 locking caveats do *not* carry over (the quadratic field bends), which is exactly
why the io translators stop midside-dropping CTETRA (io.md §7).

## 15. Superconvergent patch recovery — `recover_spr` (Round 10, owner R10-O1)

Frozen entry point (REMAINING.md; landed on this tree 2026-08-28,
`tests/test_round10_o1.py` green):

```python
from femtools.fea.recover import recover_spr   # (stress: StressResult, model)
# ZZ-SPR nodal stress/strain: patch-wise linear polynomial fit over the
# superconvergent samples, evaluated at the node. average_nodal stays 1/n_adj.
# -> NodalStressResult, same container as average_nodal (§13).
```

Reference: Zienkiewicz, O.C., Zhu, J.Z., *The superconvergent patch recovery and a
posteriori error estimates. Part 1: The recovery technique*, IJNME 33(7), 1992,
pp. 1331–1364. For the linear elements of the kernel the superconvergent sampling
points are the element **centroids** (Barlow 1976) — exactly where §10 already
recovers, so SPR consumes a centroid `StressResult` without recomputation. Per node
$a$ and stress component $c$, over the patch $\mathrm{adj}(a)$ of recovered elements
incident on $a$:

$$\sigma_c^*(x) = \mathbf{p}(x)\, \mathbf{a}_c, \qquad \mathbf{p} = [1, x, y, z],
\qquad \min_{\mathbf{a}_c} \sum_{e \in \mathrm{adj}(a)}
\left( \mathbf{p}(x_e^{(c)})\, \mathbf{a}_c - \sigma_{e,c} \right)^2,$$

then report $\sigma_c^*(x_a)$ — a small $4 \times 4$ normal-equation (or QR) solve per
patch, $O(n_{nodes})$ total. Fit in the **basic frame** for the same reason
`average_nodal` averages there (§13). Rank guards, all load-bearing: a boundary node
whose patch has fewer than 4 non-coplanar centroids (or coplanar/collinear samples —
surface and edge nodes of coarse meshes) makes $\mathbf{p}$-columns dependent — either
borrow the patch of an adjacent *interior* node and evaluate at the boundary node (the
ZZ paper's own recipe) or drop degenerate monomials / fall back to the incidence
average, but **document which** and warn once. Skipped elements do not vote, exactly
as in §13.

Exactness gate (ACCEPTANCE case 30): a constant stress state lies inside the fitted
polynomial space, so the LS fit reproduces it *regardless of sample positions* — SPR
must return the exact tensor at **every** node of a constant-stress patch
(BAR2/TET4/HEX8 suffice per the Round-10 brief; `examples/tet10_patch.py` drives the
TET4 twin of its patch). TET10 in SPR may reuse the same centroid samples or skip
TET10 with a documented message — both are contract-conforming, the example tolerates
`NotImplementedError`; the landed kernel reuses the centroid samples, measured exact
on the TET10 patch (5.1e-15). Scope guard, mirrored from §13: `average_nodal` stays the plain
1/n_adj incidence average, bit-identical — SPR is the *better field at gradients*
(that is its point: the fitted field is one order more accurate and feeds the ZZ error
estimator later, if ever), the average is the cheap default. Pitfall: never fit
higher-order polynomials than the element can support superconvergently — a quadratic
fit over linear-element centroids is noise amplification dressed as accuracy.

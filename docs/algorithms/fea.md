# FEA algorithms — elements, assembly, solvers

Spec for `femtools.fea` (owner: R1-O1). Frozen entry points per `docs/CONTRACT_API.md`:

```python
from femtools.fea.assemble import assemble_km, AssemblyResult
from femtools.fea.static import solve_static
from femtools.fea.eigen import solve_modes           # -> ModalResult
from femtools.fea.elements import available_elements
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

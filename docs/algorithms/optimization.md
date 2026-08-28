# Optimization algorithms — size (SLSQP), topology (SIMP), shape, topometry, DOE

Spec for `femtools.optimization` (owner: R1-O4). Frozen entry points:

```python
from femtools.optimization.size import size_optimize
from femtools.optimization.topology import topology_simp
from femtools.optimization.doe import latin_hypercube, full_factorial
# Round-6 addition (REMAINING.md, owner R6-O4) — see §4:
from femtools.optimization.shape import shape_optimize, ShapeResult
# Round-7 addition (REMAINING.md, owner R7-O4; pending on this tree) — see §5:
from femtools.optimization.topometry import topometry_optimize, TopometryResult
```

## 1. Size optimization — `size_optimize`

```python
size_optimize(model, design_vars, objective, constraints=(),
              method="slsqp", max_iter=50, tol=1e-6, seed=0) -> SizeResult

design_vars = [{"type": "property", "id": 1, "name": "t", "lower": 1e-3, "upper": 1e-2}, ...]
objective   = {"kind": "mass"} | {"kind": "compliance", "loads": ...} | callable(model)->float
constraints = [{"kind": "freq_min", "mode": 1, "value_hz": 50.0},
               {"kind": "mass_max", "value": 12.0},
               {"kind": "displacement_max", "dof": (17, 2), "loads": ..., "value": 1e-3}, ...]
```

NLP form: $\min_x f(x)$ s.t. $g_j(x) \le 0$, $x \in [l, u]$, solved with
`scipy.optimize.minimize(method="SLSQP")` — sequential quadratic programming with a damped-BFGS
Hessian approximation; per iteration one QP $O(n_x^3)$ plus gradient evaluations. Design
variables are the same relative $p = \theta/\theta_0$ scaling used in `updating.md` §1 —
SLSQP has no internal scaling, unscaled thickness-vs-modulus problems stall on the first
Wolfe check.

Gradients (pass `jac=`, never let SLSQP FD a modal solve):

- mass: $\partial m / \partial p_k = $ assembled density/volume block — analytic, free;
- eigenfrequency constraints: Fox–Kapoor from `updating.sensitivity` (reuse, same formula);
- compliance $c = f^\top u$: self-adjoint, $\partial c / \partial p_k = -u^\top
  \frac{\partial K}{\partial p_k} u$ — one static solve total;
- generic displacement constraints: adjoint solve $K \mu = e_{dof}$, then
  $\partial u_{dof} / \partial p_k = -\mu^\top \frac{\partial K}{\partial p_k} u$ — one extra
  solve per active constraint, not per variable.

Pitfalls: **mode switching** — "keep $f_1 \ge 50$ Hz" chases a moving target when mode order
changes; track constrained modes by MAC against the initial shapes, or bound *all* modes in the
band (robust formulation). Repeated eigenvalues make $f_r(x)$ non-smooth — SLSQP oscillates;
detect clusters and constrain the cluster minimum via the subspace derivative
(`updating.md` §3.1 pitfall). SLSQP is local: multistart from `latin_hypercube` points (§3)
is the sanctioned global strategy. Feasibility: start feasible when possible; SLSQP handles
infeasible starts poorly with tight nonlinear constraints.

## 2. Topology optimization — `topology_simp`

Minimum-compliance density method on a regular mesh of the design domain (Round 1: 2D QUAD4
grid; the element stiffness $k_e^0$ comes from `fea.elements`).

$$\min_{\rho}\ c(\rho) = f^\top u(\rho)
\quad \text{s.t.}\quad K(\rho) u = f,\quad
\frac{\sum_e \tilde\rho_e v_e}{\sum_e v_e} \le V^*,\quad
0 \le \rho_e \le 1$$

Modified SIMP interpolation (keeps $K$ nonsingular at $\rho = 0$):

$$E_e(\tilde\rho_e) = E_{min} + \tilde\rho_e^{\,p} (E_0 - E_{min}),
\qquad E_{min} = 10^{-9} E_0,\quad p = 3.$$

Compliance sensitivity (self-adjoint):
$\dfrac{\partial c}{\partial \tilde\rho_e} = -p\, \tilde\rho_e^{\,p-1} (E_0 - E_{min})\,
u_e^\top k_e^0 u_e \le 0.$

### 2.1 Filtering (mandatory — mesh independence, checkerboard control)

Density filter with radius $r_{min}$ (in element widths, default 1.5–3):

$$\tilde\rho_e = \frac{\sum_i H_{ei}\, v_i\, \rho_i}{\sum_i H_{ei}\, v_i},
\qquad H_{ei} = \max(0,\ r_{min} - \mathrm{dist}(e, i)),$$

precomputed as a sparse row-stochastic matrix $W$ ($\tilde\rho = W \rho$, $O(nnz_W)$ per
iteration; $nnz_W \approx n_e \pi r_{min}^2$). Chain rule:
$\partial c / \partial \rho = W^\top (\partial c / \partial \tilde\rho)$.
Alternative `filter="sensitivity"` (Sigmund's original) filters
$\partial c/\partial\rho$ weighted by $\rho$ — cheaper legacy option, keep both. Optional
Heaviside projection $\bar\rho = (\tanh(\beta\eta) + \tanh(\beta(\tilde\rho - \eta))) /
(\tanh(\beta\eta) + \tanh(\beta(1 - \eta)))$ with $\beta$-continuation (1 → 64, doubling every
50 iterations) for crisp 0/1 designs.

### 2.2 Optimality-criteria update

$$B_e = \frac{-\partial c / \partial \rho_e}{\lambda_v\, \partial V / \partial \rho_e},\qquad
\rho_e^{new} = \mathrm{clip}\!\left( \rho_e B_e^{\eta},\ \rho_e \pm m_{move},\ [0, 1] \right)$$

with damping $\eta = 1/2$, move limit $m_{move} = 0.2$, and $\lambda_v$ found by bisection on
the volume constraint (monotone in $\lambda_v$; ~40 bisections of an $O(n_e)$ evaluation).
Convergence: $\max_e |\Delta\rho_e| < 0.01$ or `max_iter` (default 200). MMA is the
better general-purpose optimizer but OC is the Round-1 spec: simple, robust, matches the
88-line-class reference results used in acceptance.

```python
topology_simp(nelx, nely, volfrac, penal=3.0, rmin=2.0, filter="density",
              loads=..., spcs=..., max_iter=200, seed=0) -> TopoResult(rho, compliance_history)
```

Pitfalls: $p$ too high from the start ⇒ premature 0/1 lock-in into poor local minima (use
continuation $p: 1 \to 3$ if designs look pathological); gray transition bands are the filter
doing its job — post-threshold at $\rho = 0.5$ only for reporting, never mid-run; checkerboards
appearing means the filter radius is under 1 element — enforce $r_{min} > 1$; QUAD4 + SIMP with
1-point integration adds hourglass artifacts (use full 2×2, `fea.md` §2.4); compliance must
decrease essentially monotonically under OC + filtering — a rising history is the standard
symptom of a sign error in the sensitivity or a stale factorization. Per-iteration cost is one
sparse static solve — reuse symbolic factorization across iterations (pattern is constant).

## 3. Design of experiments — `latin_hypercube`, `full_factorial`

```python
latin_hypercube(n_samples, bounds, seed=0, criterion="maximin", iterations=100) -> (n, k)
full_factorial(levels, bounds) -> (prod(levels), k)     # levels: int | sequence per dim
```

LHS: for each dimension, partition $[0,1]$ into $n$ equal strata, draw one uniform sample per
stratum, then randomly permute strata across dimensions — every 1-D projection is stratified
(the golden property: exactly one sample per stratum per dimension, tested exactly).
`criterion="maximin"`: repeat `iterations` random permutation sets, keep the design maximizing
$\min_{i \ne j} \lVert x_i - x_j \rVert$ ($O(\text{iters} \cdot n^2 k)$); `criterion=None`
returns the first draw. Map to physical bounds affinely. Determinism: everything through
`np.random.default_rng(seed)` — the same seed must reproduce the same design bit-for-bit on all
platforms (acceptance test).

Full factorial: Cartesian product of per-dimension level grids (endpoints included, linspace);
size $\prod_k L_k$ explodes combinatorially — warn above $10^5$ points. Two-level fractional
designs and orthogonal arrays are out of Round-1 scope.

Uses: multistart seeds for §1, surrogate/DOE studies over updating parameters, sensitivity
screening. Pitfall: LHS guarantees 1-D stratification only — pairwise projections can still
cluster without the maximin criterion; for $k > n$ dimensions any DOE is degenerate, warn.

## 4. Node-based shape optimization — `shape_optimize` (Round 6, owner R6-O4)

Shape optimization with **selected node coordinates as design variables** (the "natural
design variable" formulation) on a fixed mesh topology — no remeshing, no CAD
parameterization in Round-6 scope. Method class: Haftka & Grandhi, "Structural shape
optimization — a survey", *CMAME* 57 (1986) 91–106; Haftka & Gürdal, *Elements of Structural
Optimization* (3rd ed., Kluwer 1992). NLP solved with
`scipy.optimize.minimize(method="SLSQP" | "trust-constr")`, reusing the §1 machinery.

```python
shape_optimize(model, design_nodes, objective, constraints=(), *,
               directions=None,            # per-node xyz mask, default all three
               bounds=None,                # box on the coordinate perturbations
               method="slsqp",             # or "trust-constr"
               quality_min=0.2,            # min-Jacobian barrier threshold
               smoothing="laplacian",      # non-design nodes follow the boundary
               max_iter=50, tol=1e-6) -> ShapeResult
# objective: {"kind": "frequency", "mode": r, "target": f_hz | "maximize"}
#            | {"kind": "compliance", "loads": ...} | {"kind": "mass"} | callable
# ShapeResult: model (updated copy, input never mutated), x, history,
#              quality_history, converged, n_iter, message
```

Design variables are coordinate *perturbations* $s$ of the selected nodes
($x_a = x_a^0 + s_a$, optionally masked to chosen directions), scaled by a characteristic
element length so SLSQP sees $O(1)$ variables (same scaling argument as §1).

Sensitivities — semi-analytic at element level: only elements touching a moved node change,
so $\partial K / \partial s_a \approx (K_e(x + h e_a) - K_e(x - h e_a)) / (2h)$ with
$h \approx 10^{-6} L_e$, assembled over the adjacent-element patch. Then reuse the standard
adjoint/eigen formulas: Fox–Kapoor for frequencies
($\partial \lambda_r / \partial s = \phi_r^\top (\partial K/\partial s -
\lambda_r\, \partial M/\partial s)\, \phi_r$ — mind that $M$ *does* depend on shape, unlike
§1 thickness variables), self-adjoint compliance
$\partial c / \partial s = -u^\top (\partial K / \partial s)\, u$ (plus a load term when
loads ride on moved nodes — warn instead of silently ignoring). Known accuracy trap:
the semi-analytic method loses digits on rigid-rotation-dominated beam/shell elements and
the error *grows* with mesh refinement (Barthelemy & Haftka, *Structural Optimization* 2,
1990) — central differences and the relative step above are the standard mitigation; an FD
oracle on the full objective is the acceptance cross-check.

Mesh-quality barrier (mandatory — the failure mode of node-based shape is mesh tangling,
not divergence): per element the scaled Jacobian $q_e = \min_{gp} \det J_{gp} /
\det J_{gp}^0$ over the Gauss points, and either the hard constraint
$\min_e q_e \ge q_{min}$ smoothed by KS/p-norm aggregation
($-\tfrac{1}{\rho} \ln \sum_e e^{-\rho q_e}$, since $\min$ is non-smooth) or a log-barrier
term $-\mu \sum_e \ln(q_e - q_{min})$ added to the objective. An inverted element
($q_e \le 0$) must short-circuit the evaluation (return a large penalty) — never hand the
eigensolver a tangled mesh.

Laplacian smoothing (`smoothing="laplacian"`): design nodes move as the optimizer dictates;
the remaining *non-design* nodes follow by solving the graph-Laplacian system
$L_{ii} x_{int} = -L_{ib} x_{design}$ (equivalently iterating
$x_i \leftarrow \tfrac{1}{|N(i)|} \sum_{j \in N(i)} x_j$) so interior distortion is spread
over the mesh instead of accumulating in the first element ring — the spring-analogy mesh
deformation classically paired with node-based shape variables. It also suppresses the
**jagged-boundary** pathology: raw node-by-node variables produce oscillatory boundaries
(the standard objection to natural design variables in Haftka–Grandhi); smoothing acts as
the regularizing filter, exactly like the SIMP density filter in §2.1.

Pitfalls: mode switching and repeated eigenvalues for frequency objectives — track modes by
MAC against the initial shapes and constrain cluster minima (§1 pitfalls apply verbatim);
mass changes implicitly with shape even for "frequency" objectives — add a mass constraint
unless drift is intended; symmetry is broken by numerical noise unless symmetric design
nodes are linked to one variable; loads/BCs attached to moved nodes change the problem
definition (report which); fixed topology means large shape changes starve the mesh —
the quality barrier going active for many iterations is the signal to stop trusting the
result, report `quality_history`.

## 5. Topometry optimization — `topometry_optimize` (Round 7, owner R7-O4)

Element-wise sizing on an **existing** `FEModel` mesh: one design variable per element
(shell/membrane thickness $t_e$, or a density-like modulus scale $x_e$ for element types
without a thickness), minimum compliance under a volume / mean-thickness constraint.
Frozen entry point (REMAINING.md; module pending on this tree as of 2026-08-28):

```python
from femtools.optimization.topometry import topometry_optimize, TopometryResult
# design field: per-element thickness (shell properties) or density (E-scale),
# objective: min compliance under given loads; constraint: volume or mean thickness;
# update: OC or SLSQP. Works on the model's own mesh — nothing is re-meshed.
```

How it differs from its two neighbours, since all three share the compliance machinery:
`topology_simp` (§2) builds its **own** structured grid and drives fictitious densities to
0/1 with the SIMP power $p = 3$ — the answer is a *layout*. `shape_optimize` (§4) moves
**node coordinates** on a fixed mesh — the answer is a *boundary*. Topometry (the term is
Vanderplaats'; the capability shipped in Genesis) keeps the user's mesh *and* keeps the
variables physical: a per-element thickness distribution is manufacturable at intermediate
values, so **no penalization is needed** ($p = 1$) when the field is a real shell
thickness; an optional $p > 1$ is only meaningful when the field is reinterpreted as a
0/1 material indicator, which is then just element-wise SIMP on an unstructured mesh.

Sensitivities are analytic and self-adjoint, exactly as §2: with the element stiffness
split into its thickness powers — membrane $\propto t$, plate bending $\propto t^3$,
i.e. $k_e(t_e) = t_e\, k_e^m + t_e^3\, k_e^b$ (both parts integrated once at unit
thickness) —

$$\frac{\partial c}{\partial t_e} = -u_e^\top \frac{\partial k_e}{\partial t_e} u_e
= -u_e^\top \left( k_e^m + 3 t_e^2\, k_e^b \right) u_e \le 0,
\qquad \frac{\partial V}{\partial t_e} = A_e,$$

so one static solve per iteration prices every element. For the density variant replace
$t_e$ by $x_e^p$ scaling the whole $k_e^0$ (§2 formulas verbatim, $v_e$ fixed). Update:
the §2.2 optimality-criteria fixed point with bisection on the volume multiplier (the
sensitivity signs and the monotone-compliance argument carry over unchanged), or SLSQP
via the §1 machinery when extra constraints (per-element bounds beyond boxes, frequency
floors) are requested. The §2.1 density filter is available and recommended — a thickness
field is physical so checkerboarding is less poisonous than in topology, but mesh
dependence of the optimum is not, and the filter is the standard cure either way.

Gates (ACCEPTANCE Round-7 status, `tests/test_round7_o4.py`): a clamped cantilever plate
started from uniform thickness must end with strictly lower compliance at the same
volume, every $t_e$ within bounds, no inverted/degenerate element ever handed to the
solver (the mesh never moves, so this is a bounds question, not a Jacobian question),
monotone OC history, input model not mutated. References: Bendsøe & Sigmund, *Topology
Optimization* (2003) ch. 1 (sizing vs topology taxonomy); Sigmund, "A 99 line topology
optimization code written in Matlab", *SMO* 21 (2001) — OC/filter machinery reused here;
Vanderplaats, "Structural optimization for statics, dynamics and beyond", *J. Braz. Soc.
Mech. Sci. Eng.* 28 (2006) — topometry naming and scope.

## 6. Complexity summary

| Kernel | Cost |
|---|---|
| SLSQP iteration | QP $O(n_x^3)$ + 1 modal/static solve + analytic gradients |
| SIMP iteration | 1 static solve (reused pattern) + $O(nnz_W)$ filter + $O(40 n_e)$ bisection |
| Shape iteration | 1 modal/static solve + $O(n_s \bar n_{adj})$ element re-integrations + QP $O(n_s^3)$ |
| Topometry iteration | 1 static solve + $O(n_e)$ analytic gradients + $O(40 n_e)$ OC bisection (+ filter $O(nnz_W)$) |
| LHS | $O(\text{iters} \cdot n^2 k)$ maximin, $O(nk)$ plain |
| Full factorial | $O(\prod_k L_k)$ — memory bound |

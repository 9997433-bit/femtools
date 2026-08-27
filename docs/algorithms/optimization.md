# Optimization algorithms — size (SLSQP), topology (SIMP), shape, DOE

Spec for `femtools.optimization` (owner: R1-O4). Frozen entry points:

```python
from femtools.optimization.size import size_optimize
from femtools.optimization.topology import topology_simp
from femtools.optimization.doe import latin_hypercube, full_factorial
# Round-6 addition (REMAINING.md, owner R6-O4) — see §4:
from femtools.optimization.shape import shape_optimize, ShapeResult
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

## 5. Complexity summary

| Kernel | Cost |
|---|---|
| SLSQP iteration | QP $O(n_x^3)$ + 1 modal/static solve + analytic gradients |
| SIMP iteration | 1 static solve (reused pattern) + $O(nnz_W)$ filter + $O(40 n_e)$ bisection |
| Shape iteration | 1 modal/static solve + $O(n_s \bar n_{adj})$ element re-integrations + QP $O(n_s^3)$ |
| LHS | $O(\text{iters} \cdot n^2 k)$ maximin, $O(nk)$ plain |
| Full factorial | $O(\prod_k L_k)$ — memory bound |

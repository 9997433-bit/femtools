# Model updating algorithms — sensitivities, WLS/Bayesian iteration, force identification

Spec for `femtools.updating` (owner: R1-O4). Frozen entry points:

```python
from femtools.updating.sensitivity import sensitivity_matrix
from femtools.updating.updater import update_model, UpdateResult
from femtools.updating.force_id import identify_harmonic_forces
```

Method class: iterative sensitivity-based weighted least squares with Bayesian (prior-weighted)
regularization — the Friswell–Mottershead penalty-function formulation, which is what
FEMtools-class updating runs.

## 1. Parameterization

Parameters are declared as dicts (kept plain for scripting/serialization):

```python
parameters = [
    {"type": "material", "id": 1, "name": "E",   "lower": 0.5, "upper": 2.0},
    {"type": "material", "id": 1, "name": "rho"},
    {"type": "property", "id": 2, "name": "t"},          # shell thickness
    {"type": "property", "id": 3, "name": "k"},          # spring stiffness
]
```

Internally work in **relative** variables $p_k = \theta_k / \theta_k^{(0)}$ (start value
$p_k = 1$): dimensionless, uniformly scaled Jacobians, bounds map to simple box constraints.
Structural matrices are affine in most Round-1 parameters:
$K(p) = K_{res} + \sum_k p_k K_k$ exactly for `E` (per material region) and `spring_k`;
$M(p)$ likewise for `rho`. Shell thickness is polynomial: membrane $\propto t$, bending
$\propto t^3$ — apply the chain rule $\partial K / \partial p = K_m + 3 p^2 K_b$ rather than
pretending linearity (classic silent error).

## 2. Residuals

$$z(p) = W_z^{1/2} \begin{bmatrix}
\left( f_{test} - f_{fem}(p) \right) / f_{test} & \text{(relative eigenfrequencies)}\\[2pt]
\text{shape residuals: } 1 - \mathrm{MAC}_{rr}\ \text{or}\ \phi_{test} - \phi_{fem}\,\mathrm{MSF} & \\[2pt]
\mathrm{Re/Im}\, \big( H_{test}(\omega_k) - H_{fem}(\omega_k) \big) & \text{(optional FRF lines)}
\end{bmatrix}$$

Rules: pair modes with `correlation.pair_modes` **every iteration** (mode crossing during
updating is the norm, not the exception); use relative frequency residuals so all rows are
$O(10^{-2})$; MAC residuals are bounded and robust but have zero gradient at MAC = 1 (use
$1 - \sqrt{\mathrm{MAC}}$ near convergence or switch to shape-difference residuals); FRF
residuals only away from resonances of *both* models, else the Jacobian is dominated by peak
misalignment (see `correlation.md` §4 pitfall).

## 3. Sensitivities — `sensitivity_matrix(model, parameters, responses, method="analytic")`

$S_{ik} = \partial z_i / \partial p_k$, `(n_z, n_p)` dense.

### 3.1 Analytic (Fox–Kapoor), preferred for eigen-data

Mass-normalized $\phi_r$, distinct eigenvalues:

$$\frac{\partial \lambda_r}{\partial p_k} =
\phi_r^\top \left( \frac{\partial K}{\partial p_k} - \lambda_r \frac{\partial M}{\partial p_k} \right) \phi_r,
\qquad
\frac{\partial f_r}{\partial p_k} = \frac{1}{8 \pi^2 f_r} \frac{\partial \lambda_r}{\partial p_k}.$$

$\partial K / \partial p_k$ is just the parameter's element-block $K_k$ (assemble once per
parameter over its element set — reuse the assembly kernels with unit parameter). Cost:
$O(n_p \cdot m \cdot nnz_k)$, no extra solves.

Mode-shape sensitivities (needed for MAC/shape residuals):

- Modal superposition (truncated):
  $\dfrac{\partial \phi_r}{\partial p_k} = \sum_{s \ne r}
  \dfrac{\phi_s^\top \left( \partial K/\partial p_k - \lambda_r\, \partial M/\partial p_k \right) \phi_r}
        {\lambda_r - \lambda_s}\, \phi_s
  \;-\; \tfrac12 \left( \phi_r^\top \dfrac{\partial M}{\partial p_k} \phi_r \right) \phi_r$ —
  cheap, biased by truncation (keep ≥ 2× the paired modes).
- Nelson's method: exact with only the $r$-th mode, one bordered solve of the singular system
  per (mode, parameter) — use when truncation bias shows (flag `shape_method="nelson"`).

Pitfall — repeated eigenvalues: $\partial \lambda / \partial p$ is not differentiable
mode-wise; the derivatives are eigenvalues of the projected $2 \times 2$
$\Phi_d^\top (\partial K - \lambda \partial M) \Phi_d$ problem. Detect
$|\lambda_r - \lambda_s| < 10^{-6} \lambda_r$ and handle via the subspace, or the Jacobian
column is garbage that flips sign between iterations.

### 3.2 Finite differences, the oracle and fallback

Forward: $S_{:k} \approx (z(p + h e_k) - z(p)) / h$, error $O(h)$; central $O(h^2)$ at twice
the solves. Step: $h = \sqrt{\varepsilon_{mach}} \max(|p_k|, 1) \approx 1.5 \times 10^{-8}$
(forward) or $\varepsilon^{1/3} \approx 6 \times 10^{-6}$ (central) — balance truncation vs
subtractive cancellation. Cost: $n_p$ (or $2 n_p$) full modal solves. **Mode tracking is
mandatory**: after each perturbed solve, re-pair modes to the nominal set by MAC before
differencing, otherwise a mode swap injects an $O(1)$ jump into a supposedly $O(h)$ difference
(the classic FD-updating failure). Acceptance: analytic vs central-FD agree to $10^{-5}$
relative on the golden beam.

## 4. WLS / Bayesian iteration — `update_model(...)`

```python
update_model(model, parameters, measured, weights=None,
             regularization=0.0, max_iter=20, tol=1e-6,
             move_limit=0.2, method="analytic") -> UpdateResult

measured = {"freq_hz": ..., "modes": ..., "mode_dofs": [...], "frf": ...}   # any subset

class UpdateResult:
    model: FEModel            # updated copy — never mutate the input
    p: np.ndarray             # final relative parameters
    covariance: np.ndarray    # posterior parameter covariance
    history: list[dict]       # per-iteration: p, residual norm, max |dp|, pairing
    converged: bool
```

Gauss–Newton step on the weighted, prior-regularized objective
$J(p) = \Delta z^\top W_z \Delta z + (p - p_{prior})^\top W_p (p - p_{prior})$:

$$\Delta p = \left( S^\top W_z S + W_p \right)^{-1}
             \left( S^\top W_z\, \Delta z + W_p (p_{prior} - p) \right)$$

- Bayesian reading: $W_z = C_z^{-1}$ (measurement covariance, default
  $\mathrm{diag}(1/\sigma_i^2)$ from `weights`), $W_p = C_p^{-1}$ (prior parameter covariance);
  posterior covariance $C_{post} = (S^\top W_z S + W_p)^{-1}$ — return it, it is the updating
  quality metric.
- $W_p = \lambda I$ (`regularization`) recovers plain Tikhonov; $\lambda = 0$ with
  $n_z \ge n_p$ is ordinary WLS.
- Underdetermined case ($n_p > n_z$, common in element-wise updating): use the dual form
  $\Delta p = C_p S^\top \left( S C_p S^\top + C_z \right)^{-1} \Delta z$ — same solution,
  smaller system, and it makes the minimum-norm behavior explicit.

Safeguards per iteration:

1. **Move limits**: clip $|\Delta p_k| \le$ `move_limit` (default 0.2) — Gauss–Newton on
   eigenvalue residuals overshoots badly outside the linearization trust region.
2. **Bounds**: project $p$ onto $[lower, upper]$ boxes after the step (projected GN); an active
   bound for > 2 iterations should be reported — it usually means wrong parameterization.
3. **Conditioning**: SVD of $W_z^{1/2} S$; truncate $\sigma_i / \sigma_1 < 10^{-8}$; log the
   collinearity between parameter columns (angle < 5° ⇒ the pair is not separately identifiable
   from this data — report, don't silently split the correction).
4. **Line search**: halve the step while $\lVert \Delta z \rVert$ does not decrease (max 5
   halvings) — cheap insurance against pairing flips.
5. **Convergence**: $\max_k |\Delta p_k| <$ `tol` *and* residual reduction < `tol` relative;
   also stop on stagnating pairing oscillation (report `converged=False` + history).

Exactness anchor (golden case): for a uniform-$E$ structure, $\lambda_r(p) = p\, \lambda_r(1)$
exactly, so one analytic GN step recovers a +10 % E perturbation to machine precision — the 2 %
acceptance tolerance only absorbs FD noise and multi-parameter coupling
(`docs/ACCEPTANCE.md` §6).

Pitfalls: over-parameterization fits noise (watch $C_{post}$ diagonal exploding); frequency
residuals alone cannot separate global $E$ from global $\rho$ (only $E/\rho$ is observable —
the classic identifiability trap; needs mass info or shape/FRF residuals); test DOFs ≠ FEM DOFs
— reduce FEM shapes to test DOFs for residuals (never expand noisy test shapes for updating);
units: relative parameters and relative residuals keep the problem $O(1)$, do not undo this
with absolute-frequency weighting.

## 5. Harmonic force identification — `identify_harmonic_forces(H, X, lam=0.0)`

Inverse problem $X(\omega) = H(\omega) F(\omega)$ per frequency line: given measured responses
$X$ `(n_out, n_f)` and FRFs $H$ `(n_out, n_in, n_f)` (from `dynamics` or test), recover
$F$ `(n_in, n_f)`. Tikhonov-regularized least squares via SVD
$H = U \Sigma V^{\mathsf H}$:

$$\hat F(\omega) = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}
\left( u_i^{\mathsf H} X(\omega) \right) v_i
\qquad(\lambda = 0 \Rightarrow \text{pseudo-inverse}).$$

Requirements and pitfalls: overdetermination $n_{out} > n_{in}$ strongly recommended;
conditioning is worst near resonances (all columns of $H$ collinear with the dominant mode —
$\mathrm{cond}(H)$ spikes) and at antiresonances of the driving-point rows; per-line $\lambda$
selection by the L-curve corner (log–log $\lVert \hat F \rVert$ vs residual) or GCV — expose
`lam="lcurve"`; correlated output noise wants a whitening $W^{1/2}$ pre-multiplier. Return
$\hat F$, per-line condition numbers, and the chosen $\lambda(\omega)$. Golden case: noiseless
2-DOF synthetic recovers the exact force to $10^{-8}$.

## 6. Complexity summary

| Kernel | Cost |
|---|---|
| Analytic eigen-sensitivities | $O(n_p m\, nnz)$, no solves |
| Nelson shape sensitivities | $O(n_p m)$ sparse solves |
| FD sensitivities | $n_p$ (+$n_p$ central) modal solves |
| GN step | $O(n_z n_p^2 + n_p^3)$ (dense, tiny) |
| Full update | `max_iter` × (modal solve + $S$ + step) |
| Force ID | $O(n_f \cdot n_{out} n_{in} \min(n_{out}, n_{in}))$ SVDs |

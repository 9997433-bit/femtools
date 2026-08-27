# Dynamics algorithms — FRF, damping, substructuring, time integration

Spec for `femtools.dynamics` (owner: R1-O2). Frozen entry points:

```python
from femtools.dynamics.frf import modal_frf, direct_frf
from femtools.dynamics.harmonic import harmonic_response
from femtools.dynamics.mba import modal_based_assembly
from femtools.dynamics.craig_bampton import craig_bampton
from femtools.dynamics.time_domain import time_history
from femtools.dynamics.residuals import residual_vectors
# Round-4 additions (REMAINING.md, owner R4-O2) — see §9:
from femtools.dynamics.cms_free import rubin, macneal, FreeCMSResult
```

Inputs/outputs are DOF tuples `(node_id, dof_code)` resolved through
`ModalResult.dof_map` / `AssemblyResult.dof_map` (see `fea.md` §1). FRFs are **receptance**
(displacement/force) unless `kind` says otherwise; mobility $= i\omega H$, accelerance
$= -\omega^2 H$.

## 1. Damping models (shared dict format)

The model is inferred from the keys (`femtools.dynamics.damping.as_damping`);
there is no `"model"` key. Multiple key groups combine by superposition.

```python
damping = 0.01                          # bare scalar/array -> modal zeta
damping = {"zeta": 0.01}                # modal viscous, scalar or (m,) per-mode
damping = {"alpha": a, "beta": b}       # Rayleigh, C = a*M + b*K
damping = {"eta": 0.02}                 # structural/hysteretic, scalar or per-mode
damping = {"C": C_matrix}               # explicit viscous matrix (or pass the matrix)
damping = RayleighDamping(a, b)         # any DampingModel instance passes through
```

Rayleigh in modal coordinates: $\zeta_r = \dfrac{\alpha}{2\omega_r} + \dfrac{\beta \omega_r}{2}$
— fit $(\alpha, \beta)$ from two anchor $(\omega, \zeta)$ pairs; note $\alpha$ over-damps the
low end and $\beta$ the high end. Structural (hysteretic) damping multiplies stiffness:
$K_c = K(1 + i\eta)$; it is frequency-independent and **only valid for harmonic/frequency-domain
analyses** — reject `structural` in `time_history` (non-causal in time domain).

## 2. Modal FRF — `modal_frf(modal, inputs, outputs, freq_hz, damping) -> FRFResult`

Mass-normalized modes ($\Phi^\top M \Phi = I$). Receptance between input DOF $q$ and output
DOF $p$:

$$H_{pq}(\omega) = \sum_{r=1}^{m} \frac{\phi_{pr}\,\phi_{qr}}{d_r(\omega)},\qquad
d_r(\omega) = \begin{cases}
\omega_r^2 - \omega^2 + 2 i \zeta_r \omega_r \omega & \text{viscous (modal / rayleigh)}\\[2pt]
\omega_r^2 (1 + i \eta_r) - \omega^2 & \text{structural}
\end{cases}$$

Vectorized evaluation per frequency line: with $\Phi_o$ (n_out × m), $\Phi_i$ (n_in × m),

$$H(\omega_k) = \Phi_o\, \mathrm{diag}\!\big(1/d_r(\omega_k)\big)\, \Phi_i^\top
\quad\Rightarrow\quad O(n_f\, m\, n_{out}\, n_{in})\ \text{total, no}\ n\ \text{dependence}.$$

```python
class FRFResult:
    H: np.ndarray        # complex (n_out, n_in, n_freq)
    freq_hz: np.ndarray  # (n_freq,)
    inputs: list[tuple[int, int]]
    outputs: list[tuple[int, int]]
    kind: str            # "receptance" | "mobility" | "accelerance"
```

Pitfalls: (i) rigid-body modes ($\omega_r \approx 0$) make $d_r \to -\omega^2$ — keep them, they
form the mass line, but guard $\omega = 0$ with $d_r = \omega_r^2 = 0 \Rightarrow$ receptance is
singular at DC for free structures (return `inf`+warn or require `freq_hz[0] > 0`);
(ii) truncation bias — see §4 residual vectors; the acceptance case (modal vs direct, 5 % L2 on
0.2–0.8 $f_{max}$ band with 20 modes) fails if modes are truncated below ~1.5× band top without
residuals; (iii) per-mode arrays (`zeta`, `eta`) must broadcast to `m` after residual-vector
augmentation too.

`harmonic_response(model_or_modal, loads, freq_hz, damping)` is the forced-response wrapper:
same kernels, complex load vector instead of unit inputs, returns displacement spectra at
requested DOFs.

## 3. Direct FRF — `direct_frf(model, inputs, outputs, freq_hz, damping)`

Dynamic stiffness per line:

$$Z(\omega) = \begin{cases}
K + i\omega C - \omega^2 M, & C = \alpha M + \beta K \ \text{(rayleigh)}\\
K(1 + i\eta) - \omega^2 M & \text{(structural)}
\end{cases}$$

Solve $Z(\omega_k) X = F$ where $F$ is the n × n_in unit-force scatter; then
$H[:, :, k] = X[\text{output rows}, :]$. One complex sparse LU per line:
$O(n_f \cdot \mathrm{LU}(n))$ — this is the expensive oracle used to validate `modal_frf`.
Implementation notes: $Z$ is complex **symmetric**, not Hermitian — plain SuperLU
(`splu`) is correct; reuse the symbolic pattern across lines (same sparsity ∀ω) via
`splu(Z_k)` with precomputed CSC structure; batch RHS in one solve call. `modal` damping has no
physical $C$ matrix: synthesize $C = \Phi^{-\top} \mathrm{diag}(2\zeta_r\omega_r) \Phi^{-1}$ only
implicitly by refusing `modal` for `direct_frf` and documenting that comparisons use
`rayleigh` or `structural` in both functions (acceptance does exactly this).

Pitfall: at $\omega$ exactly on an undamped resonance with $\zeta = \eta = 0$, $Z$ is singular —
require nonzero damping or off-resonance grid.

## 4. Residual vectors — `residual_vectors(model, modal, inputs) -> ModalResult`

Static correction for modal truncation (Dickens/MacNeal). For unit loads $F$ at the input DOFs:

$$R = K^{-1} F - \sum_{r=1}^{m} \frac{\phi_r (\phi_r^\top F)}{\omega_r^2}
\qquad\text{(attachment-mode residual flexibility)}.$$

Then (1) M-orthogonalize $R$ against retained $\Phi$: $R \leftarrow R - \Phi (\Phi^\top M R)$;
(2) drop near-null columns (SVD, $\sigma_i / \sigma_1 < 10^{-8}$ — inputs close to existing mode
content produce rank deficiency); (3) solve the small projected eigenproblem
$(R^\top K R)\, v = \mu\, (R^\top M R)\, v$ and append pseudo-modes $R v$ with pseudo-frequencies
$\sqrt{\mu}/2\pi$, mass-normalized. The augmented `ModalResult` drops straight into `modal_frf`.

Free–free structures: $K^{-1}$ does not exist — use the shifted operator
$(K + \sigma M)^{-1}$ with small $\sigma > 0$ and project out rigid-body content (inertia-relief
attachment modes). Pitfall: forgetting this makes residual vectors silently explode along RBMs.

## 5. Craig–Bampton — `craig_bampton(model, boundary, n_modes) -> CBResult`

Partition active DOFs into interior $i$ and boundary $b$ (`boundary`: list of
`(node_id, dof_code)`).

- Constraint modes: $\Psi = -K_{ii}^{-1} K_{ib}$ (one factorization, $n_b$ solves).
- Fixed-interface normal modes: $K_{ii} \Phi = M_{ii} \Phi \Lambda$, keep $m$ lowest,
  mass-normalized.

$$T = \begin{bmatrix} \Phi & \Psi \\ 0 & I \end{bmatrix},\qquad
\hat K = T^\top K T = \begin{bmatrix} \Lambda & 0 \\ 0 & K_{bb} - K_{bi} K_{ii}^{-1} K_{ib} \end{bmatrix},\qquad
\hat M = T^\top M T = \begin{bmatrix} I & \hat M_{qb} \\ \hat M_{qb}^\top & \hat M_{bb} \end{bmatrix}$$

with $\hat M_{qb} = \Phi^\top (M_{ib} + M_{ii}\Psi)$ and
$\hat M_{bb} = M_{bb} + M_{bi}\Psi + \Psi^\top M_{ib} + \Psi^\top M_{ii} \Psi$. The $\hat K$
boundary block is the Guyan/Schur complement; $\hat K$ has no coupling block — a nonzero one is
an implementation bug (good unit test).

```python
class CBResult:
    K_red: np.ndarray; M_red: np.ndarray      # (m + n_b) square, dense
    T: np.ndarray                             # (n, m + n_b) recovery basis
    boundary: list[tuple[int, int]]
    fixed_interface_freq_hz: np.ndarray
```

Truncation rule: keep fixed-interface modes to ≥ 1.5–2× the analysis band. Exactness invariant
(golden test): with **all** interior modes kept, CB is a change of basis — reduced eigenvalues
match the full model to ~1e-10 relative. Pitfalls: large $n_b$ makes $\Psi$ a dense
$n_i \times n_b$ block (memory) and degrades reduction quality — consider characteristic
constraint-mode (interface) reduction later; ill-conditioning if boundary DOFs include
mechanism-like massless rotations.

## 6. Modal-based assembly (MBA) and SDM — `modal_based_assembly`

Couple component modal models (test- or FE-derived) through connector elements — the FEMtools
"modal-based assembly" workflow; the single-component special case is classical SDM (structural
dynamics modification).

Components $A, B$ with mass-normalized $(\Lambda_A, \Phi_A)$, $(\Lambda_B, \Phi_B)$ given at
their physical connection DOFs. Generalized coordinates $q = (q_A, q_B)$:

$$\hat M = I,\qquad \hat K_0 = \mathrm{blkdiag}(\Lambda_A, \Lambda_B).$$

A connector (spring $k_c$, damper $c_c$, or point mass) between DOF $a$ (in A) and DOF $b$
(in B) adds, with the relative-motion row $w^\top = [\Phi_A[a,:],\ -\Phi_B[b,:]]$:

$$\Delta \hat K = k_c\, w w^\top \quad (\text{rank-1 per connector}),\qquad
\Delta \hat M = m_c\, \tilde w \tilde w^\top,\ \tilde w^\top = [\Phi_A[a,:],\ \Phi_B[b,:]] .$$

Rigid connections: penalty $k_c \to$ large (bounded by conditioning, see pitfall) or exact
dual assembly with Lagrange multipliers on $B q = 0$ compatibility rows (preferred; solve the
constrained eigenproblem via null-space basis $q = N z$). Then solve the small dense
generalized eigenproblem $(\hat K_0 + \Delta\hat K)\, v = \mu (I + \Delta\hat M)\, v$
($O((m_A + m_B)^3)$) and expand $\Phi_{new} = \mathrm{blkdiag}(\Phi_A, \Phi_B)\, V$.

SDM: identical with one component — re-eigen
$(\Lambda + \Phi^\top \Delta K \Phi)\, v = \mu (I + \Phi^\top \Delta M \Phi)\, v$.

Pitfalls: **truncation dominates accuracy** — a stiff connector exercises static flexibility the
kept modes may not span; augment components with residual vectors at connection DOFs (§4) before
coupling (this is the difference between a toy and a usable MBA). Penalty springs beyond
~$10^4 \times$ the largest component modal stiffness destroy the eigenproblem conditioning.
Test-derived shapes are not exactly mass-normalized — scale errors propagate quadratically into
coupled frequencies.

## 7. Time integration — `time_history`

```python
time_history(model_or_modal, loads, t, damping, method="newmark",  # or "modal"
             beta=0.25, gamma=0.5, u0=None, v0=None) -> ThResult(u, v, a, t)
```

`loads`: dict DOF → `ndarray (n_t,)` force samples on the uniform grid `t`.

### 7.1 Newmark-β (physical coordinates)

Predictors, then one linear solve per step ($M \ddot u + C \dot u + K u = f$):

$$\tilde u = u_n + \Delta t\, v_n + \Delta t^2 (\tfrac12 - \beta) a_n,\qquad
  \tilde v = v_n + \Delta t (1 - \gamma) a_n$$
$$\big(M + \gamma \Delta t\, C + \beta \Delta t^2 K\big)\, a_{n+1} = f_{n+1} - C \tilde v - K \tilde u$$
$$u_{n+1} = \tilde u + \beta \Delta t^2 a_{n+1},\qquad v_{n+1} = \tilde v + \gamma \Delta t\, a_{n+1}.$$

Defaults $\gamma = \tfrac12, \beta = \tfrac14$ (average acceleration): unconditionally stable,
zero algorithmic damping, period elongation $\approx (\omega \Delta t)^2 / 12$. Constant
$\Delta t$, $K$, $C$, $M$ → factorize the effective matrix once (sparse LU), $O(n_t)$
back-substitutions. Initial acceleration from $M a_0 = f_0 - C v_0 - K u_0$. Accuracy rule:
$\Delta t \le 1/(20 f_{max}^{interest})$. Pitfalls: $\gamma > \tfrac12$ without matching $\beta$
adds first-order numerical damping (sometimes wanted — expose it, don't hardcode); undamped
free structures drift in rigid-body coordinates (exact behavior, not a bug); zero-mass DOFs
(massless springs) make $M$ singular but the effective matrix is still fine for $\beta > 0$.

### 7.2 Modal time history (exact piecewise-linear / ramp-invariant)

Project to modal SDOFs $\ddot q_r + 2\zeta_r \omega_r \dot q_r + \omega_r^2 q_r = p_r(t) =
\phi_r^\top f(t)$. For force linear within each step, the discrete update is **exact**
(Duhamel over one step with a ramp), the classic recurrence:

$$q_{n+1} = A q_n + B \dot q_n + C p_n + D p_{n+1},\qquad
 \dot q_{n+1} = A' q_n + B' \dot q_n + C' p_n + D' p_{n+1}$$

with coefficients built from $e^{-\zeta\omega\Delta t}$, $\sin/\cos(\omega_d \Delta t)$,
$\omega_d = \omega\sqrt{1-\zeta^2}$ (see any standard derivation; implement once, test against
the closed-form step response $q(t) = (1 - e^{-\zeta\omega t}(\cos\omega_d t +
\tfrac{\zeta\omega}{\omega_d}\sin\omega_d t))/\omega^2$). Unconditionally stable, error only
from linear force interpolation; complexity $O(m\, n_t)$ — orders faster than Newmark for
long records. Handle $\omega_r = 0$ (rigid body) analytically:
$q_{n+1} = q_n + \Delta t \dot q_n + \Delta t^2 (p_n/3 + p_{n+1}/6)$,
$\dot q_{n+1} = \dot q_n + \Delta t (p_n + p_{n+1})/2$. Pitfall: truncation again — physical
response needs enough modes; offer static correction with residual vectors.

## 8. Complexity summary

| Kernel | Cost |
|---|---|
| `modal_frf` | $O(n_f\, m\, n_{out} n_{in})$ |
| `direct_frf` | $O(n_f)$ × complex sparse LU($n$) |
| `residual_vectors` | 1 LU + $n_{in}$ solves + small eig |
| `craig_bampton` | 1 LU($K_{ii}$) + $n_b$ solves + ARPACK($m$) |
| `rubin` / `macneal` (§9) | eig($m$ kept) + 1 LU($K$) + $n_b$ solves + dense $O(n_b^3)$ |
| `modal_based_assembly` | $O((\sum m_c)^3)$ dense eig |
| Newmark | 1 LU + $n_t$ back-subs |
| Modal TH | $O(m\, n_t)$ |

## 9. Free-interface CMS — MacNeal / Rubin (Round 4, owner R4-O2)

Spec for `femtools.dynamics.cms_free`, frozen in `.agent_workspace/REMAINING.md`.
Signature and conventions **mirror `craig_bampton`** (matrix-level, integer DOF indices,
reduced coordinate order `[boundary..., modal...]` — `examples/cms_rubin.py` couples
superelements assuming exactly this ordering):

```python
def rubin(K, M, boundary_dofs, n_modes=0, *, interior_dofs=None) -> FreeCMSResult
def macneal(K, M, boundary_dofs, n_modes=0, *, interior_dofs=None) -> FreeCMSResult

class FreeCMSResult:        # CraigBamptonResult-like
    K: np.ndarray; M: np.ndarray   # (n_b + m) square, order [boundary, modal]
    T: np.ndarray                  # (ndof, n_b + m) in the original DOF ordering
    boundary_dofs: np.ndarray
    free_freq_hz: np.ndarray       # kept free-interface (elastic) mode frequencies
```

### 9.1 Residual-flexibility basis

The component is analyzed with the interface **free** (its own supports, if any, stay
applied). Keep the $m$ lowest free-interface modes $(\Lambda, \Phi)$, mass-normalized,
and correct the truncation with the residual flexibility restricted to the boundary
columns $b$:

$$G_{res} = K^{-1} - \Phi \Lambda^{-1} \Phi^\top,\qquad G_b = G_{res}[:, b].$$

The raw basis $u = \Phi q + G_b f_b$ uses interface *forces* as coordinates; components
couple on interface *displacements*, so eliminate $f_b$ via
$u_b = \Phi_b q + G_{bb} f_b$:

$$u = \underbrace{G_b G_{bb}^{-1}}_{T_b}\, u_b
 + \underbrace{\left(\Phi - G_b G_{bb}^{-1} \Phi_b\right)}_{T_q}\, q,\qquad
T = [\,T_b\ \ T_q\,].$$

- **Rubin**: full Galerkin projection $K_{red} = T^\top K T$, $M_{red} = T^\top M T$ —
  the residual attachment shapes carry their (second-order) inertia.
- **MacNeal**: same $T$ but the residual coordinates are massless. In $(f_b, q)$
  coordinates $\hat M = \mathrm{diag}(0, I)$; transforming with
  $W = \left[\begin{smallmatrix} G_{bb}^{-1} & -G_{bb}^{-1}\Phi_b \\ 0 & I \end{smallmatrix}\right]$
  gives $M_{red} = W^\top \hat M W$, whose **interface block is singular**. Consumers must
  solve the coupled pencil with QZ (`scipy.linalg.eig`) and discard the infinite
  eigenvalues — plain `eigh` raises on the non-SPD mass. Rubin's $M_{red}$ is SPD.

Free–free components (rigid modes present): $K^{-1}$ does not exist — use inertia-relief
residual flexibility, $G = P^\top K_c^{-1} P$ with the rigid-body force projector
$P = I - M R (R^\top M R)^{-1} R^\top$ and a temporarily constrained $K_c$, and keep the
rigid modes in $\Phi$ (they belong to $\Lambda^{-1}$ only through the elastic partition).
The acceptance demo deliberately uses two *supported* components (fixed–fixed beam split
at midspan) so this path is optional for the first landing; raise `NotImplementedError`
rather than returning garbage if it is deferred.

### 9.2 Coupling recipe and measured accuracy

Primal assembly on shared boundary DOFs: stack $[u_b, q_A, q_B]$, add each component's
reduced blocks at $[u_b, q_A]$ resp. $[u_b, q_B]$ (see `couple()` in
`examples/cms_rubin.py`). Measured on the fixed–fixed 20×BEAM2 beam split at midspan
(8 kept modes + 6 interface DOFs per component, first 4 coupled modes vs the unsplit
model): max relative frequency error **1.9e-7 (Rubin)**, **7.7e-5 (MacNeal)**,
7.7e-5–2.9e-5 for the Craig–Bampton baseline through the same harness. Rubin beating CB
at equal mode count is the expected textbook ranking; MacNeal trades a little accuracy
for the sparser massless-residual model. Acceptance (case 18) uses generous headroom:
Rubin < 1 %, MacNeal < 3 % on the first four coupled modes. A sharper component-level
invariant (pinned by `tests/test_round4_cms.py`): the kept free-interface modes lie in
$\mathrm{span}(T)$, so the Rubin-reduced component reproduces their eigenvalues to
~1e-8 relative — exactness of the *retained* modes, the free-interface analogue of the
CB all-modes check.

Pitfalls: truncation rule is the same 1.5–2× band as CB, but free-interface component
modes converge to coupled modes *slower* than fixed-interface ones when the interface is
stiff — residual flexibility is what closes the gap, never ship the method without it
(plain mode-displacement coupling is 10–100× worse); $G_{bb}$ inherits the conditioning
of the master selection (all-rotation interfaces on fine meshes are nearly singular);
mixing Rubin and MacNeal components in one assembly is fine (the harness only assumes
boundary-first ordering); test-derived $\Phi$ that is not mass-normalized corrupts
$\Lambda^{-1}$ quadratically.

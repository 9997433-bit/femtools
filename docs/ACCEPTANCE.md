# Acceptance — golden analytical cases and tolerances

Expands the tolerance table in `docs/CONTRACT_API.md` with the exact formulas each golden test
checks against. Tests live in `tests/` (owner R1-G1/R2-G1); recommended test ids are listed so
failures map back here. Conventions: $c = \sqrt{E/\rho}$ (bar wave speed), $f = \omega/2\pi$,
all modes mass-normalized ($\Phi^\top M \Phi = I$).

## Measured status (merged tree, 2026-08-27)

Verified by running `examples/*.py` and `pytest tests/` against the merged
`cursor/femtools-sota-d551` tree (17 passed, 3 perf skips). Checked = measured
passing with the quoted numbers; unchecked = not yet exercised by a test or
example on this tree.

- [x] **1a** axial bar, 2-node discrete — `tests/test_golden_fea.py::test_two_node_axial_bar_frequency` green
- [ ] **1b** axial bar N-element dispersion — no test/example yet
- [ ] **1c** axial bar mesh-converged continuum — no test/example yet
- [x] **2** cantilever EB, 10 BEAM2 — `examples/cantilever_beam.py`: 6 lowest bending modes
      (both planes, 16.71–439.95 Hz) max rel err **2.55e-4** (tol 2e-2);
      `tests/test_golden_fea.py::test_euler_bernoulli_cantilever_first_three_modes_per_bending_plane` green
- [x] **3a** mass normalization — `examples/cantilever_beam.py`:
      $\max|\Phi^\top M \Phi - I|$ = **6.66e-16** (tol 1e-8); `tests/test_mass_normalization.py` green
- [ ] **3b** stiffness orthogonality — not asserted separately yet
- [x] **4a–4c** MAC identities — `examples/mac_demo.py`: identity/scale-invariance dev
      **2.22e-16** (tol 1e-12/1e-10); `tests/test_mac.py` (2 tests) green; Hungarian pairing
      recovers a shuffled+perturbed set 5/5 (`tests/test_pairing.py` green)
- [ ] **5** cantilever effective mass — no test/example yet
- [x] **6** updating: recover E (+10 %) — `examples/update_youngs.py`: relative E error
      **1.76e-10** in 3 iterations (tol 2e-2), converged, WLS/Gauss–Newton;
      `tests/test_updating.py` green
- [ ] **7a** SDOF modal FRF closed form — no test/example yet
- [x] **7b** modal vs direct FRF — `examples/frf_synthesis.py`: 10×BEAM2 cantilever,
      20 modes, Rayleigh ζ≈1 %, band 783–3132 Hz (0.2–0.8 f_max): rel L2
      **1.40 %** (tip driving point) / **0.98 %** (midspan transfer), tol 5 %;
      `tests/test_frf.py` green
- [x] **8** EFI toy — `examples/pretest_efi.py`: 10-DOF chain, 2 target modes, 4 sensors
      → kept-row AutoMAC off-diag **0.0502** (EFI) / **0.0416** (MAC elimination), tol < 0.15;
      `tests/test_efi.py` green
- [ ] **9** free–free beam rigid modes — no test/example yet
- [ ] **10** Craig–Bampton exactness — no test/example yet
- [ ] **11** static bar/cantilever tip — no test/example yet
- [ ] **12** Newmark SDOF period error — no test/example yet
- [ ] **13** force identification — no test/example yet
- [x] **14** LHS stratification — `tests/test_doe.py` (bounded, stratified, seeded) green
- [ ] **15** RBPE synthetic rigid body — no test/example yet
- [ ] **16** FDD synthetic 2-DOF — no test/example yet
- [x] **SIMP** (small-mesh smoke level only) — `tests/test_topology.py` green; the full
      60×20 MBB criteria of §8 remain unmeasured

## 0. Master table

| # | Case | Exact reference | Metric | Tol |
|---|------|-----------------|--------|-----|
| 1a | Axial bar, 2-node fixed–free, consistent M | $\omega = \sqrt{3}\,c/L$ (discrete closed form) | rel freq | 1e-8 |
| 1b | Axial bar, N-element fixed–free | discrete dispersion, §1.2 | rel freq (all modes) | 1e-8 |
| 1c | Axial bar, fixed–fixed, mesh-converged (≥100 el) | $f_1 = \dfrac{1}{2L}\sqrt{\dfrac{E}{\rho}}$ | rel freq | 1e-4 |
| 2 | Cantilever, ≥10 BEAM2, first 3 bending | $\beta_n L$ roots 1.875104, 4.694091, 7.854757 (§2) | rel freq | 2e-2 |
| 3a | Mass normalization | $\Phi^\top M \Phi = I$ | max abs dev | 1e-8 |
| 3b | Stiffness orthogonality | $\Phi^\top K \Phi = \Lambda$ | rel dev | 1e-6 |
| 4a | MAC self | $\mathrm{MAC}(x,x) = 1$ | $1 - \min \mathrm{diag}$ | 1e-12 |
| 4b | MAC scale/phase invariance | $\mathrm{MAC}(\alpha x, \beta y) = \mathrm{MAC}(x, y)$ | abs dev | 1e-12 |
| 4c | MAC orthonormal basis | $\mathrm{MAC}(Q, Q) = I$ | max offdiag | 1e-10 |
| 5 | Cantilever effective mass, transverse | fractions ≈ 0.613, 0.188, 0.065 (§5) | abs dev | 1e-2 |
| 6 | Updating: recover E (+10 % seeded error) | $\lambda_r \propto E$ exactly (§6) | rel E error | 2e-2 |
| 7a | Modal FRF, 1-DOF | $H = (k - m\omega^2 + ic\omega)^{-1}$ exactly (§7) | rel | 1e-12 |
| 7b | Modal vs direct FRF, ≥20 modes, light damping | same damping model both sides | rel L2 on 0.2–0.8 $f_{max}$ | 5e-2 |
| 8 | EFI toy (10 DOF, 2 target modes, 4 sensors) | AutoMAC of kept rows | max offdiag | < 0.15 |
| 9 | Free–free beam | 6 rigid modes at ~0 Hz; elastic per §2 free–free roots | $f_{rb} < 10^{-4} f_{el,1}$; rel 2e-2 | — |
| 10 | Craig–Bampton, all interior modes kept | exact change of basis | rel eigenvalue dev | 1e-10 |
| 11 | Static: bar tip $u = FL/EA$; cantilever tip $\delta = FL^3/3EI$ | Hermite/linear exact for end loads | rel | 1e-12 |
| 12 | Newmark SDOF ($\gamma{=}\tfrac12, \beta{=}\tfrac14$) | period error $\approx (\omega\Delta t)^2/12$ (§7.3) | bounded by 2× estimate | — |
| 13 | Force ID, noiseless 2-DOF | $\hat F = H^{+} X$ exact | rel | 1e-8 |
| 14 | LHS stratification | 1 sample per stratum per dim, seeded reproducibility | exact | — |
| 15 | RBPE synthetic rigid body | recover $(m, c, J)$ (§8) | rel | 1e-8 |
| 16 | FDD synthetic 2-DOF | peak at $f_r$ within PSD resolution; shape MAC | $\Delta f \le df$; MAC > 0.99 | — |

## 1. Axial bar

### 1.1 Continuum references

Longitudinal vibration of a uniform bar, wave speed $c = \sqrt{E/\rho}$:

- fixed–fixed and free–free (elastic): $f_n = \dfrac{n}{2L}\sqrt{\dfrac{E}{\rho}}$, $n = 1, 2, \ldots$
  — in particular the headline golden value $\boxed{f_1 = \frac{1}{2L}\sqrt{E/\rho}}$;
- fixed–free: $f_n = \dfrac{2n-1}{4L}\sqrt{\dfrac{E}{\rho}}$.

These are *mesh-convergence* targets (case 1c): a 2-node model cannot and must not be tested
against them at 1e-8.

### 1.2 Discrete closed forms (what 1e-8 actually checks)

For $N$ equal 2-node elements of length $h = L/N$ with **consistent** mass, the interior
difference equation $\frac{EA}{h}(-u_{j-1} + 2u_j - u_{j+1}) = \omega^2 \frac{\rho A h}{6}
(u_{j-1} + 4 u_j + u_{j+1})$ is solved exactly by $u_j = \sin(j\gamma)$, giving the discrete
dispersion relation

$$\omega_k = \frac{c}{h} \sqrt{ \frac{6 \left( 1 - \cos\gamma_k \right)}{2 + \cos\gamma_k} },
\qquad
\gamma_k = \begin{cases}
\dfrac{(2k-1)\pi}{2N} & \text{fixed–free},\\[6pt]
\dfrac{k\pi}{N} & \text{fixed–fixed}.
\end{cases}$$

Checks: $N = 1$, fixed–free ⇒ $\gamma = \pi/2$ ⇒ $\omega = \sqrt{3}\,c/L$ (case 1a, the
contract's "2-node vs analytical" row); $\gamma \to 0$ recovers the continuum formulas above
(consistent mass converges from above, error $O(\gamma^2/ ...) \sim O(N^{-2})$ per mode). With
lumped mass the dispersion is $\omega_k = \frac{2c}{h} \sin(\gamma_k/2)$ (converges from below);
the golden tests pin the consistent-mass branch. Test ids:
`test_fea_golden.py::test_bar_2node_discrete`, `::test_bar_dispersion[N=4,8,16]`,
`::test_bar_converged_continuum`.

## 2. Euler–Bernoulli cantilever (and free–free) beam

Bending frequencies of a uniform EB beam:

$$f_n = \frac{(\beta_n L)^2}{2\pi L^2} \sqrt{\frac{E I}{\rho A}}$$

Cantilever characteristic equation $\cos(\beta L)\cosh(\beta L) = -1$, first roots:

| $n$ | $\beta_n L$ |
|---|---|
| 1 | 1.8751040687 |
| 2 | 4.6940911330 |
| 3 | 7.8547574382 |
| $n \ge 4$ | $\approx (2n - 1)\pi/2$ |

Free–free: $\cos(\beta L)\cosh(\beta L) = +1$, roots 4.7300407449, 7.8532046241,
10.9956078381 (case 9). Acceptance: ≥10 BEAM2 elements, consistent mass, no rotary inertia →
first three bending frequencies within 2 % (they converge $O(h^4)$; at 10 elements the actual
error is ≲0.1 %, the 2 % headroom absorbs section/orientation variations). For non-square
sections both bending planes appear ($I_y \ne I_z$): compare against the union of the two
analytic families sorted ascending — see `examples/cantilever_beam.py`.
Test id: `test_fea_golden.py::test_cantilever_eb`.

## 3. Eigen-solution identities

$\max |\Phi^\top M \Phi - I| \le 10^{-8}$ (contract row) and
$\max_r |\phi_r^\top K \phi_r - \lambda_r| / \lambda_r \le 10^{-6}$ on every golden model.
Rigid-body eigenvalues clamp to 0, never NaN (`fea.md` §6.1).

## 4. MAC identities

With $\mathrm{MAC}_{ij} = |\phi_i^{\mathsf H} \psi_j|^2 / ((\phi_i^{\mathsf H}\phi_i)
(\psi_j^{\mathsf H}\psi_j))$:

- (4a) reflexivity: $\mathrm{MAC}(x, x) = 1$ for any $x \ne 0$, real or complex;
- (4b) invariance: $\mathrm{MAC}(\alpha x, \beta y) = \mathrm{MAC}(x, y)$ for all
  $\alpha, \beta \in \mathbb{C} \setminus \{0\}$ — includes sign flips and complex phase;
- (4c) orthonormal basis: columns of any $Q$ with $Q^{\mathsf H} Q = I$ give
  $\mathrm{MAC}(Q, Q) = I$ (off-diag ≤ 1e-10; generate $Q$ by `np.linalg.qr` of a seeded
  random matrix);
- bounds: $0 \le \mathrm{MAC} \le 1$ always (Cauchy–Schwarz) — property-test with random
  complex vectors (hypothesis, seeded).

Note 4c uses *unit-weighted* orthonormality; M-orthogonal mode sets satisfy the analogous
identity for POC/XOR (`correlation.md` §3), not for MAC. Test ids:
`test_correlation_golden.py::test_mac_self`, `::test_mac_invariance`, `::test_mac_orthonormal`.

## 5. Effective mass (cantilever, transverse)

$L = \Phi^\top M R$, $m_{\mathrm{eff},r} = L_r^2$, completeness
$\sum_r L_r L_r^\top = R^\top M R$ (checked to 1e-6 with all modes of a small model).
Classical transverse fractions for a uniform cantilever (any $E, \rho, L$ — dimensionless):

| mode | $m_{\mathrm{eff}} / m_{total}$ |
|---|---|
| 1 | 0.6131 |
| 2 | 0.1883 |
| 3 | 0.0647 |

Tolerance 1e-2 absolute at ≥20 elements. Test id: `test_pretest_golden.py::test_effmass_cantilever`.

## 6. Updating recovery

Golden construction: take the 2-parameter cantilever, multiply material 1's $E$ by 1.10, solve
modes → these frequencies are the synthetic "test" data; reset the model to $E_0$ and run
`update_model` on parameter $E$. Exactness anchor: when the whole stiffness scales with one
parameter and $M$ is independent of it,

$$K(p) = p K_0 \ \Rightarrow\ \lambda_r(p) = p\, \lambda_r(1)\ \ \forall r
\quad\Rightarrow\quad
\frac{\partial \lambda_r}{\partial p} = \lambda_r \ \ \text{exactly},$$

so a single analytic Gauss–Newton step lands on $p = 1.10$ to machine precision; the 2 %
acceptance tolerance only allows for FD sensitivities and multi-parameter runs (e.g. $E$ +
spring $k$ with correlated columns). Must also assert: `UpdateResult.converged`, monotone
residual history, input model not mutated. Test id:
`test_updating_golden.py::test_recover_youngs`. Force identification (case 13): synthetic
$X = H F_{true}$, noiseless ⇒ $\hat F$ within 1e-8 relative.

## 7. FRF golden cases

### 7.1 SDOF closed form (case 7a)

Mass-normalized single mode $\phi = 1/\sqrt{m}$, $\omega_0^2 = k/m$,
$\zeta = c / (2\sqrt{km})$ makes the modal sum **algebraically identical** to

$$H(\omega) = \frac{1}{k - m\omega^2 + i c \omega},$$

so `modal_frf` on a 1-DOF model must match to 1e-12 — this pins sign conventions, damping
insertion, and receptance/accelerance kinds before any multi-DOF comparison.

### 7.2 Modal vs direct (case 7b, contract row)

Same physical damping on both sides (`rayleigh` or `structural` — `modal` has no assembled $C$,
see `dynamics.md` §3), ≥20 modes on a ≥60-DOF beam, band $[0.2, 0.8] f_{max}$ where
$f_{max} = $ 20th frequency. Metric:
$\lVert H_{modal} - H_{direct} \rVert_2 / \lVert H_{direct} \rVert_2 \le 0.05$ per FRF.
With residual vectors enabled the same metric must pass 1e-3 (regression-guard the
improvement). Test ids: `test_dynamics_golden.py::test_frf_sdof_exact`,
`::test_modal_vs_direct`.

### 7.3 Time integration (case 12)

Newmark average-acceleration on an undamped SDOF, $\omega \Delta t = 0.1$, 100 cycles: no
amplitude decay (energy drift < 1e-6 relative), period elongation within 2× the estimate
$(\omega\Delta t)^2 / 12$. Modal time history: exact match (1e-10) to the closed-form damped
step response $q(t) = \big(1 - e^{-\zeta\omega t}(\cos\omega_d t +
\tfrac{\zeta\omega}{\omega_d}\sin\omega_d t)\big)/\omega^2$ at the sample instants, since the
ramp-invariant recurrence is exact for piecewise-linear loads.

## 8. Remaining module goldens

- **EFI toy (case 8, contract row)**: 10-DOF fixed–free spring–mass chain (unit masses/springs;
  modes from a dense `eigh` oracle), 2 target modes, select 4 sensors →
  AutoMAC off-diag of the kept rows < 0.15. The implemented
  `EFIResult.history` records `(n_remaining, min E_D)` per elimination step
  (the minimum leverage grows monotonically as the weakest candidates are
  dropped — measured 0.040 → 0.450 on the toy chain); a log-det trace is not
  exposed directly. See `examples/pretest_efi.py`.
- **Craig–Bampton exactness (case 10)**: any golden model, keep *all* interior modes → reduced
  eigenvalues match full-model eigenvalues to 1e-10 relative; $\hat K$ coupling block exactly 0.
- **LHS (case 14)**: for every dimension, the $n$ samples occupy $n$ distinct strata
  (exact integer check); identical output for identical seed; maximin criterion never decreases
  the min pairwise distance vs the plain draw.
- **SIMP**: 60×20 MBB half-beam, volfrac 0.5, $r_{min} = 2$ → compliance history monotone
  decreasing (tol 1e-9 per step), volume constraint met to 1e-6, final design symmetric under
  the mesh's load/BC symmetry.
- **RBPE (case 15)**: rigid block ($m = 10$ kg, $c = (0.1, 0.05, 0.2)$ m, principal
  $J = \mathrm{diag}(0.5, 0.8, 1.0)$ kg·m²) on 6 soft springs; analytic 6-DOF FRFs sampled on
  a mass-line band → all 10 parameters within 1e-8 relative; $J_G$ SPD and triangle
  inequalities hold.
- **FDD (case 16)**: 2-DOF system, white-noise excitation (seeded), Welch with
  $n_{perseg}$ giving ≥10 lines across each half-power band → peaks within one frequency bin,
  shape MAC vs analytic > 0.99; EFDD damping within 20 % relative (leakage-limited — this is a
  sanity bound, not a precision claim).
- **p-LSCF / LSCE**: synthetic 5-mode FRF matrix from known poles/residues, no noise → poles
  recovered to 1e-6 relative in $f$, 1e-3 absolute in $\zeta$ at the correct model order;
  with 1 % noise, stabilization diagram contains all 5 physical poles flagged stable.

## 9. Determinism requirements (all tests)

Fixed seeds via `np.random.default_rng(seed)` only; eigenvector sign convention per
`fea.md` §6.2; degenerate-subspace comparisons via MAC/S2MAC, never entrywise; no test may
depend on ARPACK iteration counts or wall time.

# Correlation algorithms — MAC family, orthogonality, FRF correlation, pairing

Spec for `femtools.correlation` (owner: R1-O3). Frozen entry points:

```python
from femtools.correlation.mac import mac_matrix, comac, poc
from femtools.correlation.pairing import pair_modes
from femtools.correlation.frf_corr import frac, csac, csf
from femtools.correlation.orthogonality import cross_orthogonality
# Round-4 expansion operators (owner R4-O3) and the Round-10 composition
# (owner R10-O3; landed on this tree) — see §6:
from femtools.correlation.expansion import expand_guyan, expand_serep
from femtools.correlation.expansion import expanded_mac
```

All mode-shape arguments are column-major: `phi` is `(n_dof, n_modes)`, complex allowed
everywhere (test shapes are complex; FE shapes real). Both sets must live on the **same DOF
rows** — caller reduces FE shapes to test DOFs (or expands test shapes) *before* calling;
these kernels do no geometry matching.

## 1. MAC — `mac_matrix(phi_a, phi_b=None) -> (m_a, m_b)`

$$\mathrm{MAC}_{ij} =
\frac{\left| \phi_{a,i}^{\mathsf H}\, \phi_{b,j} \right|^2}
     {\left( \phi_{a,i}^{\mathsf H} \phi_{a,i} \right)\left( \phi_{b,j}^{\mathsf H} \phi_{b,j} \right)}
\in [0, 1]$$

($^{\mathsf H}$ = conjugate transpose; bounded by Cauchy–Schwarz). `phi_b=None` → AutoMAC.
Vectorized: `num = np.abs(phi_a.conj().T @ phi_b)**2`, divide by the outer product of squared
column norms — $O(n\, m_a m_b)$, no loops.

Identities (golden tests, see `docs/ACCEPTANCE.md` §3):
$\mathrm{MAC}(x, x) = 1$; scale/phase invariance
$\mathrm{MAC}(\alpha x, \beta y) = \mathrm{MAC}(x, y)\ \forall \alpha,\beta \in \mathbb{C}\setminus\{0\}$;
for a matrix $Q$ with orthonormal columns, $\mathrm{MAC}(Q, Q) = I$ to $10^{-12}$.

Pitfalls: (i) MAC is a *unit-weighted* correlation — it is **not** an orthogonality check;
M-orthogonal modes of a strongly non-uniform mass distribution can show large off-diagonal MAC
(use POC/XOR for orthogonality statements); (ii) spatial aliasing: with few sensor DOFs,
distinct modes look collinear and MAC saturates near 1 — this is exactly what
`pretest.eliminate_by_mac` guards; (iii) forgetting the conjugate on complex test shapes
biases MAC low; (iv) zero columns → 0/0: raise, don't return NaN.

## 2. CoMAC — `comac(phi_a, phi_b) -> (n_dof,)`

Coordinate MAC over $L$ **paired** columns (pair first with `pair_modes`, pass aligned arrays):

$$\mathrm{CoMAC}_j =
\frac{\left( \sum_{l=1}^{L} \left| \phi_{a,jl}\, \overline{\phi_{b,jl}} \right| \right)^2}
     {\sum_{l=1}^{L} \left| \phi_{a,jl} \right|^2\ \sum_{l=1}^{L} \left| \phi_{b,jl} \right|^2 }
\in [0, 1]$$

Low CoMAC flags the DOFs responsible for poor global correlation (bad sensor, wrong
orientation/sign, local modeling error). Requires each pair scaled consistently first —
normalize columns (unit norm + phase alignment via MSF, the modal scale factor
$\mathrm{MSF}_l = \phi_{a,l}^{\mathsf H}\phi_{b,l} / \phi_{b,l}^{\mathsf H}\phi_{b,l}$, applied to
$\phi_b$). Pitfall: with $L = 1$, CoMAC is 1 everywhere except sign flips — needs several pairs
to be meaningful.

## 3. Orthogonality — `poc` / `cross_orthogonality`

Mass-weighted checks against the FE (reduced) mass matrix $M_r$ on the shared DOF set:

$$\mathrm{XOR} = \Phi_t^{\mathsf H} M_r\, \Phi_f \qquad(\text{cross-orthogonality, signed}),$$
$$\mathrm{POC}_{ij} = \frac{\left| \phi_{t,i}^{\mathsf H} M_r\, \phi_{f,j} \right|^2}
{\left( \phi_{t,i}^{\mathsf H} M_r \phi_{t,i} \right)\left( \phi_{f,j}^{\mathsf H} M_r \phi_{f,j} \right)}
\qquad(\text{normalized, MAC-like}).$$

With mass-normalized sets and a consistent reduction, $\mathrm{XOR} \approx I$; the classical
acceptance gate (NASA-STD-5002 style) is diag ≥ 0.9 (POC ≥ 0.95) and off-diag ≤ 0.10.

$M_r$ must come from a reduction to the test DOF set $a$:

- Guyan: $T = \begin{bmatrix} I \\ -K_{oo}^{-1} K_{oa} \end{bmatrix}$, $M_r = T^\top M T$ —
  static, degrades when omitted-DOF inertia matters (rule: valid below
  $\approx 0.5\,\sqrt{\lambda_1(K_{oo}, M_{oo})}/2\pi$);
- SEREP: $T = \Phi_n (\Phi_{n,a})^{+}$ over the target modes — reproduces exactly those modes,
  $M_r$ can be rank-deficient if $n_a < m$ or sensors are poorly placed
  ($\Phi_{n,a}$ ill-conditioned → pseudo-inverse noise blow-up; check
  $\mathrm{cond}(\Phi_{n,a})$ and warn > $10^3$).

Signature: `cross_orthogonality(phi_t, phi_f, M_r, normalized=False)`;
`poc(phi_t, phi_f, M_r)` returns the normalized form. Complexity $O(nnz(M_r) m + n m^2)$.
Pitfall: mixing row orderings between $M_r$ and the shape arrays is silent and catastrophic —
both functions require an explicit shared `dofs: list[(node_id, dof_code)]` argument or
pre-aligned arrays, never positional guessing.

## 4. FRF correlation — `frac`, `csac`, `csf`

Let $H_a, H_x$ be analytical/experimental FRF arrays `(n_out, n_in, n_f)` on a **shared**
frequency grid (interpolate first; complex linear interpolation on the real/imag parts).

FRAC — per response/reference pair, correlation over frequency:

$$\mathrm{FRAC}_{pq} =
\frac{\left| \sum_k H_{a,pq}(\omega_k)\, \overline{H_{x,pq}(\omega_k)} \right|^2}
     {\sum_k \left| H_{a,pq}(\omega_k) \right|^2\ \sum_k \left| H_{x,pq}(\omega_k) \right|^2}
\quad\to\ (n_{out}, n_{in}).$$

CSAC — per frequency line, correlation over space (stack outputs for each input, or all
out×in pairs into one vector $h(\omega_k)$):

$$\mathrm{CSAC}(\omega_k) =
\frac{\left| h_a(\omega_k)^{\mathsf H} h_x(\omega_k) \right|^2}
     {\left( h_a^{\mathsf H} h_a \right)\left( h_x^{\mathsf H} h_x \right)}\quad\to\ (n_f,).$$

CSF (cross signature scale factor) — amplitude agreement per line, ≤ CSAC-style bound:

$$\mathrm{CSF}(\omega_k) =
\frac{2 \left| h_a(\omega_k)^{\mathsf H} h_x(\omega_k) \right|}
     {h_a^{\mathsf H} h_a + h_x^{\mathsf H} h_x}\quad\to\ (n_f,).$$

CSAC measures shape correlation (insensitive to global scaling), CSF punishes level mismatch;
both = 1 ∀ω iff the FRF sets are identical up to nothing at all.

Pitfalls: FRF correlation collapses under small **frequency shifts** — a 1 % resonance shift
sends CSAC to ~0 near that peak even for a perfect shape match. Standard practice: report
CSAC/CSF as diagnostics, drive updating with eigenfrequency + MAC residuals, or pre-align peaks.
Damping mismatch dominates amplitude near resonance → CSF dips there first. Never correlate on
dB values inside these formulas (they are defined on complex FRFs); log-magnitude comparisons
are a separate plot-level tool.

## 5. Mode pairing — `pair_modes` (incl. double modes)

```python
# implemented API (femtools.correlation.pairing)
pair_modes(phi_a, phi_b, freq_a=None, freq_b=None, *,
           method="greedy" | "hungarian" | "auto",
           mac_threshold=0.0, freq_tol=None, weights=None)
    -> PairingResult    # iterable of ModePair(index_a, index_b, mac, freq_a, freq_b)
```

`method="hungarian"` solves the rectangular linear assignment maximizing total MAC
(`scipy`-style Jonker–Volgenant, $O(\max(m_a, m_b)^3)$); `"auto"` uses it up to 512 modes
and falls back to greedy above. Gates: pairs with $\mathrm{MAC} < \texttt{mac\_threshold}$
are rejected, and `freq_tol` forbids candidates with $|\Delta f| / f_b$ beyond the window.
Prefer the optimal assignment when modes are close in shape: greedy max-MAC pairing
deadlocks on crossed modes (classic switch of modes 2/3 after updating steps).

**Double / repeated modes.** For (near-)degenerate pairs (symmetric structures: cylinders,
plates, discs), eigenvectors are only defined up to an arbitrary rotation inside the 2-D
subspace: pairwise MAC between two perfect models can approach 0. Handling:

1. Cluster columns whose frequencies coincide within `cluster_tol_rel` (default $10^{-3}$).
2. Between an atom $\phi$ and a cluster subspace $\Psi = [\psi_1 \ldots \psi_c]$, score with the
   subspace MAC (S2MAC), i.e. the squared cosine of the first principal angle:

$$\mathrm{S2MAC}(\phi, \Psi) =
\frac{\phi^{\mathsf H}\, \Psi \left( \Psi^{\mathsf H} \Psi \right)^{-1} \Psi^{\mathsf H}\, \phi}
     {\phi^{\mathsf H} \phi} \in [0, 1].$$

3. Cluster-to-cluster: match via orthogonal Procrustes — SVD of
   $Q_a^{\mathsf H} Q_b = U \Sigma V^{\mathsf H}$ (orthonormalized cluster bases);
   $\sigma_i = \cos\theta_i$ principal angles; report
   $\mathrm{MAC}_{sub} = \prod \sigma_i^2$ (or min $\sigma_i^2$) and the rotated pairing
   $Q_a U \leftrightarrow Q_b V$ so downstream per-mode comparisons see aligned vectors.

Pitfalls: near-degenerate but not exactly equal frequencies (broken symmetry) may or may not
cluster — expose `cluster_tol_rel`; assignment can pair everything even when half the modes are
missing from the test set (always post-filter); complex test shapes should be
phase-normalized (rotate each column so its largest component is real-positive) before any
report meant for humans.

## 6. Expansion and expanded MAC — `expand_serep` / `expanded_mac` (Round 10, owner R10-O3)

Shape expansion moves a mode set measured on a sensor subset back onto the full FE DOF
set so correlation runs where the model lives, not where the sensors happened to be.
The Round-4 operators (`expand_guyan`, `expand_serep`, both →
`ExpansionResult`) implement the two classical transformations — Guyan's static
recovery (fea.md §9.1) and SEREP (O'Callahan, Avitabile & Riemer, *System Equivalent
Reduction Expansion Process*, Proc. 7th IMAC, 1989; fea.md §9.3):

$$\psi_{full} = \Phi\, \Phi_m^{+}\, \psi_m, \qquad \Phi_m = \Phi[\text{master rows}, :],$$

i.e. fit the measured shape in least squares by the FE basis restricted to the
measured DOFs, then evaluate the fit everywhere. `ExpansionResult.residual` reports
the per-mode misfit at the masters — the part of the measurement the truncated basis
cannot represent; keep the basis well *smaller* than the sensor count, or the fit is
square, the residual is zero by construction and the noise gets expanded along with
the shape.

The Round-10 composition (landed on this tree, `tests/test_round10_o3.py` green):

```python
expanded_mac(phi_test, modes, master, *, reference=None, weights=None,
             n_modes=None, rcond=1e-12, ...) -> ExpandedMACResult
# expand_serep + mac_matrix in one call: MAC of the expanded test shapes
# against the full-DOF FE modes (or an explicit `reference` set).
# Unpacks as (mac, expansion); np.asarray(result) is the MAC table.
```

Why compose at all: a MAC on the measured DOFs alone saturates under spatial aliasing
(§1 pitfall ii) — with a few dozen sensors, genuinely different shapes look collinear.
Expanding first moves the comparison onto the complete model. The result carries the
MAC table, the underlying `ExpansionResult` (so `residual` stays readable alongside
the correlation), and the diagnostics `diagonal_error`, `max_off_diagonal`,
`identity_error`.

**The identity fixed point** (ACCEPTANCE case 32) is what makes the composition
trustworthy: feed the FE modes *restricted to the master rows* back in —
`expanded_mac(fe_modes[master], fe_modes, master)` — and, for a full-column-rank
$\Phi_m$, SEREP reproduces them exactly ($\Phi \Phi_m^{+} \Phi_m = \Phi$), so the MAC
diagonal is 1 to round-off: `diagonal_error` ≤ 1e-10, SEREP `residual` = 0. Two
distinct claims hide in "the table is the identity", and the result type separates
them deliberately:

- `diagonal_error` — the SEREP self-check, reference-independent. A departure is a
  defect of the master set (too few sensors, rank-deficient $\Phi_m$, basis truncated
  below the modes being fitted), never a correlation result.
- `identity_error` — additionally requires the reference modes to be mutually
  uncorrelated *under the MAC weighting*. True for an orthonormal basis and for
  mass-normalized FE modes with `weights=M` (the mass-weighted MAC); the **unweighted**
  table instead collapses onto the plain AutoMAC of the FE modes, whose off-diagonal
  belongs to the mode set, not to the expansion (§1 pitfall i — MAC is not an
  orthogonality check). Measured on the case-2 cantilever (6 modes, 18 random
  masters): `diagonal_error` 2.2e-16, `weights=M` `identity_error` 4.4e-16,
  unweighted off-diagonal 0.240 = the AutoMAC.

What the fixed point does *not* prove: SEREP can only produce shapes inside
$\mathrm{span}(\Phi)$, so expanded *test* shapes are filtered onto the basis and their
MAC against the FE modes is optimistic by construction — read `residual` alongside the
table before believing a diagonal. `expand_serep` / `expand_guyan` numerics are
contractually unchanged by Round 10 (the composition adds no formula); `mac_matrix`
stays §1's.

## 7. Complexity summary

| Kernel | Cost |
|---|---|
| `mac_matrix` | $O(n\, m_a m_b)$ |
| `comac` | $O(n\, L)$ |
| `poc` / `cross_orthogonality` | $O(nnz(M_r)\, m + n\, m^2)$ (+reduction cost upstream) |
| `frac` / `csac` / `csf` | $O(n_{out} n_{in} n_f)$ |
| `pair_modes` | MAC + $O(m^3)$ assignment (+SVDs of cluster size, negligible) |
| `expand_serep` / `expanded_mac` | pinv $O(n_m m^2)$ + $O(n\, m\, n_m)$ + MAC |

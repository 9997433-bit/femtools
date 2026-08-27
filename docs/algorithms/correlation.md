# Correlation algorithms — MAC family, orthogonality, FRF correlation, pairing

Spec for `femtools.correlation` (owner: R1-O3). Frozen entry points:

```python
from femtools.correlation.mac import mac_matrix, comac, poc
from femtools.correlation.pairing import pair_modes
from femtools.correlation.frf_corr import frac, csac, csf
from femtools.correlation.orthogonality import cross_orthogonality
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
pair_modes(freq_a, phi_a, freq_b, phi_b, mac_min=0.5, w_freq=0.0, freq_tol_rel=None)
    -> list[ModePair]   # ModePair(i_a, i_b, mac, dfreq_rel)
```

Cost matrix combining shape and frequency proximity:

$$C_{ij} = w_f\, \frac{|f_{a,i} - f_{b,j}|}{\max(f_{a,i}, \epsilon)} + (1 - w_f)\,(1 - \mathrm{MAC}_{ij}),$$

solved as a rectangular linear assignment (`scipy.optimize.linear_sum_assignment`,
Jonker–Volgenant, $O(\max(m_a, m_b)^3)$). Post-filter: drop pairs with
$\mathrm{MAC} < \texttt{mac\_min}$ or $|\Delta f|/f > \texttt{freq\_tol\_rel}$ (if given).
Greedy max-MAC pairing is *not* acceptable: it deadlocks on crossed modes (classic switch of
modes 2/3 after updating steps).

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

## 6. Complexity summary

| Kernel | Cost |
|---|---|
| `mac_matrix` | $O(n\, m_a m_b)$ |
| `comac` | $O(n\, L)$ |
| `poc` / `cross_orthogonality` | $O(nnz(M_r)\, m + n\, m^2)$ (+reduction cost upstream) |
| `frac` / `csac` / `csf` | $O(n_{out} n_{in} n_f)$ |
| `pair_modes` | MAC + $O(m^3)$ assignment (+SVDs of cluster size, negligible) |

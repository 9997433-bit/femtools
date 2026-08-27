# MPE / RBPE algorithms — p-LSCF, FDD/EFDD, LSCE, rigid-body properties

Spec for `femtools.mpe` and `femtools.rbpe` (owner: R1-O4). Frozen entry points:

```python
from femtools.mpe.p_lscf import poly_lscf
from femtools.mpe.fdd import fdd, efdd
from femtools.mpe.lsce import lsce
from femtools.rbpe.rbfit import rigid_body_properties
# Round-4 additions (REMAINING.md, owner R4-O4) — see §5–§6:
from femtools.mpe.frf_estimation import estimate_h1, estimate_h2, coherence
from femtools.mpe.ssi import ssi_cov
# Round-6 addition (REMAINING.md, owner R6-O4) — see §7:
from femtools.mpe.ssi import ssi_data
```

Common conventions: FRFs `H (n_out, n_in, n_f)` on `freq_hz`; continuous-time poles
$\lambda_r = -\zeta_r \omega_r + i \omega_r \sqrt{1 - \zeta_r^2}$; reported
$f_r = |\lambda_r| / 2\pi$, $\zeta_r = -\mathrm{Re}\,\lambda_r / |\lambda_r|$. Physical poles
come in conjugate pairs with $\mathrm{Re}\,\lambda_r < 0$; everything else on a stabilization
diagram is mathematical garbage to be filtered, not fixed.

## 1. p-LSCF (PolyMAX-class) — `poly_lscf`

Poly-reference least-squares complex frequency-domain estimator on a right matrix-fraction
description (RMFD). Per output row $o$ ($1 \times n_i$):

$$H_o(\omega_k) \approx B_o(\Omega_k)\, A(\Omega_k)^{-1},\qquad
B_o(\Omega) = \sum_{j=0}^{n} \Omega^j \beta_{oj},\quad
A(\Omega) = \sum_{j=0}^{n} \Omega^j \alpha_j,$$

with $\beta_{oj} \in \mathbb{R}^{1 \times n_i}$, $\alpha_j \in \mathbb{R}^{n_i \times n_i}$, and
the **discrete-time basis** $\Omega_k = e^{-i \omega_k \Delta t}$, $\Delta t = 1 / (2 f_{band}^{hi})$
after shifting the band to start at 0. This basis is the whole point: powers of $i\omega$
(continuous basis) produce Vandermonde conditioning that explodes past order ~10, while
$|\Omega_k| = 1$ keeps the normal equations tame at orders 50+, and real-valued coefficient
blocks keep them real.

Linearized weighted LS (errors-in-equation): minimize
$\sum_{o,k} \lVert W_o(\omega_k) \left( \beta_o(\Omega_k) - H_o(\omega_k)\, \alpha(\Omega_k) \right) \rVert_F^2$.
Per output, normal-equation blocks over the basis vector
$\gamma_k = [1, \Omega_k, \ldots, \Omega_k^n]$:

$$R_o = \mathrm{Re}\sum_k \Gamma_k^{\mathsf H} \Gamma_k,\quad
S_o = \mathrm{Re}\sum_k \Gamma_k^{\mathsf H} \Upsilon_{ok},\quad
T_o = \mathrm{Re}\sum_k \Upsilon_{ok}^{\mathsf H} \Upsilon_{ok}$$

($\Gamma_k = \gamma_k \otimes I$ acting on $\beta_o$, $\Upsilon_{ok} = \gamma_k \otimes
(-H_o(\omega_k))$ acting on the stacked $\alpha$). Eliminate every $\beta_o$
($\beta_o = -R_o^{-1} S_o \alpha$), accumulate the reduced normal matrix

$$M = 2 \sum_{o=1}^{n_o} \left( T_o - S_o^{\mathsf H} R_o^{-1} S_o \right)
\in \mathbb{R}^{(n+1) n_i \times (n+1) n_i},$$

impose the parameter constraint $\alpha_n = I_{n_i}$ (highest order), and solve the remaining
blocks in LS: $\alpha_{0:n-1} = -M_{[0:n, 0:n]}^{-1} M_{[0:n, n]}$. The
highest-order constraint is deliberate: it biases mathematical poles to the **stable** side,
which is what makes PolyMAX-class stabilization diagrams famously clean; constraining
$\alpha_0 = I$ scatters them across the unit circle (document both, default `constraint="high"`).

Poles and participations: companion matrix of $A(z)$
($n \cdot n_i$ square, block-companion of $\alpha_n^{-1}\alpha_j$), eigenvalues $z_r$ →
$\lambda_r = \ln(z_r) / \Delta t$ (+ band-shift restore); modal participation factors
$l_r^\top$ = last $n_i$ rows of the corresponding eigenvectors. Discard $|z_r|$ ≈ 0/∞ and
positive-real-axis artifacts.

Mode shapes — LSFD second stage: with poles and participations fixed, fit residues linearly
over the band with lower/upper residual terms:

$$H(\omega) \approx \sum_r \left( \frac{\phi_r l_r^\top}{i\omega - \lambda_r}
+ \frac{\bar\phi_r \bar l_r^\top}{i\omega - \bar\lambda_r} \right)
- \frac{LR}{\omega^2} + UR,$$

one complex LS per output row, $O(n_o n_f (2m + 2 n_i)^2)$.

Stabilization diagram (`orders=range(2, n_max)`): a pole at order $n$ is *stable* if a pole at
$n-1$ matches within $\Delta f / f < 1\%$, $\Delta\zeta / \zeta < 5\%$, participation MAC
> 0.95. Return the full diagram (`StabDiagram`: poles per order + flags) plus auto-selected
stable columns; never auto-select silently without exposing the diagram.

```python
poly_lscf(H, freq_hz, order_max, band=None, weights=None) -> PlscfResult
# PlscfResult: poles, participation, shapes, lower_upper_residuals, stab (diagram)
```

Complexity: normal equations $O(n_o n_f (n\, n_i)^2)$ dominant; reduction and solve
$O(n_o (n n_i)^3)$; companion eig $O((n n_i)^3)$ per order (reuse accumulated $M$ blocks for
nested orders — the $R, S, T$ sums are order-independent Toeplitz-structured; exploit or at
least cache). Pitfalls: weighting — use coherence-based $W_o$ or $1/\mathrm{var}$ if available,
uniform otherwise (amplitude weighting $1/|H|$ over-fits antiresonances); band edges leak —
always fit with LR/UR residuals; low damping + short $\Delta t$ mismatch: verify
$f_{band}^{hi} < 1/(2\Delta t)$ (Shannon on the mapped band).

## 2. FDD / EFDD — output-only (OMA)

`fdd(y=None, fs=None, psd=None, freq_hz=None, nperseg=4096, overlap=0.5) -> FddResult`

Estimate the output PSD matrix $G_{yy}(\omega_k)$ ($n_o \times n_o$, Hermitian PSD) by
Welch/periodogram averaging (Hann window, 50 % overlap, one-sided). At each line take the SVD

$$G_{yy}(\omega_k) = U_k S_k U_k^{\mathsf H};$$

under broadband excitation and light damping, near a mode $s_1(\omega)$ peaks and
$u_1(\omega_{peak}) \approx \phi_r$ (unscaled). Peak picking on $s_1$ (prominence-based,
`scipy.signal.find_peaks`) gives $f_r$ and shapes; close modes surface in $s_2$ — return all
singular value tracks, not just the first.

EFDD (damping + refined frequency): around each peak, collect the *SDOF bell* — lines where
$\mathrm{MAC}(u_1(\omega), u_1(\omega_{peak})) > 0.8$; take $s_1$ over the bell as the SDOF
auto-PSD, inverse-FFT to the modal autocorrelation $r(\tau)$, then:
$\zeta_r$ from the log decrement of successive extrema
($\delta = \ln(r_i / r_{i+2\pi})$ linear fit → $\zeta = \delta / \sqrt{4\pi^2 + \delta^2}$),
$f_d$ from a linear fit of zero-crossing times, $f_r = f_d / \sqrt{1 - \zeta^2}$. Fit only the
correlation segment between ~0.95·max and noise floor (~10 % tail), excluding lag 0.

Pitfalls: **frequency resolution** — bias in $\zeta$ from leakage is the dominant EFDD error;
require $\ge 10$ lines across the half-power bandwidth
($n_{perseg} \gtrsim 10 f_s / (2 \zeta f_r)$), warn otherwise; harmonics (rotating machinery)
masquerade as ultra-low-damping modes — flag candidate lines whose bell kurtosis is
non-Gaussian; shapes are unscaled (no mass normalization without extra info — mass-change or
known-input methods out of Round-1 scope); deterministic tests must fix the RNG of the
synthetic excitation *and* the Welch segmentation.

Complexity: Welch $O(n_o^2 n_t \log n_{perseg})$; SVDs $O(n_f n_o^3)$.

## 3. LSCE — `lsce`

Least-squares complex exponential (time domain, poly-reference Prony). Inputs: impulse
responses `h (n_out, n_in, n_t)` (from inverse FFT of FRFs — apply exponential window
$e^{-a t}$ to suppress wraparound leakage, then correct damping: $\zeta_{true}\omega_r =
\zeta_{est}\omega_r - a$). Model: $h_{pq}(j \Delta t) = \sum_{r=1}^{2m} A_{pq r} z_r^j$,
$z_r = e^{\lambda_r \Delta t}$. The $z_r$ are common roots of the AR polynomial
$\sum_{i=0}^{2m} \beta_i z^i = 0$, $\beta_{2m} = 1$, estimated globally by stacking Hankel rows
over **all** response/reference pairs:

$$\sum_{i=0}^{2m-1} \beta_i\, h_{pq}(j + i) = -h_{pq}(j + 2m)
\quad \forall\, p, q,\ j = 0 \ldots n_t - 2m - 1,$$

one LS solve $O(N_{rows} (2m)^2)$; roots via companion matrix $O((2m)^3)$; then residues by
linear LS (LSFD as in §1, or time-domain Vandermonde). Stabilization diagram over $m$ exactly
as §1. Pitfalls: heavily biased at low signal-to-noise (Prony's classic weakness — overspecify
the order 2–3× and rely on stabilization); $\Delta t$ aliasing folds out-of-band modes in
(band-pass + decimate first); the LS matrix is Hankel-structured and ill-conditioned for long
records — normalize IRF amplitude and prefer SVD-based solve (`lstsq`, rcond pinned).

## 4. RBPE — `rigid_body_properties`

Estimate the 10 rigid-body parameters — mass $m$, CoG $c$, inertia tensor $J$ (6) — from
measured FRFs in the **mass-line band**: well above the suspension modes
($\ge \sqrt{10} \times f_{susp}$) and well below the first elastic mode
($\le 0.5 \times f_{el,1}$), where the structure responds as a rigid body on soft springs and
accelerance is flat.

```python
rigid_body_properties(frf, freq_hz, sensors, excitations, band_hz,
                      ref_point=(0,0,0), weights=None) -> RbpeResult
# sensors:    [(position xyz, direction unit vector), ...]  per output row
# excitations:[(position xyz, direction unit vector), ...]  per input column
# RbpeResult: mass, cog, inertia_cog (3x3), inertia_ref, residual, per_line_cond
```

Kinematics: rigid acceleration state $a = (a_0, \alpha) \in \mathbb{C}^6$ at reference $x_0$
gives sensor accelerations $a_p = e_p^\top (a_0 + \alpha \times r_p)$, i.e. rows
$e_p^\top [\, I_3\ \ -\tilde r_p \,]$ stacked into $E \in \mathbb{R}^{n_o \times 6}$
($r_p$ = sensor position − $x_0$). Per frequency line and excitation $q$ (unit force
$e_q$ at $r_q$, so generalized load $g_q = (e_q,\ r_q \times e_q)$, force-scaled by the FRF
definition):

1. Recover the rigid motion state by LS: $\hat a_q(\omega_k) = E^{+}\, \omega_k^2\, H_{:,q}(\omega_k) \cdot (-1)$
   (receptance → accelerance sign convention); requires $n_o \ge 6$ with
   $\mathrm{cond}(E) \lesssim 10^2$ — sensors must span 3D, non-coplanar, mixed directions.
2. Newton–Euler about $x_0$ is **linear in the 10 unknowns**
   $\theta = (m,\ m c,\ J_{xx}, J_{yy}, J_{zz}, J_{xy}, J_{xz}, J_{yz})$:

$$\begin{bmatrix} m I_3 & -m \tilde c \\ m \tilde c & J_{x_0} \end{bmatrix}
\begin{bmatrix} a_0 \\ \alpha \end{bmatrix} = g_q
\quad\Longrightarrow\quad
A(\hat a_q(\omega_k))\, \theta = g_q,$$

   stack $A$ over all lines in the band × all excitations (real and imaginary parts
   separately), solve one overdetermined real LS. In the ideal band $\hat a$ is
   frequency-flat, so lines act as repeated measurements — weight by coherence or $1/\sigma^2$
   when available.
3. Post-process: $c = (m c) / m$; translate inertia to the CoG (Huygens–Steiner):

$$J_G = J_{x_0} - m \left( |c|^2 I_3 - c\, c^\top \right).$$

   Validity checks to return: $J_G$ symmetric positive definite, triangle inequalities on
   principal inertias ($I_1 + I_2 \ge I_3$ etc.), residual per excitation.

Golden case: noiseless synthetic FRFs generated from a known $(m, c, J)$ 6-DOF rigid model on
soft springs recover all 10 parameters to $10^{-8}$ relative inside the band
(`docs/ACCEPTANCE.md` §8).

Pitfalls: band selection is the #1 error source — automate a flatness scan (pick the widest
band where $|\omega^2 H|$ slope < few %/octave) but always report the band used; geometry errors
(sensor positions/directions) enter the estimate linearly and don't average out — sanity-check
$E$ against the measured rigid shapes; suspension-mode tails bias $m$ low and elastic-mode
tails bias $J$ high (symptom: parameters drift when the band edges move — report the drift);
force direction errors at the exciter corrupt $g_q$ — prefer multiple excitation locations.

## 5. H1 / H2 / coherence — `femtools.mpe.frf_estimation` (Round 4, owner R4-O4)

Welch-averaged FRF estimators from measured force and response records. Data layout
follows `fdd.cross_spectral_density`: records are `(n_channels, n_samples)` (tall arrays
auto-transposed), `x` may be 1-D for a single input. Cross-spectrum convention is
scipy's `csd(x, y)` $= E[X^* Y]$.

```python
estimate_h1(x, y, fs, *, nperseg=1024, noverlap=None, window="hann")
    -> (freq_hz, H)        # H (n_out, n_in, n_f) complex
estimate_h2(x, y, fs, *, ...) -> (freq_hz, H)
coherence(x, y, fs, *, ...)   -> (freq_hz, gamma2)   # gamma2 (n_out, n_in, n_f) real
```

Tuple returns match `cross_spectral_density`; a small `FrfEstimate` dataclass with
`.freq_hz` / `.H` is also acceptable — `examples/h1_ssi.py` tolerates both, but pick one
and keep it. With auto/cross spectra $G_{xx}, G_{yy}, G_{xy}$ from **one shared
segmentation** (same `nperseg`, overlap, window, detrend):

$$H_1 = \frac{G_{xy}}{G_{xx}},\qquad
  H_2 = \frac{G_{yy}}{G_{yx}} = \frac{G_{yy}}{\overline{G_{xy}}},\qquad
  \gamma^2 = \frac{|G_{xy}|^2}{G_{xx} G_{yy}} = \frac{|H_1|}{|H_2|} \le 1 .$$

$H_1$ is unbiased under output noise (underestimates at resonances under *input* noise),
$H_2$ under input noise (overestimates at antiresonances under output noise); the
identities $|H_1| \le |H_2|$ and $\gamma^2 = |H_1|/|H_2|$ are line-wise mathematical
facts given shared spectra and are the acceptance pins (case 19; measured dev 7.8e-16 —
a violation means the three estimators segment differently, which is a bug, not noise).
True MIMO ($n_{in} > 1$ simultaneous, partially correlated inputs) needs the full input
CSD matrix inverse per line, $H_1 = G_{yx} G_{xx}^{-1}$ — guard $\mathrm{cond}(G_{xx})$
and defer if not needed; the Round-4 example is single-input.

Pitfalls: **leakage** biases $|H_1|$ low at light-damped resonances — require ≥ 10 lines
across the half-power bandwidth ($n_{perseg} \gtrsim 10 f_s / (2\zeta f_r)$, same rule as
EFDD §2); coherence drops from noise, nonlinearity *and* leakage alike — don't read it as
a pure SNR meter near peaks; Welch `scaling` cancels in all three ratios but keep
`"density"` for consistency; sampled-data validation — an estimator fed ZOH-sampled
records recovers the *discrete-time* FRF, which differs from the continuous model by a
half-sample delay and sinc rolloff (measured on the 3-DOF chain at $f_s = 64$ Hz: 4.2 %
median complex error vs the ZOH-exact FRF, but 32 % vs the continuous receptance —
compare magnitudes or the ZOH-exact reference, as `examples/h1_ssi.py` does).
`tests/test_round4_mpe.py` (R4-G1) pins the noise-free case: both estimators recover a
known discrete IIR filter's `freqz` response to 2 % with coherence > 0.995.

## 6. SSI-cov — `femtools.mpe.ssi` (Round 4, owner R4-O4)

Covariance-driven stochastic subspace identification, the output-only complement of
p-LSCF/LSCE. Under unmeasured broadband (white) excitation the output correlations of an
LTI system factor like Markov parameters:

$$R_k = E\big[y_{t+k}\, y_t^\top\big] = C A^{k-1} G,$$

so the block-Hankel matrix of correlations factors into observability × stochastic
controllability:

$$\mathcal H = \begin{bmatrix} R_1 & R_2 & \cdots & R_i\\ R_2 & R_3 & \cdots & R_{i+1}\\
\vdots & & & \vdots\\ R_i & R_{i+1} & \cdots & R_{2i-1}\end{bmatrix}
= \mathcal O_i\, \mathcal C_i .$$

Algorithm: unbiased correlation estimates $\hat R_k = \frac{1}{n-k}\sum_t y_{t+k} y_t^\top$
for $k = 1..2i$; SVD $\mathcal H = U S V^\top$, keep $n_s = 2 \cdot \texttt{order}$
singular values, $\mathcal O = U_1 S_1^{1/2}$; then $C = \mathcal O[:l,:]$ ($l$ =
channels) and the shift invariance $A = \mathcal O[:-l,:]^{+}\, \mathcal O[l:,:]$.
Eigendecomposition $A \Psi = \Psi \mathrm{diag}(z_r)$ gives poles
$\lambda_r = f_s \ln z_r$ and (unscaled) shapes $\phi_r = C \psi_r$. Filter to physical
poles exactly as §1/§3: conjugate pairs, $\mathrm{Re}\,\lambda < 0$, $\zeta$ window;
run a stabilization sweep over orders when requested.

```python
ssi_cov(data, fs, *, order=None, n_modes=None, block_rows=None, orders=None,
        f_range=None, stabilization=True, weighting="none", ref_channels=None,
        ...) -> ModalParameterResult
# data (n_ch, n_samples); order = STATE dimension (2x the number of mode pairs;
# NOTE: differs from lsce, whose order counts pole pairs), default 2*n_modes + 10;
# with stabilization=True (default) the identification sweeps orders 2..order and
# keeps the poles that stabilize, so give order 2-3x headroom over the physical
# state count; n_modes trims to the most persistent/dominant modes and f_range
# gates the acceptance band; block_rows i defaults to max(10, ceil(2*order/n_ref))
# and must satisfy i*n_ref >= order; i/fs should span ~half a period of the lowest
# mode.
```

Returns the shared `mpe.common.ModalParameterResult` (ascending `freq_hz`, `damping`,
`mode_shapes (n_ch, n_modes)`). Measured against the merged kernel (3-mode chain, 600 s
at 64 Hz, 2 % output noise, `order=20, n_modes=3, f_range=(1, 12)` in
`examples/h1_ssi.py`): frequency errors ≤ 0.10 %, damping errors ≤ 11 %, MAC vs truth
≥ 0.9998, spurious poles rejected by the stabilization sweep + ζ-window +
nearest-frequency match (acceptance case 20 allows 2 % / 50 %). The bare-minimum
`order=6` is a trap: it leaves the sweep only orders {2, 4, 6} and the clustering
returns a single stabilized pole.

Pitfalls: shapes are **unscaled** (no mass normalization without known inputs — same
caveat as FDD §2); overspecify the order 2–3× and rely on stabilization, exactly like
Prony/LSCE; biased (divide-by-$n$) correlation estimates shift damping — use unbiased
$1/(n-k)$; harmonics masquerade as ζ ≈ 0 poles (flag, don't fit); damping estimates need
record lengths of ≳ 500 cycles of the lowest mode for ~10 % scatter; seeded
`synthetic_response` is the deterministic test source.

## 7. SSI-DATA — `femtools.mpe.ssi.ssi_data` (Round 6, owner R6-O4)

Data-driven stochastic subspace identification (N4SID-class): Van Overschee & De Moor,
*Subspace Identification for Linear Systems: Theory — Implementation — Applications*
(Kluwer, 1996), ch. 3 (stochastic case); reference-based variant Peeters & De Roeck,
"Reference-based stochastic subspace identification for output-only modal analysis",
*MSSP* 13(6), 1999. Same problem as §6 — output-only poles/shapes under unmeasured broadband
excitation — but the Hankel compression step differs: instead of estimating correlations
$\hat R_k$ first, project the raw data directly.

Build the output block-Hankel with $2i$ block rows and $j = n_t - 2i + 1$ columns
(scaled $1/\sqrt j$), split into past and future halves:

$$Y_{0|2i-1} = \frac{1}{\sqrt j}
\begin{bmatrix} y_0 & y_1 & \cdots & y_{j-1}\\
\vdots & & & \vdots\\
y_{2i-1} & y_{2i} & \cdots & y_{2i+j-2}\end{bmatrix}
= \begin{bmatrix} Y_p \\ Y_f \end{bmatrix},$$

then the orthogonal projection of the future row space onto the past row space

$$\mathcal P_i = Y_f / Y_p = Y_f Y_p^\top \left( Y_p Y_p^\top \right)^{+} Y_p .$$

The main stochastic identification theorem (Van Overschee & De Moor, ch. 3) states
$\mathcal P_i = \mathcal O_i \hat X_i$: the projection factors into the extended
observability matrix and the forward Kalman-filter state sequence — the data-driven
counterpart of §6's $\mathcal H = \mathcal O_i\, \mathcal C_i$. Numerics: **never form
$Y Y^\top$**. Compute one LQ factorization of the stacked Hankel,
$[Y_p; Y_f] = L Q^\top$ with orthonormal $Q$; then $\mathcal P_i = L_{21} Q_1^\top$, and
since the column space is unchanged by the orthonormal right factor, the SVD can act on the
small triangular block $L_{21}$ ($(il) \times (il)$) directly. Weighted SVD
$W_1 \mathcal P_i W_2 = U S V^\top$; weighting variants UPC ($W_1 = W_2 = I$ — the Round-6
default), PC, and CVA ($W_1 = (Y_f Y_f^\top)^{-1/2}$, better separation of close/weak modes
at extra cost). Keep `order` singular values, $\mathcal O_i = W_1^{-1} U_1 S_1^{1/2}$, and
from there the path is **identical to §6** and shared in code: $C = \mathcal O[:l,:]$, shift
invariance $A = \mathcal O[:-l,:]^{+} \mathcal O[l:,:]$, poles $\lambda_r = f_s \ln z_r$,
unscaled shapes $\phi_r = C \psi_r$, conjugate-pair/$\zeta$-window filtering and the
stabilization sweep.

```python
ssi_data(data, fs, *, order=None, n_modes=None, block_rows=None, orders=None,
         f_range=None, stabilization=True, ref_channels=None, ...)
    -> ModalParameterResult          # same result type and defaults as ssi_cov (§6)
```

cov vs data, when to use which: `ssi_cov` compresses the record to $2i$ correlation
matrices first — cheapest, and fine for long clean records; `ssi_data` works on the raw
data through one LQ, is numerically better conditioned (never squares the data, works on
triangular factors) and statistically somewhat more efficient on short/noisy records (the
projection is a conditional-mean, i.e. Kalman, estimate rather than a truncated correlation
sequence). Both return unscaled shapes and share every downstream convention, so results on
the same record must agree to well within the acceptance gates (`docs/ACCEPTANCE.md` §10,
case 23). Reference channels (Peeters–De Roeck): replace $Y_p$ by the reference-row
sub-Hankel — cost drops from $O(j(2il)^2)$ toward $O(j(i(l+r))^2)$ and noisy channels stop
polluting the conditioning; `ref_channels=` mirrors §6.

Pitfalls: all of §6's (unscaled shapes; overspecify `order` 2–3× and let stabilization
choose; harmonics masquerade as $\zeta \approx 0$ poles; ≳500 cycles of the lowest mode for
~10 % damping scatter; `block_rows` must satisfy $i\,l \ge$ order with $i/f_s$ spanning
about half a period of the lowest mode) plus the data-driven specials: de-mean/detrend every
channel first (a DC offset is a unit-circle pole that eats one order and biases the lowest
mode); keep the $1/\sqrt j$ normalization consistent between the projection and any
singular-value-based order diagnostics (poles are invariant to it, singular values are
not); never materialize $Q$ ($ (2il) \times j$ — economy-mode `scipy.linalg.qr` on the
transposed data returns only the triangular factor needed); memory scales with $j \cdot il$
for the Hankel itself — build it as a strided view, not a copy, for long records.

## 8. Complexity summary

| Kernel | Cost |
|---|---|
| `poly_lscf` normal eqs | $O(n_o n_f (n\, n_i)^2)$; solve/companion $O((n\, n_i)^3)$ |
| `fdd` | Welch $O(n_o^2 n_t \log n_{seg})$ + $O(n_f n_o^3)$ SVDs |
| `efdd` | + per-peak IFFT and linear fits, negligible |
| `lsce` | LS $O(n_o n_i n_t (2m)^2)$ + roots $O((2m)^3)$ |
| `estimate_h1/h2`, `coherence` | Welch $O(n_o n_i n_t \log n_{seg})$ |
| `ssi_cov` | correlations $O(n_o^2\, i\, n_t)$ + SVD $O((i\, n_o)^3)$ |
| `ssi_data` | LQ $O(j\,(2 i n_o)^2)$ + SVD $O((i\, n_o)^3)$ — LQ dominates |
| `rigid_body_properties` | $O(n_f n_i (n_o \cdot 6 + 6 \cdot 10))$ LS stacks |

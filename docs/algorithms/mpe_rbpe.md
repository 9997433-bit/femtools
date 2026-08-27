# MPE / RBPE algorithms — p-LSCF, FDD/EFDD, LSCE, rigid-body properties

Spec for `femtools.mpe` and `femtools.rbpe` (owner: R1-O4). Frozen entry points:

```python
from femtools.mpe.p_lscf import poly_lscf
from femtools.mpe.fdd import fdd, efdd
from femtools.mpe.lsce import lsce
from femtools.rbpe.rbfit import rigid_body_properties
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

## 5. Complexity summary

| Kernel | Cost |
|---|---|
| `poly_lscf` normal eqs | $O(n_o n_f (n\, n_i)^2)$; solve/companion $O((n\, n_i)^3)$ |
| `fdd` | Welch $O(n_o^2 n_t \log n_{seg})$ + $O(n_f n_o^3)$ SVDs |
| `efdd` | + per-peak IFFT and linear fits, negligible |
| `lsce` | LS $O(n_o n_i n_t (2m)^2)$ + roots $O((2m)^3)$ |
| `rigid_body_properties` | $O(n_f n_i (n_o \cdot 6 + 6 \cdot 10))$ LS stacks |

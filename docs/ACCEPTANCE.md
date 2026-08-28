# Acceptance — golden analytical cases and tolerances

Expands the tolerance table in `docs/CONTRACT_API.md` with the exact formulas each golden test
checks against. Tests live in `tests/` (owner R1-G1/R2-G1); recommended test ids are listed so
failures map back here. Conventions: $c = \sqrt{E/\rho}$ (bar wave speed), $f = \omega/2\pi$,
all modes mass-normalized ($\Phi^\top M \Phi = I$).

## Measured status (merged tree, 2026-08-27, Round-5 re-run)

Verified by running all eight `examples/*.py` (8/8 PASS) and `pytest tests/` (65 passed,
3 perf skips) against the merged `cursor/femtools-remaining-d551` tree with the real
Round-4 kernels landed. The three Round-4 examples (`guyan_serep.py`, `cms_rubin.py`,
`h1_ssi.py`) now run against `femtools.fea.reduction`, `femtools.dynamics.cms_free`,
`femtools.mpe.frf_estimation` and `femtools.mpe.ssi` — see the Round-4 block below.
Checked = measured passing with the quoted numbers; unchecked = not yet exercised by a
test or example on this tree.

- [x] **1a** axial bar, 2-node discrete — `tests/test_golden_fea.py::test_two_node_axial_bar_frequency` green
- [ ] **1b** axial bar N-element dispersion — no test/example yet
- [ ] **1c** axial bar mesh-converged continuum — no test/example yet
- [x] **2** cantilever EB, 10 BEAM2 — `examples/cantilever_beam.py`: 6 lowest bending modes
      (both planes, 16.71–439.95 Hz) max rel err **2.55e-4** (tol 2e-2);
      `tests/test_golden_fea.py::test_euler_bernoulli_cantilever_first_three_modes_per_bending_plane` green
- [x] **3a** mass normalization — `examples/cantilever_beam.py`:
      $\max|\Phi^\top M \Phi - I|$ = **2.22e-16** (tol 1e-8); `tests/test_mass_normalization.py` green
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
- [x] **HEX8 anti-locking** (`fea.md` §2.5/§8, not in the master table) —
      `tests/test_hex8_verification.py` (3 tests) green: one-through-thickness bending ratio
      **0.9855** with the default incompatible modes (tol > 0.95) vs **0.6428** with `"full"`;
      distorted patch test **5.0e-16** (tol 1e-10); free–free block exactly 6 rigid-body modes

Round-4 cases (17–20): measured 2026-08-27 by running the three Round-4 examples against
the real merged kernels (R4-O1/O2/O4). The numbers below supersede the dry-run figures
that were previously quoted from throwaway reference implementations.

- [x] **17** Guyan/IRS/SEREP — `examples/guyan_serep.py` PASS (6/6 checks): Guyan static
      exactness **9.7e-13** (tol 1e-8); SEREP freq **2.3e-13** / shape recon **7.4e-16**
      (tol 1e-6 / 1e-8); Guyan upper bound margin **+1.05e-2 Hz**, mean rel err Guyan
      1.35e-2 → IRS **3.78e-4**
- [x] **18** free-interface CMS — `examples/cms_rubin.py` PASS (3/3 checks): first 4
      coupled modes vs unsplit reference, Rubin **1.85e-7** (tol 1e-2), MacNeal
      **7.71e-5** (tol 3e-2), Craig–Bampton baseline **2.93e-5** (tol 1e-2)
- [x] **19** H1/H2/coherence — `examples/h1_ssi.py` part 1 PASS: max |H1|/|H2| **0.9996**
      (≤ 1), γ² identity dev **6.7e-16**, median coherence 0.961, H1 vs ZOH-exact FRF
      median rel err **4.2 %** (tol 10 %), all three |H1| peaks within tolerance
- [x] **20** SSI-cov — `examples/h1_ssi.py` part 2 PASS: 3/3 poles from the
      stabilisation sweep (`order=20, n_modes=3`), rel f err ≤ **0.10 %** (tol 2 %),
      rel ζ err ≤ **11 %** (tol 50 %), MAC ≥ **0.9998** (tol > 0.9)

### Round-6 status (2026-08-28, R7-F3 re-audit — kernels landed, cases 21–28 measured)

All eight `examples/*.py` re-run on this tree: **8/8 PASS** (unchanged), and `pytest tests/`
is green (**191 passed, 3 perf skips**). Every Round-6 frozen API that the 2026-08-27 audit
found absent now imports, and cases 21–28 are **measured passing**. Test files:
`tests/test_round6_o4.py` (21 tests — cases 21–23), `tests/test_round6_io.py` (18 tests —
cases 24–25), `tests/test_round6_o1.py` (31 tests — case 28 + FEA golden regressions);
cases 26–27 have no dedicated pytest yet and were measured by direct probe (numbers below).

- [x] **21** UQ first-order covariance + seeded MC — `femtools.updating.uq`
      (`parameter_covariance`, `monte_carlo_update`, `UQResult`) imports;
      `tests/test_round6_o4.py` green: closed-form LS covariance reproduced to 1e-12,
      sandwich collapse for $W = C_z^{-1}$ to 1e-10, Gauss–Markov inflation for unit
      weighting, prior shrinkage, seeded-MC reproducibility (exact) and MC-vs-first-order
      agreement on a linear problem, residual/start-point resampling on the beam.
- [x] **22** shape optimization — `femtools.optimization.shape` (`shape_optimize`,
      `ShapeResult`) imports; `tests/test_round6_o4.py` green: two-bar-arch $f_1$ raise
      and compliance drop, mesh-quality barrier stop, QUAD4 plate stays valid while
      improving, `element_size_ratios` flags a folded shell, impossible requests rejected.
      The §10.2 cantilever-plate example construction stays frozen (examples note below).
- [x] **23** SSI-DATA — `femtools.mpe.ssi.ssi_data` imports; `tests/test_round6_o4.py`
      green (SDOF recovery, cross-method vs `ssi_cov` on a 2-DOF record, CVA weighting +
      reference channels, argument validation, `block_hankel`). §10.3 measured verbatim on
      this tree (2026-08-28, case-20 records: seed 11, `order=20, n_modes=3,
      f_range=(1, 12)`): rel f err ≤ **1.3e-3** (tol 2e-2), rel ζ err ≤ **14 %** (tol
      50 %), MAC vs truth ≥ **0.9997** (tol > 0.9); cross-method `ssi_data` vs `ssi_cov`
      |Δf|/f ≤ **3.5e-4** (tol 5e-3), cross-MAC ≥ **0.99999** (tol > 0.99).
- [x] **24** Abaqus INP round-trip — `femtools.io.inp.read_inp` imports;
      `tests/test_round6_io.py` green: HEX8 cube reads and assembles, shell sections and
      boundary forms, beam general/rect/circ sections, truss area, unknown-keyword
      warn-once with the material surviving, malformed decks raise, write→read
      round-trip on the cube/plate/beam deck.
- [x] **25** LS-DYNA K round-trip + cross-format identity — `femtools.io.kfile.read_k`
      imports; `tests/test_round6_io.py` green: fixed and free formats, degenerate
      tria-as-quad and solid degeneracies (incl. TET10 two-line midside drop), TC/RC
      constraint-code lookup table, beam resultant/rect/tube sections;
      `test_acceptance_cube_plate_beam_as_inp_and_k` pins the §10.4 same-deck
      INP-vs-K $K$/$M$ identity.
- [x] **26** NMD / MACX identities — `femtools.correlation.mac.nmd` / `.macx` import;
      no dedicated pytest yet — measured by direct probe (2026-08-28):
      $\mathrm{nmd}(x, x)$ = **0.0** exactly, range $[0, 1]$ holds,
      $\max|\mathrm{nmd}^2 + \mathrm{MAC} - 1|$ = **0.0** (tol 1e-12); `macx` ≡
      `mac_matrix` on real modes to **0.0**, complex-scale invariance dev **1.1e-16**
      (tol 1e-12). `mac_matrix` numerics unchanged (`tests/test_mac.py` green).
- [x] **27** modal strain/kinetic energy — `femtools.dynamics.energy` imports;
      no dedicated pytest yet — measured by direct probe on the case-2 cantilever
      (2026-08-28): max rel $|\mathrm{MSE}_r - \lambda_r/2|$ = **5.3e-13** (tol 1e-10).
      Convention note: the implemented `modal_kinetic_energy` is the *unit-amplitude*
      generalized KE $\mathrm{diag}(\Phi^\top M \Phi)/2$ — measured
      $\max|\mathrm{MKE}_r - 1/2|$ = **2.2e-16** — so the §10.6 equality reads
      $\mathrm{MSE}_r = \omega_r^2\,\mathrm{MKE}_r$, measured to **5.3e-13** rel.
      Element split (`element_modal_energy`) closes on the totals to **1.5e-15**
      (tol 1e-10, `meta["closure"]`).
- [x] **28** shell drilling 6-RBM — per-node rotational frames landed (R6-O1);
      `tests/test_round6_o1.py` green. `shell_drilling_orientation_gap` measured on this
      tree (2026-08-28): QUAD4 **6 / 6** aligned/oblique zero modes (first elastic
      34.412 Hz both orientations, rel dev 2.6e-13), TRIA3 **6 / 6** (33.775 Hz, rel dev
      6.0e-13), `oblique_warned = 0`, 16 framed nodes on the oblique plate; HEX8 98.6 %
      tip ratio, MITC4, BEAM2 EB and `solve_static(enforced=)` goldens unchanged
      (same test file).

Consequence for examples: the §10.2–§10.4 constructions are now exercised by the Round-6
test files (and the §10.3 verbatim probe above), so `examples/shape_plate.py`,
`examples/ssi_data_oma.py` and `examples/read_inp_k.py` remain optional; the example set
deliberately stays at the same eight this round — the only new examples sanctioned by the
Round-7 brief (`examples/rbe2_rigid.py`, `examples/topometry_plate.py`) are blocked on
kernels that do not import yet (Round-7 status below).

### Round-7 status (2026-08-28, R7-F3 audit)

Round-7 frozen APIs (`REMAINING.md` / `ROUND7_BRIEF.md`), checked by import on this tree.
The R7-O2 kernels already import locally (`dynamics.superelement.dump_cms` / `load_cms`,
`psd_response(base_accel=)`); the rows below are **pending**, none importable yet:

- [ ] **R7 stress recovery** — `femtools.fea.recover` (`recover_stress`,
      `recover_strain`, `StressResult`; BAR2/BEAM2/QUAD4/TRIA3/HEX8/TET4 centroids;
      constant-strain patch ≤ 1e-12, `fea.md` §10) — **pending**, module absent
- [ ] **R7 RBE2 condensation** — `femtools.fea.mpc` (`apply_rbe2`,
      `ConstraintTransform`; `assemble_km` honoring `model.rbe2`; welded free–free pair
      keeps 6 RBM; a rigid offset beam carries moment, `fea.md` §11) — **pending**,
      module absent (the `core.model.RBE2` container and `FEModel.add_rbe2` are merged
      and stable)
- [ ] **R7 write_cdb / write_k** — `femtools.io.cdb.write_cdb` /
      `femtools.io.kfile.write_k` (round-trip the Round-6 HEX8/QUAD4/BEAM2 acceptance
      decks through `assemble_km`) — **pending**, symbols absent (both readers import)
- [ ] **R7 topometry** — `femtools.optimization.topometry` (`topometry_optimize`,
      `TopometryResult`; element-wise thickness/density field on an *existing* mesh,
      min compliance under a volume/mean-thickness constraint; a cantilever plate must
      beat the uniform start with no inverted elements, `optimization.md` §5) —
      **pending**, module absent
- [ ] **R7 map_nearest_nodes** — `femtools.correlation.dofmap.map_nearest_nodes`
      (two translated copies of the same 8-node cube match 1–1 with
      distance = translation) — **pending**, symbol absent (module imports)

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
| 17a | Guyan static exactness (load at masters only) | Schur-complement identity (§9.1) | rel dev at masters | 1e-8 |
| 17b | SEREP square reproduction ($n_m = m$) | $T = \Phi \Phi_m^{-1}$ exact | rel freq / shape recon | 1e-6 / 1e-8 |
| 17c | Guyan bounds + IRS improvement | Rayleigh–Ritz; O'Callahan correction | $f_{red} \ge f_{full}$; mean err(IRS) ≤ mean err(Guyan) | — |
| 18 | Free-interface CMS, split fixed–fixed beam | coupled vs unsplit 20×BEAM2 (§9.2) | rel freq, first 4 | 1e-2 Rubin / 3e-2 MacNeal |
| 19 | H1/H2/coherence identities | $\gamma^2 = \lvert H_1\rvert/\lvert H_2\rvert \le 1$, shared segmentation | max dev | 1e-8 |
| 20 | SSI-cov synthetic 3-DOF chain (seeded) | known $(f_r, \zeta_r, \phi_r)$ (§9.4) | rel $f$ / rel $\zeta$ / MAC | 2e-2 / 5e-1 / > 0.9 |
| 21 | UQ: first-order covariance + seeded MC | sandwich $\to (S^\top C_z^{-1} S)^{-1}$ identity; $\sigma_{\hat p} = 2\sigma_f/\sqrt{n_z}$ closed form (§10.1) | identity dev / rel $\sigma_{\hat p}$ / MC vs first-order | 1e-10 / 1e-6 / 3e-1 |
| 22 | Shape optimization, cantilever plate (§10.2) | accepted-step objective history monotone; scaled Jacobian floor; mass constraint | monotone / $\min_e q_e$ / rel mass dev | — / ≥ $q_{min}$ / 1e-6 |
| 23 | SSI-DATA on the case-20 records (§10.3) | same truth as case 20 + cross-method vs `ssi_cov` | rel $f$ / rel $\zeta$ / MAC; cross $\Delta f/f$ / cross-MAC | 2e-2 / 5e-1 / > 0.9; 5e-3 / > 0.99 |
| 24 | Abaqus INP round-trip (§10.4) | minimal C3D8/S4/B31 deck → `FEModel` → `assemble_km` | counts, $E, \nu, \rho$, SPC masks exact; assembles | exact / no crash |
| 25 | LS-DYNA K round-trip + cross-format identity (§10.4) | same deck as K cards; same cube INP vs K | as case 24; $\max\lvert\Delta K\rvert, \lvert\Delta M\rvert$ after DOF matching | exact / 1e-12 rel |
| 26 | NMD / MACX identities (§10.5) | $\mathrm{nmd} = \sqrt{1 - \mathrm{MAC}}$ (brief-frozen); MACX = MAC on real modes | max dev | 1e-12 |
| 27 | Modal strain/kinetic energy (§10.6) | $\mathrm{MSE}_r = \mathrm{MKE}_r = \lambda_r / 2$ (mass-normalized); element sums = totals | rel dev | 1e-10 |
| 28 | Shell drilling 6-RBM contract (§10.7) | oblique free–free flat plate: 6 zero modes; elastic spectrum orientation-invariant | zero count / rel $f_{el}$ dev / goldens | = 6 / 1e-8 / unchanged |

Rows 21–28 are Round-6 contracts; as of the 2026-08-28 R7-F3 re-audit their kernels all
import on this tree and every row is **measured passing** (status block above) —
constructions in §10.

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

## 9. Round-4 goldens (reduction, free-interface CMS, H1/H2, SSI-cov)

Constructions live in the three Round-4 examples; every "measured" number below was
obtained by running them against the real merged kernels on
`cursor/femtools-remaining-d551` (2026-08-27), so they are regression facts for this
tree.

### 9.1 Reduction (case 17, `examples/guyan_serep.py`)

10-element BEAM2 cantilever, 60 free DOFs, 6 masters (uy/uz at $x = 0.3L, 0.6L, L$),
compared on the 6 lowest (all-bending) modes; formulas in `fea.md` §9.

- 17a: unit tip load (a master DOF) → Guyan-reduced static solution equals the full
  solution at the masters. Measured 1.1e-12 relative; tol 1e-8.
- 17b: SEREP with 6 masters = 6 kept modes reproduces the kept frequencies (measured
  2.3e-13 rel, tol 1e-6) and shapes ($\max|\Phi - T\Phi_m|$ 7.4e-16 rel, tol 1e-8).
- 17c: every Guyan frequency ≥ full-model frequency (Rayleigh–Ritz, measured margin
  +1.05e-2 Hz on mode 1); mean rel err improves Guyan 1.35e-2 → IRS 3.78e-4. The IRS
  assertion is on the *mean* — individual high modes need not improve monotonically.
  Guyan first-mode error < 1 % (measured 6.3e-4).

### 9.2 Free-interface CMS (case 18, `examples/cms_rubin.py`)

Fixed–fixed 20×BEAM2 beam split at midspan into two 10-element supported components
(A clamped at $x=0$, B at $x=L$), 8 kept free-interface modes + 6 interface DOFs each.
The `FreeCMSResult` generalized coordinates are [kept modes..., residual modes...] (no
physical DOF among them), so Rubin/MacNeal components are coupled with
`cms_free.free_interface_assembly`: rigid ties on the 6 shared *physical* interface
DOFs, eliminated through the null space of the compatibility matrix, with MacNeal's
massless residual block condensed statically (see `dynamics.md` §9.1). The
Craig–Bampton baseline, whose reduced coordinates do start with the physical boundary
DOFs, is assembled primally on the shared node and solved with QZ. Reference: the
unsplit model, which itself matches the analytic $\cosh\beta L\cos\beta L = 1$ roots
(4.7300407449, 7.8532046241, …) to example precision. Measured max rel err on the first
4 coupled modes: Rubin **1.85e-7** (tol 1e-2), MacNeal **7.71e-5** (tol 3e-2),
Craig–Bampton baseline **2.93e-5** (tol 1e-2).

### 9.3 H1/H2/coherence (case 19, `examples/h1_ssi.py` part 1)

3-DOF spring–mass chain ($k = 1000$ N/m, $m = 1$ kg → 2.240/6.276/9.069 Hz, $\zeta = 2\%$),
ZOH-exact simulation at 64 Hz for 1200 s, seeded white-noise force at DOF 1, 10 % output
noise; Welch `nperseg=8192` (17 averages, ≥ 17 lines across each half-power band).
Identities pinned on the 1–12 Hz band: $\max |H_1|/|H_2| \le 1$ (measured 0.9996) and
$\max|\gamma^2 - |H_1|/|H_2||$ ≤ 1e-8 nominal (measured 6.7e-16 — any visible deviation
means inconsistent segmentation between the three estimators). Accuracy: median complex
error vs the **ZOH-exact discrete FRF** < 10 % (measured 4.2 %); vs the continuous
receptance only magnitudes are comparable (half-sample delay — measured 3.0 % median
magnitude dev, quoted as a note, not a gate); each $|H_1|$ peak within
$\max(2\,df,\ 1.5\%)$ of the true frequency.

### 9.4 SSI-cov (case 20, `examples/h1_ssi.py` part 2)

Output-only records from `femtools.mpe.synthetic.synthetic_response` (same chain shapes,
600 s at 64 Hz, seed 11, 2 % noise), `ssi_cov(..., order=20, n_modes=3, f_range=(1, 12))`
→ nearest-frequency match to truth. `ssi_cov` sweeps model orders up to `order` and keeps
the poles that stabilise across orders, so `order` needs headroom above the 6 physical
states (the bare-minimum `order=6` leaves too few sweep points and the stabilisation
clustering returns a single pole). Gates: rel frequency error < 2 % (measured ≤ 1.0e-3),
rel damping error < 50 % (measured ≤ 11 %), MAC > 0.9 when shapes are returned (measured
≥ 0.9998). Spurious poles are rejected by the sweep and the `n_modes`/`f_range`
selection, and the nearest-frequency matching tolerates any that remain.

## 10. Round-6 golden constructions (kernels landed — cases 21–28 measured)

The 2026-08-27 R6-F3 audit found none of the Round-6 frozen APIs importable; as of the
2026-08-28 R7-F3 re-audit they all import and cases 21–28 are **measured passing**
(status block above, test files `tests/test_round6_o1.py` / `test_round6_io.py` /
`test_round6_o4.py`). The constructions below stay as the normative reference for the
gates those tests pin.
Formulas live in the matching `docs/algorithms/` files (updating.md §6, optimization.md §4,
mpe_rbpe.md §7, io.md).

### 10.1 UQ — first-order covariance and Monte Carlo (case 21)

Reuse the §6 E-recovery construction (single relative parameter $p$ on a uniform-$E$
cantilever, $n_z = 6$ relative frequency residuals). Identity gate:
`parameter_covariance` with $W_z = C_z^{-1}$, $W_p = 0$ must return
$(S^\top C_z^{-1} S)^{-1}$ to 1e-10 (sandwich collapse). Closed form: since
$f_r(p) = \sqrt{p}\, f_r(1)$ gives $\partial z_i / \partial p = -1/2$ at $\hat p = 1$,
i.i.d. relative noise $\sigma_f = 10^{-3}$ yields $\sigma_{\hat p} = 2\sigma_f / \sqrt{n_z}$
exactly — hit it to 1e-6. `monte_carlo_update` (200 samples, required seed, e.g. 0): sample
covariance within 30 % of first-order (sampling scatter $\approx \sqrt{2/199} \approx 10\%$
leaves headroom), identical output for identical seed (exact), input model not mutated.

### 10.2 Shape — cantilever plate (case 22, optional `examples/shape_plate.py`)

Clamped rectangular QUAD4 plate; design variables = in-plane coordinates of the free-edge
node column; objective: maximize $f_1$ under a total-mass equality constraint (tapering
toward the root is the physically expected optimum). Gates: objective history over
*accepted* steps monotone (SLSQP line-search rejections excluded), final $f_1$ strictly
above the initial value, $\min_e q_e \ge q_{min}$ for every evaluated design (no inverted
element ever reaches the eigensolver), mass constraint met to 1e-6 relative, input model
not mutated, `ShapeResult.converged` true.

### 10.3 SSI-DATA (case 23, optional `examples/ssi_data_oma.py`)

Feed `ssi_data` the *identical* records, orders and gates as case 20 (§9.4:
`synthetic_response`, 600 s at 64 Hz, seed 11, 2 % noise, `order=20, n_modes=3,
f_range=(1, 12)`) — same result type as `ssi_cov` by contract. Beyond the case-20 truth
gates, the cross-method gate is the point: `ssi_data` vs `ssi_cov` pole frequencies within
0.5 % and shape cross-MAC > 0.99 on all three modes (both methods share the SVD +
shift-invariance path, so disagreement isolates the Hankel/projection stage).

### 10.4 INP / K text subsets (cases 24–25, optional `examples/read_inp_k.py`)

Minimal decks per `io.md` §3: one HEX8 cube (8 nodes), one QUAD4 patch, one BEAM2 member,
with materials, sections and one boundary card each. Gates: node/element counts and ids
exact; $E, \nu, \rho$, thickness and beam $A, I$ exact; SPC masks exact (including the
K-file `TC/RC` lookup-table codes and Abaqus `first, last` DOF ranges); `assemble_km`
completes on both models; the same cube written as INP and as K assembles identical
$K$ and $M$ to 1e-12 relative after DOF matching (the translators carry no physics).
Degenerate-connectivity retyping (K-file tet-as-collapsed-solid → `TET4`,
tria-as-collapsed-quad → `TRIA3`) must be exercised by at least one element in the deck.

### 10.5 NMD / MACX (case 26)

`nmd` is frozen by the Round-6 brief as $\sqrt{1 - \mathrm{MAC}}$ (elementwise on the MAC
matrix): gates $\mathrm{nmd}(x, x) = 0$, range $[0, 1]$, and consistency
$\mathrm{nmd}^2 + \mathrm{MAC} = 1$ to 1e-12. `macx` (extended MAC for complex modes, using
both $\phi$ and $\bar\phi$) must reduce to `mac_matrix` on real mode sets to 1e-12 and be
invariant under complex scaling of either argument — `mac_matrix`/`fmac` numerics on real
modes are contractually unchanged.

### 10.6 Modal energies (case 27)

On any golden model with mass-normalized $\Phi$: $\mathrm{MSE}_r = \tfrac12 \phi_r^\top K
\phi_r = \lambda_r / 2$ and $\mathrm{MKE}_r = \tfrac12 \omega_r^2 \phi_r^\top M \phi_r =
\lambda_r / 2$, so $\mathrm{MSE}_r = \mathrm{MKE}_r$ to 1e-10 relative on every elastic
mode (rigid-body modes: both exactly 0). With per-element output enabled, element
contributions must sum to the totals to 1e-10 relative — the partition, not the total, is
the diagnostic content.

Implemented convention (R6, `femtools.dynamics.energy`): `modal_kinetic_energy` returns
the *unit-amplitude* generalized KE $\mathrm{diag}(\Phi^\top M \Phi)/2 = 1/2$ per
mass-normalized mode (the $\omega_r^2$ factor is deliberately not folded in), so the
equality above is checked as $\mathrm{MSE}_r = \omega_r^2\,\mathrm{MKE}_r$ — the
equipartition a normal mode satisfies by definition. Measured 2026-08-28 on the case-2
cantilever: $\max$ rel $|\mathrm{MSE}_r - \lambda_r/2|$ = 5.3e-13,
$\max|\mathrm{MKE}_r - 1/2|$ = 2.2e-16, element-split closure 1.5e-15
(`element_modal_energy(...).meta["closure"]`).

### 10.7 Shell drilling 6-RBM contract (case 28)

`fea.verification.shell_drilling_orientation_gap(etype)` builds a free–free flat plate
twice — normal along a global axis, and rotated to a generic orientation — and counts
near-zero frequencies. Pre-R6-O1 reproduction (2026-08-27): QUAD4 **6 aligned / 7
oblique**, TRIA3 **6 / 7**, `oblique_warned = 1` — the fictitious drilling mechanism was
present and correctly warned about. Contract, **met on this tree** since R6-O1's per-node
rotational frames (local 3-axis = averaged shell normal, drilling auto-constrained for
arbitrary orientation): `oblique_zero_modes` = **6** for both element types, first elastic
frequency unchanged between orientations to 1e-8 relative, no warning, and the existing
goldens untouched (HEX8 98.6 % tip ratio, 6 RBM on the cube, MITC4 thin plate, patch
tests, BEAM2 EB, `solve_static(enforced=)`). Measured 2026-08-28: QUAD4 **6 / 6**
(first elastic 34.412 Hz both orientations, rel dev 2.6e-13), TRIA3 **6 / 6**
(33.775 Hz, rel dev 6.0e-13), `oblique_warned = 0`, 16 framed nodes on the oblique plate;
pinned by `tests/test_round6_o1.py` (31 tests).

## 11. Determinism requirements (all tests)

Fixed seeds via `np.random.default_rng(seed)` only; eigenvector sign convention per
`fea.md` §6.2; degenerate-subspace comparisons via MAC/S2MAC, never entrywise; no test may
depend on ARPACK iteration counts or wall time.

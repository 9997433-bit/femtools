# State of the art — algorithmic basis and references

`femtools` implements published, public-domain-of-knowledge algorithms. Every numerical method
in the codebase traces to the peer-reviewed literature or standard textbooks listed here — not
to any commercial product's source code or manuals. Where the commercial FEMtools product
offers a capability, we implement the *published* algorithm family that capability is built
on, from the primary sources.

Implementation-facing notes state what femtools adopts from each reference and any deliberate
deviations.

## 1. Correlation metrics

* **MAC.** Allemang, R.J., Brown, D.L., *A Correlation Coefficient for Modal Vector Analysis*,
  Proc. 1st IMAC, 1982, pp. 110–116. Review and pitfalls: Allemang, R.J., *The Modal Assurance
  Criterion — Twenty Years of Use and Abuse*, Sound and Vibration, 37(8), 2003, pp. 14–23.
  femtools uses the complex-conjugate form
  `MAC(i,j) = |φ_aᵢᴴ φ_bⱼ|² / ((φ_aᵢᴴ φ_aᵢ)(φ_bⱼᴴ φ_bⱼ))`, valid for real and complex shapes.
* **CoMAC.** Lieven, N.A.J., Ewins, D.J., *Spatial Correlation of Mode Shapes: The Coordinate
  Modal Assurance Criterion (COMAC)*, Proc. 6th IMAC, 1988. Per-DOF correlation over paired
  mode sets; femtools requires modes to be paired (§ mode pairing) before CoMAC.
* **Cross-orthogonality / POC.** Mass-weighted checks `Φ_aᵀ M Φ_b` against identity, standard
  in FE/test verification practice (e.g. NASA-STD-5002 style criteria: diagonal ≥ 0.9,
  off-diagonal ≤ 0.1 for validated models). Requires reduction of M to test DOFs (§4) or
  expansion of test shapes.
* **FRF correlation.** FRAC (frequency response assurance criterion) and shape/amplitude
  variants: Heylen, W., Lammens, S., *FRAC: A Consistent Way of Comparing Frequency Response
  Functions*, Proc. Int. Conf. on Identification in Engineering Systems, 1996; textbook
  treatment in Heylen, W., Lammens, S., Sas, P., *Modal Analysis Theory and Testing*,
  KU Leuven, 1997. femtools implements FRAC per FRF pair, CSAC/CSF (cross-signature assurance
  criterion / scale factor) per frequency line.
* **Mode pairing.** MAC-based assignment; femtools solves the rectangular assignment problem
  on `1 − MAC` (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`) with a
  configurable frequency-deviation penalty, rather than greedy row-max, to avoid duplicate
  pairings on closely spaced modes.

## 2. Pretest design

* **Effective Independence (EFI).** Kammer, D.C., *Sensor Placement for On-Orbit Modal
  Identification and Correlation of Large Space Structures*, Journal of Guidance, Control,
  and Dynamics, 14(2), 1991, pp. 251–259. Iterative removal of candidate DOFs by smallest
  effective-independence value `E_d = diag(Φ (ΦᵀΦ)⁻¹ Φᵀ)`, maintaining the determinant of the
  Fisher information matrix. femtools re-evaluates `E_d` after every removal (exact update via
  rank-one downdate) instead of one-shot ranking.
* **Modal effective mass** for target-mode selection in base-driven structures: standard
  formulation, e.g. Wijker, J., *Mechanical Vibrations in Spacecraft Design*, Springer, 2004,
  ch. on modal effective mass; participation factors from rigid-body vectors
  `L = Φᵀ M R`, effective mass `Lᵢ²/mᵢ` per direction.
* **Kinetic-energy ranking.** Nodal kinetic energy `diag(Φ M Φᵀ)`-type per-DOF measures as a
  complementary heuristic (discussion in Kammer's work and in Heylen–Lammens–Sas).

## 3. Model updating and force identification

* **Sensitivity (gradient) updating.** Friswell, M.I., Mottershead, J.E., *Finite Element
  Model Updating in Structural Dynamics*, Kluwer Academic Publishers, 1995. Tutorial:
  Mottershead, J.E., Link, M., Friswell, M.I., *The sensitivity method in finite element model
  updating: A tutorial*, Mechanical Systems and Signal Processing, 25(7), 2011,
  pp. 2275–2296. femtools implements the weighted least-squares / Bayesian-weighted iteration
  `Δθ = (SᵀW_ε S + W_θ)⁻¹ SᵀW_ε r` with parameter and residual weighting, trust-region-style
  step limiting, and convergence on weighted residual norm.
* **Statistical weighting.** Collins, J.D., Hart, G.C., Hasselman, T.K., Kennedy, B.,
  *Statistical Identification of Structures*, AIAA Journal, 12(2), 1974, pp. 185–190
  (covariance-weighted estimation femtools exposes as `W_ε`, `W_θ`).
* **Eigenvalue/eigenvector derivatives.** Fox, R.L., Kapoor, M.P., *Rates of Change of
  Eigenvalues and Eigenvectors*, AIAA Journal, 6(12), 1968, pp. 2426–2429 (eigenvalue
  sensitivities, exact and cheap: `∂λᵢ/∂θ = φᵢᵀ(∂K/∂θ − λᵢ ∂M/∂θ)φᵢ` for mass-normalized φ).
  Nelson, R.B., *Simplified Calculation of Eigenvector Derivatives*, AIAA Journal, 14(9),
  1976, pp. 1201–1205 (single-mode-shape derivatives without full modal basis; used for MAC
  and shape residual sensitivities). Finite-difference fallback is retained for verification.
* **Regularization.** Tikhonov-type side constraints for ill-posed parameter sets; discussion
  in Friswell–Mottershead (1995) ch. on ill-conditioning and in the 2011 tutorial.
* **Force identification.** Harmonic force reconstruction by pseudo-inversion of the FRF
  matrix at measured responses, with truncated-SVD regularization; classical treatment in
  Ewins (see §5) and the inverse-problem literature. femtools reports singular-value spectra
  so users can audit conditioning.

## 4. Reduction and expansion

* **Guyan (static) reduction.** Guyan, R.J., *Reduction of Stiffness and Mass Matrices*,
  AIAA Journal, 3(2), 1965, p. 380.
* **IRS.** O'Callahan, J., *A Procedure for an Improved Reduced System (IRS) Model*,
  Proc. 7th IMAC, 1989 (adds inertial correction to Guyan).
* **SEREP.** O'Callahan, J., Avitabile, P., Riemer, R., *System Equivalent Reduction Expansion
  Process (SEREP)*, Proc. 7th IMAC, 1989 (modal-basis reduction/expansion, exact at retained
  modes; femtools' default for shape expansion in cross-orthogonality).
* femtools implements each as an explicit transformation matrix `T` with
  `K_r = TᵀKT, M_r = TᵀMT`, shared between pretest (test-DOF system matrices) and correlation
  (expansion of measured shapes).

## 5. Structural dynamics: FRF synthesis, CMS

* **Modal FRF synthesis + residual terms.** Ewins, D.J., *Modal Testing: Theory, Practice and
  Application*, 2nd ed., Research Studies Press, 2000 — receptance synthesis
  `H_jk(ω) = Σᵢ (φ_jᵢ φ_kᵢ) / (ωᵢ² − ω² + 2iζᵢωᵢω)` plus lower/upper residual terms (residual
  mass and residual flexibility) for out-of-band modes. Also Craig, R.R., Kurdila, A.J.,
  *Fundamentals of Structural Dynamics*, 2nd ed., Wiley, 2006. femtools implements modal ζ,
  Rayleigh (α, β → equivalent modal ζᵢ = α/(2ωᵢ) + βωᵢ/2), and structural damping η variants,
  and validates against direct dynamic-stiffness inversion (`docs/CONTRACT_API.md` tolerance:
  5% relative L2 over 0.2–0.8 f_max with 20 modes).
* **Craig–Bampton CMS.** Craig, R.R., Bampton, M.C.C., *Coupling of Substructures for Dynamic
  Analyses*, AIAA Journal, 6(7), 1968, pp. 1313–1319 — fixed-interface normal modes +
  constraint modes. Free-interface alternatives for R2: MacNeal, R.H., *A Hybrid Method of
  Component Mode Synthesis*, Computers & Structures, 1(4), 1971, pp. 581–601; Rubin, S.,
  *Improved Component-Mode Representation for Structural Dynamic Analysis*, AIAA Journal,
  13(8), 1975, pp. 995–1006 (residual-flexibility attachment modes).
* **Eigen solution.** Lehoucq, R.B., Sorensen, D.C., Yang, C., *ARPACK Users' Guide*, SIAM,
  1998 — implicitly restarted Lanczos, shift-invert for the constrained generalized problem.

## 6. Modal parameter estimation (EMA/OMA)

* **LSCE.** Brown, D.L., Allemang, R.J., Zimmerman, R., Mergeay, M., *Parameter Estimation
  Techniques for Modal Analysis*, SAE Technical Paper 790221, 1979 — least-squares complex
  exponential on impulse-response data (via inverse FFT of FRFs).
* **p-LSCF / PolyMAX-class.** Guillaume, P., Verboven, P., Vanlanduit, S., Van der Auweraer,
  H., Peeters, B., *A Poly-Reference Implementation of the Least-Squares Complex
  Frequency-Domain Estimator*, Proc. 21st IMAC, 2003; Peeters, B., Van der Auweraer, H.,
  Guillaume, P., Leuridan, J., *The PolyMAX Frequency-Domain Method: A New Standard for Modal
  Parameter Estimation?*, Shock and Vibration, 11(3–4), 2004, pp. 395–409. femtools implements
  the poly-reference right matrix-fraction model with real-valued reduced normal equations,
  pole extraction from the companion matrix, and LSFD residue estimation; stabilization
  diagrams are R2.
* **FDD / EFDD.** Brincker, R., Zhang, L., Andersen, P., *Modal Identification of Output-Only
  Systems Using Frequency Domain Decomposition*, Smart Materials and Structures, 10(3), 2001,
  pp. 441–445; damping via SDOF autocorrelation fitting: Brincker, R., Ventura, C.E.,
  Andersen, P., *Damping Estimation by Frequency Domain Decomposition*, Proc. 19th IMAC, 2001.
  Textbook: Brincker, R., Ventura, C., *Introduction to Operational Modal Analysis*, Wiley,
  2015.
* **SSI (planned R3+).** Van Overschee, P., De Moor, B., *Subspace Identification for Linear
  Systems: Theory — Implementation — Applications*, Kluwer, 1996.

## 7. Rigid-body property extraction

* Bretl, J., Conti, P., *Rigid Body Mass Properties from Test Data*, Proc. 5th IMAC, 1987
  (mass-line / inertia identification from low-frequency FRFs).
* Survey and method comparison: Schedlinski, C., Link, M., *A Survey of Current Inertia
  Parameter Identification Methods*, Mechanical Systems and Signal Processing, 15(1), 2001,
  pp. 189–211. femtools implements the frequency-domain mass-line least-squares fit for the
  10 rigid-body parameters (m, CoG, inertia tensor) with condition-number reporting.

## 8. Optimization

* **SIMP topology optimization.** Bendsøe, M.P., *Optimal Shape Design as a Material
  Distribution Problem*, Structural Optimization, 1(4), 1989, pp. 193–202; Bendsøe, M.P.,
  Sigmund, O., *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
  Compact reference implementation style: Sigmund, O., *A 99 Line Topology Optimization Code
  Written in Matlab*, Structural and Multidisciplinary Optimization, 21(2), 2001, pp. 120–127.
  femtools uses penalization p=3 with optimality-criteria updates and a density filter —
  Bourdin, B., *Filters in Topology Optimization*, IJNME, 50(9), 2001, pp. 2143–2158;
  Bruns, T.E., Tortorelli, D.A., *Topology Optimization of Non-Linear Elastic Structures and
  Compliant Mechanisms*, CMAME, 190(26–27), 2001, pp. 3443–3459.
* **Size optimization.** Gradient-based NLP on the updating parameter protocol (scipy
  SLSQP/trust-constr), with the eigen-sensitivities of §3.
* **DOE.** McKay, M.D., Beckman, R.J., Conover, W.J., *A Comparison of Three Methods for
  Selecting Values of Input Variables in the Analysis of Output from a Computer Code*,
  Technometrics, 21(2), 1979, pp. 239–245 (Latin hypercube sampling; femtools requires an
  explicit seed).

## 9. Numerical acceptance philosophy

Acceptance is defined so that a failure always identifies a *code* defect, never an ambiguous
modeling discrepancy:

1. **Identity-level checks at near machine precision.** Properties that hold exactly in exact
   arithmetic — mass orthonormality `‖ΦᵀMΦ − I‖_max ≤ 1e-8`, `MAC(Φ,Φ)` diagonal within
   `1e-12` of 1, symmetry of assembled K/M — are tested at tolerances dominated only by
   accumulated roundoff, independent of mesh quality.
2. **Discretization-aware comparisons.** Comparisons against continuum theory (Euler–Bernoulli
   cantilever frequencies) use tolerances that reflect the element formulation's convergence
   rate at the prescribed mesh (2% for 10 BEAM2 elements on the first 3 bending modes), while
   cases where the discrete model *is* the theory (2-node axial bar single mode) demand 1e-8
   relative accuracy. A tolerance must never absorb both discretization and implementation
   error at once.
3. **Cross-method consistency.** Independent computational paths for the same physical
   quantity must agree: modal-superposition FRF vs. direct dynamic-stiffness FRF (≤5% relative
   L2 over 0.2–0.8 f_max, 20 modes, light damping); semi-analytic sensitivities vs. central
   finite differences; ARPACK vs. dense `eigh` on small problems.
4. **Round-trip invariance.** I/O translators are tested write→read→compare on the typed
   objects (ids, coordinates, shape values), with exact equality for integers/strings and
   1 ULP-scale tolerances for floats through text formats.
5. **Inverse-method recovery.** Estimators are validated on synthetic data generated by the
   forward code with known ground truth and controlled noise: updating recovers a 10% E
   perturbation within 2%; EFI-selected sensor sets keep off-diagonal MAC < 0.15 on the toy
   case; p-LSCF/FDD recover synthetic poles within stated frequency/damping tolerances.
   Noise realizations use fixed seeds.
6. **Determinism.** Two runs of the full test suite must produce identical numbers; any
   nondeterministic tolerance ("flaky") test is treated as a defect.
7. **No silent degradation.** Post-conditions (normalization, bounds like MAC ∈ [0, 1+ε],
   Hermitian spectra) are asserted inside library code, not only in tests, so acceptance
   failures cannot be masked by downstream renormalization.

The concrete Round-1 tolerance table lives in `docs/CONTRACT_API.md` and is binding for all
implementations; `docs/ACCEPTANCE.md` (owned by R1-F3) elaborates the golden cases.

## Non-infringement note

References above are public conference papers, journal articles, and textbooks. No text,
figures, code, or file-format documentation from commercial FEMtools (Dynamic Design
Solutions) or any other proprietary product is reproduced here or in the implementation.
UNV and Nastran BDF support is implemented from publicly documented dataset/card layouts.

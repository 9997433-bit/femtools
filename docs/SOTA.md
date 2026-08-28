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
* **NMD / MACX (merged Round 6 — `correlation.mac.nmd` / `macx`, PRODUCT_MAP R6).**
  Normalized modal difference: a MAC-derived distance metric from the MAC-extension family
  reviewed in Allemang, R.J., *The Modal Assurance Criterion — Twenty Years of Use and Abuse*
  (cited above); the Round-6 brief freezes the form `nmd = sqrt(1 − MAC)`. Extended MAC for
  complex modes, correlating against both `φ` and `conj(φ)` so complex-conjugate pair
  ambiguity does not depress the metric: Vacher, P., Jacquier, B., Bucharles, A., *Extensions
  of the MAC criterion to complex modes*, Proc. ISMA 2010, pp. 2713–2725. The real-mode
  `mac_matrix` / `fmac` numerics are unchanged by these additions.
* **Mode pairing.** MAC-based assignment; femtools solves the rectangular assignment problem
  on `1 − MAC` (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`) with a
  configurable frequency-deviation penalty, rather than greedy row-max, to avoid duplicate
  pairings on closely spaced modes.
* **Nearest-node FE↔test geometry mapping (Round 7 —
  `correlation.dofmap.map_nearest_nodes`, PRODUCT_MAP R7).**
  Nearest-neighbor search over node coordinates
  via k-d trees: Bentley, J.L., *Multidimensional Binary Search Trees Used for Associative
  Searching*, Communications of the ACM, 18(9), 1975, pp. 509–517; expected-logarithmic
  best-match queries: Friedman, J.H., Bentley, J.L., Finkel, R.A., *An Algorithm for Finding
  Best Matches in Logarithmic Expected Time*, ACM Transactions on Mathematical Software,
  3(3), 1977, pp. 209–226 (the algorithm behind `scipy.spatial.cKDTree`, which femtools
  wraps). Input is a test grid `(n, 3)` against model/FE coordinates; output `(fe_ids,
  distances)`, so two translated copies of the same mesh must match 1–1 with distance equal
  to the translation. Downstream DOF selection stays typed through `DOFMap` — geometric
  proximity never silently becomes a DOF-order assumption.

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
* **Parameter uncertainty / first-order covariance (merged Round 6 — `updating.uq`,
  PRODUCT_MAP R6).** First-order propagation of the residual covariance through the
  weighted least-squares solution: with sensitivity `S` and residual covariance `Cov(ε)`, the
  estimator gain `G = (SᵀW_ε S + W_θ)⁻¹ SᵀW_ε` gives `Cov(θ) = G Cov(ε) Gᵀ` — the linearized
  parameter-uncertainty analysis of Friswell–Mottershead (1995), building on the statistical
  identification framework of Collins–Hart–Hasselman–Kennedy (1974) cited above. Complemented
  by seeded Monte Carlo re-updating over perturbed residuals (`monte_carlo_update`, explicit
  `seed` per the determinism policy); no Bayesian samplers beyond this are in scope.
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
  modes; intended default for shape expansion in cross-orthogonality).
* Implementation merged in Round 4 (`docs/PRODUCT_MAP.md` R4 — `fea.reduction`,
  `correlation.expansion`): each as an explicit transformation matrix `T` with
  `K_r = TᵀKT, M_r = TᵀMT`, shared between pretest (test-DOF system matrices)
  and correlation (expansion of measured shapes via `expand_guyan` / `expand_serep`).

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
  constraint modes. Free-interface alternatives (merged Round 4, `docs/PRODUCT_MAP.md`
  R4 — `dynamics.cms_free.rubin` / `macneal`):
  MacNeal, R.H., *A Hybrid Method of Component Mode Synthesis*, Computers & Structures,
  1(4), 1971, pp. 581–601; Rubin, S., *Improved Component-Mode Representation for Structural
  Dynamic Analysis*, AIAA Journal, 13(8), 1975, pp. 995–1006 (residual-flexibility
  attachment modes).
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
  diagrams are built by `mpe.common.stabilization_diagram` / `select_physical_poles`.
* **FDD / EFDD.** Brincker, R., Zhang, L., Andersen, P., *Modal Identification of Output-Only
  Systems Using Frequency Domain Decomposition*, Smart Materials and Structures, 10(3), 2001,
  pp. 441–445; damping via SDOF autocorrelation fitting: Brincker, R., Ventura, C.E.,
  Andersen, P., *Damping Estimation by Frequency Domain Decomposition*, Proc. 19th IMAC, 2001.
  Textbook: Brincker, R., Ventura, C., *Introduction to Operational Modal Analysis*, Wiley,
  2015.
* **SSI (merged Round 4 — `mpe.ssi.ssi_cov`, covariance-driven).** Van Overschee, P.,
  De Moor, B., *Subspace Identification for Linear Systems: Theory — Implementation —
  Applications*, Kluwer, 1996. The same monograph is the basis for the **data-driven**
  variant (merged Round 6 — `mpe.ssi.ssi_data`, PRODUCT_MAP R6): N4SID-class
  identification by orthogonal projection of the future-output block-Hankel row space onto
  the past-output row space, followed by the same SVD + shift-invariance realization step
  as the covariance-driven path (and returning the same result type). Application to
  output-only modal analysis: Peeters, B., De Roeck, G., *Reference-Based Stochastic
  Subspace Identification for Output-Only Modal Analysis*, Mechanical Systems and Signal
  Processing, 13(6), 1999, pp. 855–878.

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
* **Shape optimization (merged Round 6 — `optimization.shape.shape_optimize`, PRODUCT_MAP
  R6).** Haftka, R.T., Grandhi, R.V., *Structural shape optimization — a survey*,
  Computer Methods in Applied Mechanics and Engineering, 57(1), 1986, pp. 91–106; textbook
  treatment: Haftka, R.T., Gürdal, Z., *Elements of Structural Optimization*, 3rd ed.,
  Kluwer, 1992. Selected node coordinates as design variables on the same scipy
  SLSQP/trust-constr backends as size optimization, with a mesh-quality safeguard
  (Laplacian-smoothness / minimum-Jacobian barrier) against the element-distortion failure
  mode that the survey literature identifies as the central difficulty of shape variables.
* **Topometry — element-wise sizing on a fixed mesh (Round 7 —
  `optimization.topometry.topometry_optimize`, PRODUCT_MAP R7).** Bendsøe–Sigmund
  (2003, cited above) organize structural optimization into sizing, shape, and topology
  design: *topometry* keeps the mesh and connectivity fixed and treats a per-element
  thickness (or density) as the design variable — element-by-element sizing in the
  Bendsøe–Sigmund taxonomy — whereas *topology* optimization (our `topology_simp`) decides
  material existence on a grid it builds itself. The min-compliance formulation,
  optimality-criteria update, and density filtering reuse the published machinery already
  cited for SIMP (Sigmund 2001; Bourdin 2001), applied to an existing `FEModel` mesh with a
  volume / mean-thickness constraint (OC or scipy SLSQP). The name "topometry" for this
  variant is from the public conference literature — Leiva, J.P., *Topometry Optimization:
  A New Capability to Perform Element by Element Sizing Optimization of Structures*,
  Proc. 10th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, 2004
  (AIAA 2004-4595) — cited for terminology only; no commercial-manual material is used.
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

## 10. Merged-code gap (Round-2 closure and residual distances)

Known distances between the merged code and the state of the art it targets. Status tags in
`docs/PRODUCT_MAP.md` point here; a row tagged R1/R2/R4/R6/R7/R8/R9/R10 is merged and tested but may still carry
one of these caveats. Capabilities that are absent (rather than imperfect) are the R5+
and N/A rows of the product map, not repeated here. Round 9 close-out promotions are
covered in §13; Round 10 (Cycle D opening) is covered in §14.

### Closed in Round 2

* **HEX8 shear locking — fixed.** The Round-1 trilinear HEX8 with full 2×2×2 Gauss quadrature
  locked in bending (~66% of the reference tip deflection on a single-element-thick
  cantilever). Round 2 implemented incompatible (bubble) modes — Wilson, E.L., Taylor, R.L.,
  Doherty, W.P., Ghaboussi, J., *Incompatible Displacement Models*, in *Numerical and
  Computer Methods in Structural Mechanics*, Academic Press, 1973; static condensation of the
  internal modes per element. The same case now recovers **98.6%** of the reference
  deflection, and a free-free mesh still yields exactly 6 rigid-body modes.
* **UNV material/property cards — closed for femtools round trips.** `write_unv` appends a
  private dataset 30000 (legal user range; conforming readers skip unknown numbers) holding
  the material/property tables as JSON, and `read_unv` restores them, so a
  femtools→femtools UNV round trip no longer loses E/ν/ρ or section data.

### Residual caveats on merged code

* **HEX8 under mesh distortion.** Incompatible modes are calibrated on parallelepiped
  elements; with strong skew (skew factor → 0.4) the single-layer bending ratio degrades
  from 0.986 toward ~0.36 — still better than the locking element, but mesh-quality limits
  apply. The assumed-strain stabilization of Simo, J.C., Rifai, M.S., *A Class of Mixed
  Assumed Strain Methods and the Method of Incompatible Modes*, IJNME, 29(8), 1990,
  pp. 1595–1638, is the known distortion-robust upgrade (not implemented).
* **`bbar` HEX8 variant.** The B-bar (mean-dilatation) option is too soft in thin bending
  (~2.13× reference deflection); it is intended only for near-incompressible, thick
  components.
* **Truncated-FRF acceptance is a 20-mode statement.** The ≤5% relative-L2 band of
  `docs/CONTRACT_API.md` (0.2–0.8 f_max, `retained_band`) is contractual for 20 retained
  modes with light damping; with fewer than ~20 modes or ζ ≈ 0.02 the 5% band may not be
  met and residual-term compensation (§5) is required.
* **UNV materials are invisible to third parties.** Dataset 30000 is femtools-private by
  design; external UFF readers skip it, so materials/properties still do not transfer to
  other tools via UNV (BDF and `.ftproj` remain the lossless routes).
* **BDF midside nodes.** HEX20 import still collapses to HEX8 (midside nodes
  dropped, one aggregated warning per card type). **TET10 is first-class since
  Round 10** (§14): a 10-node CTETRA keeps all 10 nodes as etype `TET10`.
* **DAQ: hardware acquisition is N/A by design** (hardware and vendor-licensing scope, not an
  algorithmic gap). The supported substitute is synthetic test data with controlled noise and
  fixed seeds — `femtools.dynamics.synthetic` and `femtools.mpe.synthetic` — which is what the
  MPE/OMA validation cases of §6 and §9.5 consume.
* **Static typing.** `py.typed` ships and the public API has a full static view
  (`TYPE_CHECKING` re-exports), but mypy still reports a few dozen informational findings in
  internal modules; the CI mypy step is non-blocking.

## 11. FEA kernels — Round 7 (stress recovery, rigid constraints)

Round 7 (Cycle C's first round) merged two fea-layer additions frozen in
`.agent_workspace/REMAINING.md`; both are tagged **R7** in `docs/PRODUCT_MAP.md`.
References are textbooks and journal/conference papers only.

* **Linear stress/strain recovery (`fea.recover.recover_stress` / `recover_strain` →
  `StressResult`).** Element stress from the displacement solution, `σ = D B u` evaluated at
  sampling points inside each element: Cook, R.D., Malkus, D.S., Plesha, M.E., Witt, R.J.,
  *Concepts and Applications of Finite Element Analysis*, 4th ed., Wiley, 2002 (stress
  computation and sampling-point accuracy); Bathe, K.-J., *Finite Element Procedures*,
  Prentice Hall, 1996 (2nd ed., K.J. Bathe, 2014) — calculation of stresses from the
  isoparametric displacement field and their convergence behavior. Superconvergent/optimal
  sampling at the reduced Gauss points: Barlow, J., *Optimal Stress Locations in Finite
  Element Models*, International Journal for Numerical Methods in Engineering, 10(2), 1976,
  pp. 243–251. femtools reports one stress/strain state per element at the centroid (or the
  average of the Gauss-point values) for BAR2, BEAM2, QUAD4, TRIA3, HEX8, TET4 — linear
  elasticity only, no plasticity. Acceptance is a constant-strain patch test at 1e-12, in
  the spirit of MacNeal, R.H., Harder, R.L., *A Proposed Standard Set of Problems to Test
  Finite Element Accuracy*, Finite Elements in Analysis and Design, 1(1), 1985, pp. 3–20:
  a linear displacement patch is represented exactly by these formulations, so the
  tolerance is roundoff-only per §9.1. Nodal extrapolation / smoothed recovery
  (Zienkiewicz–Zhu superconvergent patch recovery) is the known upgrade for nodal stress
  fields and stayed out of Round 7's scope; Round 8 adds plain nodal *averaging* — see
  §12, which also states why that is deliberately not SPR.
* **RBE2 rigid constraints by master–slave elimination (`fea.mpc.apply_rbe2` /
  `ConstraintTransform`; the data container `core.model.RBE2` / `FEModel.add_rbe2` is
  already merged).** Multipoint constraints imposed by the transformation (master–slave /
  null-space) method — eliminate dependent DOFs through `u = T q` and reduce
  `K_r = Tᵀ K T`, `M_r = Tᵀ M T` — keeping symmetry and avoiding the eigenvalue pollution
  of penalty approaches: textbook treatment in Cook–Malkus–Plesha–Witt (2002), constraint
  chapters, and Bathe (1996), imposition of constraints in the displacement-based
  formulation. The dependent node follows the independent node by the small-rotation
  rigid-body relation `u_d = u_i + θ_i × (x_d − x_i)`, which supplies the rows of `T`;
  `assemble_km` applies `T` once, honoring `model.rbe2` (or an explicit `mpc=` transform).
  Acceptance per the Round-7 brief: a free-free model with two nodes welded by an RBE2
  keeps exactly 6 rigid-body modes, and a rigid offset beam transmits the moment. The
  `RBE2` *card* layout is read from publicly documented card descriptions, like every other
  BDF card femtools parses; no commercial-manual text is used.

## 12. Round 8 — interpolation constraints, nodal averaging, text drivers

Round 8 (Cycle C's second round) merged the APIs of `.agent_workspace/REMAINING.md`
(Round 8 section). Every row is tagged **R8** in `docs/PRODUCT_MAP.md`. References
are public textbooks and journal papers only.

* **RBE3 interpolation constraint (`fea.mpc.apply_rbe3` → the shared
  `ConstraintTransform`; container `core.model.RBE3` / `FEModel.add_rbe3` already
  merged).** The constraint *imposition* machinery is the same master–slave /
  transformation (null-space) method already referenced for RBE2 in §11 — eliminate the
  dependent DOFs through `u = T q`, reduce `K_r = Tᵀ K T`, `M_r = Tᵀ M T`, keep symmetry,
  and avoid the eigenvalue pollution of penalty approaches: Cook, R.D., Malkus, D.S.,
  Plesha, M.E., Witt, R.J., *Concepts and Applications of Finite Element Analysis*,
  4th ed., Wiley, 2002 (constraint and transformation chapters); Zienkiewicz, O.C.,
  Taylor, R.L., Zhu, J.Z., *The Finite Element Method: Its Basis and Fundamentals*,
  6th ed., Elsevier Butterworth-Heinemann, 2005 (multipoint constraints by
  transformation / master–slave elimination, vs. penalty and Lagrange-multiplier
  alternatives). What differs from RBE2 is the constraint *content*: the dependent
  (reference) node's listed components follow the **weighted average** of the
  independent nodes' displacements, `u_d = Σᵢ wᵢ u_i / Σᵢ wᵢ` (equal weights by
  default, or `RBE3.weights`) — an interpolation constraint, **not** the RBE2 rigid-weld
  kinematics `u_d = u_i + θ_i × (x_d − x_i)` of §11, and with no penalty springs. Because
  the elimination is virtual-work consistent, the force conjugate to the eliminated DOFs
  redistributes as `f_q = Gᵀ f` (Cook et al., work-equivalent load transformation): a
  force applied at the dependent node is spread over the independents in proportion to
  the weights — equal weights give equal translational force shares — and the constraint
  adds no artificial stiffness paths between the independents, so a free–free assembly
  keeps exactly 6 rigid-body modes (the Round-8 acceptance gate). `assemble_km` composes
  `model.rbe3` with `model.rbe2` into one transform; `mpc=False` still disables all MPCs.
  The `RBE3` *card* layout (refgrid / refc / wt, c, g lists) is read from publicly
  documented card descriptions, like every other BDF card femtools parses.
* **Nodal stress averaging (`fea.recover.average_nodal`).** Unweighted direct averaging
  of the element-centroid stresses of §11 onto incident nodes (each node receives the
  mean of its `n_adj` adjacent element values, weight `1/n_adj`) — the classical
  baseline for smoothing the inter-element-discontinuous stress field; smoothing of
  discontinuous FE functions goes back to Hinton, E., Campbell, J.S., *Local and Global
  Smoothing of Discontinuous Finite Element Functions Using a Least Squares Method*,
  IJNME, 8(3), 1974, pp. 461–480, with textbook treatment of nodal averaging in
  Cook et al. (2002) and Zienkiewicz–Taylor–Zhu (cited above). Acceptance keeps the
  patch-test discipline of §9.1/§11: on a constant-stress patch every element carries
  the same centroid value, so the average is exact at every node to roundoff. This is
  deliberately **not** Zienkiewicz–Zhu superconvergent patch recovery —
  Zienkiewicz, O.C., Zhu, J.Z., *A Simple Error Estimator and Adaptive Procedure for
  Practical Engineering Analysis*, IJNME, 24(2), 1987, pp. 337–357 (the ZZ error
  estimator built on a recovered field), and *The Superconvergent Patch Recovery and A
  Posteriori Error Estimates. Part 1: The Recovery Technique*, IJNME, 33(7), 1992,
  pp. 1331–1364 — which fits local polynomial patches to values sampled at the
  superconvergent (Barlow, §11) points and is the known higher-accuracy upgrade for
  nodal fields and error estimation; SPR remains out of scope.
* **ANSYS / Abaqus optional text drivers (`drivers.ansys.AnsysCdbDriver`,
  `drivers.abaqus.AbaqusInpDriver`).** No new numerical algorithm — concrete adapters on
  the PEP 544 `SolverDriver` protocol (`docs/ARCHITECTURE.md` §10), built exclusively on
  the public **text** translators: `write_cdb` / `write_inp` for input, the existing
  `.pch`/`.unv` text readers for modal results. Binary results stay N/A: a `.rst` or
  `.odb` path raises `SolverError` naming the format, exactly as OP2 is handled for
  Nastran; no proprietary binary parser exists in the codebase and tests never require
  a vendor install.
* **Remaining Round-8 names introduce no new algorithms.** `dynamics.frf.dump_frf` /
  `load_frf` are npz persistence for `FRFResult` (bit-identical `H`/`freq_hz` round trip,
  analogous to the R7 `dump_cms`); `correlation.dofmap.mapped_mode_matrix` is row
  selection of FE mode shapes at the `map_nearest_nodes` ids of §1;
  `viz.plots.plot_stress` is presentation of §11 recovered stresses;
  `updating.updater.update_from_static` composes the §3 sensitivity updating with the
  R7 static-displacement responses. Each is covered by the references already cited for
  the machinery it wraps.

## 13. Round 9 — close-out promotions, persistence, mapped MAC, SOL 101 text punch

Round 9 (Cycle C's third and closing round) merged on this tree; its rows are
tagged **R9** in `docs/PRODUCT_MAP.md`. **No new numerical algorithm enters the
codebase in this round** — every Round-9 name is a contract promotion, a
composition, or a persistence/presentation layer over machinery already referenced
in §1–§12, so no new citations are required.

* **`fea.mpc.apply_mpc` (contract promotion).** The public composer of `model.rbe2` +
  `model.rbe3` into the single `ConstraintTransform` that `assemble_km` already
  applies. The constraint *content* is exactly §11 (RBE2 small-rotation rigid weld)
  and §12 (RBE3 weighted average); the *imposition* is the same master–slave /
  null-space transformation of Cook et al. (2002) and Zienkiewicz–Taylor–Zhu (2005),
  both cited above. Round 9 freezes the import and pins composition gates (empty
  tables → identity/no-op; single-table calls bit-identical to `apply_rbe2` /
  `apply_rbe3`; mixed RBE2-off-RBE3 chains keep exactly 6 free–free rigid-body modes;
  overlapping dependent DOFs raise). No kinematic or numerical change.
* **`updating.responses.static_stress_response` (contract promotion).** The recovered
  stresses of §11 (`fea.recover.recover_stress`) plugged into the §3 sensitivity
  updating as residuals — the Friswell–Mottershead machinery is unchanged. The
  recovery gate is *displacement-driven* (enforced displacement rather than dead
  load): on a statically determinate member a dead-load stress σ = F/A is independent
  of E and carries no parameter information, so only the displacement-driven residual
  can identify the modulus.
* **`correlation.dofmap.mapped_mac` (convenience wrap — explicitly *not* a new
  correlation metric).** One call composing `map_nearest_nodes` (§1, k-d tree
  nearest-neighbor mapping) + `mapped_mode_matrix` (§12, row selection) + `mac_matrix`
  (§1, Allemang 1982). The MAC formula and the nearest-node distances are exactly
  those of the functions it wraps; the gate (two translated copies of the same block →
  mapped-MAC diagonal of 1) is the R8 `mapped_mode_matrix` gate re-run through the
  one-call surface.
* **`dynamics.random.dump_psd` / `load_psd` (persistence, no algorithm).** npz
  dump/load of `PSDResult`, directly analogous to the R8 `dump_frf` and the R7
  `dump_cms` (§12): the stored spectra and `freq_hz` are bit-identical after a load
  round trip, and the `psd_response` numerics (§5 synthesis; Miles / base-acceleration
  checks of the R7 row) are untouched.
* **Nastran SOL 101 static punch — still text, still no binaries.** The
  `NastranPunchDriver` (PRODUCT_MAP R7; `docs/ARCHITECTURE.md` §10) gains a static
  path: a public SOL 101 case control requesting `DISPLACEMENT(PUNCH)=ALL` on the
  write side, and a punch `$DISPLACEMENTS` **text** parser feeding the driver's
  `read_static` on the read side. Everything remains text punch, exactly like the R7
  SOL 103 modal path; tests stub the executable. **Still no OP2, no RST, no ODB** —
  the closed binary dumps stay N/A (§10 and the product map), and Round 9 does not
  reopen them.
* **CLI / script / GUI polish (presentation only).** The `dump-frf` / `load-frf` /
  `update-static` commands, the `UPDATE STATIC` script verb, and the GUI stress table
  over the existing `/api/stress` endpoint are surfaces over already-cited machinery;
  no algorithmic content, and `import femtools.viz` still never requires pyvista.

## 14. Round 10 — Cycle D opening: TET10, ZZ-SPR, ERA, residual flexibility, expanded MAC, CTETRA10 / punch stresses

Round 10 opens Cycle D. Its API is frozen in `.agent_workspace/REMAINING.md` (Round 10
section) / `ROUND10_BRIEF.md`, and every Round-10 row in `docs/PRODUCT_MAP.md` is tagged
**R10** — merged, tested, and a stable top-level export. References are public journal
papers and textbooks only, as everywhere else in this document.

* **TET10 — 10-node quadratic tetrahedron (`fea.elements.tet10`, etype `"TET10"`).**
  Standard isoparametric quadratic tet (4 corner + 6 midside nodes) from the public
  textbook literature: Zienkiewicz, O.C., Taylor, R.L., Zhu, J.Z., *The Finite Element
  Method: Its Basis and Fundamentals*, 6th ed., Elsevier Butterworth-Heinemann, 2005
  (quadratic tetrahedra, shape functions and numerical integration); Bathe, K.-J.,
  *Finite Element Procedures*, Prentice Hall, 1996 (isoparametric solid formulation);
  Cook, R.D., Malkus, D.S., Plesha, M.E., Witt, R.J., *Concepts and Applications of
  Finite Element Analysis*, 4th ed., Wiley, 2002 (solid elements and quadrature rules) —
  all three already cited in §11–§12. Stiffness by the typical 4-point tetrahedral Gauss
  rule; consistent (or documented lumped) mass. Acceptance keeps the §9/§11 patch-test
  discipline (MacNeal–Harder): the quadratic basis contains every linear displacement
  field, so the constant-strain patch stays exact to roundoff, and a free-free single
  TET10 carries exactly 6 rigid-body modes (solid nodes carry 3 translational DOFs).
  Deliberately unchanged: the HEX8 Wilson–Taylor incompatible-modes default and its
  98.6% bending golden, and the §10 distortion caveat — the Simo–Rifai assumed-strain
  (EAS) upgrade stays **not implemented** in Round 10.
* **ZZ-SPR — superconvergent patch recovery (`fea.recover.recover_spr`).**
  Zienkiewicz, O.C., Zhu, J.Z., *The Superconvergent Patch Recovery and A Posteriori
  Error Estimates. Part 1: The Recovery Technique*, IJNME, 33(7), 1992, pp. 1331–1364 —
  cited in §12 as the known not-implemented upgrade to plain nodal averaging; Round 10
  implements it. A linear polynomial is fitted per nodal patch (the elements incident on
  a node) to stresses sampled at the superconvergent points — Barlow, J., *Optimal
  Stress Locations in Finite Element Models*, IJNME, 10(2), 1976, pp. 243–251 (§11):
  the centroids for the linear elements — then evaluated at the node. A constant-stress
  patch stays exact at every node (§9.1 discipline). For TET10 the brief allows the same
  centroid samples or skipping TET10 in SPR, documented either way by the implementing
  agent. `average_nodal` (§12) deliberately stays plain 1/n_adj averaging and is *not*
  SPR; the two coexist as distinct functions.
* **ERA — Eigensystem Realization Algorithm (`mpe.era.era`).** Juang, J.N., Pappa, R.S.,
  *An Eigensystem Realization Algorithm for Modal Parameter Identification and Model
  Reduction*, Journal of Guidance, Control, and Dynamics, 8(5), 1985, pp. 620–627.
  Minimal state-space realization from Markov parameters / impulse responses (or IRFs
  obtained from FRFs), in the Ho–Kalman lineage — Ho, B.L., Kalman, R.E., *Effective
  Construction of Linear State-Variable Models from Input/Output Functions*,
  Regelungstechnik, 14(12), 1966, pp. 545–548: block-Hankel matrix, SVD into
  observability/controllability factors, shift-invariance for `A`, then poles from
  `eig(A)` and mode shapes through the output matrix `C`. Returns the same
  `mpe.common.ModalParameterResult` container as LSCE (§6, Brown et al. 1979) and SSI
  (§6, Van Overschee–De Moor 1996); stabilization over a model-order range reuses
  `mpe.common.stabilization_diagram`. Acceptance per §9.5: synthetic 2-DOF data with
  known truth — frequencies within one spectral line, shape MAC > 0.99 — with the
  existing p-LSCF/SSI/LSCE goldens unchanged.
* **Residual flexibility as a public FRF correction
  (`dynamics.residuals.residual_flexibility`).** No new algorithm: the static
  residual-flexibility / upper-residual machinery is already cited in §5 — MacNeal
  (1971), Rubin (1975), and the residual-term treatment of Ewins, *Modal Testing*,
  2nd ed., 2000. Round 10 freezes a public function returning the residual-flexibility
  block (retained-mode content stripped) shaped for `modal_frf(..., upper_residual=...)`,
  over the `residual_vectors` machinery merged in Round 1; the
  `ResidualVectorResult.residual_flexibility` attribute is not renamed. This is the
  compensation route the §10 truncated-FRF caveat already points to; the contractual
  20-mode 5% statement of `docs/CONTRACT_API.md` itself is unchanged, as are the
  Rubin 0.028% and 20-mode FRF goldens.
* **SEREP-expanded MAC (`correlation.expansion.expanded_mac`).** A composition,
  explicitly *not* a new correlation metric: `expand_serep` (§4, O'Callahan–Avitabile–
  Riemer 1989) followed by `mac_matrix` (§1, Allemang–Brown 1982), with the numerics of
  both wrapped functions byte-for-byte unchanged. The gate exploits the SEREP property
  that expansion is exact at retained modes: expanding an FE mode set onto *itself*
  through a master subset must return an identity MAC (diagonal 1, off-diagonal ~0) for
  the retained modes.
* **CTETRA10 kept + punch `$STRESSES` — text I/O only, no new algorithm.**
  `io.bdf.read_bdf` maps a 10-node `CTETRA` to the new TET10 with all 10 node ids and
  `write_bdf` emits it back; a 4-node CTETRA stays TET4, and **HEX20 still warns and
  drops to HEX8** (§10). `io.pch.read_pch_stress` parses public punch `$STRESSES` /
  `$ELEMENT STRESSES` text blocks (80-column punch, the same conventions as `read_pch`
  and the §13 `$DISPLACEMENTS` static reader), skipping other block types the same
  tolerant way; tests stub executables and never require a Nastran install. Everything
  is publicly documented card/punch *text*, like every other femtools translator —
  **still no OP2, no RST, no ODB** (§10 and the product map N/A rows stand).

## Non-infringement note

References above are public conference papers, journal articles, and textbooks. No text,
figures, code, or file-format documentation from commercial FEMtools (Dynamic Design
Solutions) or any other proprietary product is reproduced here or in the implementation.
UNV and Nastran BDF support is implemented from publicly documented dataset/card layouts.

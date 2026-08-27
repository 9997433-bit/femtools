# Pretest algorithms — target modes, sensor selection, mass loading

Spec for `femtools.pretest` (owner: R1-O3). Frozen entry points:

```python
from femtools.pretest.target_modes import effective_mass, select_target_modes
from femtools.pretest.efi import effective_independence
from femtools.pretest.sensor import eliminate_by_mac, nodal_kinetic_energy
```

Workflow this module serves: pick *target modes* worth measuring → pick *sensor DOFs* that make
those modes distinguishable (well-conditioned $\Phi_s$, low off-diagonal AutoMAC) → check the
instrumentation does not perturb the structure (mass loading). Candidate sets are
`(node_id, dof_code)` lists resolved through `ModalResult.dof_map`.

## 1. Effective modal mass — `effective_mass(modal, model) -> EffMassResult`

Rigid-body influence matrix $R \in \mathbb{R}^{n \times 6}$ about reference point $x_0$
(default: model CoG): for a node at position $r_k$ (relative to $x_0$), its 6-DOF block is

$$R_k = \begin{bmatrix} I_3 & -\tilde r_k \\ 0 & I_3 \end{bmatrix},
\qquad \tilde r = \begin{bmatrix} 0 & -r_z & r_y \\ r_z & 0 & -r_x \\ -r_y & r_x & 0 \end{bmatrix},$$

rows restricted to active DOFs. Modal participation factors and effective mass, with
mass-normalized $\Phi$:

$$L = \Phi^\top M R \in \mathbb{R}^{m \times 6},\qquad
  m_{\mathrm{eff}, r d} = L_{rd}^2 .$$

Completeness identity (golden test):
$\sum_{r=1}^{\infty} L_r L_r^\top = R^\top M R = M_{rb}$, the 6×6 rigid-body mass matrix
(total mass on translational diagonal, inertia about $x_0$ on rotational). Report per-mode
fractions $m_{\mathrm{eff},rd} / (M_{rb})_{dd}$ and the *missing mass*
$M_{rb} - \sum_{r=1}^{m} L_r L_r^\top$ as the truncation indicator.

`select_target_modes(modal, model, threshold=0.05, band_hz=None)` returns modes whose effective
mass fraction exceeds `threshold` in any direction (∪ any mode inside `band_hz`). Classical
cantilever check: transverse fractions ≈ 61.3 %, 18.8 %, 6.5 % for modes 1–3
(`docs/ACCEPTANCE.md` §5).

Pitfalls: effective mass ranks *global, base-excitation-relevant* modes — local modes
(brackets, panels) score ≈ 0 yet may be test-critical: `select_target_modes` must accept a
manual include list. Rotational effective masses depend on $x_0$; report $x_0$ with the result.
For free–free models the 6 rigid-body modes carry ~100 % effective mass — exclude
$f < f_{rb,tol}$ modes from the ranking.

## 2. Effective Independence — `effective_independence(phi, n_sensors) -> EfiResult`

Kammer's EFI. Sensor-candidate shape matrix $\Phi_s \in \mathbb{R}^{n_s \times m}$ (rows =
candidate DOFs, columns = target modes). Fisher information matrix and per-sensor leverage:

$$Q = \Phi_s^\top \Phi_s,\qquad
  E_D = \mathrm{diag}\!\left( \Phi_s\, Q^{-1} \Phi_s^\top \right) \in [0, 1]^{n_s},\qquad
  \textstyle\sum_j E_{D,j} = m .$$

$E_{D,j}$ is the fractional contribution of sensor $j$ to the rank/determinant of $Q$
(deleting sensor $j$ scales $\det Q$ by exactly $1 - E_{D,j}$ — rank-one downdate identity).
Backward elimination:

```
while n_current > n_sensors:
    compute E_D                    # solve Q X = Phi_s.T; E_D = rowsum(Phi_s * X.T)
    drop argmin(E_D)               # ties: drop the one with smaller row norm
```

Never form $Q^{-1}$ explicitly: the implementation computes $E_D$ from a thin SVD
($E_{D,j} = \lVert U_{j,:} \rVert^2$), which avoids squaring the condition number. Cost per
sweep $O(n_s m^2 + m^3)$, total $O((n_s - n_{target})(n_s m^2))$ — trivial for realistic
sizes. EFI is the greedy maximizer of $\det Q$ (D-optimality); the implementation records
the minimum retained leverage per step rather than a log-det trace.

```python
# implemented API (femtools.pretest.efi)
effective_independence(phi, n_sensors, *, candidate_dofs=None, mass=None,
                       freq_hz=None, method="efi" | "efi-dpr", keep=None) -> EFIResult

class EFIResult:
    dofs: np.ndarray           # selected candidate ids, ranked by final E_D desc
    selected: np.ndarray       #   (alias of dofs)
    efi: np.ndarray            # final E_D of the kept set, same order
    index: np.ndarray          # row positions of the kept candidates in phi
    ranking, removed           # all candidates by elimination order / dropped ids
    ed_initial: np.ndarray     # E_D of the full candidate set before elimination
    history: list[tuple]       # (n_remaining, min E_D) per elimination step
```

Pitfalls: (i) stopping below $n_{sensors} = m$ makes $Q$ singular — hard error; (ii) EFI
*clusters* sensors on high-amplitude regions and ignores redundancy against sensor failure —
offer the kinetic-energy-weighted variant (EVP: rank by $E_{D,j} \times \mathrm{NKE}_j$, §4)
and a minimum mutual distance option; (iii) candidate DOFs with near-zero response across all
targets destabilize $Q^{-1}$ early — prefilter rows with
$\lVert \phi_j \rVert < 10^{-6} \max_j \lVert \phi_j \rVert$; (iv) triaxial hardware: eliminate
per *node triplet* (sum the three DOF leverages) when `group_triax=True`.

## 3. MAC-based elimination — `eliminate_by_mac(phi, n_sensors, ...)`

Objective: keep the AutoMAC of the retained sensor set close to identity (distinguishability —
the quantity the contract's acceptance row checks: off-diag < 0.15 on the toy case). Greedy
backward elimination:

```
while n_current > n_sensors:
    for each remaining sensor j:            # tentative removal
        score_j = max offdiag AutoMAC(Phi_s without row j)
    remove argmin_j score_j
```

Naive cost $O(n_s^2 m^2 n)$ per sweep; cheap enough with rank-one Gram updates
($G = \Phi_s^\top \Phi_s$ and per-column norms downdated by the removed row:
$G \leftarrow G - \phi_j \phi_j^\top$), giving $O(n_s m^2)$ per sweep.

```python
eliminate_by_mac(phi, n_sensors, max_offdiag=None) -> MacElimResult
# MacElimResult: selected (kept indices), removal_order, max_offdiag (final)
```

Fail loudly (raise) if the final value exceeds a given `max_offdiag`.
EFI and MAC-elimination are complementary (D-optimality vs pairwise distinguishability);
FEMtools-class workflows run EFI then verify with AutoMAC — mirror that in examples.

## 4. Nodal kinetic energy — `nodal_kinetic_energy(modal, model) -> (n_dof, m)`

Per DOF $j$ and mode $r$ (mass-normalized shapes):

$$\mathrm{NKE}_{jr} = \phi_{jr} \left( M \phi_r \right)_j,
\qquad \sum_j \mathrm{NKE}_{jr} = 1 .$$

Use the consistent $M$ (off-diagonal terms can make individual entries slightly negative —
that is physical bookkeeping, sum per node before ranking; with a lumped $M$,
$\mathrm{NKE}_{jr} = m_{jj} \phi_{jr}^2 \ge 0$). Aggregations: per node (sum over its DOFs),
per mode, or total ($\sum_r$, optionally weighted). Uses: sensor ranking robust to mass
distribution (unlike raw $|\phi|$), exciter placement (high NKE = good controllability of that
mode), EVP weighting for EFI (§2). Cost: one sparse mat-mat, $O(nnz(M)\, m)$.

## 5. Mass loading of instrumentation

Accelerometer of mass $m_a$ attached at node $k$: first-order Rayleigh-quotient sensitivity with
mass-normalized shapes,

$$\frac{\Delta f_r}{f_r} \approx -\tfrac12\, m_a \sum_{d \in \{x,y,z\}} \phi_{(k,d),r}^2 ,$$

i.e. worst at high-displacement points of high modes — exactly where sensors go. Screening rule:
flag any mode with predicted $|\Delta f / f|$ over `tol` (default $0.1\%$) for the summed sensor
set $\sum_a m_a \phi^2$. Verification path (no new API): attach `MASS` elements
(`model.add_element(type="MASS", ...)`, see `fea.md` §2.6) at the selected sensors and re-run
`solve_modes`; compare with `correlation.pair_modes`. Pitfalls: the first-order formula ignores
mode-shape rotation (adequate below ~1 % shifts, not beyond); cable/roving-mass effects during
the test differ from the static prediction — recommend the re-solve check in every pretest
report; base-mounted (seismic) accelerometer masses on the fixture side do not load the
structure — only count sensors on flexible DOFs.

## 6. Complexity summary

| Kernel | Cost |
|---|---|
| `effective_mass` | $O(nnz(M) \cdot 6 + n m)$ |
| `effective_independence` | $O((n_s - n_{tgt}) \cdot n_s m^2)$ |
| `eliminate_by_mac` | $O((n_s - n_{tgt}) \cdot n_s m^2)$ with Gram downdates |
| `nodal_kinetic_energy` | $O(nnz(M)\, m)$ |
| Mass-loading screen | $O(n_{sens} m)$ (+1 modal re-solve for verification) |

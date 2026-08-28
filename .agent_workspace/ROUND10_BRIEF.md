# Round 10 任务简报（Cycle D 第一轮）

## Close-out

All ten Round-10 agents delivered. Kernels and tests are merged on
`cursor/femtools-cycle-d-d551`. Parent glue: top-level `_EXPORTS` (149 names:
143 + `tet10`, `recover_spr`, `read_pch_stress`, `era`, `expanded_mac`,
`residual_flexibility`), PRODUCT_MAP R10-wip → R10, ACCEPTANCE Round-10 rows
measured. Goldens unchanged (HEX8 98.6%, Rubin 0.028%, H1/H2=γ², tilted-shell
6 RBM, 10% E recovery). DAQ / OP2 / RST / ODB remain N/A. HEX20 still drops.

The per-agent briefs below are historical.

---

Base: `cursor/femtools-cycle-d-d551` from `origin/main` @ Round 9 close
(`f30108d`, `_EXPORTS` 143, pytest 481/3, 13/13 examples).
Do not regress goldens (HEX8 98.6%, Rubin 0.028%, H1/H2=γ², tilted-shell 6 RBM,
displacement-driven 10% E recovery ~1e-9).
No DAQ, no OP2/RST/ODB, no EAS-30, no CAD, no copyrighted manuals.

Frozen names: `.agent_workspace/REMAINING.md` (Round 10 section).
File ownership: `.agent_workspace/FILE_OWNERSHIP.md`.

Parent seed (already on this branch): `core.model.ELEMENT_NODE_COUNTS` includes
`"TET10": (10,)` and `_ELEMENT_NEEDS_PROPERTY` lists `TET10`. Do **not** rewrite
`RBE2`/`RBE3` dataclasses. Do **not** edit `femtools/__init__.py` `_EXPORTS`
(parent glue after all ten land).

## R10-O1 — TET10 + ZZ-SPR (cloud)

Register a 10-node quadratic tetrahedron and freeze superconvergent patch recovery.

```python
from femtools.fea.elements import tet10   # also registered as etype "TET10"
from femtools.fea.recover import recover_spr
```

TET10 (required):
- Shape functions of the 10-node tet (4 corners + 6 midsides). Public textbooks:
  Zienkiewicz & Taylor / Bathe / Cook isoparametric solids. Typical 4-point tet
  quadrature for stiffness; consistent or well-documented lumped mass.
- Constant-strain patch remains exact (quadratic contains linear). Free-free
  single TET10: exactly 6 rigid-body modes (solids carry 3 translational DOFs).
- `recover_stress` / `recover_strain` at the element centroid (or averaged
  Gauss) for TET10. Add TET10 to `verification.PATCH_TYPES` and a patch mesh
  builder so existing parametrized patch tests cover it.
- Import `tet10` from `fea.elements` so the registry is populated. Aliases
  (`CTETRA10`, `C3D10`) ok if cheap.
- Do **not** change HEX8 default (Wilson–Taylor incompatible modes). Do **not**
  implement EAS-30. HEX8 98.6%, MITC4, drilling 6 RBM, `solve_static(enforced=)`
  stay. `average_nodal` stays 1/n_adj, not SPR.

`recover_spr` (required): Zienkiewicz, O.C., Zhu, J.Z., *The superconvergent
patch recovery and a posteriori error estimates. Part 1: The recovery
technique*, IJNME 33(7), 1992, pp. 1331–1364. For linear elements the
superconvergent samples are the centroids (Barlow). Fit a linear polynomial
over the patch of elements incident on a node; evaluate at the node.
Constant-stress patch stays exact at every node. Distinct from `average_nodal`.
TET10 SPR may use the same centroid samples or skip TET10 in SPR (document it).

Gates (`tests/test_round10_o1.py`):
- TET10 constant-strain patch ≤ 1e-12 (stress and strain)
- free-free TET10 (or a small tet mesh): exactly 6 RBM
- `recover_spr` constant-stress patch exact at nodes (BAR2/TET4/HEX8 enough)
- HEX8 bending ratio still ≥ 0.98 (do not retune); 6 RBM on tilted shells

Do **not** edit `io/**`, `core.model` RBE dataclasses, or `femtools/__init__.py`.
You may edit `fea/elements/__init__.py` and `fea/verification.py`.

## R10-F2 — CTETRA10 + punch stresses (cloud)

```python
from femtools.io.bdf import read_bdf, write_bdf
from femtools.io.pch import read_pch_stress
```

BDF: 10-node `CTETRA` → `type="TET10"` with all 10 node ids (parent already
seeded `ELEMENT_NODE_COUNTS["TET10"]`). `write_bdf` emits 10-node CTETRA for
TET10. **HEX20 still warn+drop to HEX8** (aggregated warning). 4-node CTETRA
stays TET4. If you also own CDB/K tet10 cards: keep 10 nodes as TET10 the same
way; HEX20 still drops. Do not invent copyrighted decks. Public card layouts
only.

`read_pch_stress`: parse public punch **`$STRESSES`** / `$ELEMENT STRESSES`
text blocks (80-column punch, same conventions as `read_pch` / `read_pch_static`)
into a small container (element ids + Voigt tensors, or a `StressResult` if
that type already fits). Skip eigenvector / `$DISPLACEMENTS` the same tolerant
way. **No OP2.** Tests must not require Nastran. Stub like R7/R9.

UNV dataset 2414 (analysis data at elements/nodes — public UFF) is optional if
cheap. Do not break datasets 55/58/2412/30000.

Tests: `tests/test_round10_io.py`. Keep `test_round7_io.py` / `test_round8_io.py`
/ `test_round9_io.py` green. Existing `test_round6_io.py` TET10-drop assertions
are **G1's** to update — do not edit that file.

Do **not** edit `femtools/__init__.py`. Do not implement TET10 stiffness (O1).

## R10-O4 — Juang–Pappa ERA (cloud)

```python
from femtools.mpe.era import era
```

Eigensystem Realization Algorithm: Juang, J.N., Pappa, R.S., *An Eigensystem
Realization Algorithm for Modal Parameter Identification and Model Reduction*,
J. Guidance, Control, and Dynamics, 8(5), 1985, pp. 620–627.

Inputs: Markov parameters / impulse responses, or IRFs from `irf_from_frf`.
Reuse `mpe.ssi.block_hankel` if it fits; SVD → observability/controllability →
`(A, B, C)` → poles from `eig(A)`, shapes from `C ψ`. Return
`mpe.common.ModalParameterResult` (same container as LSCE/SSI). Stabilization
over a model-order range is welcome but a single-order path is enough for the
gate.

Gates (`tests/test_round10_o4.py`):
- synthetic 2-DOF (use `mpe.synthetic` / a known 2-DOF IRF): identified
  frequencies within one spectral-line `df` of truth; MAC of recovered shapes
  vs truth > 0.99
- existing `poly_lscf` / `ssi_data` / `lsce` / `parameter_covariance` /
  topometry / 10% modal-E goldens unchanged

Do not edit `fea/**` or `io/**`. Do not edit `femtools/__init__.py`.

## R10-O2 — residual flexibility for FRF (local, no git)

```python
from femtools.dynamics.residuals import residual_flexibility
```

Public function returning the static residual-flexibility **block** suitable as
`modal_frf(..., upper_residual=...)`. `residual_vectors` already computes
`ResidualVectorResult.residual_flexibility`; freeze a function that returns the
`(n_out, n_in)` (or `(ndof, n_force)`) matrix after stripping retained-mode
content (MacNeal / Ewins upper residual). Do not rename the attribute; add the
function.

Gate: few retained modes + `upper_residual=residual_flexibility(...)` lowers
relative L2 vs `direct_frf` compared with the same truncated `modal_frf`
without the residual. Do not change the 20-mode 5% FRF golden or Rubin 0.028%.

Tests: `tests/test_round10_o2.py` (exclusive). Do not edit `tests/test_frf.py`.

Do **not** run git. Do not edit `femtools/__init__.py`.

## R10-O3 — expanded MAC (local, no git)

```python
from femtools.correlation.expansion import expanded_mac
```

Compose `expand_serep` + `mac_matrix`. Expanding an FE mode set onto **itself**
through a master subset must yield a MAC that is the identity (diagonal 1,
off-diagonal ~0) for the retained modes.

```python
# FE modes restricted to masters, expanded with SEREP using the same Phi
# → mac_matrix(expanded, original_fe_modes) ≈ I
```

Return a small result type (MAC table + ExpansionResult) or `(mac, expansion)`.
Do not change `expand_serep` / `expand_guyan` numerics. Guyan 0.028% Rubin is
not yours; do not touch dynamics.

Tests: `tests/test_round10_o3.py` (exclusive). Do not edit `tests/test_mac.py`.

Do **not** run git. Do not edit `femtools/__init__.py`.

## R10-F4 — CLI / script / GUI / viz (local, no git)

CLI (lazy-fail like `recover-stress` when the kernel is missing):
- `dump-psd` / `load-psd` over existing `dump_psd`/`load_psd` (R9 kernel)
- `era` wrapping `mpe.era.era` when it imports
- `recover-spr` wrapping `recover_spr` when it imports
- `expanded-mac` optional if cheap

Script verbs for the same. GUI: a small ERA / SPR / expanded-MAC surface if
the page already has a pattern for MAC/stress; do not require new JS frameworks.

Keep matplotlib default; `import femtools.viz` never requires pyvista.
Do not commit `*.png`. Do not run git. Do not edit `femtools/__init__.py`.

## R10-F1 — PRODUCT_MAP / SOTA / ARCHITECTURE / README (local, no git)

Tag new Round-10 rows **R10-wip** (parent glue retags R10-wip → R10).
Do **not** add unlanded names to `_EXPORTS` / `__all__`. Docstring may mention
Round 10 as in progress.

Update `docs/PRODUCT_MAP.md` (TET10 in the element-library row or a new row;
ZZ-SPR; ERA; expanded MAC; residual_flexibility function; CTETRA10; punch
stress). SOTA: new Cycle D section with Juang–Pappa 1985, Zienkiewicz–Zhu 1992,
quadratic tet textbooks; keep §10 caveats that still apply (HEX8 distortion /
EAS-30, `bbar`, truncated FRF 20-mode statement, UNV 30000, HEX20 drop, mypy
non-blocking). TET10 midside-drop caveat is **closed for CTETRA10** once F2
lands — mark it R10-wip, not closed.

Do not run git.

## R10-F3 — algorithms / ACCEPTANCE / examples (local, no git)

New example(s), e.g. `examples/tet10_patch.py` and/or `examples/era_2dof.py`,
that print a `PASS`/`FAIL` line like the existing 13. Do not change measured
R7–R9 ACCEPTANCE numbers. Add Round-10 rows with the gates (patch 1e-12, ERA
df/MAC, expanded MAC identity, residual-flexibility L2 improvement) as
unchecked until parent measures.

Update `docs/algorithms/**` (fea TET10/SPR, mpe ERA, correlation expanded MAC,
dynamics residual_flexibility, io CTETRA10 / `$STRESSES`).

Do not run git. Do not commit pngs.

## R10-G1 — golden tests (local, no git)

Pin goldens. Do **not** create `tests/test_round10_*.py`.

If TET10 is first-class in `ELEMENT_NODE_COUNTS` and F2 keeps 10-node CTETRA:
update `tests/test_round6_io.py` (and any sibling) that currently asserts
`TET10 -> TET4` drop so BDF/CDB keep TET10; HEX20 drop stays. If F2 has not
landed yet, skip or xfail that one assertion with a comment — do not fail
the suite on a missing kernel.

Add `tests/test_expanded_mac.py` and/or `tests/test_residual_flexibility.py`
**only if** those modules import on this tree; otherwise leave a skip.

HEX8 98.6%, Rubin 0.028%, H1/H2=γ², 6 RBM, 10% E must stay green.
Do not run git.

## R10-G2 — benchmarks / probe / scripts (local, no git)

`scripts/probe_boundaries.py`: skip missing kernels (TET10 assemble, `era`,
`recover_spr`, `expanded_mac`, `residual_flexibility`, `read_pch_stress`).
When present, probe: TET10 patch, ERA 2-DOF, expanded MAC identity, residual
L2 improvement, CTETRA10 round-trip. Do not require Nastran/ANSYS/Abaqus.
Benchmarks: optional TET10 vs TET4 timing, non-gating.

Do not run git.

MODEL_SLUG: claude-opus-5-thinking-high-fast

# R11-O5F — Round 11 alignment (C bucket + acceptance test debt)

Branch `cursor/r11-o5f-align-d551`, based on `cursor/r11-align-docs-d551` @ `6673ece`
(i.e. `main` @ `6fb7b4e` plus the alignment brief). No PR opened, no merge.

Three commits:

| commit | contents |
|---|---|
| `bf34b2f` | C1/C2/C4/C5/C6 — `docs/PRODUCT_MAP.md`, `docs/SOTA.md`, `docs/algorithms/fea.md`, `README.md` |
| `00f479b` | C3 + case 11 + `tests/test_round11_o5f_acceptance.py` (cases 3b, 5, 11, 13, 15, 16) |
| `a82b799` | Round-11 status block in `docs/ACCEPTANCE.md` |

**No file under `src/` was touched.** `git diff origin/main --stat` lists only
`README.md`, `docs/*`, `tests/test_round11_o5f_acceptance.py` and the two
`.agent_workspace` files inherited from the base branch.

## 1. C-bucket items — all seven done

**C1 — PRODUCT_MAP cross-cutting section.** The paragraph said `_EXPORTS` held
"143 stable names" and that the six Round-10 frozen names were "**not** in
`_EXPORTS`/`__all__` yet: the dict stays at 143 stable names until the parent glue
confirms the Round-10 merge and promotes them". That whole clause is deleted. The
count now reads 149, and the sentence in its place says Round 10 added `tet10`,
`recover_spr`, `read_pch_stress`, `era`, `expanded_mac` and `residual_flexibility`
"once those modules merged, taking the dict from 143 to its current 149 stable names",
followed by "CI resolves every `__all__` entry, so the count is checked, not asserted."

**C2 — legend, transitional tag.** "The transitional **R10** tag is retired" →
"The transitional **R10-wip** tag is retired", matching the intact R4-wip / R7-wip /
R9-wip sentences. The **R10** merged-status row in the legend table is untouched.

**C4 — R5+ legend row.** Marked retired rather than deleted, because "R5+" is still
referenced in running prose in three places (PRODUCT_MAP legend paragraph, the
cross-cutting section, `docs/SOTA.md` §10) and a reader hitting those needs the key.
The row now reads *"Retired after Round 6 — direction fixed, API not frozen in that
cycle. Round 6 landed every R5+ row, so no row in this file carries the tag any more;
it is kept here only to read the prose above and `docs/SOTA.md` §10."*

**C5 — CLI rows.** The Round-1 "Command-line surface" row gains `gui` and now says it
lists "17 of the **22** typer commands; the other 5 are the Round-10 row below"; the
Round-10 CLI row closes with "these 5 complete the 22-command typer app". Verified
against the live app:

```
22 ['dump-frf', 'dump-psd', 'era', 'estimate-frf', 'expanded-mac', 'frf', 'gui',
    'load-frf', 'load-psd', 'mac', 'plot-stress', 'pretest', 'read-mesh',
    'recover-spr', 'recover-stress', 'reduce', 'report-mac', 'script',
    'solve-modes', 'update', 'update-static', 'write-mesh']
```

`gui` landed in Round 1 (`git log -S '@app.command("gui")'` → `2d695a2`, the Round-1
integration commit), so it belongs on the R1 row, not on a later one.

**C6 — historical element lists.** Cross-references added, history left alone; none of
these paragraphs now claims TET10 existed in Round 1 or Round 7.

| location | added |
|---|---|
| PRODUCT_MAP element-library R1 row | "The quadratic solid TET10 is the Round-10 row below (`docs/SOTA.md` §14); this row is the Round-1 linear library" |
| PRODUCT_MAP recovery R7 row | "…and TET10 recovery is the Round-10 row below (`docs/SOTA.md` §14)" |
| `docs/SOTA.md` §11 | "the Round-7 element list; TET10 joined it in Round 10, see §14" |
| `docs/algorithms/fea.md` §10 | "That element list is Round 7's. TET10 joined the dispatch table in Round 10 — see §14 for the element, §15 for `recover_spr`, and `docs/SOTA.md` §14 for the SPR caveat." |
| `docs/algorithms/fea.md` §2.6 | "The quadratic answer to those caveats, TET10, is the Round-10 element of §14; this section is the Round-1 linear library." |

**C3 — ACCEPTANCE orphan sentence.** The dangling `status block until the parent
measures the merged tree — constructions in §12.` at L373 is gone. Rather than drop the
`§12` pointer with it, the surviving sentence now ends "…parent-measured on the merged
Cycle-D tree (status block above) — constructions in §12", which is what the Round-10
edit was evidently reaching for.

**Case 11 — checked.** Now `[x]`, pointing first at `examples/update_static.py`, whose
tip deflection versus $FL^3/3EI$ I re-ran on this tree: **2.31e-13** (the example's own
tolerance is 1e-9). I also added assertions for *both* halves of the row, since the
master table asks for the bar tip as well as the cantilever tip — BAR2 chain versus
$FL/EA$ measures **0.0** (bit-exact) and the 8-element BEAM2 tip versus $FL^3/3EI$
measures **1.8e-13**, both at the row's 1e-12 tolerance.

## 2. Optional E-bucket tests — all five done, plus two extras

`tests/test_round11_o5f_acceptance.py`, 298 lines, 9 tests, all green. Every
construction is the one already written down in ACCEPTANCE (§5, §6, §8), public
formulas only, `default_rng` with a fixed seed wherever anything is stochastic. **No
kernel numerics were changed to make anything pass** — every test passed against the
kernels as they are, first run except for one fixture problem of my own (below).

| case | test | measured | tolerance |
|---|---|---|---|
| 3b stiffness orthogonality | 16×BEAM2, 6 modes, $\Phi^\top K \Phi$ vs $\Lambda$ | diag rel **3.2e-12**, off-diag **9.0e-15** of $\max\lambda$ | 1e-6 |
| 5 effective mass | 20×BEAM2 transverse fractions **0.61308 / 0.18830 / 0.06473** vs the §5 table 0.6131 / 0.1883 / 0.0647 | max abs dev **3.2e-5** | 1e-2 |
| 5 completeness | $\sum_r L_r L_r^\top$ vs $R^\top M R$, complete free–free rod mode set | rel **2.8e-16** | 1e-6 |
| 11 static, bar | BAR2 chain tip vs $FL/EA$ | **0.0** | 1e-12 |
| 11 static, beam | 8×BEAM2 tip vs $FL^3/3EI$ | **1.8e-13** | 1e-12 |
| 13 force ID | noiseless 2-DOF, 33 lines, seeded complex $F_{true}$, $X = HF_{true}$ | rel **1.1e-15** | 1e-8 |
| 15 RBPE | §8 block on 6 soft springs, 24 sensors / 6 drives, 6–15 Hz | mass rel **1.2e-15**, CoG **2.8e-16**, inertia **6.1e-15**; $J_G$ SPD + triangle inequalities hold | 1e-8 |
| 15 RBPE control | the same data with `mount_k` omitted | mass **15.99** kg vs 10.0 — the apparent negative mass $K/\omega^2$ | > 1e-3 off |
| 16 FDD | seeded 2-DOF white-noise record (5 / 13 Hz, ζ = 1 %, 6 ch, 256 Hz, 600 s), Welch `nperseg` 2048, $df$ = 0.125 Hz | peaks **4.99927 / 12.97986** Hz, $\lvert\Delta f\rvert$ **7.3e-4** / **2.0e-2** Hz; shape MAC **0.99999** / **0.99984** | $\le df$; MAC > 0.99 |

The two extras beyond the requested five are the RBPE `mount_k` control (it is what
makes the 1e-8 headline meaningful — without the suspension correction the same data
reads 60 % high) and the effective-mass completeness identity, which ACCEPTANCE §5
asks for alongside the fraction table.

One modelling note worth recording, because it looks like a kernel defect and is not.
The completeness identity $\sum_r L_r L_r^\top = R^\top M R$ can only be exercised on an
**unconstrained** model: the right-hand side counts the mass sitting on the clamped
DOFs, which the mode set cannot span, so on a cantilever it misses by ~7 % by
construction. It also cannot be exercised on a beam at all, because BEAM2 carries no
rotary inertia about its own axis — `solve_modes(n_modes=30)` on a 4-element cantilever
returns 24 finite modes, the other 6 DOFs being massless. The test therefore uses a
free–free BAR2 rod whose axial DOFs are the complete free set, and the identity then
closes to 2.8e-16. The docstring says so, so the next reader does not re-derive it.

## 3. Verification

| check | result |
|---|---|
| `python3 -c "from femtools import _EXPORTS; print(len(_EXPORTS))"` | **149** |
| `ruff check .` (whole tree, not just touched files) | **All checks passed** |
| `pytest` | **599 passed / 3 skipped** (was 590/3; +9 new) |
| `examples/*.py` | **15/15 PASS** |
| `scripts/probe_boundaries.py` | **39/39 pass** |

Goldens explicitly re-confirmed, none regressed: HEX8 one-through-thickness bending
ratio **0.9855** with incompatible modes and the distorted patch at 5.0e-16
(`tests/test_hex8_verification.py`, 3 tests green); free-interface CMS
`examples/cms_rubin.py` 3/3 with rubin max rel err 1.85e-07, macneal 7.71e-05,
craig_bampton 2.93e-05; `examples/h1_ssi.py` 16/16 (H1/H2 = γ² identity and SSI-cov);
6 rigid-body modes in the shell, HEX8 and TET10 suites; 10 % E recovery in both
`examples/update_youngs.py` and `examples/update_static.py`.

## 4. Scope discipline

Nothing was added from the forbidden list: no DAQ, no OP2/RST/ODB, no EAS-30, no CAD,
no HEX20, no UNV 2414, no CMIF. `src/femtools/__init__.py` was not opened.

Two edits sit just outside the literal seven C-bucket items; both are the same class of
error as C1 (a merged Round-10 capability still described as pending) and are called out
here so the parent can drop them if unwanted:

1. `README.md` line 42 still said "a 10-node quadratic TET10 is Round-10 WIP" and
   "ZZ-SPR recovery is Round-10 WIP". Both are merged; the WIP wording is gone and SPR
   is now named in the `fea` row.
2. `README.md`'s `cli` row listed 14 commands with no count and its `mpe` row omitted
   ERA. The CLI row now gives the count (22) and points at `femtools --help`; ERA is
   named in the `mpe` row.

## 5. Still open (not in this round's scope)

Acceptance rows with merged kernels and no numerical test remain: **1b** (bar discrete
dispersion), **1c** (bar mesh convergence), **7a** (SDOF modal FRF closed form),
**9** (free–free *beam* — the 6-RBM half is covered several times over by the shell,
HEX8 and TET10 suites, the elastic free–free Euler–Bernoulli roots are not),
**10** (Craig–Bampton exactness with *all* interior modes retained — `cms_rubin.py` has
a 2.93e-5 baseline at 8 retained modes, which is not the identity the row asks for),
**12** (Newmark period-error bound, half-covered by the round-5 Newmark-vs-exact
comparison), and the full 60×20 MBB SIMP criteria. These are listed in the new
Round-11 status block in `docs/ACCEPTANCE.md` so they do not get lost.

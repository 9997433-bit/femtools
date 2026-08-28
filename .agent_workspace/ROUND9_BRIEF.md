# Round 9 任务简报（Cycle C 第三轮 — 打磨收口）

Base: `cursor/femtools-cycle-c-d551` (Round 8 closed, pytest 397/3, `_EXPORTS` 138).
Do not regress goldens (HEX8 98.6%, Rubin 0.028%, H1/H2=γ², tilted-shell 6 RBM, 10% E recovery).
No DAQ, no OP2/RST/ODB.

Frozen names: `.agent_workspace/REMAINING.md` (Round 9 section).
File ownership: `.agent_workspace/FILE_OWNERSHIP.md`.

Round 9 is the Cycle-C close-out: promote two already-landed kernels to the frozen
contract, add one mapped-MAC convenience, thicken CLI/script/GUI, and keep every
example green. Do **not** reopen N/A rows. Do **not** invent ZZ-SPR, EAS-30, or
binary result parsers.

Parent glue after all ten land: top-level `_EXPORTS` for the frozen names,
PRODUCT_MAP R9-wip → R9, ACCEPTANCE measured, then PR #3 marked ready.

## R9-O1 — `apply_mpc` contract (cloud)

`apply_mpc` already exists in `fea.mpc` and is what `assemble_km` uses. Freeze it
as the public composer:

```python
from femtools.fea.mpc import apply_mpc
```

Do not rewrite `RBE2`/`RBE3` dataclasses (`core.model`). Do not change RBE2
kinematics or RBE3 weighted-average content.

Gates (pin in `tests/test_round9_o1.py`):
- empty `rbe2`+`rbe3` → identity transform (or no-op equivalent)
- empty `rbe3` → `apply_rbe2` G **bit-identical**
- empty `rbe2` → `apply_rbe3` G **bit-identical**
- mixed RBE2 hanging off an RBE3 reference (or the reverse) still **exactly 6**
  free–free rigid-body modes
- overlapping dependent DOF raises
- `assemble_km(..., mpc=False)` still disables both tables
- HEX8 98.6%, MITC4, drilling 6 RBM, `solve_static(enforced=)` stay

Do **not** edit `femtools/__init__.py`.

## R9-F2 — SOL 101 static punch (cloud)

`NastranPunchDriver` is SOL 103 only; `read_pch` returns `ModalResult` and skips
`$DISPLACEMENTS`. Add a **text** static path:

- `read_pch` (or a sibling `read_pch_static`) parses public punch `$DISPLACEMENTS`
  into a `StaticResult` (or a small vector+ids container the driver returns).
- `NastranPunchDriver.write_input(..., sol=101)` (or an equivalent explicit
  static method) emits a public SOL 101 case control requesting
  `DISPLACEMENT(PUNCH)=ALL`. Default remains SOL 103 — do not break R7 tests.
- `read_static` on the driver reads that punch. Missing executable / nonzero /
  timeout still `SolverError`. **No Nastran binary in tests.** Stub like R7.
- OP2 stays N/A.

Tests: `tests/test_round9_io.py`. Keep `test_round7_io.py` / `test_round8_io.py` green.
Do **not** edit `femtools/__init__.py`.

## R9-O4 — `static_stress_response` contract (cloud)

`static_stress_response` already exists in `updating.responses`. Freeze the
import; pin a displacement-driven 10% E recovery from a **stress** residual
(enforced tip displacement or equivalent — dead-load σ=F/A is independent of E).

```python
from femtools.updating.responses import static_stress_response
```

Gates (`tests/test_round9_o4.py`):
- BAR2/HEX8 constant-stress patch: parameter recovered to ~1e-9 or better
  when the residual is displacement-driven
- `update_from_static` displacement path (R8, 4.4e-16) unchanged
- `parameter_covariance` / topometry / 10% modal-E goldens unchanged

Do not edit `fea/**` or `io/**`. Do not edit `femtools/__init__.py`.

## R9-O2 — PSD dump/load (local, no git)

```python
from femtools.dynamics.random import dump_psd, load_psd
```

npz dump/load of `PSDResult`, analogous to `dump_frf` / `dump_cms`. Auto-spectra
(or the stored `S` block) and `freq_hz` **bit-identical** after load. Do not
change `psd_response` / Miles / Rubin numerics.

## R9-O3 — mapped MAC convenience (local, no git)

```python
from femtools.correlation.dofmap import mapped_mac
```

One-call wrap of `map_nearest_nodes` + `mapped_mode_matrix` + `mac_matrix`.
Gate: two translated copies of the same block → mapped-MAC diagonal 1.
Do not change `mac_matrix` real-mode numerics or `map_nearest_nodes` distances.

## R9-F4 — CLI / script / GUI polish (local, no git)

- CLI: `femtools dump-frf` / `load-frf` (lazy-fail like `recover-stress` if the
  kernel is missing) and `femtools update-static` wrapping `update_from_static`.
- Script: `UPDATE STATIC` (and `DUMP FRF` if cheap). Do not break `SOLVE STATIC`,
  `SET`, `RECOVER STRESS`, `ADD RBE2`/`ADD RBE3`.
- GUI: the HTML page (`gui/page.py`) must **display** `GET /api/stress` (table
  of von Mises / components) after a static solve; keep 400 handling. Optionally
  include a stress plot in the plot list when a static result exists.
- `import femtools.viz` still must not require pyvista.

## R9-F1 — docs (local, no git)

PRODUCT_MAP rows **R9-wip** until symbols import on THIS tree. **Do not** add
`__all__` names for missing modules (CI resolves every export). `apply_mpc` and
`static_stress_response` already import — still leave top-level `_EXPORTS` to
parent glue. Update README module table (it still reads like Round 1). Ignore
example-generated `frf_synthesis.png`. Architecture: SOL 101 is still **text**
punch; still no OP2/RST/ODB.

## R9-F3 — notes + examples (local, no git)

Keep all 11 examples PASS. Add kernel-backed demos only if the names import:
`examples/update_static.py` (10% E from tip deflection) and/or
`examples/mapped_mac.py` (translated-grid MAC diagonal 1). Skip/`sys.exit(0)
from None` if import fails. Algorithm notes if needed. ACCEPTANCE.md: Round-7/8
rows stay measured; add Round-9 pending rows.

## R9-G1 — existing tests (local, no git)

Do **not** create `tests/test_round9_*.py`. Pin `apply_mpc` / `static_stress_response`
(both already import). Pin RBE3 validation, H1/H2, CDB, RBE2 goldens. No
importorskip for missing R9 kernel modules that already live on this tree.

## R9-G2 — probes (local, no git)

Extend `scripts/probe_boundaries.py` for `apply_mpc`, `static_stress_response`,
`dump_psd`, `mapped_mac`, Nastran SOL 101/`read_static` — **skip** if absent.
Existing probes stay green.

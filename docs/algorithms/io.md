# IO translators — Abaqus INP and LS-DYNA keyword (K) text subsets

Spec for `femtools.io.inp` and `femtools.io.kfile` (owner: R6-F2), plus the Round-7
Nastran BDF extensions of §4 (owner: R7-F2). Frozen entry points:

```python
from femtools.io.inp import read_inp      # write_inp is optional
from femtools.io.kfile import read_k
# Round-7 (REMAINING.md; landed and measured — ACCEPTANCE Round-7 status) — see §4:
from femtools.io.cdb import write_cdb
from femtools.io.kfile import write_k
from femtools.io.bdf import read_bdf      # INCLUDE + RBE2, §4
# Round-8 (REMAINING.md, owner R8-F2; pending on this tree) — see §5–§6:
from femtools.io.bdf import read_bdf, write_bdf   # grow RBE3, §5
from femtools.drivers.ansys import AnsysCdbDriver
from femtools.drivers.abaqus import AbaqusInpDriver
```

Scope guard (REMAINING.md): **public text card layouts only** — the card grammars below are
documented in the public Abaqus Keywords Reference and the LS-DYNA Keyword User's Manual
Vol. I. Commercial binary result dumps (OP2/RST/ODB) stay explicitly out of scope, in every
round. Both readers map into `core.model.FEModel` and follow the conventions established by
`io.cdb`/`io.bdf`: unknown cards are skipped with *one aggregated warning per keyword name*
(never a hard error — real decks are full of out-of-scope history/step/control cards),
parsing is deterministic and single-pass over lines, and the resulting model must survive
`assemble_km` on a tiny HEX8/QUAD4/BEAM deck (`docs/ACCEPTANCE.md` §10, cases 24–25).

## 1. Abaqus INP subset — `read_inp`

### 1.1 Grammar

Keyword lines start with `*` (`*KEYWORD, PARAM=VALUE, FLAG, ...`), comment lines with `**`;
keywords and parameter names are **case-insensitive** and blanks around commas are legal —
normalize (`upper().strip()` per token) before dispatch. Data lines follow their keyword
until the next `*` line; values are comma-separated; a data line ending in a comma
continues on the next line (element connectivity routinely wraps — even 8-node cards may be
split). Model definition ends where step definitions begin: parsing **stops at the first
`*STEP`** — anything inside steps (loads, step-level `*BOUNDARY`, output requests) is
analysis history, not model, and is out of the Round-6 subset.

### 1.2 Supported cards → `FEModel`

| card | data layout | maps to |
|---|---|---|
| `*NODE` | `id, x, y, z` (missing coords = 0) | `FEModel` nodes |
| `*ELEMENT, TYPE=..., ELSET=...` | `eid, n1, n2, ...` (wrapping allowed) | elements; `ELSET=` implicitly creates the element set |
| `*MATERIAL, NAME=...` | container — suboptions below belong to it | material entry |
| `*ELASTIC` | `E, nu` (isotropic, first temperature row) | material `E`, `nu` |
| `*DENSITY` | `rho` | material `rho` |
| `*SOLID SECTION, ELSET=, MATERIAL=` | (data line unused for 3-D) | element → `solid` property → material binding |
| `*SHELL SECTION, ELSET=, MATERIAL=` | line 1: `t, n_int` | `shell` property, thickness `t` |
| `*BEAM GENERAL SECTION, ELSET=, SECTION=GENERAL` | line 1: `A, I11, I12, I22, J`; line 2: section-axis direction; line 3: `E, G` | `beam` property (A, Iyy, Izz, J) + material (this card carries its own elastic constants — no `MATERIAL=`) |
| `*BOUNDARY` (model-level) | `node/nset, dof_first[, dof_last[, value]]` or `node/nset, TYPE` | SPCs; nonzero `value` → enforced displacement (`solve_static(enforced=)`) |
| `*NSET, NSET=...[, GENERATE]` / `*ELSET, ELSET=...[, GENERATE]` | ids (≤16 per line) or `first, last, increment` | named sets in `FEModel.sets` |

Element type map (topological; formulation differences stay femtools's business):
`C3D8`/`C3D8R` → `HEX8`, `C3D4` → `TET4`, `S4`/`S4R` → `QUAD4`, `S3`/`S3R` → `TRIA3`,
`B31` → `BEAM2`, `T3D2` → `BAR2`. Reduced-integration variants (`...R`) collapse onto the
same femtools type with one aggregated warning — the integration scheme is chosen by the
femtools element formulation (`fea.md` §2), not imported.

DOF conventions transfer directly: Abaqus DOFs 1–6 are $u_x, u_y, u_z, r_x, r_y, r_z$ —
identical to `DOF_LABELS`, so `*BOUNDARY` component ranges (`first, last` inclusive) map to
the 6-bool SPC mask without translation. Named boundary types worth accepting: `ENCASTRE`
(1–6) and `PINNED` (1–3); the symmetry names (`XSYMM` etc.) may be warned-and-skipped.
C3D8 node ordering (bottom face counter-clockwise, then top face) matches femtools `HEX8`;
S4/S3 orientation conventions likewise.

### 1.3 Parsing rules that actually bite

- Material binding is a **two-step indirection**: element → `ELSET` → section card →
  `MATERIAL=` name. Build the model only after the whole file is read; sections may precede
  or follow their materials and element sets (forward references are legal).
- `*ELASTIC` and `*DENSITY` bind to the *most recent* `*MATERIAL` — a suboption before any
  `*MATERIAL` is a deck error, report with line number.
- `GENERATE` ranges are inclusive with default increment 1.
- `*SHELL SECTION` ≠ `*SHELL GENERAL SECTION` (the latter carries stiffness matrices, out
  of subset — aggregated warning).
- Multiple temperature rows under `*ELASTIC`: take the first row, warn once (femtools
  materials are temperature-independent).
- Node/element ids are arbitrary positive integers, not necessarily contiguous — never
  index arrays by raw id (same rule as `io.cdb`).

## 2. LS-DYNA keyword subset — `read_k`

### 2.1 Grammar

Cards start with `*KEYWORD_NAME` (case-insensitive), `$` starts a comment line, `*END`
terminates input. Data cards are **fixed-width**: unless a card documents otherwise, 8
fields × 10 characters. The two exceptions in this subset are load-bearing:

- `*NODE`: `NID` in columns 1–8 (I8), `X, Y, Z` in three **16-character** fields
  (E16.0), then `TC, RC` in 8-character fields — a naive `split()` breaks the moment two
  coordinate fields touch, so slice fixed columns;
- element cards (`*ELEMENT_...`): all-I8 fields (`10I8` for the classic one-line solid
  layout).

Free format is also legal anywhere: if a data line contains a comma, split on commas
instead of slicing — the presence test per line is the documented two-format rule. Long
format (keyword suffixed `+`, doubled field widths) and I10 format (suffix `%`) are out of
the Round-6 subset — detect the suffix and fail with a clear message rather than mis-slicing
silently.

### 2.2 Supported cards → `FEModel`

| card | data layout | maps to |
|---|---|---|
| `*NODE` | `NID(I8), X, Y, Z(3×E16), TC, RC(2×I8)` | nodes; `TC`/`RC` constraint codes → SPCs |
| `*ELEMENT_SOLID` | `EID, PID, N1..N8` (10I8, one line) | `HEX8`; degenerate tet (`N5=N6=N7=N8=N4`) → `TET4` |
| `*ELEMENT_SHELL` | `EID, PID, N1..N4` | `QUAD4`; collapsed quad (`N4=N3`) → `TRIA3` |
| `*ELEMENT_BEAM` | `EID, PID, N1, N2, N3` (N3 = orientation node) | `BEAM2` (orientation from N3) |
| `*MAT_ELASTIC` | `MID, RO, E, PR, ...` | isotropic material |
| `*SECTION_SOLID` | `SECID, ELFORM, ...` | `solid` property (binding only) |
| `*SECTION_SHELL` | card 1: `SECID, ELFORM, ...`; card 2: `T1..T4, ...` | `shell` property, uniform t = T1 (warn if T1..T4 differ) |
| `*SECTION_BEAM` | card 1: `SECID, ELFORM, ...`; card 2 (ELFORM=2, resultant): `A, ISS, ITT, IRR, SA` | `beam` property (A, Iyy, Izz, J) |
| `*PART` | line 1: title; line 2: `PID, SECID, MID, ...` | the PID → (SECID, MID) indirection |
| `*BOUNDARY_SPC_NODE` | `NID, CID, DOFX, DOFY, DOFZ, DOFRX, DOFRY, DOFRZ` (0/1 flags) | SPCs |

### 2.3 Parsing rules that actually bite

- The **PART indirection** is the structural difference from Abaqus: elements reference
  only `PID`; `*PART` binds `PID → (SECID, MID)`. Cards appear in any order and forward
  references are legal — accumulate raw cards first, resolve bindings after `*END`/EOF.
- Degenerate connectivity conventions must be detected and *retyped*, never passed
  through: a tet written as a collapsed solid or a tria as a collapsed quad produces a
  zero-Jacobian `HEX8`/`QUAD4` that poisons the assembled stiffness
  (`core.validation` flags it, but the reader should map to `TET4`/`TRIA3` outright).
- `*NODE` `TC`/`RC` codes are a **lookup table, not bit flags**: 0 = free, 1 = x, 2 = y,
  3 = z, 4 = x&y, 5 = y&z, 6 = z&x, 7 = x,y,z (the classic mistake is treating 4 as a
  bitmask for z). `RC` uses the same table for rotations.
- `*BOUNDARY_SPC_NODE` per-DOF 0/1 flags translate directly to the SPC 6-bool mask; `CID`
  (constraint coordinate system) other than 0 is out of subset — warn and apply globally.
- `*SECTION_BEAM` card-2 meaning depends on `ELFORM`: only the resultant formulation
  (ELFORM = 2) exposes `A, ISS, ITT, IRR` directly; integrated formulations (ELFORM = 1)
  give cross-section *dimensions* instead — out of subset, aggregated warning.
- `*INCLUDE` (nested decks) is out of subset — warn, do not chase files.
- `$` comments and blank lines may appear between the cards of a multi-card keyword
  (e.g. between `*PART` title and data) — strip them before counting card lines.

## 3. Round-trip acceptance and cross-format identity

Both readers must produce models that pass `assemble_km` without crashing on a minimal
deck containing one HEX8 cube, one QUAD4 patch and one BEAM2 member with materials,
sections and at least one boundary card (`docs/ACCEPTANCE.md` §10, cases 24–25). The
stronger cross-check: the *same* cube meshed once as INP (`C3D8` + `*SOLID SECTION`) and
once as K (`*ELEMENT_SOLID` + `*PART`/`*SECTION_SOLID`/`*MAT_ELASTIC`) must assemble
identical $K$ and $M$ up to DOF ordering — the translators carry no physics, so any
discrepancy is a parsing bug by construction. `write_inp`, if provided, must round-trip
`read_inp(write_inp(model))` to an equivalent model (same invariant `io.unv`/`io.bdf`
already pin).

Complexity: one pass over the file, $O(n_{lines})$ with $O(1)$ keyword dispatch; memory is
the model itself plus the raw-card store for the K-file two-phase resolve.

## 4. Nastran BDF extensions — `INCLUDE` and `RBE2` (Round 7, owner R7-F2)

`read_bdf` (Round 1, stable) grew two features frozen in REMAINING.md — landed on this
tree and measured (ACCEPTANCE Round-7 status). Both are public card/statement layouts
(MSC/NX Quick Reference Guide); no copyrighted deck content is reproduced anywhere.

### 4.1 `INCLUDE` statement resolution

Grammar: the statement keyword `INCLUDE` (case-insensitive) followed by a filename,
quoted with `'...'` or `"..."` (quotes optional for paths without blanks); it may appear
anywhere card-sized input is legal. Wrapped multi-line quoted filenames exist in the wild
but are out of the Round-7 subset — a clear error beats a silent mis-parse. Resolution
rules, all load-bearing:

- **Relative paths resolve against the directory of the *including* file**, never the
  process CWD — nested includes chain through their own directories, which is the whole
  point of deck-relative includes.
- **Depth ≤ 8**: a ninth nested level raises `BdfError` carrying the include chain.
- **Cycle-safe**: keep the stack of currently-open files as `os.path.realpath`-normalized
  paths; re-entering a file already on the stack raises with the cycle spelled out.
  (Note: a *diamond* — the same file included twice sequentially — is not a cycle; it is
  legal and simply parsed twice, duplicate-id errors then surface as such.)
- A missing file raises with the including file and line number.

Implementation shape: the line reader becomes a stack of `(path, line_iterator)` sources
and included text is spliced in place; everything downstream (free/small/large-field card
parsing, continuations) is untouched and continuation cards may not straddle an include
boundary. Errors keep reporting `(file, line)` per source, not a global line count.

### 4.2 `RBE2` card

Public small/large/free-field layout:

```text
RBE2  EID  GN  CM  GM1  GM2  GM3 ...  (ALPHA)
```

`GN` is the **independent** grid, `CM` the constrained component codes as a digit string
of 1..6 (e.g. `123456`), `GM1..GMi` the dependent grids, continuing over standard
continuation lines. The optional trailing `ALPHA` (thermal expansion coefficient) is
recognized and skipped with one aggregated warning — femtools carries no thermal strain.
Disambiguation of the tail is the documented one: grid ids are integers, `ALPHA` is a
real (contains `.` or an exponent).

Maps directly onto the merged, stable container — **do not invent a parallel table**:

```python
model.add_rbe2(id=EID, independent=GN, dependents=(GM1, ..., GMi),
               components=tuple(int(c) for c in str(CM)))
```

which is the same `core.model.RBE2` list that `fea.mpc.apply_rbe2` consumes for the
condensation $u = Tq$ (`fea.md` §11) — reading a deck and assembling it therefore honors
the rigid constraints with no extra plumbing. Validation (duplicate id, missing nodes,
independent ∈ dependents) lives in the container, not the parser. `RBE3` (interpolation,
*not* rigid) was optional Round-7 scope (warn-and-skip); it becomes required Round-8
scope in §5 below — and it must never be stored as an `RBE2`.

## 5. Nastran BDF `RBE3` card (Round 8, owner R8-F2)

Frozen in REMAINING.md — pending on this tree as of 2026-08-28: `read_bdf` parses `RBE3`
into the merged, stable `FEModel.add_rbe3` container, and `write_bdf` emits it. Public
small/large/free-field layout (MSC/NX Quick Reference Guide; no copyrighted deck content
reproduced):

```text
RBE3  EID  (blank)  REFGRID  REFC  WT1  C1  G1,1  G1,2 ...
      WT2  C2  G2,1  G2,2 ...  ("UM" / ALPHA tails, see below)
```

`REFGRID` is the **dependent** (reference) grid — note the direction is the opposite of
`RBE2`'s `GN` — and `REFC` its constrained component digits (1..6, packed, e.g. `123`).
The tail is a repeating sequence of weighted groups: a real weight `WTi`, a component
digit string `Ci`, then the independent grids `Gi,j` the pair applies to, continuing over
standard continuation lines. Field disambiguation inside the tail is type-driven, same
rule as the RBE2 `ALPHA` tail: grids are integers, weights are reals (contain `.` or an
exponent), component strings are digit-packed integers following a weight.

Mapping onto the container — **do not invent a parallel table**:

```python
model.add_rbe3(
    id=EID, dependent=REFGRID,
    components=tuple(int(c) for c in str(REFC)),
    independents=(G11, G12, ..., G21, ...),          # groups flattened, deck order
    independent_components=tuple(int(c) for c in str(C1)),
    weights=(WT1, WT1, ..., WT2, ...),               # one entry per grid
)
```

which is the same `core.model.RBE3` list `fea.mpc.apply_rbe3` consumes (`fea.md` §12).
Validation (duplicate id, missing nodes, dependent ∈ independents, component ranges,
weight count) lives in the container, not the parser. Subset boundaries, all taking the
standard **one aggregated `UserWarning`** path (never silent, never a hard error):

- heterogeneous per-group component lists (`C2 != C1`) — the container carries a single
  `independent_components`; decks that mix component sets per group are out of subset;
- the `UM` continuation (explicit dependent-DOF selection) and trailing `ALPHA`
  (thermal expansion) — recognized and skipped, like the RBE2 `ALPHA`;
- zero or negative weights — the deck is warned about and the card skipped rather than
  normalized into something the writer cannot round-trip.

`write_bdf` emits one `RBE3` per record — `EID, blank, REFGRID, REFC`, then the weighted
groups (equal weights `1.0` when `weights is None`) — such that
`read_bdf(write_bdf(model))` round-trips the `model.rbe3` table exactly (the invariant
`io.bdf` already pins for nodes/elements/SPCs/RBE2).

## 6. Solver text drivers — `AnsysCdbDriver`, `AbaqusInpDriver` (Round 8, owner R8-F2)

Frozen in REMAINING.md — pending on this tree as of 2026-08-28. Two more concrete
adapters for the `drivers.base.SolverDriver` protocol (PEP 544 structural typing, merged
Round 4; `docs/ARCHITECTURE.md` §10), joining Round-7's `NastranPunchDriver`. Both are
**text-only**: models cross the boundary as text decks the io layer already writes, and
results come back as text the io layer already reads. femtools ships no proprietary
binary parsers, in any round.

```python
from femtools.drivers.ansys import AnsysCdbDriver     # drivers/ansys.py
from femtools.drivers.abaqus import AbaqusInpDriver   # drivers/abaqus.py
# both re-exported from femtools.drivers (drivers/__init__.py);
# femtools/__init__.py is deliberately untouched.
```

Contract, identical shape for both (the four-step exchange loop of `drivers.base`):

| step | `AnsysCdbDriver` | `AbaqusInpDriver` |
|---|---|---|
| `write_input` | `io.cdb.write_cdb` (§ACCEPTANCE cases 24–25 deck subset) | `io.inp.write_inp` (§1 subset) |
| `is_available` | `shutil.which` over the executable and public aliases (`ansys`, `mapdl`) — must not raise | `shutil.which("abaqus")` — must not raise |
| `run` | launch executable, block; raise `SolverError` on **missing executable**, **nonzero exit**, or **timeout** | same conventions |
| `read_modal` | existing **text** readers only: `.pch` (`io.pch.read_pch`) or `.unv` (`io.unv.read_unv`) | same: `.unv` / `.pch` text only |

Binary results stay N/A with a *named* refusal: handing a `.rst` path to
`AnsysCdbDriver.read_modal` raises `SolverError` explicitly naming RST as not available
(pointing at the text alternatives), and `.odb` likewise for `AbaqusInpDriver` — a clear
error beats a silent misparse of a binary header. This mirrors the PRODUCT_MAP N/A row
(OP2/RST/ODB) rather than weakening it.

Testing convention (frozen): **no ANSYS or Abaqus install is ever required.** Tests stub
the executable exactly like the Round-7 Nastran tests — a shell script on `PATH` that
emits a canned text result file — so `is_available`/`run`/`read_modal` are exercised
end-to-end, and the `SolverError` paths (missing executable, nonzero exit, timeout, RST/
ODB refusal) are pinned without any commercial binary. Translation losses in
`write_input` follow the io convention: aggregated `UserWarning`s, never silent.

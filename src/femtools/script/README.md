# FSL — Femtools Scripting Language

FSL is the original command language of femtools. It is a small,
line-oriented batch language designed for building FE models and driving
analyses from text. It is an independent design and **not** a copy of any
proprietary scripting product.

```text
# cantilever bar, axial modes
NEW PROJECT bar
ADD NODE 1 0 0 0
ADD NODE 2 1 0 0
ADD MAT 1 TYPE=isotropic E=210e9 NU=0.3 RHO=7850
ADD PROP 1 TYPE=bar MAT=1 A=1e-4
ADD ELEM 1 TYPE=BAR2 NODES=1,2 PROP=1
SPC 1 ALL
SOLVE MODES N=5
MAC
SAVE bar.ftproj
```

Run it with the Python API:

```python
from femtools.script.engine import ScriptEngine
engine = ScriptEngine().run_file("bar.fsl")
print(engine.results["modes"].freq_hz)
```

or from the command line:

```bash
femtools script bar.fsl
femtools script -c "NEW PROJECT; ADD NODE 1 0 0 0; PRINT SUMMARY"
```

## Lexical rules

- A script is a sequence of **statements** separated by newlines and/or
  semicolons (`;`).
- `#` starts a comment that runs to the end of the line (unless inside a
  double-quoted string).
- Blank statements are ignored.
- Tokens are whitespace-separated. Double quotes group tokens that
  contain spaces (`SAVE "my project.ftproj"`).
- **Keywords are case-insensitive** (`solve modes` == `SOLVE MODES`).
  Values (names, file paths) keep their case.
- Numbers are parsed as integers when possible, otherwise as floats
  (scientific notation such as `210e9` is supported).
- **Options** are `KEY=value` tokens and may appear in any order after the
  positional arguments. A value containing commas is a **list**
  (`NODES=1,2,3`).

## Grammar (EBNF)

```ebnf
script     = { statement , ( ";" | newline ) } ;
statement  = new | add_node | add_mat | add_prop | add_elem
           | spc | solve | mac | save | print ;

new        = "NEW" , "PROJECT" , [ name ] ;
add_node   = "ADD" , "NODE" , id , number , number , number ;
add_mat    = "ADD" , ("MAT" | "MATERIAL") , id , { option } ;
add_prop   = "ADD" , ("PROP" | "PROPERTY") , id , "TYPE=" ptype ,
             [ "MAT=" id ] , { option } ;
add_elem   = "ADD" , ("ELEM" | "ELEMENT") , id , "TYPE=" etype ,
             "NODES=" idlist , [ "PROP=" id ] , { option } ;
spc        = "SPC" , id , ( mask | "ALL" | "FREE" | "DOF=" idlist ) ;
solve      = "SOLVE" , "MODES" , [ "N=" int ] , [ "SHIFT=" number ] ,
             [ "NAME=" name ] ;
mac        = "MAC" , [ "A=" name ] , [ "B=" name ] , [ "NAME=" name ] ;
save       = "SAVE" , path ;
print      = "PRINT" , [ "SUMMARY" | "RESULTS" ] ;

option     = key , "=" , value ;
idlist     = id , { "," , id } ;
mask       = six characters from {"0","1"} , e.g. "111000" ;
```

## Command reference

### `NEW PROJECT [name]`

Creates a fresh, empty `FEModel` and clears all stored results. Every
model-building command requires an active project.

### `ADD NODE <id> <x> <y> <z>`

Adds a node with integer id at coordinates `(x, y, z)`.

### `ADD MAT <id> [TYPE=isotropic] [E=..] [NU=..] [RHO=..] [G=..] ...`

Adds a material. `TYPE` defaults to `isotropic`. All other options are
forwarded to `FEModel.add_material` (`E` and `G` uppercase, other names
lowercase, per the femtools core convention).

### `ADD PROP <id> TYPE=<bar|beam|shell|solid|...> [MAT=<mat_id>] [A=..] [I=..] ...`

Adds a property card. `MAT` maps to `material_id`. Section options are
normalized to the canonical femtools spellings regardless of the case
you type: `IY`/`IZ` become `Iy`/`Iz`, `T`/`THICKNESS`/`H` become `t`,
`K`/`STIFFNESS` become `k` (spring), `M`/`MASS` become `m`, `C` becomes
`c` (damper), `NSM` stays `nsm`. Other short symbolic names (`A`, `I`,
`I1`, `I2`, `J`, `EA`, ...) are forwarded uppercase, longer names
lowercase.

### `ADD ELEM <id> TYPE=<etype> NODES=<n1,n2,...> [PROP=<prop_id>]`

Adds an element. `TYPE` is uppercased (`bar2` -> `BAR2`); Round-1 types
are `BAR2`, `BEAM2`, `TRUSS2D`, `QUAD4`, `TRIA3`, `HEX8`, `TET4`, `MASS`,
`SPRING`, `DAMPER`. `NODES` is a comma list of node ids and `PROP` maps
to `property_id`.

### `SPC <node_id> <constraint>`

Applies a single-point constraint on the six nodal DOFs
(tx, ty, tz, rx, ry, rz). The constraint is one of:

| Form           | Example          | Meaning                          |
|----------------|------------------|----------------------------------|
| 0/1 mask       | `SPC 1 111000`   | constrain tx, ty, tz             |
| `ALL`          | `SPC 1 ALL`      | constrain all six DOFs           |
| `FREE`         | `SPC 1 FREE`     | constrain nothing (placeholder)  |
| DOF list       | `SPC 1 DOF=1,2,6`| constrain 1-based DOF numbers    |

### `SOLVE MODES [N=<n>] [SHIFT=<hz2>] [NAME=<name>]`

Runs the real eigen solver (`femtools.fea.eigen.solve_modes`) on the
active model. `N` defaults to 10, `SHIFT` to 0.0. The `ModalResult` is
stored in `engine.results[NAME]` (default name `"modes"`) and becomes the
default operand for `MAC`.

### `MAC [A=<result>] [B=<result>] [NAME=<name>]`

Computes the Modal Assurance Criterion matrix between two stored modal
results (`femtools.correlation.mac.mac_matrix`). Both operands default to
the most recent `SOLVE MODES` result (self-MAC). The matrix is stored in
`engine.results[NAME]` (default `"mac"`).

### `SAVE <path>`

Saves the active project with `femtools.io.project.save_project`
(`.ftproj` JSON/npz hybrid). Quote paths containing spaces.

### `PRINT [SUMMARY|RESULTS]`

Convenience output: `SUMMARY` (default) prints entity counts of the
active model; `RESULTS` lists stored result names and types.

## Errors

All parse and execution failures raise `femtools.script.ScriptError`,
which reports the offending statement and its 1-based index within the
script. Commands that need sibling modules (`SOLVE MODES` -> `femtools.fea`,
`MAC` -> `femtools.correlation`, `SAVE` -> `femtools.io`,
`NEW PROJECT` -> `femtools.core`) raise a clear `ScriptError` if the
module is not installed, rather than failing at import time.

## Embedding

`ScriptEngine` is dependency-injectable for testing: pass
`model_factory=` (a callable accepting `name=`) to avoid the
`femtools.core` import, and read `engine.model`, `engine.results` and
`engine.log` after `run()` / `run_file()` / `execute()`.

## Model-file loading (`femtools.script.loading`)

Next to the FSL engine, this package hosts the model-file loader shared
by the CLI and the GUI:

```python
from femtools.script import load_model_file
loaded = load_model_file("beam.ftproj")   # or .json / .unv / .bdf / .inp / .k
loaded.model      # bare FEModel (Project/UnvData containers unwrapped)
loaded.results    # named results stored with the file, if any
```

Abaqus `.inp` and LS-DYNA `.k`/`.key` decks dispatch to the optional
`femtools.io.inp` / `femtools.io.kfile` text translators; when a build
ships without them the loader raises `ImportError` (CLI exit code 3)
instead of failing at import time.

`.json` models use a plain-JSON schema built on the public
`FEModel.add_*` API (`nodes`, `materials`, `properties`, `elements`,
`spcs`); the full schema is documented in the `femtools.script.loading`
module docstring.

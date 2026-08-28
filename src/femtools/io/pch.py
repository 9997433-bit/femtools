"""Nastran punch file (.pch) translator for real modal and static results.

The punch file is the 80-column text sibling of the OP2: each line carries
72 characters of data plus a sequence number in columns 73-80.  This module
reads and writes the real-eigenvector subset produced by
``DISPLACEMENT(PUNCH) = ALL`` in a SOL 103 run and reads the real static
``$DISPLACEMENTS`` blocks the same request punches in a SOL 101 run (an
original parser -- no Nastran or pyNastran code involved; **no binary OP2
support**, by design):

::

    $TITLE   = CANTILEVER                                                  1
    $SUBTITLE= SOL 103                                                     2
    $LABEL   =                                                             3
    $EIGENVALUE =  3.947842E+05  MODE =     1                              4
    $EIGENVECTOR                                                           5
             1       G      0.000000E+00      0.000000E+00  ...            6
    -CONT-                  0.000000E+00      0.000000E+00  ...            7

* ``$EIGENVALUE`` headers carry the eigenvalue ``lambda = omega**2`` in
  (rad/s)^2 -- the same convention as :class:`~femtools.core.results.
  ModalResult`; natural frequencies are recovered as
  ``sqrt(max(lambda, 0)) / (2 pi)``.
* ``G`` (grid) points carry 6 components over two lines (T1 T2 T3 /
  ``-CONT-`` R1 R2 R3); ``S`` (scalar) points carry one value, mapped to
  local DOF 0.
* Reading is tolerant: sequence numbers are optional, fields are
  whitespace-split (the fixed 18-character fields always leave separators),
  Fortran ``D`` exponents are accepted, and modes may list different node
  sets (missing entries are zero-filled with a warning).
* Complex-eigenvector blocks (``$REAL-IMAGINARY OUTPUT`` /
  ``$MAGNITUDE-PHASE OUTPUT``) and non-eigenvector result blocks
  (``$DISPLACEMENTS``, ``$SPCF``, ...) are skipped with one warning per
  kind, never an error.

:func:`read_pch_static` is the SOL 101 sibling of :func:`read_pch`: it
reads the public static punch layout below into a
:class:`~femtools.core.results.StaticResult` (one column per
``$SUBCASE``), skipping eigenvector and complex blocks the same tolerant
way :func:`read_pch` skips ``$DISPLACEMENTS``:

::

    $TITLE   = CANTILEVER                                                  1
    $SUBTITLE= SOL 101                                                     2
    $LABEL   =                                                             3
    $DISPLACEMENTS                                                         4
    $REAL OUTPUT                                                           5
    $SUBCASE ID =           1                                              6
             1       G      0.000000E+00      0.000000E+00  ...            7
    -CONT-                  0.000000E+00      0.000000E+00  ...            8

:func:`read_pch_stress` (Round 10) reads the element-stress sibling a
``STRESS(PUNCH) = ALL`` request punches -- the public ``$STRESSES`` /
``$ELEMENT STRESSES`` text blocks -- into a :class:`PchStressResult`
(element ids + Voigt stress tensors, one slab per ``$SUBCASE``).  Two
data-line shapes are read, both 80-column punch text with the usual
``-CONT-`` continuations:

* the labeled solid layout (CTETRA/CHEXA/CPENTA): component labels
  ``X Y Z XY YZ ZX`` each followed by their value; the first occurrence
  of each label (the ``CENTER`` group) is kept and the per-corner
  repeats, direction cosines and principal values (``A``/``B``/``C``)
  are ignored::

      $ELEMENT STRESSES                                                  4
      $REAL OUTPUT                                                       5
      $SUBCASE ID =           1                                          6
      $ELEMENT TYPE =          39  CTETRA                                7
             1           0GRID CS  4 GP                                  8
      -CONT-  CENTER  X   1.829032E+02  XY  -9.212549E+00   A  ...       9
      -CONT-          Y   1.093623E+02  YZ  -4.290556E+00   B  ...      10
      -CONT-          Z   1.093812E+02  ZX   1.107610E+00   C  ...      11

* plain rows of up to 6 values, read in Voigt order
  ``xx yy zz xy yz zx`` (missing trailing components are zero; entries
  with more than 6 values -- type-specific 2-D layouts femtools does not
  decode -- keep the first 6 with one aggregated warning)::

      $STRESSES                                                          4
      $SUBCASE ID =           1                                          5
             1                  1.000000E+02      2.000000E+01  ...      6
      -CONT-                    5.000000E+00      0.000000E+00  ...      7

``$EIGENVECTOR`` / ``$DISPLACEMENTS`` / complex-output blocks are
skipped with one warning per kind, never an error -- the same tolerant
policy as the other two readers.  **No OP2**: femtools ships no binary
result parsers, by design.

What the format cannot carry (documented losses):

* generalized (modal) mass -- Nastran punch has no such record;
  :func:`read_pch` returns 1.0 per mode (the Nastran default ``MASS``
  eigenvector normalization), and :func:`write_pch` warns when it drops
  non-unit modal masses.  Modal damping is likewise not representable.
* complex mode shapes -- :func:`write_pch` refuses them (use UNV dataset 55,
  which has a complex form).

:func:`write_pch` emits the layout above with sequence numbers and 9-decimal
mantissas (still inside the standard 18-character fields, so fixed-column
and free readers both parse it; plain 6-decimal Nastran output round-trips
at its own precision).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import FileFormatError
from ..core.results import DofPair, ModalResult, StaticResult

__all__ = [
    "read_pch",
    "read_pch_static",
    "read_pch_stress",
    "write_pch",
    "PchStressResult",
    "PchError",
]


class PchError(FileFormatError):
    """Raised for malformed punch content (a :class:`ValueError` via
    :class:`~femtools.core.errors.FileFormatError`)."""


_EIGENVALUE_RE = re.compile(
    r"\$\s*EIGENVALUE\s*=\s*([-+]?[0-9.]+(?:[EeDd][-+]?\d+)?)\s+MODE\s*=\s*(\d+)"
)

_SUBCASE_RE = re.compile(r"\$\s*SUBCASE\s+(?:ID\s*)?=\s*(\d+)")

_ELEMENT_TYPE_RE = re.compile(r"\$\s*ELEMENT\s+TYPE\s*=?\s*(\d+)?\s*(\S+)?")

#: labeled stress-component tokens of the public solid punch layout -> Voigt slot
_VOIGT_SLOT = {
    "X": 0, "Y": 1, "Z": 2,
    "XY": 3, "YX": 3, "YZ": 4, "ZY": 4, "ZX": 5, "XZ": 5,
}  # fmt: skip

#: point-type codes: components carried per point
_POINT_NDOF = {"G": 6, "S": 1, "E": 1, "M": 1}

#: $-headers that flag complex output (femtools reads real modes only)
_COMPLEX_MARKERS = ("$REAL-IMAGINARY", "$MAGNITUDE-PHASE")


def _strip_seq(raw: str) -> str:
    """Drop the sequence-number field (columns 73-80) when present."""
    line = raw.rstrip("\n")
    return line[:72] if len(line) > 72 else line


def _to_float(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

#: one parsed result block: node id -> (point type, component values)
_PointBlock = dict[int, tuple[str, list[float]]]


def _read_point_line(
    stripped: str,
    lineno: int,
    path: Path,
    points: _PointBlock,
    last_point: list[float] | None,
) -> list[float]:
    """Parse one punch data line (point or ``-CONT-``) into ``points``.

    Returns the value list continuation lines should extend.
    """
    toks = stripped.split()
    try:
        if toks[0] == "-CONT-":
            if last_point is None:
                raise PchError(
                    "-CONT- line without a point line", file=path.name, line=lineno
                )
            last_point.extend(_to_float(t) for t in toks[1:])
            return last_point
        nid = int(toks[0])
        ptype = toks[1].upper() if len(toks) > 1 and toks[1].isalpha() else "G"
        first = 2 if len(toks) > 1 and toks[1].isalpha() else 1
        values = [_to_float(t) for t in toks[first:]]
        points[nid] = (ptype, values)
        return values
    except PchError:
        raise
    except (ValueError, IndexError) as exc:
        raise PchError(
            f"cannot parse punch data line {lineno}: {stripped!r}",
            file=path.name,
            line=lineno,
        ) from exc


def _stack_point_blocks(
    blocks: list[_PointBlock], path: Path
) -> tuple[tuple[DofPair, ...], np.ndarray, bool]:
    """Union (node, dof) row labels over blocks and stack them column-wise.

    Returns ``(dof_index, matrix, uneven)`` where ``matrix`` is
    ``(n_dof, len(blocks))`` with missing entries zero-filled and ``uneven``
    flags blocks that list different node sets.
    """
    ndof_by_node: dict[int, int] = {}
    for points in blocks:
        for nid, (ptype, values) in points.items():
            n = _POINT_NDOF.get(ptype)
            if n is None:
                raise PchError(f"unknown point type {ptype!r} for node {nid}", file=path.name)
            n = max(n, min(len(values), 6))
            ndof_by_node[nid] = max(ndof_by_node.get(nid, 0), n)

    dof_index: list[DofPair] = []
    row_of: dict[DofPair, int] = {}
    for nid in sorted(ndof_by_node):
        for d in range(ndof_by_node[nid]):
            row_of[(nid, d)] = len(dof_index)
            dof_index.append((nid, d))

    matrix = np.zeros((len(dof_index), len(blocks)))
    uneven = False
    for j, points in enumerate(blocks):
        if len(points) != len(ndof_by_node):
            uneven = True
        for nid, (_ptype, values) in points.items():
            for d, v in enumerate(values[:6]):
                matrix[row_of[(nid, d)], j] = v
    return tuple(dof_index), matrix, uneven


class _ModeBlock:
    """One $EIGENVALUE/$EIGENVECTOR pair while being parsed."""

    __slots__ = ("eigenvalue", "number", "points", "is_complex")

    def __init__(self, eigenvalue: float, number: int) -> None:
        self.eigenvalue = eigenvalue
        self.number = number
        self.points: dict[int, tuple[str, list[float]]] = {}  # nid -> (type, values)
        self.is_complex = False


def read_pch(path: str | Path) -> ModalResult:
    """Read real frequencies and mode shapes from a Nastran punch file.

    Returns a :class:`~femtools.core.results.ModalResult` whose rows are
    labelled by ``dof_index``: 6 DOFs per grid (``G``) point, 1 per scalar
    (``S``/``E``/``M``) point, nodes ascending.  ``generalized_mass`` is set
    to 1.0 (see module docstring).  Raises :class:`PchError` when the file
    contains no readable real eigenvector data.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    modes: list[_ModeBlock] = []
    current: _ModeBlock | None = None
    in_vector = False
    complex_section = False  # sticky until a "$REAL OUTPUT" header
    skipped_blocks: dict[str, int] = {}
    n_complex = 0
    last_point: list[float] | None = None

    def flush() -> None:
        nonlocal current, in_vector, n_complex, last_point
        if current is not None:
            if current.is_complex:
                n_complex += 1
            elif current.points:
                modes.append(current)
        current = None
        in_vector = False
        last_point = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_seq(raw)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            upper = stripped.upper()
            m = _EIGENVALUE_RE.match(upper)
            if m:
                flush()
                current = _ModeBlock(_to_float(m.group(1)), int(m.group(2)))
                current.is_complex = complex_section
                continue
            # output-type markers describe the *following* blocks
            if upper.startswith(_COMPLEX_MARKERS):
                flush()
                complex_section = True
                continue
            if upper.startswith("$REAL OUTPUT"):
                flush()
                complex_section = False
                continue
            if upper.startswith("$EIGENVECTOR"):
                if current is None:
                    if complex_section:
                        n_complex += 1  # complex header layout we did not parse
                        continue
                    raise PchError(
                        "$EIGENVECTOR without a preceding $EIGENVALUE header",
                        file=path.name,
                        line=lineno,
                    )
                in_vector = True
                last_point = None
                continue
            if upper.startswith(("$TITLE", "$SUBTITLE", "$LABEL", "$SUBCASE", "$POINT")):
                continue
            # any other result block ($DISPLACEMENTS, $SPCF, stresses, ...)
            name = upper[1:].split("=")[0].strip() or "?"
            skipped_blocks[name] = skipped_blocks.get(name, 0) + 1
            in_vector = False
            last_point = None
            continue

        if not in_vector or current is None or current.is_complex:
            continue  # data of a skipped/complex block

        last_point = _read_point_line(stripped, lineno, path, current.points, last_point)

    flush()

    if n_complex:
        warnings.warn(
            f"read_pch({path.name}): skipped {n_complex} complex-eigenvector block(s); "
            "only real modes are read (use UNV dataset 55 for complex shapes)",
            UserWarning,
            stacklevel=2,
        )
    for name, count in sorted(skipped_blocks.items()):
        if name.startswith("DISPLACEMENT"):
            hint = " -- read_pch_static reads static displacement blocks"
        elif "STRESS" in name:
            hint = " -- read_pch_stress reads element stress blocks"
        else:
            hint = ""
        warnings.warn(
            f"read_pch({path.name}): skipped non-eigenvector block {name} (x{count}){hint}",
            UserWarning,
            stacklevel=2,
        )
    if not modes:
        raise PchError("no real eigenvector data found in punch file", file=path.name)

    dof_index, shapes, uneven = _stack_point_blocks([blk.points for blk in modes], path)
    if uneven:
        warnings.warn(
            f"read_pch({path.name}): modes list different node sets; "
            "missing entries were zero-filled",
            UserWarning,
            stacklevel=2,
        )

    eigenvalues = np.array([blk.eigenvalue for blk in modes])
    freq_hz = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)
    return ModalResult(
        freq_hz=freq_hz,
        eigenvalues=eigenvalues,
        modes=shapes,
        generalized_mass=np.ones(len(modes)),
        dof_index=tuple(dof_index),
    )


class _CaseBlock:
    """One $DISPLACEMENTS block (a static subcase) while being parsed."""

    __slots__ = ("subcase", "points", "is_complex")

    def __init__(self, subcase: int | None) -> None:
        self.subcase = subcase
        self.points: _PointBlock = {}
        self.is_complex = False


def read_pch_static(path: str | Path) -> StaticResult:
    """Read real static displacements from a Nastran punch file (SOL 101).

    Parses the public ``$DISPLACEMENTS`` blocks a
    ``DISPLACEMENT(PUNCH) = ALL`` request punches in a static run into a
    :class:`~femtools.core.results.StaticResult`.  Rows are labelled by
    ``dof_index`` exactly like :func:`read_pch` (6 DOFs per ``G`` point, 1
    per scalar point, nodes ascending); one column per ``$SUBCASE``, with
    ``load_case`` carrying the subcase ids (1-based position when a block
    has no ``$SUBCASE`` header).  A single subcase returns a 1-D ``u``.
    Eigenvector blocks, complex-output blocks and other result kinds
    (``$SPCF``, stresses, ...) are skipped with one warning per kind;
    a file with no readable displacement data raises :class:`PchError`.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    cases: list[_CaseBlock] = []
    current: _CaseBlock | None = None
    complex_section = False  # sticky until a "$REAL OUTPUT" header
    pending_subcase: int | None = None
    skipped_blocks: dict[str, int] = {}
    n_complex = 0
    n_eigen = 0
    last_point: list[float] | None = None

    def flush() -> None:
        nonlocal current, last_point, n_complex
        if current is not None:
            if current.is_complex:
                n_complex += 1
            elif current.points:
                cases.append(current)
        current = None
        last_point = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_seq(raw)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            upper = stripped.upper()
            m = _SUBCASE_RE.match(upper)
            if m:
                sid = int(m.group(1))
                if current is not None and not current.points:
                    current.subcase = sid  # $SUBCASE follows $DISPLACEMENTS
                elif current is not None:
                    # next subcase without a repeated $DISPLACEMENTS header
                    flush()
                    current = _CaseBlock(sid)
                    current.is_complex = complex_section
                else:
                    pending_subcase = sid  # $SUBCASE precedes $DISPLACEMENTS
                continue
            if upper.startswith("$DISPLACEMENT"):
                flush()
                current = _CaseBlock(pending_subcase)
                current.is_complex = complex_section
                pending_subcase = None
                last_point = None
                continue
            # output-type markers appear between $DISPLACEMENTS and the data
            if upper.startswith(_COMPLEX_MARKERS):
                complex_section = True
                if current is not None and not current.points:
                    current.is_complex = True
                else:
                    flush()
                continue
            if upper.startswith("$REAL OUTPUT"):
                complex_section = False
                if current is not None and not current.points:
                    current.is_complex = False
                continue
            if upper.startswith("$EIGENVALUE"):
                flush()
                n_eigen += 1
                continue
            if upper.startswith("$EIGENVECTOR"):
                flush()
                continue
            if upper.startswith(("$TITLE", "$SUBTITLE", "$LABEL", "$POINT")):
                continue
            # any other result block ($SPCF, stresses, ...)
            name = upper[1:].split("=")[0].strip() or "?"
            skipped_blocks[name] = skipped_blocks.get(name, 0) + 1
            flush()
            continue

        if current is None or current.is_complex:
            continue  # data of a skipped/eigenvector/complex block

        last_point = _read_point_line(stripped, lineno, path, current.points, last_point)

    flush()

    if n_eigen:
        warnings.warn(
            f"read_pch_static({path.name}): skipped {n_eigen} eigenvector block(s) "
            "-- read_pch reads modal punch content",
            UserWarning,
            stacklevel=2,
        )
    if n_complex:
        warnings.warn(
            f"read_pch_static({path.name}): skipped {n_complex} complex displacement "
            "block(s); only real static output is read",
            UserWarning,
            stacklevel=2,
        )
    for name, count in sorted(skipped_blocks.items()):
        hint = " -- read_pch_stress reads element stress blocks" if "STRESS" in name else ""
        warnings.warn(
            f"read_pch_static({path.name}): skipped non-displacement block {name} "
            f"(x{count}){hint}",
            UserWarning,
            stacklevel=2,
        )
    if not cases:
        raise PchError(
            "no real static $DISPLACEMENTS data found in punch file", file=path.name
        )

    dof_index, u, uneven = _stack_point_blocks([case.points for case in cases], path)
    if uneven:
        warnings.warn(
            f"read_pch_static({path.name}): subcases list different node sets; "
            "missing entries were zero-filled",
            UserWarning,
            stacklevel=2,
        )

    load_case = [
        case.subcase if case.subcase is not None else j + 1 for j, case in enumerate(cases)
    ]
    if len(cases) == 1:
        return StaticResult(u=u[:, 0], dof_index=dof_index, load_case=load_case[0])
    return StaticResult(u=u, dof_index=dof_index, load_case=tuple(load_case))


# ---------------------------------------------------------------------------
# element stresses ($STRESSES / $ELEMENT STRESSES)
# ---------------------------------------------------------------------------


@dataclass
class PchStressResult:
    """Element stresses read from punch ``$STRESSES`` blocks.

    Attributes
    ----------
    element_ids:
        Element identifiers, ascending; row ``i`` of ``stress`` belongs to
        ``element_ids[i]``.
    stress:
        Voigt tensors ordered ``xx yy zz xy yz zx``: ``(n_elements, 6)``
        for a single subcase, ``(n_elements, 6, n_cases)`` when the file
        carries several ``$SUBCASE`` blocks (missing entries zero-filled
        with a warning, like :func:`read_pch_static`).
    load_case:
        The ``$SUBCASE`` id (or ids, one per slab); 1-based block position
        when a block has no ``$SUBCASE`` header.
    etypes:
        ``element id -> punch element-type name`` for elements preceded by
        a ``$ELEMENT TYPE`` header (empty when the file has none).
    """

    element_ids: tuple[int, ...] = ()
    stress: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    load_case: int | tuple[int, ...] = 1
    etypes: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.element_ids)

    @property
    def n_elements(self) -> int:
        return len(self.element_ids)

    @property
    def n_cases(self) -> int:
        return 1 if self.stress.ndim == 2 else int(self.stress.shape[2])

    def index_of(self, element_id: int) -> int:
        """Row of one element id (:class:`KeyError` when absent)."""
        try:
            return self.element_ids.index(int(element_id))
        except ValueError:
            raise KeyError(f"element {element_id} not in the punch stress data") from None

    @property
    def von_mises(self) -> np.ndarray:
        """Von Mises equivalent stress: ``(n_elements,)`` per subcase slab."""
        s = self.stress
        dev = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
        shear = s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
        return np.sqrt(0.5 * dev + 3.0 * shear)


def _try_float(tok: str) -> float | None:
    try:
        return _to_float(tok)
    except ValueError:
        return None


class _StressBlock:
    """One $STRESSES block (a subcase) while being parsed."""

    __slots__ = ("subcase", "elements", "etypes", "is_complex")

    def __init__(self, subcase: int | None) -> None:
        self.subcase = subcase
        self.elements: dict[int, list[str]] = {}  # eid -> data tokens
        self.etypes: dict[int, str] = {}
        self.is_complex = False


def _stress_voigt(eid: int, toks: list[str], path: Path) -> tuple[np.ndarray, bool]:
    """Token list of one element entry -> ``(6 Voigt components, truncated)``.

    Labeled entries (any ``X``/``XY``/... token followed by a value, the
    public solid layout) keep the first occurrence of each component --
    the ``CENTER`` group -- and ignore everything else (grid repeats,
    direction cosines, principal values).  Plain entries must be numbers:
    up to 6 are read in Voigt order, more than 6 flags ``truncated``.
    """
    vals = np.zeros(6)
    labeled = any(
        t.upper() in _VOIGT_SLOT and i + 1 < len(toks) and _try_float(toks[i + 1]) is not None
        for i, t in enumerate(toks)
    )
    if labeled:
        filled = [False] * 6
        i = 0
        while i < len(toks):
            slot = _VOIGT_SLOT.get(toks[i].upper())
            if slot is not None and i + 1 < len(toks):
                v = _try_float(toks[i + 1])
                if v is not None:
                    if not filled[slot]:  # first occurrence = the CENTER group
                        vals[slot] = v
                        filled[slot] = True
                    i += 2
                    continue
            i += 1
        return vals, False
    floats: list[float] = []
    for t in toks:
        v = _try_float(t)
        if v is None:
            raise PchError(
                f"cannot parse stress value {t!r} for element {eid}", file=path.name
            )
        floats.append(v)
    n = min(len(floats), 6)
    vals[:n] = floats[:6]
    return vals, len(floats) > 6


def read_pch_stress(path: str | Path) -> PchStressResult:
    """Read real element stresses from a Nastran punch file.

    Parses the public ``$STRESSES`` / ``$ELEMENT STRESSES`` blocks a
    ``STRESS(PUNCH) = ALL`` request punches into a
    :class:`PchStressResult`: element ids ascending, one ``(n, 6)`` Voigt
    slab per ``$SUBCASE`` (see the module docstring for the two data-line
    shapes read).  Eigenvector blocks, ``$DISPLACEMENTS``, complex-output
    blocks and other result kinds are skipped with one warning per kind,
    never an error; a file with no readable stress data raises
    :class:`PchError`.  Text punch only -- **no OP2**, by design.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    blocks: list[_StressBlock] = []
    current: _StressBlock | None = None
    complex_section = False  # sticky until a "$REAL OUTPUT" header
    pending_subcase: int | None = None
    current_etype = ""
    skipped_blocks: dict[str, int] = {}
    n_complex = 0
    n_eigen = 0
    last_tokens: list[str] | None = None

    def flush() -> None:
        nonlocal current, last_tokens, n_complex
        if current is not None:
            if current.is_complex:
                n_complex += 1
            elif current.elements:
                blocks.append(current)
        current = None
        last_tokens = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_seq(raw)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            upper = stripped.upper()
            m = _SUBCASE_RE.match(upper)
            if m:
                sid = int(m.group(1))
                if current is not None and not current.elements:
                    current.subcase = sid  # $SUBCASE follows $STRESSES
                elif current is not None:
                    # next subcase without a repeated $STRESSES header
                    flush()
                    current = _StressBlock(sid)
                    current.is_complex = complex_section
                else:
                    pending_subcase = sid  # $SUBCASE precedes $STRESSES
                continue
            em = _ELEMENT_TYPE_RE.match(upper)
            if em:  # annotates the following entries; not a block boundary
                current_etype = (em.group(2) or em.group(1) or "").strip()
                continue
            if upper.startswith(("$STRESS", "$ELEMENT STRESS")):
                flush()
                current = _StressBlock(pending_subcase)
                current.is_complex = complex_section
                pending_subcase = None
                current_etype = ""
                continue
            # output-type markers appear between $STRESSES and the data
            if upper.startswith(_COMPLEX_MARKERS):
                complex_section = True
                if current is not None and not current.elements:
                    current.is_complex = True
                else:
                    flush()
                continue
            if upper.startswith("$REAL OUTPUT"):
                complex_section = False
                if current is not None and not current.elements:
                    current.is_complex = False
                continue
            if upper.startswith("$EIGENVALUE"):
                flush()
                n_eigen += 1
                continue
            if upper.startswith("$EIGENVECTOR"):
                flush()
                continue
            if upper.startswith(("$TITLE", "$SUBTITLE", "$LABEL", "$POINT")):
                continue
            # any other result block ($DISPLACEMENTS, $SPCF, forces, ...)
            name = upper[1:].split("=")[0].strip() or "?"
            skipped_blocks[name] = skipped_blocks.get(name, 0) + 1
            flush()
            continue

        if current is None or current.is_complex:
            continue  # data of a skipped/eigenvector/complex block

        toks = stripped.split()
        if toks[0] == "-CONT-":
            if last_tokens is None:
                raise PchError(
                    "-CONT- line without an element line", file=path.name, line=lineno
                )
            last_tokens.extend(toks[1:])
            continue
        try:
            eid = int(toks[0])
        except ValueError as exc:
            raise PchError(
                f"cannot parse punch stress line {lineno}: {stripped!r}",
                file=path.name,
                line=lineno,
            ) from exc
        last_tokens = list(toks[1:])
        current.elements[eid] = last_tokens
        if current_etype:
            current.etypes.setdefault(eid, current_etype)

    flush()

    if n_eigen:
        warnings.warn(
            f"read_pch_stress({path.name}): skipped {n_eigen} eigenvector block(s) "
            "-- read_pch reads modal punch content",
            UserWarning,
            stacklevel=2,
        )
    if n_complex:
        warnings.warn(
            f"read_pch_stress({path.name}): skipped {n_complex} complex stress "
            "block(s); only real element stresses are read",
            UserWarning,
            stacklevel=2,
        )
    for name, count in sorted(skipped_blocks.items()):
        hint = (
            " -- read_pch_static reads static displacement blocks"
            if name.startswith("DISPLACEMENT")
            else ""
        )
        warnings.warn(
            f"read_pch_stress({path.name}): skipped non-stress block {name} (x{count}){hint}",
            UserWarning,
            stacklevel=2,
        )
    if not blocks:
        raise PchError(
            "no real $STRESSES element stress data found in punch file", file=path.name
        )

    all_ids = sorted({eid for blk in blocks for eid in blk.elements})
    row_of = {eid: i for i, eid in enumerate(all_ids)}
    stress = np.zeros((len(all_ids), 6, len(blocks)))
    truncated: dict[int, None] = {}
    uneven = False
    etypes: dict[int, str] = {}
    for j, blk in enumerate(blocks):
        if len(blk.elements) != len(all_ids):
            uneven = True
        for eid, toks in blk.elements.items():
            stress[row_of[eid], :, j], over = _stress_voigt(eid, toks, path)
            if over:
                truncated[eid] = None
        for eid, name in blk.etypes.items():
            etypes.setdefault(eid, name)
    if truncated:
        eids = list(truncated)
        shown = ", ".join(map(str, eids[:5])) + (", ..." if len(eids) > 5 else "")
        warnings.warn(
            f"read_pch_stress({path.name}): {len(eids)} element entrie(s) [{shown}] "
            "carried more than 6 values (a type-specific layout femtools does not "
            "decode); the first 6 were read as Voigt components",
            UserWarning,
            stacklevel=2,
        )
    if uneven:
        warnings.warn(
            f"read_pch_stress({path.name}): subcases list different element sets; "
            "missing entries were zero-filled",
            UserWarning,
            stacklevel=2,
        )

    load_case = [
        blk.subcase if blk.subcase is not None else j + 1 for j, blk in enumerate(blocks)
    ]
    if len(blocks) == 1:
        return PchStressResult(
            element_ids=tuple(all_ids),
            stress=stress[:, :, 0],
            load_case=load_case[0],
            etypes=etypes,
        )
    return PchStressResult(
        element_ids=tuple(all_ids),
        stress=stress,
        load_case=tuple(load_case),
        etypes=etypes,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _is_modal_like(obj: object) -> bool:
    """Anything with the ModalResult data attributes (duck typing: accepts
    both ``femtools.core.results.ModalResult`` and the richer
    ``femtools.fea.eigen.ModalResult``)."""
    return all(
        hasattr(obj, a) for a in ("freq_hz", "eigenvalues", "modes", "generalized_mass")
    )


def _coerce_path_modal(first: object, second: object) -> tuple[Path, ModalResult]:
    """Accept both ``write_pch(path, modal)`` and ``write_pch(modal, path)``."""
    if _is_modal_like(first) and isinstance(second, (str, Path)):
        return Path(second), first  # type: ignore[return-value]
    if _is_modal_like(second) and isinstance(first, (str, Path)):
        return Path(first), second  # type: ignore[return-value]
    raise TypeError(
        "expected (path, modal) or (modal, path); "
        f"got {type(first).__name__} and {type(second).__name__}"
    )


def _dof_index_of(modal: ModalResult) -> tuple[DofPair, ...] | None:
    """Row labels of a modal result: ``dof_index`` when present, otherwise
    derived from a solver ``dof_map`` (femtools.fea convention: 6 DOFs per
    node, nodes in map order)."""
    dof_index = getattr(modal, "dof_index", None)
    if dof_index is not None:
        return tuple((int(n), int(d)) for n, d in dof_index)
    dof_map = getattr(modal, "dof_map", None)
    if dof_map is not None:
        return tuple(
            (int(nid), d) for nid in dof_map.node_ids for d in range(dof_map.dofs_per_node)
        )
    return None


def _real_matrix(modal: ModalResult) -> np.ndarray:
    modes = np.asarray(modal.modes)
    if np.iscomplexobj(modes):
        if np.max(np.abs(modes.imag), initial=0.0) > 0.0:
            raise PchError(
                "punch eigenvectors are real; cannot write complex mode shapes "
                "(write_unv dataset 55 supports complex modes)"
            )
        modes = modes.real
    return np.asarray(modes, dtype=float)


def write_pch(
    path: str | Path | ModalResult,
    modal: ModalResult | str | Path | None = None,
    *,
    title: str = "FEMTOOLS MODAL EXPORT",
    subtitle: str = "",
    label: str = "",
) -> None:
    """Write real frequencies and mode shapes as a Nastran punch file.

    Accepts ``write_pch(path, modal)`` or ``write_pch(modal, path)``, where
    ``modal`` may be a :class:`femtools.core.results.ModalResult` or a
    solver result such as :class:`femtools.fea.eigen.ModalResult` (rows are
    labelled by ``dof_index`` or, failing that, by the solver ``dof_map``).
    Every referenced node is written as a ``G`` point with 6 components,
    zero-filling DOFs absent from the row labels.  Non-unit generalized
    masses and modal damping are dropped with a warning (the format has no
    record for them); complex shapes raise :class:`PchError`.
    """
    out, modal = _coerce_path_modal(path, modal)
    dof_index = _dof_index_of(modal)
    if dof_index is None:
        raise PchError(
            "write_pch requires (node, dof) row labels: modal.dof_index or modal.dof_map"
        )
    modes = _real_matrix(modal)
    if len(dof_index) != modes.shape[0]:
        raise PchError(
            f"modal has {modes.shape[0]} rows but {len(dof_index)} (node, dof) labels"
        )
    eigenvalues = np.asarray(modal.eigenvalues)
    if np.iscomplexobj(eigenvalues):
        if np.max(np.abs(eigenvalues.imag), initial=0.0) > 0.0:
            raise PchError("punch $EIGENVALUE records are real; eigenvalues are complex")
        eigenvalues = eigenvalues.real
    if not np.allclose(modal.generalized_mass, 1.0):
        warnings.warn(
            "write_pch: punch files cannot carry generalized mass; "
            "readers will assume mass-normalized modes (1.0)",
            UserWarning,
            stacklevel=2,
        )
    damping = getattr(modal, "damping", None)
    if damping is not None and np.any(np.asarray(damping) != 0.0):
        warnings.warn(
            "write_pch: modal damping is not representable in a punch file; dropped",
            UserWarning,
            stacklevel=2,
        )

    # rows grouped per node (6 slots, absent DOFs stay zero)
    node_rows: dict[int, list[tuple[int, int]]] = {}  # nid -> [(dof, row), ...]
    for row, (nid, dof) in enumerate(dof_index):
        node_rows.setdefault(int(nid), []).append((int(dof), row))
    node_ids = sorted(node_rows)

    lines: list[str] = []
    seq = 0

    def emit(body: str) -> None:
        nonlocal seq
        seq += 1
        lines.append(f"{body:<72.72s}{seq:>8d}")

    emit(f"$TITLE   = {title.upper()}")
    emit(f"$SUBTITLE= {subtitle.upper()}")
    emit(f"$LABEL   = {label.upper()}")
    for j in range(modal.n_modes):
        emit(f"$EIGENVALUE = {float(eigenvalues[j]):16.9E}  MODE = {j + 1:5d}")
        emit("$EIGENVECTOR")
        for nid in node_ids:
            vals = [0.0] * 6
            for dof, row in node_rows[nid]:
                vals[dof] = float(modes[row, j])
            emit(f"{nid:>10d}{'G':>8s}" + "".join(f"{v:18.9E}" for v in vals[:3]))
            emit(f"{'-CONT-':<18s}" + "".join(f"{v:18.9E}" for v in vals[3:]))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Nastran punch file (.pch) translator for real modal results.

The punch file is the 80-column text sibling of the OP2: each line carries
72 characters of data plus a sequence number in columns 73-80.  This module
reads and writes the real-eigenvector subset produced by
``DISPLACEMENT(PUNCH) = ALL`` in a SOL 103 run (an original parser -- no
Nastran or pyNastran code involved; **no binary OP2 support**, by design):

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
from pathlib import Path

import numpy as np

from ..core.errors import FileFormatError
from ..core.results import DofPair, ModalResult

__all__ = ["read_pch", "write_pch", "PchError"]


class PchError(FileFormatError):
    """Raised for malformed punch content (a :class:`ValueError` via
    :class:`~femtools.core.errors.FileFormatError`)."""


_EIGENVALUE_RE = re.compile(
    r"\$\s*EIGENVALUE\s*=\s*([-+]?[0-9.]+(?:[EeDd][-+]?\d+)?)\s+MODE\s*=\s*(\d+)"
)

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

        toks = stripped.split()
        try:
            if toks[0] == "-CONT-":
                if last_point is None:
                    raise PchError(
                        "-CONT- line without a point line", file=path.name, line=lineno
                    )
                last_point.extend(_to_float(t) for t in toks[1:])
            else:
                nid = int(toks[0])
                ptype = toks[1].upper() if len(toks) > 1 and toks[1].isalpha() else "G"
                first = 2 if len(toks) > 1 and toks[1].isalpha() else 1
                values = [_to_float(t) for t in toks[first:]]
                current.points[nid] = (ptype, values)
                last_point = values
        except PchError:
            raise
        except (ValueError, IndexError) as exc:
            raise PchError(
                f"cannot parse punch data line {lineno}: {stripped!r}",
                file=path.name,
                line=lineno,
            ) from exc

    flush()

    if n_complex:
        warnings.warn(
            f"read_pch({path.name}): skipped {n_complex} complex-eigenvector block(s); "
            "only real modes are read (use UNV dataset 55 for complex shapes)",
            UserWarning,
            stacklevel=2,
        )
    for name, count in sorted(skipped_blocks.items()):
        warnings.warn(
            f"read_pch({path.name}): skipped non-eigenvector block {name} (x{count})",
            UserWarning,
            stacklevel=2,
        )
    if not modes:
        raise PchError("no real eigenvector data found in punch file", file=path.name)

    # -- union DOF index over all modes ------------------------------------
    ndof_by_node: dict[int, int] = {}
    for blk in modes:
        for nid, (ptype, values) in blk.points.items():
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

    shapes = np.zeros((len(dof_index), len(modes)))
    uneven = False
    for j, blk in enumerate(modes):
        if len(blk.points) != len(ndof_by_node):
            uneven = True
        for nid, (_ptype, values) in blk.points.items():
            for d, v in enumerate(values[:6]):
                shapes[row_of[(nid, d)], j] = v
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

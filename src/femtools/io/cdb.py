"""ANSYS coded-database (.cdb) translator -- blocked NBLOCK/EBLOCK subset.

Reads the archive files written by MAPDL ``CDWRITE`` (an original parser --
no ANSYS or pymapdl code involved; **no binary .rst support, by design**).
Supported records:

=================  ==========================================================
record             use
=================  ==========================================================
NBLOCK             nodes (blocked, fixed-format; rotation angles ignored)
EBLOCK             elements (blocked; SOLID, COMPACT and default key --
                   COMPACT records inherit the current TYPE/MAT/REAL/SECNUM)
ET / ETBLOCK       element type number -> ANSYS element name
MP / MPDATA        isotropic material data (EX, NUXY/PRXY, DENS, GXY, ALPX)
R / RMORE/RLBLOCK  real constants (shell thickness, spring k, mass m, BEAM4);
                   both the interactive ``R,NSET,V1..V6`` and the archive
                   ``R,NSET,LOC,STLOC,V1,V2,V3`` forms are read
SECTYPE/SECDATA/   shell section thickness (layered sections are summed)
SECBLOCK
D                  single-point constraints (UX..ROTZ, ALL)
F                  nodal loads (FX..MZ)
CSYS               warning only (coordinates are assumed global cartesian)
=================  ==========================================================

Every other command is ignored silently (a .cdb is full of bookkeeping
commands); structurally relevant but unsupported records (``CP``, ``CE``,
``CERIG``, ``RBE3``) raise one aggregated ``UserWarning`` each.

Element type mapping (ANSYS -> femtools), by element *name number*, so both
``ET,1,185`` and ``ET,1,SOLID185`` resolve:

==================  ========  =========================================
ANSYS               femtools  notes
==================  ========  =========================================
LINK1               TRUSS2D   2-D spar
LINK8, LINK180      BAR2
BEAM3               BEAM2     A/IZZ from real constants (2-D element:
                              Iy=Iz, J=Iy+Iz fallbacks applied)
BEAM4               BEAM2     section from real constants
BEAM188             BEAM2     orientation node K honoured; section
                              values are NOT derived from SECDATA
SHELL41/63/181      QUAD4     TRIA3 when degenerate (duplicate node)
SHELL93/281         QUAD4     corner nodes; midside nodes dropped
SOLID45/185         HEX8      TET4 when degenerate; wedge/pyramid skipped
SOLID95/186         HEX8      corner nodes; midside nodes dropped
SOLID92/187         TET10     all 10 nodes kept (Round 10); degenerate
                              records drop their midsides to TET4
SOLID285            TET4
MASS21              MASS      mass = real constant 1 (MASSX)
COMBIN14            SPRING    k = real constant 1; axial (no DOF pins)
==================  ========  =========================================

Like the BDF reader, degradations (midside drops, skipped wedges, missing
section data) are reported as aggregated ``UserWarning`` s -- never an
error.  Fixed-format lines are sliced by the Fortran format line that
heads each block (``(3i9,6e21.13e3)``...), because negative values can
touch the neighbouring field and whitespace splitting would be wrong.

A .cdb carries no unit information: the returned model keeps the default
SI :class:`~femtools.core.units.UnitSystem` and consistency is the
caller's responsibility.

:func:`write_cdb` (Round 7) emits the same subset -- one ``ET`` per ANSYS
element number, ``MP`` material records, ``R``/``RMORE`` real constants,
``SECTYPE``/``SECDATA`` shell sections, one blocked ``NBLOCK``/``EBLOCK``
(SOLID key) pair and ``D``/``F`` records -- so a written archive reads
back through :func:`read_cdb`.  Element type mapping on write: HEX8 ->
SOLID185, TET4 -> SOLID285, TET10 -> SOLID187 (nodes 9-10 on the EBLOCK
continuation line), QUAD4/TRIA3 -> SHELL181 (TRIA3 as the
degenerate quad), BEAM2 -> BEAM4 (orientation vectors become extra K
nodes appended after the real nodes), BAR2 -> LINK180, TRUSS2D -> LINK1,
MASS -> MASS21, SPRING -> COMBIN14.  Documented losses (aggregated
``UserWarning`` s, never silent): DAMPER elements, SPRING DOF pins,
beam ``kappa``, non-structural mass, RBE2 tables, enforced-SPC /
load set ids other than 1, nodal output systems (``cd``) and the model
name (:func:`read_cdb` names the model after the file).  Property ids
are not carried by the format: the reader re-synthesizes one property
per (kind, MAT, REAL, SECNUM) combination.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from ..core.errors import FileFormatError
from ..core.model import FEModel

__all__ = ["read_cdb", "write_cdb", "CdbError"]


class CdbError(FileFormatError):
    """Raised for malformed coded-database content (a :class:`ValueError`
    via :class:`~femtools.core.errors.FileFormatError`)."""


# ---------------------------------------------------------------------------
# Fortran fixed-format machinery
# ---------------------------------------------------------------------------

_FMT_ITEM = re.compile(r"(\d*)([iefga])(\d+)(?:\.\d+)?(?:e\d+)?", re.IGNORECASE)

_Field = int | float | str | None


def _parse_format(line: str, lineno: int) -> list[tuple[str, int]]:
    """``(3i9,6e21.13e3)`` -> ``[('i', 9)]*3 + [('f', 21)]*6``.

    ``a`` (alphanumeric) descriptors are kept as kind ``'a'``: ETBLOCK is
    written as ``(2i9,19a9)``, carrying the KEYOPT values as text fields.
    """
    body = line.strip()
    if not (body.startswith("(") and body.endswith(")")):
        raise CdbError(f"expected a Fortran format line, got {line!r}", line=lineno)
    fields: list[tuple[str, int]] = []
    for part in body[1:-1].split(","):
        m = _FMT_ITEM.fullmatch(part.strip())
        if m is None:
            raise CdbError(f"unsupported format item {part.strip()!r} in {body}", line=lineno)
        rep = int(m.group(1) or 1)
        code = m.group(2).lower()
        kind = code if code in ("i", "a") else "f"
        fields.extend([(kind, int(m.group(3)))] * rep)
    return fields


def _to_float(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def _fields(line: str, fmt: list[tuple[str, int]], lineno: int) -> list[_Field]:
    """Slice one fixed-format line; blank fields become ``None``."""
    out: list[_Field] = []
    pos = 0
    for kind, width in fmt:
        raw = line[pos : pos + width].strip()
        pos += width
        try:
            if not raw:
                out.append(None)
            elif kind == "i":
                out.append(int(raw))
            elif kind == "a":
                out.append(raw)
            else:
                out.append(_to_float(raw))
        except ValueError as exc:
            raise CdbError(
                f"cannot parse field {raw!r} at line {lineno}: {line!r}", line=lineno
            ) from exc
    return out


def _first_int(fields: list[_Field]) -> int | None:
    return fields[0] if fields and isinstance(fields[0], int) else None


def _n_present(fields: list[_Field]) -> int:
    """Number of fields actually on the line (trailing blanks excluded)."""
    n = len(fields)
    while n and fields[n - 1] is None:
        n -= 1
    return n


def _read_extra_nodes(
    lines: list[str],
    i: int,
    fmt: list[tuple[str, int]],
    node_fields: list[int],
    needed: int,
    full: bool,
) -> int:
    """Consume EBLOCK continuation lines carrying additional node numbers.

    Per the CDB spec a record spills onto the next line only when the
    previous line was full, so ``full`` gates the loop -- a record that
    simply lists fewer nodes than the element's maximum (e.g. a BEAM188
    without its orientation node) must not swallow the following record.
    Returns the new line index.
    """
    n_lines = len(lines)
    while full and len(node_fields) < needed and i < n_lines:
        nxt = lines[i]
        if re.match(r"\s*[A-Za-z]", nxt):  # next command reached
            break
        extra = _fields(nxt, fmt, i + 1)
        first = _first_int(extra)
        if first is None or first < 0:  # block terminator
            break
        i += 1
        n = _n_present(extra)
        node_fields += [v if isinstance(v, int) else 0 for v in extra[:n]]
        full = n == len(fmt)
    return i


# ---------------------------------------------------------------------------
# element catalogue (keyed by the ANSYS element *name number*)
# ---------------------------------------------------------------------------

#: kind -> handling; n_expected is the full ANSYS node count (for compact
#: EBLOCK continuation detection)
_ANSYS_ELEMENTS: dict[int, tuple[str, int]] = {
    1: ("truss2d", 2),  # LINK1
    8: ("bar", 2),  # LINK8
    180: ("bar", 2),  # LINK180
    3: ("beam", 2),  # BEAM3 (2-D)
    4: ("beam", 3),  # BEAM4 (K = orientation node)
    188: ("beam", 3),  # BEAM188 (K = orientation node)
    41: ("shell", 4),  # SHELL41
    63: ("shell", 4),  # SHELL63
    181: ("shell", 4),  # SHELL181
    93: ("shell_quad", 8),  # SHELL93
    281: ("shell_quad", 8),  # SHELL281
    45: ("hex", 8),  # SOLID45
    185: ("hex", 8),  # SOLID185
    95: ("hex_quad", 20),  # SOLID95
    186: ("hex_quad", 20),  # SOLID186
    92: ("tet_quad", 10),  # SOLID92
    187: ("tet_quad", 10),  # SOLID187
    285: ("tet", 4),  # SOLID285
    21: ("mass", 1),  # MASS21
    14: ("spring", 2),  # COMBIN14
}

_MP_LABELS = frozenset(
    {"EX", "EY", "EZ", "NUXY", "NUYZ", "NUXZ", "PRXY", "PRYZ", "PRXZ",
     "DENS", "GXY", "GYZ", "GXZ", "ALPX", "ALPY", "ALPZ"}
)  # fmt: skip

_D_LABELS = {"UX": 0, "UY": 1, "UZ": 2, "ROTX": 3, "ROTY": 4, "ROTZ": 5}
_F_LABELS = {"FX": 0, "FY": 1, "FZ": 2, "MX": 3, "MY": 4, "MZ": 5}

_COUPLING_CMDS = frozenset({"CP", "CE", "CERIG", "RBE3", "CPBLOCK", "CEBLOCK"})


def _dedupe(nodes: list[int]) -> list[int]:
    """Drop duplicate node ids preserving order (ANSYS degenerate shapes)."""
    return list(dict.fromkeys(nodes))


class _ElemRecord:
    __slots__ = ("eid", "etype", "mat", "real", "sec", "nodes")

    def __init__(self, eid: int, etype: int, mat: int, real: int, sec: int,
                 nodes: list[int]) -> None:  # fmt: skip
        self.eid = eid
        self.etype = etype
        self.mat = mat
        self.real = real
        self.sec = sec
        self.nodes = nodes


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def read_cdb(path: str | Path) -> FEModel:
    """Read an ANSYS coded-database archive into an :class:`FEModel`.

    See the module docstring for the supported record subset and the
    element mapping table.  All records are collected first and the model
    is built in dependency order, so record order in the file does not
    matter (beyond ANSYS's own NBLOCK-before-EBLOCK convention).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    et_map: dict[int, int] = {}  # type id -> ANSYS name number
    raw_nodes: list[tuple[int, float, float, float]] = []
    n_rotated = 0
    records: list[_ElemRecord] = []
    mp: dict[int, dict[str, float]] = {}  # mat id -> label -> value
    rconst: dict[int, list[float]] = {}
    last_r_id: int | None = None
    # current element attributes (TYPE/MAT/REAL/SECNUM commands); a COMPACT
    # EBLOCK carries only node numbers and inherits these
    cur_attr = {"TYPE": 1, "MAT": 1, "REAL": 1, "SECNUM": 0}
    sec_kind: dict[int, str] = {}
    shell_t: dict[int, float] = {}
    current_sec: int | None = None
    spc_entries: list[tuple[int, int, float]] = []  # node, local dof, value
    load_entries: list[tuple[int, int, float]] = []  # node, local dof, value
    notes: list[str] = []
    coupling: dict[str, int] = {}
    csys_warned = False

    def _int_tok(toks: list[str], j: int, default: int | None = None) -> int | None:
        if j >= len(toks) or not toks[j].strip():
            return default
        try:
            return int(float(toks[j]))
        except ValueError as exc:
            raise CdbError(f"cannot parse integer {toks[j]!r} in {toks[0]} record") from exc

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].split("!", 1)[0]
        i += 1
        if not line.strip():
            continue
        toks = [t.strip() for t in line.split(",")]
        cmd = toks[0].upper()

        # ---- nodes ---------------------------------------------------------
        if cmd == "NBLOCK":
            fmt = _parse_format(lines[i], i + 1)
            i += 1
            n_ints = sum(1 for k, _ in fmt if k == "i")
            while i < n_lines:
                row = lines[i]
                if re.match(r"\s*N\s*,", row, re.IGNORECASE):  # "N,R5.3,LOC,-1" trailer
                    i += 1
                    break
                f = _fields(row, fmt, i + 1)
                nid = _first_int(f)
                if nid is None or nid < 0:
                    i += 1
                    break
                reals = [float(v) if v is not None else 0.0 for v in f[n_ints:]]
                reals += [0.0] * (6 - len(reals))
                if any(reals[3:6]):
                    n_rotated += 1
                raw_nodes.append((nid, reals[0], reals[1], reals[2]))
                i += 1

        # ---- elements ------------------------------------------------------
        elif cmd == "EBLOCK":
            key = toks[2].upper() if len(toks) > 2 else ""
            solid = key.startswith("SOLID")
            compact = key.startswith("COMPACT")
            fmt = _parse_format(lines[i], i + 1)
            i += 1
            while i < n_lines:
                f = _fields(lines[i], fmt, i + 1)
                first = _first_int(f)
                if first is None or first < 0:
                    i += 1
                    break
                i += 1
                present = _n_present(f)
                full = present == len(fmt)
                ints = [v if isinstance(v, int) else 0 for v in f]
                ints += [0] * max(0, 19 - len(ints))
                if solid:
                    mat, etype, real, sec = ints[0], ints[1], ints[2], ints[3]
                    nnodes, eid = ints[8], ints[10]
                    # zero placeholders kept so the field count matches nnodes
                    node_fields = ints[11:present]
                    i = _read_extra_nodes(lines, i, fmt, node_fields, nnodes, True)
                    nodes = [v for v in node_fields[:nnodes] if v]
                elif compact:
                    # field 1 = element number, the rest are node numbers;
                    # attributes come from the current TYPE/MAT/REAL/SECNUM
                    eid = ints[0]
                    etype, mat = cur_attr["TYPE"], cur_attr["MAT"]
                    real, sec = cur_attr["REAL"], cur_attr["SECNUM"]
                    node_fields = ints[1:present]
                    expected = _ANSYS_ELEMENTS.get(et_map.get(etype, -1), (None, 0))[1]
                    i = _read_extra_nodes(lines, i, fmt, node_fields, expected, full)
                    nodes = [v for v in node_fields if v]
                else:
                    eid, etype, real, mat = ints[0], ints[1], ints[2], ints[3]
                    sec = 0
                    node_fields = ints[5:present]
                    expected = _ANSYS_ELEMENTS.get(et_map.get(etype, -1), (None, 0))[1]
                    i = _read_extra_nodes(lines, i, fmt, node_fields, expected, full)
                    nodes = [v for v in node_fields if v]
                records.append(_ElemRecord(eid, etype, mat, real, sec, nodes))

        # ---- element types ---------------------------------------------------
        elif cmd == "ET":
            itype = _int_tok(toks, 1)
            ename = toks[2] if len(toks) > 2 else ""
            m = re.search(r"(\d+)\s*$", ename)
            if itype is not None and m:
                et_map[itype] = int(m.group(1))
        elif cmd == "ETBLOCK":
            fmt = _parse_format(lines[i], i + 1)
            i += 1
            while i < n_lines:
                row = lines[i]
                if re.match(r"\s*[A-Za-z(]", row):  # next command reached
                    break
                i += 1
                if re.match(r"\s*-1\b", row):  # block trailer
                    break
                f = _fields(row, fmt, i)
                itype = _first_int(f)
                if itype is None or itype < 0:
                    break
                if len(f) > 1 and isinstance(f[1], int):
                    et_map[itype] = f[1]

        # ---- materials --------------------------------------------------------
        elif cmd in ("MP", "MPDATA"):
            lab_idx = next((j for j, t in enumerate(toks) if t.upper() in _MP_LABELS), None)
            if lab_idx is None:
                continue  # e.g. unsupported label (KXX, MURX, ...)
            label = toks[lab_idx].upper()
            mat_id = _int_tok(toks, lab_idx + 1)
            if mat_id is None:
                continue
            if cmd == "MP":
                value_tok = toks[lab_idx + 2] if len(toks) > lab_idx + 2 else ""
            else:  # MPDATA ..., lab, mat, stloc, c1
                stloc = _int_tok(toks, lab_idx + 2, 1) or 1
                if stloc > 1:
                    notes.append(
                        f"material {mat_id}: only the first temperature point of "
                        f"{label} is used"
                    )
                    continue
                value_tok = toks[lab_idx + 3] if len(toks) > lab_idx + 3 else ""
            if value_tok:
                mp.setdefault(mat_id, {}).setdefault(label, _to_float(value_tok))

        # ---- real constants ------------------------------------------------------
        elif cmd == "R":
            args = toks[1:]
            # CDWRITE inserts a revision token ("R5.0", newer "UNBL")
            if args and re.fullmatch(r"R\d+(\.\d+)?|UNBL", args[0].upper()):
                args = args[1:]
            rtoks = [cmd, *args]
            rid = _int_tok(rtoks, 1)
            if rid is not None:
                if len(args) > 1 and args[1].upper() == "LOC":
                    # archive-file style: R,NSET,LOC,STLOC,VAL1,VAL2,VAL3
                    stloc = _int_tok(rtoks, 3, 1) or 1
                    vals = [_to_float(t) if t else 0.0 for t in args[3:]]
                    table = rconst.setdefault(rid, [])
                    table.extend([0.0] * (stloc - 1 + len(vals) - len(table)))
                    for k, v in enumerate(vals):
                        table[stloc - 1 + k] = v
                else:  # interactive style: R,NSET,VAL1,...,VAL6
                    vals = [_to_float(t) if t else 0.0 for t in args[1:]]
                    rconst.setdefault(rid, []).extend(vals)
                last_r_id = rid
        elif cmd == "RMORE":
            if last_r_id is not None:
                # RMORE always continues at constants 7-12 (then 13-18, ...),
                # even when the R line carried fewer than 6 values
                table = rconst[last_r_id]
                start = -(-len(table) // 6) * 6
                table.extend([0.0] * (start - len(table)))
                table.extend(_to_float(t) if t else 0.0 for t in toks[1:])
        elif cmd == "RLBLOCK":
            n_sets = _int_tok(toks, 1, 0) or 0
            fmt1 = _parse_format(lines[i], i + 1)
            fmt2 = _parse_format(lines[i + 1], i + 2)
            i += 2
            for _ in range(n_sets):
                f = _fields(lines[i], fmt1, i + 1)
                i += 1
                if not isinstance(f[0], int) or not isinstance(f[1], int):
                    raise CdbError(f"malformed RLBLOCK set at line {i}", line=i)
                rid, n_items = f[0], f[1]
                vals = [float(v) if v is not None else 0.0 for v in f[2:]]
                while len(vals) < n_items and i < n_lines:
                    more = _fields(lines[i], fmt2, i + 1)
                    i += 1
                    vals += [float(v) if v is not None else 0.0 for v in more]
                vals = vals[:n_items] + [0.0] * max(0, n_items - len(vals))
                rconst[rid] = vals
                last_r_id = rid

        # ---- sections ---------------------------------------------------------------
        elif cmd == "SECTYPE":
            # SECTYPE, SECID, Type[, Subtype, Name]  (an "R5.0" revision token
            # may be inserted after the command name by CDWRITE)
            args = toks[1:]
            if args and re.fullmatch(r"R\d+(\.\d+)?", args[0].upper()):
                args = args[1:]
            sec_id = _int_tok(args, 0)
            if sec_id is not None:
                sec_kind[sec_id] = args[1].upper() if len(args) > 1 else ""
                current_sec = sec_id
        elif cmd == "SECDATA":
            if current_sec is not None and sec_kind.get(current_sec) == "SHELL" and len(toks) > 1:
                try:
                    shell_t[current_sec] = shell_t.get(current_sec, 0.0) + _to_float(toks[1])
                except ValueError:
                    notes.append(f"SECDATA for section {current_sec} not understood; skipped")
        elif cmd == "SECBLOCK":
            if current_sec is not None and sec_kind.get(current_sec) == "SHELL":
                n_layers = _int_tok(toks, 1, 0) or 0
                total = 0.0
                try:
                    for _ in range(n_layers):
                        total += _to_float(lines[i].split(",")[0])
                        i += 1
                    shell_t[current_sec] = total
                except (ValueError, IndexError):
                    notes.append(f"SECBLOCK for section {current_sec} not understood; skipped")

        # ---- constraints and loads ------------------------------------------------
        elif cmd == "D":
            nid = _int_tok(toks, 1)
            lab = toks[2].upper() if len(toks) > 2 else ""
            value = _to_float(toks[3]) if len(toks) > 3 and toks[3] else 0.0
            if nid is not None:
                if lab == "ALL":
                    spc_entries.extend((nid, d, value) for d in range(6))
                elif lab in _D_LABELS:
                    spc_entries.append((nid, _D_LABELS[lab], value))
                else:
                    notes.append(f"D constraint label {lab!r} at node {nid} not supported")
        elif cmd == "F":
            nid = _int_tok(toks, 1)
            lab = toks[2].upper() if len(toks) > 2 else ""
            value = _to_float(toks[3]) if len(toks) > 3 and toks[3] else 0.0
            if nid is not None:
                if lab in _F_LABELS:
                    load_entries.append((nid, _F_LABELS[lab], value))
                else:
                    notes.append(f"F load label {lab!r} at node {nid} not supported")

        # ---- current element attributes (consumed by COMPACT EBLOCKs) --------------
        elif cmd in ("TYPE", "MAT", "REAL", "SECNUM"):
            val = _int_tok(toks, 1)
            if val is not None:
                cur_attr[cmd] = val

        # ---- things we refuse politely -----------------------------------------------
        elif cmd in _COUPLING_CMDS:
            coupling[cmd] = coupling.get(cmd, 0) + 1
        elif cmd == "CSYS" and not csys_warned:
            if (_int_tok(toks, 1, 0) or 0) != 0:
                notes.append(
                    "CSYS selects a non-global system; NBLOCK coordinates are "
                    "read as global cartesian regardless"
                )
                csys_warned = True
        # every other command: bookkeeping, silently ignored

    # -- build the model ----------------------------------------------------
    model = FEModel(name=path.stem)

    for nid, x, y, z in raw_nodes:
        model.add_node(id=nid, xyz=(x, y, z))
    if n_rotated:
        notes.append(
            f"nodal rotation angles (THXY/THYZ/THZX) on {n_rotated} node(s) ignored; "
            "displacement output systems are not carried over"
        )

    for mid in sorted(mp):
        d = mp[mid]
        e = d.get("EX")
        g = d.get("GXY")
        if e is None and g is None:
            notes.append(f"material {mid}: no EX/GXY data; material skipped")
            continue
        model.add_material(
            id=mid,
            type="isotropic",
            E=e,
            nu=d.get("NUXY", d.get("PRXY")),
            rho=d.get("DENS"),
            G=g,
            alpha=d.get("ALPX"),
        )

    # -- synthesize properties, one per (kind, mat, real, sec) combination ----
    prop_ids: dict[tuple, int] = {}
    missing_mats: set[int] = set()
    beam_sections: set[int] = set()
    beam3_reals: set[int] = set()
    missing_shell_t: set[int] = set()

    def _rc(rid: int, k: int) -> float:
        vals = rconst.get(rid, [])
        return float(vals[k]) if k < len(vals) else 0.0

    def _property_for(kind: str, rec: _ElemRecord, ansys_num: int) -> int:
        base = {"truss2d": "bar", "shell_quad": "shell", "hex_quad": "solid",
                "hex": "solid", "tet": "solid", "tet_quad": "solid"}.get(kind, kind)  # fmt: skip
        key = (base, rec.mat, rec.real, rec.sec)
        if key in prop_ids:
            return prop_ids[key]
        pid = len(prop_ids) + 1
        extra = {"ansys_type": ansys_num, "ansys_real": rec.real, "ansys_secnum": rec.sec}
        if rec.mat not in model.materials:
            missing_mats.add(rec.mat)
        if base == "solid":
            model.add_property(id=pid, type="solid", material_id=rec.mat,
                               check_refs=False, **extra)  # fmt: skip
        elif base == "shell":
            t = shell_t.get(rec.sec)
            if t is None and rec.real in rconst:  # legacy shells: thickness = RC 1
                t = _rc(rec.real, 0)
            if not t:
                missing_shell_t.add(rec.sec or rec.real)
                t = 0.0
            model.add_property(id=pid, type="shell", material_id=rec.mat, t=t,
                               check_refs=False, **extra)  # fmt: skip
        elif base == "beam":
            if ansys_num == 4:  # BEAM4: R = AREA, IZZ, IYY, ..., IXX(8th)
                model.add_property(id=pid, type="beam", material_id=rec.mat,
                                   A=_rc(rec.real, 0), Iy=_rc(rec.real, 2),
                                   Iz=_rc(rec.real, 1), J=_rc(rec.real, 7),
                                   check_refs=False, **extra)  # fmt: skip
            elif ansys_num == 3:  # BEAM3 (2-D): R = AREA, IZZ, HEIGHT -- no IYY/IXX
                beam3_reals.add(rec.real)
                izz = _rc(rec.real, 1)
                # BEAM2's own missing-inertia fallbacks (Iy=Iz, J=Iy+Iz),
                # applied here because the model layer requires the fields
                model.add_property(id=pid, type="beam", material_id=rec.mat,
                                   A=_rc(rec.real, 0), Iy=izz, Iz=izz,
                                   J=2.0 * izz, check_refs=False, **extra)  # fmt: skip
            else:  # BEAM188: section integration data, not derivable here
                beam_sections.add(rec.sec)
                model.add_property(id=pid, type="beam", material_id=rec.mat, A=0.0,
                                   Iy=0.0, Iz=0.0, J=0.0, check_refs=False, **extra)  # fmt: skip
        elif base == "bar":
            model.add_property(id=pid, type="bar", material_id=rec.mat,
                               A=_rc(rec.real, 0), check_refs=False, **extra)  # fmt: skip
        elif base == "spring":
            model.add_property(id=pid, type="lumped", k=_rc(rec.real, 0), attrs=extra)
        else:  # mass
            model.add_property(id=pid, type="lumped", m=_rc(rec.real, 0), attrs=extra)
        prop_ids[key] = pid
        return pid

    unknown_et: dict[int, int] = {}
    unsupported: dict[int, int] = {}
    degenerate_skips: dict[str, list[int]] = {}
    midside_drops: dict[int, list[int]] = {}

    for rec in records:
        ansys_num = et_map.get(rec.etype)
        if ansys_num is None:
            unknown_et[rec.etype] = unknown_et.get(rec.etype, 0) + 1
            continue
        info = _ANSYS_ELEMENTS.get(ansys_num)
        if info is None:
            unsupported[ansys_num] = unsupported.get(ansys_num, 0) + 1
            continue
        kind, _n_expected = info
        nodes = rec.nodes
        ftype: str
        orientation = None
        if kind in ("bar", "truss2d"):
            ftype, nodes = ("BAR2" if kind == "bar" else "TRUSS2D"), nodes[:2]
        elif kind == "beam":
            ftype = "BEAM2"
            if len(nodes) > 2 and nodes[2] in model.nodes and nodes[0] in model.nodes:
                orientation = model.nodes[nodes[2]].xyz - model.nodes[nodes[0]].xyz
            nodes = nodes[:2]
        elif kind in ("shell", "shell_quad"):
            corners = _dedupe(nodes[:4])
            if kind == "shell_quad" and len(nodes) > 4:
                midside_drops.setdefault(ansys_num, []).append(rec.eid)
            if len(corners) == 4:
                ftype, nodes = "QUAD4", corners
            elif len(corners) == 3:
                ftype, nodes = "TRIA3", corners
            else:
                degenerate_skips.setdefault("degenerate shell", []).append(rec.eid)
                continue
        elif kind in ("hex", "hex_quad", "tet", "tet_quad"):
            if kind == "tet_quad" and len(_dedupe(nodes)) == 10:
                # complete quadratic tet: first-class TET10 since Round 10
                # (SOLID92/187 order I..L corners then the 6 midsides, the
                # same convention as the 10-node CTETRA)
                model.add_element(
                    id=rec.eid,
                    type="TET10",
                    nodes=tuple(nodes[:10]),
                    property_id=_property_for(kind, rec, ansys_num),
                )
                continue
            if kind in ("hex_quad", "tet_quad") and len(nodes) > (
                8 if kind == "hex_quad" else 4
            ):
                midside_drops.setdefault(ansys_num, []).append(rec.eid)
            corners = _dedupe(nodes[: 8 if kind.startswith("hex") else 4])
            if len(corners) == 8:
                ftype, nodes = "HEX8", corners
            elif len(corners) == 4:
                ftype, nodes = "TET4", corners
            else:
                degenerate_skips.setdefault(
                    "wedge/pyramid solid (no femtools element)", []
                ).append(rec.eid)
                continue
        elif kind == "mass":
            ftype, nodes = "MASS", nodes[:1]
        else:  # spring
            ftype, nodes = "SPRING", _dedupe(nodes)[:2]
        model.add_element(
            id=rec.eid,
            type=ftype,
            nodes=tuple(nodes),
            property_id=_property_for(kind, rec, ansys_num),
            orientation=orientation,
        )

    # -- constraints / loads --------------------------------------------------
    spc_grouped: dict[tuple[int, float], list[bool]] = {}
    for nid, dof, value in spc_entries:
        spc_grouped.setdefault((nid, value), [False] * 6)[dof] = True
    for (nid, value), mask in sorted(spc_grouped.items()):
        if nid in model.nodes:
            model.add_spc(node_id=nid, mask=mask, value=value, sid=1)
        else:
            notes.append(f"D constraint references undefined node {nid}; skipped")

    load_grouped: dict[int, list[float]] = {}
    for nid, dof, value in load_entries:
        load_grouped.setdefault(nid, [0.0] * 6)[dof] += value
    for nid, vec in sorted(load_grouped.items()):
        if nid not in model.nodes:
            notes.append(f"F load references undefined node {nid}; skipped")
            continue
        force = np.array(vec[:3]) if any(vec[:3]) else None
        moment = np.array(vec[3:]) if any(vec[3:]) else None
        if force is not None or moment is not None:
            model.add_load(node_id=nid, force=force, moment=moment, sid=1)

    # -- aggregated diagnostics -------------------------------------------------
    for etype, count in sorted(unknown_et.items()):
        notes.append(f"element type {etype} has no ET record; skipped {count} element(s)")
    for num, count in sorted(unsupported.items()):
        notes.append(f"unsupported ANSYS element {num}; skipped {count} element(s)")
    for num, eids in sorted(midside_drops.items()):
        shown = ", ".join(map(str, eids[:5])) + (", ..." if len(eids) > 5 else "")
        notes.append(
            f"ANSYS element {num}: midside nodes dropped on {len(eids)} element(s) "
            f"[{shown}]; quadratic accuracy is lost"
        )
    for what, eids in sorted(degenerate_skips.items()):
        shown = ", ".join(map(str, eids[:5])) + (", ..." if len(eids) > 5 else "")
        notes.append(f"skipped {len(eids)} {what} element(s) [{shown}]")
    if missing_mats:
        notes.append(
            f"element(s) reference material id(s) {sorted(missing_mats)} with no "
            "MP data; properties keep the dangling reference"
        )
    if missing_shell_t:
        notes.append(
            f"no thickness found for shell section/real id(s) {sorted(missing_shell_t)}; "
            "t = 0.0 written (fix before solving)"
        )
    if beam3_reals:
        notes.append(
            f"BEAM3 real set(s) {sorted(beam3_reals)} carry A and IZZ only (2-D "
            "element; RC 3 is the stress-recovery HEIGHT, not IYY); the BEAM2 "
            "fallbacks Iy=Iz and J=Iy+Iz were applied"
        )
    if beam_sections:
        notes.append(
            f"BEAM188 section(s) {sorted(beam_sections)}: A/Iy/Iz/J are not derived "
            "from SECDATA; beam properties are zero (fix before solving)"
        )
    for cmd_name, count in sorted(coupling.items()):
        notes.append(f"coupling/constraint-equation record {cmd_name} (x{count}) not supported")
    for note in notes:
        warnings.warn(f"read_cdb({path.name}): {note}", UserWarning, stacklevel=2)
    return model


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

#: femtools element type -> ANSYS element name number (writer side).
_WRITE_NUM: dict[str, int] = {
    "HEX8": 185,  # SOLID185
    "TET4": 285,  # SOLID285
    "TET10": 187,  # SOLID187 (10 nodes; EBLOCK spills onto a second line)
    "QUAD4": 181,  # SHELL181
    "TRIA3": 181,  # SHELL181, degenerate quad (n3 repeated)
    "BEAM2": 4,  # BEAM4 (section from real constants, K orientation node)
    "BAR2": 180,  # LINK180
    "TRUSS2D": 1,  # LINK1
    "MASS": 21,  # MASS21
    "SPRING": 14,  # COMBIN14
}

_D_NAMES = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")
_F_NAMES = ("FX", "FY", "FZ", "MX", "MY", "MZ")


def _gv(v: float | None) -> str:
    """Full-precision value for free-format command records (MP, R, ...)."""
    return format(0.0 if v is None else float(v), ".17g")


def _i9(v: int, what: str) -> str:
    s = f"{int(v):9d}"
    if len(s) > 9:
        raise CdbError(f"{what} {v} does not fit the blocked i9 field")
    return s


def write_cdb(path: str | Path | FEModel, model: FEModel | str | Path | None = None) -> None:
    """Write an :class:`FEModel` as an ANSYS coded-database archive.

    Accepts ``write_cdb(path, model)`` or ``write_cdb(model, path)``.

    Emits exactly the record subset :func:`read_cdb` parses (see the
    module docstring for the element mapping and the documented losses):
    ``ET`` records (one TYPE per ANSYS element number), free-format
    ``MP`` materials, ``R``/``RMORE`` real-constant sets and
    ``SECTYPE``/``SECDATA`` shell sections keyed by the femtools property
    id, a fixed-format ``NBLOCK``/``EBLOCK`` (SOLID key) pair and
    ``D``/``F`` constraint/load records.  BEAM2 orientation vectors are
    materialized as BEAM4 K nodes: one extra grid per distinct
    ``(end A, orientation)`` position, ids continuing after the real
    nodes (they read back as unattached nodes; the assembler drops their
    DOFs).  Every degradation raises one aggregated ``UserWarning``.
    """
    from ._compat import coerce_path_model

    out_path, model = coerce_path_model(path, model)
    notes: list[str] = []

    # -- plan elements (and the beam orientation nodes they need) -----------
    next_nid = max(model.nodes, default=0) + 1
    orient_ids: dict[tuple[float, float, float], int] = {}
    extra_nodes: list[tuple[int, float, float, float]] = []
    # (eid, mat, ansys_num, real, sec, nodes)
    planned: list[tuple[int, int, int, int, int, list[int]]] = []
    dropped_dampers: list[int] = []
    dropped_pins: list[int] = []

    for eid in sorted(model.elements):
        el = model.elements[eid]
        if el.type == "DAMPER":
            dropped_dampers.append(eid)
            continue
        num = _WRITE_NUM[el.type]
        pid = el.property_id or 0
        prop = model.properties.get(pid)
        mat = prop.material_id if prop is not None and prop.material_id is not None else 0
        real, sec = pid, 0
        nodes = list(el.nodes)
        if el.type == "BEAM2" and el.orientation is not None:
            xyz = model.nodes[el.nodes[0]].xyz + np.asarray(el.orientation, dtype=float)
            key = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            k = orient_ids.get(key)
            if k is None:
                k = next_nid
                next_nid += 1
                orient_ids[key] = k
                extra_nodes.append((k, *key))
            nodes.append(k)
        elif el.type in ("QUAD4", "TRIA3"):
            real, sec = 0, pid
            if el.type == "TRIA3":
                nodes.append(nodes[2])  # ANSYS degenerate-quad convention
        elif el.type == "SPRING" and el.dofs is not None:
            dropped_pins.append(eid)
        planned.append((eid, mat, num, real, sec, nodes))

    used_nums = sorted({num for _, _, num, _, _, _ in planned})
    type_of = {num: tid for tid, num in enumerate(used_nums, start=1)}

    lines: list[str] = [
        f"/COM, femtools coded-database export -- model: {model.name}",
        "/PREP7",
    ]

    # -- element types --------------------------------------------------------
    for num in used_nums:
        lines.append(f"ET,{type_of[num]},{num}")

    # -- materials -------------------------------------------------------------
    for mid in sorted(model.materials):
        mtl = model.materials[mid]
        if mtl.type != "isotropic":
            warnings.warn(
                f"write_cdb: material {mid} is {mtl.type}; written as isotropic "
                "MP records with E1/nu12",
                UserWarning,
                stacklevel=2,
            )
            e, nu, g = mtl.E1, mtl.nu12, None
        else:
            e, nu, g = mtl.E, mtl.nu, mtl.G
        if e is not None:
            lines.append(f"MP,EX,{mid},{_gv(e)}")
        elif g is not None:
            lines.append(f"MP,GXY,{mid},{_gv(g)}")
        if nu is not None:
            lines.append(f"MP,NUXY,{mid},{_gv(nu)}")
        if mtl.rho is not None:
            lines.append(f"MP,DENS,{mid},{_gv(mtl.rho)}")
        if mtl.alpha is not None:
            lines.append(f"MP,ALPX,{mid},{_gv(mtl.alpha)}")
        if mtl.damping:
            notes.append(f"material {mid}: structural damping GE has no MP record; dropped")

    # -- real constants / sections (keyed by the femtools property id) ---------
    for pid in sorted(model.properties):
        prop = model.properties[pid]
        if prop.type == "beam":
            # BEAM4 layout: R1 AREA, R2 IZZ, R3 IYY / RMORE: R7 ISTRN, R8 IXX
            lines.append(f"R,{pid},{_gv(prop.A)},{_gv(prop.Iz)},{_gv(prop.Iy)}")
            lines.append(f"RMORE,0.,{_gv(prop.J)}")
            if prop.kappa is not None:
                notes.append(f"beam property {pid}: shear factor kappa not written")
        elif prop.type == "bar":
            lines.append(f"R,{pid},{_gv(prop.A)}")
            if prop.J:
                notes.append(f"bar property {pid}: torsion constant J not written")
        elif prop.type == "shell":
            lines.append(f"SECTYPE,{pid},SHELL")
            lines.append(f"SECDATA,{_gv(prop.t)}")
        elif prop.type == "lumped":  # MASS21 / COMBIN14 read their value from RC 1
            value = prop.m if prop.m is not None else prop.k
            if prop.m is not None and prop.k is not None:
                notes.append(
                    f"lumped property {pid} carries both m and k; real constant 1 "
                    "holds m (a COMBIN14 sharing this set reads it as stiffness)"
                )
            if value is None:
                value = 0.0
                notes.append(f"lumped property {pid}: damping-only value not written")
            lines.append(f"R,{pid},{_gv(value)}")
        if prop.nsm:
            notes.append(f"property {pid}: non-structural mass has no CDB record; dropped")

    # -- nodes ------------------------------------------------------------------
    lines.append("NBLOCK,6,SOLID")
    lines.append("(3i9,6e21.13e3)")
    all_nodes = [(nid, *model.nodes[nid].xyz) for nid in model.node_ids()]
    for nid, x, y, z in [*all_nodes, *extra_nodes]:
        ints = _i9(nid, "node id") + _i9(0, "field") + _i9(0, "field")
        lines.append(ints + "".join(f"{float(v):21.13E}" for v in (x, y, z)))
    lines.append("N,R5.3,LOC,       -1,")
    if any(model.nodes[nid].cd for nid in model.nodes):
        notes.append("nodal output systems (cd) have no NBLOCK record; dropped")

    # -- elements ----------------------------------------------------------------
    if planned:
        lines.append(f"EBLOCK,19,SOLID,,{len(planned)}")
        lines.append("(19i9)")
        for eid, mat, num, real, sec, nodes in planned:
            # SOLID-key record: 11 attribute fields + the first 8 nodes fill
            # the 19-field line; further nodes (TET10) spill onto full-width
            # continuation lines, exactly as the reader consumes them.
            fields = [mat, type_of[num], real, sec, 0, 0, 0, 0, len(nodes), 0, eid, *nodes[:8]]
            lines.append("".join(_i9(v, "EBLOCK field") for v in fields))
            for start in range(8, len(nodes), 19):
                lines.append(
                    "".join(_i9(v, "EBLOCK field") for v in nodes[start : start + 19])
                )
        lines.append(_i9(-1, "terminator"))

    # -- constraints and loads ------------------------------------------------------
    for spc in model.spcs:
        if not any(spc.mask):
            continue
        if spc.sid != 1:
            notes.append("SPC set ids other than 1 collapse to set 1 on read")
        if all(spc.mask):
            lines.append(f"D,{spc.node_id},ALL,{_gv(spc.value)}")
        else:
            for d, on in enumerate(spc.mask):
                if on:
                    lines.append(f"D,{spc.node_id},{_D_NAMES[d]},{_gv(spc.value)}")
    for load in model.loads:
        if load.sid != 1:
            notes.append("load set ids other than 1 collapse to set 1 on read")
        for d, value in load.as_dof_values():
            lines.append(f"F,{load.node_id},{_F_NAMES[d]},{_gv(value)}")

    if model.rbe2:
        notes.append(
            f"{len(model.rbe2)} RBE2 table(s) dropped (CERIG/CE records are outside "
            "the readable subset)"
        )
    if dropped_dampers:
        notes.append(
            f"{len(dropped_dampers)} DAMPER element(s) dropped (COMBIN14 reads back "
            "as a spring; no damper mapping in this subset)"
        )
    if dropped_pins:
        notes.append(
            f"SPRING DOF pins dropped on {len(dropped_pins)} element(s) "
            "(COMBIN14 reads back as an axial spring)"
        )

    lines.append("FINISH")
    for note in dict.fromkeys(notes):
        warnings.warn(f"write_cdb: {note}", UserWarning, stacklevel=2)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Abaqus input-file (.inp) translator -- keyword TEXT subset.

An original parser built from the publicly documented keyword layouts
(same policy as the BDF/CDB translators: **no** vendor code, **no**
binary .odb support, by design).  Supported keywords:

=========================  ==================================================
keyword                    use
=========================  ==================================================
``*HEADING``               first data line -> model name
``*NODE``                  ``id, x, y, z`` (``NSET=`` collected)
``*ELEMENT``               ``TYPE=`` required (``ELSET=`` collected); see
                           the element table below
``*NSET`` / ``*ELSET``     id lists (``GENERATE`` supported; previously
                           defined set names may appear in the data lines)
``*MATERIAL``              named material, filled by the following options
``*ELASTIC``               ``E, nu`` (TYPE=ISOTROPIC only; first row used)
``*DENSITY``               ``rho`` (first row used)
``*SOLID SECTION``         solids: material binding; T3D2 trusses: the
                           data line holds the cross-section area
                           (default 1.0, as documented)
``*SHELL SECTION``         data line 1: thickness
``*BEAM SECTION``          ``SECTION=RECT|CIRC``: dimensions -> A/I11/
                           I22/J via the standard closed-form section
                           formulas; data line 2: local 1-axis direction
``*BEAM GENERAL SECTION``  data line 1: ``A, I11, I12, I22, J``; data
                           line 2: local 1-axis (blank -> default);
                           data line 3: ``E, G``; density from the
                           ``DENSITY=`` parameter (an auto material is
                           created per section)
``*BOUNDARY``              ``node/nset, first[, last[, value]]`` or a
                           named type (ENCASTRE, PINNED,
                           {X,Y,Z}SYMM/ASYMM)
=========================  ==================================================

Element type mapping (Abaqus -> femtools):

=================  ========  =============================================
Abaqus TYPE        femtools  notes
=================  ========  =============================================
C3D8, C3D8R/I      HEX8
C3D4               TET4
C3D10, C3D10M      TET10     all 10 nodes kept (Round 10)
S4, S4R            QUAD4
S3, S3R            TRIA3
B31                BEAM2     orientation = section local 1-axis (n1)
T3D2               BAR2
=================  ========  =============================================

Abaqus section moments map to femtools as ``I11 -> Iy``, ``I22 -> Iz``:
the section's first axis ``n1`` is stored as the femtools beam
``orientation`` vector (which spans the local x-y plane, i.e. points
along the local y axis), so bending *about* the 1-axis -- deflection in
the 2-direction -- is bending about local y.  A missing torsion
constant falls back to ``J = I11 + I22`` with an aggregated warning
(same practice as the CDB BEAM3 reader).

Keyword lines and data lines ending in a comma continue on the next
line; ``**`` comment lines are skipped; keywords and parameters are
case-insensitive.  Unknown keywords (``*STEP``, ``*STATIC``, ...) are
ignored with **one aggregated** ``UserWarning`` per keyword name (like
the BDF midside drop) and do not terminate a ``*MATERIAL`` definition
(so unsupported material options such as ``*PLASTIC`` pass through
harmlessly); malformed content raises :class:`InpError`, a
:class:`~femtools.core.errors.FileFormatError`.  An .inp carries no
unit information: the returned model keeps the default SI
:class:`~femtools.core.units.UnitSystem`.

``write_inp`` emits the same subset: nodes, elements grouped per
property (auto ``ELSET`` per property id), ``*MATERIAL``/``*ELASTIC``/
``*DENSITY``, the matching section keyword per property type and
``*BOUNDARY`` lines; lumped MASS/SPRING/DAMPER elements and nodal loads
have no mapping in this subset and are skipped with a warning.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import FileFormatError
from ..core.model import FEModel
from ..core.sets import ElementSet, NodeSet

__all__ = ["read_inp", "write_inp", "InpError"]


class InpError(FileFormatError):
    """Raised for malformed Abaqus input-file content (a :class:`ValueError`
    via :class:`~femtools.core.errors.FileFormatError`)."""


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

#: Keywords after which a *MATERIAL definition is still open (its options).
_MATERIAL_OPTIONS = frozenset({"ELASTIC", "DENSITY"})


@dataclass
class _Block:
    """One keyword block: ``*KEYWORD, PARAM=VALUE, FLAG`` + its data lines.

    A blank physical line inside a block is kept as an empty row: it is
    positionally meaningful on ``*BEAM GENERAL SECTION`` (a blank second
    data line selects the default local 1-axis).
    """

    keyword: str
    params: dict[str, str]
    data: list[list[str]] = field(default_factory=list)
    line: int = 0


def _join_continuations(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Merge lines that end with a comma into the following line.

    A keyword line never continues a data line: a trailing comma followed
    by ``*KEYWORD`` (tolerated by Abaqus for a trailing empty field) flushes
    the pending line instead of swallowing the keyword.
    """
    out: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for lineno, line in lines:
        if pending is not None:
            if line.lstrip().startswith("*"):
                out.append(pending)
            else:
                lineno, line = pending[0], pending[1] + " " + line.strip()
            pending = None
        if line.rstrip().endswith(","):
            pending = (lineno, line.rstrip())
        else:
            out.append((lineno, line))
    if pending is not None:  # trailing comma at EOF: keep what we have
        out.append(pending)
    return out


def _parse_keyword_line(line: str, lineno: int) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in line.split(",")]
    keyword = " ".join(parts[0].lstrip("*").upper().split())
    if not keyword:
        raise InpError(f"empty keyword line {line!r}", line=lineno)
    params: dict[str, str] = {}
    for p in parts[1:]:
        if not p:
            continue
        key, eq, value = p.partition("=")
        params[" ".join(key.upper().split())] = value.strip().strip('"') if eq else ""
    return keyword, params


def _blocks(text: str) -> list[_Block]:
    physical = [
        (i + 1, raw.rstrip())
        for i, raw in enumerate(text.splitlines())
        if not raw.lstrip().startswith("**")
    ]
    blocks: list[_Block] = []
    for lineno, line in _join_continuations(physical):
        stripped = line.strip()
        if stripped.startswith("*"):
            keyword, params = _parse_keyword_line(stripped, lineno)
            blocks.append(_Block(keyword, params, line=lineno))
        elif not stripped:
            if blocks:
                blocks[-1].data.append([])  # blank row (positional placeholder)
        else:
            if not blocks:
                raise InpError(f"data line before any keyword: {stripped!r}", line=lineno)
            blocks[-1].data.append([t.strip() for t in line.split(",")])
    if not blocks:
        raise InpError("no Abaqus keywords found (not an .inp deck?)")
    return blocks


def _f(tok: str, lineno: int, what: str) -> float:
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise InpError(f"cannot parse {tok!r} as a real number ({what})", line=lineno) from exc


def _i(tok: str, lineno: int, what: str) -> int:
    try:
        return int(tok)
    except ValueError as exc:
        raise InpError(f"cannot parse {tok!r} as an integer ({what})", line=lineno) from exc


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

#: Abaqus element TYPE -> (femtools type, node count).
_ELEMENT_TYPES: dict[str, tuple[str, int]] = {
    "C3D8": ("HEX8", 8),
    "C3D8R": ("HEX8", 8),
    "C3D8I": ("HEX8", 8),
    "C3D4": ("TET4", 4),
    "C3D10": ("TET10", 10),
    "C3D10M": ("TET10", 10),
    "S4": ("QUAD4", 4),
    "S4R": ("QUAD4", 4),
    "S3": ("TRIA3", 3),
    "S3R": ("TRIA3", 3),
    "B31": ("BEAM2", 2),
    "T3D2": ("BAR2", 2),
}

#: Named *BOUNDARY types -> constrained local DOFs (0-based).
_BOUNDARY_TYPES: dict[str, tuple[int, ...]] = {
    "ENCASTRE": (0, 1, 2, 3, 4, 5),
    "PINNED": (0, 1, 2),
    "XSYMM": (0, 4, 5),
    "YSYMM": (1, 3, 5),
    "ZSYMM": (2, 3, 4),
    "XASYMM": (1, 2, 3),
    "YASYMM": (0, 2, 4),
    "ZASYMM": (0, 1, 5),
}

#: Default beam local 1-axis when the section's direction line is blank.
_DEFAULT_N1 = (0.0, 0.0, -1.0)


@dataclass
class _MaterialRec:
    name: str
    E: float | None = None
    nu: float | None = None
    rho: float | None = None


@dataclass
class _SectionRec:
    kind: str  # "solid" | "shell" | "beam"
    elset: str
    material: str | None
    block: _Block


def _rect_torsion(a: float, b: float) -> float:
    """Saint-Venant torsion constant of an ``a x b`` rectangle
    (Roark/Timoshenko closed form; long side ``L``, short side ``s``)."""
    length, s = (a, b) if a >= b else (b, a)
    if s <= 0.0 or length <= 0.0:
        return 0.0
    return length * s**3 * (1.0 / 3.0 - 0.21 * (s / length) * (1.0 - s**4 / (12.0 * length**4)))


def read_inp(path: str | Path) -> FEModel:
    """Read an Abaqus input-file TEXT subset into an :class:`FEModel`.

    Keyword blocks are collected first and the model is built in
    dependency order (nodes, materials, sections/properties, elements,
    boundaries, sets), so block order in the file does not matter --
    except that ``*ELASTIC`` / ``*DENSITY`` bind to the preceding
    ``*MATERIAL``, as Abaqus defines.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = _blocks(text)

    heading: str | None = None
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: dict[int, tuple[str, tuple[int, ...]]] = {}  # eid -> (type, nodes)
    nsets: dict[str, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    set_case: dict[str, str] = {}  # upper name -> first spelling seen
    materials: dict[str, _MaterialRec] = {}
    sections: list[_SectionRec] = []
    boundary_rows: list[tuple[str, tuple[int, ...], float, int]] = []
    unsupported: dict[str, int] = {}
    skipped_types: dict[str, int] = {}
    notes: list[str] = []
    current_material: _MaterialRec | None = None

    def _set_key(name: str, table: dict[str, list[int]]) -> str:
        key = name.upper()
        set_case.setdefault(key, name)
        table.setdefault(key, [])
        return key

    def _expand_ids(block: _Block, table: dict[str, list[int]], what: str) -> list[int]:
        ids: list[int] = []
        if "GENERATE" in block.params:
            for row in block.data:
                toks = [t for t in row if t]
                if not toks:
                    continue
                if not 2 <= len(toks) <= 3:
                    raise InpError(
                        f"{what} GENERATE rows need first, last[, increment]", line=block.line
                    )
                first = _i(toks[0], block.line, what)
                last = _i(toks[1], block.line, what)
                inc = _i(toks[2], block.line, what) if len(toks) == 3 else 1
                if inc <= 0 or last < first:
                    raise InpError(
                        f"{what} GENERATE row {toks} is not an increasing range",
                        line=block.line,
                    )
                ids.extend(range(first, last + 1, inc))
            return ids
        for row in block.data:
            for tok in row:
                if not tok:
                    continue
                try:
                    ids.append(int(tok))
                except ValueError:
                    ref = tok.upper()
                    if ref in table:
                        ids.extend(table[ref])
                    else:
                        raise InpError(
                            f"{what}: unknown id or set name {tok!r}", line=block.line
                        ) from None
        return ids

    # -- pass 1: collect raw records ------------------------------------------
    for block in blocks:
        kw = block.keyword
        if kw == "MATERIAL":
            name = block.params.get("NAME")
            if not name:
                raise InpError("*MATERIAL requires a NAME parameter", line=block.line)
            if name.upper() in materials:
                raise InpError(f"duplicate material name {name!r}", line=block.line)
            current_material = _MaterialRec(name=name)
            materials[name.upper()] = current_material
            continue
        if kw == "ELASTIC":
            if current_material is None:
                raise InpError("*ELASTIC outside a *MATERIAL definition", line=block.line)
            mtype = block.params.get("TYPE", "ISOTROPIC").upper()
            if mtype != "ISOTROPIC":
                notes.append(
                    f"*ELASTIC, TYPE={mtype} on material {current_material.name!r} "
                    "not supported (isotropic only); elastic data ignored"
                )
                continue
            rows = [r for r in block.data if any(r)]
            if not rows or not rows[0][0]:
                raise InpError("*ELASTIC has no data", line=block.line)
            current_material.E = _f(rows[0][0], block.line, "*ELASTIC E")
            if len(rows[0]) > 1 and rows[0][1]:
                current_material.nu = _f(rows[0][1], block.line, "*ELASTIC nu")
            if len(rows) > 1:
                notes.append(
                    f"*ELASTIC on material {current_material.name!r}: temperature "
                    "dependence ignored (first row used)"
                )
            continue
        if kw == "DENSITY":
            if current_material is None:
                raise InpError("*DENSITY outside a *MATERIAL definition", line=block.line)
            rows = [r for r in block.data if any(r)]
            if not rows or not rows[0][0]:
                raise InpError("*DENSITY has no data", line=block.line)
            current_material.rho = _f(rows[0][0], block.line, "*DENSITY rho")
            if len(rows) > 1:
                notes.append(
                    f"*DENSITY on material {current_material.name!r}: temperature "
                    "dependence ignored (first row used)"
                )
            continue

        if kw == "HEADING":
            first = next((r for r in block.data if any(r)), None)
            if first is not None and heading is None:
                heading = ", ".join(t for t in first if t)
        elif kw == "NODE":
            nset_key = (
                _set_key(block.params["NSET"], nsets) if block.params.get("NSET") else None
            )
            for row in block.data:
                if not any(row):
                    continue
                if not row[0]:
                    raise InpError("*NODE row without a node id", line=block.line)
                nid = _i(row[0], block.line, "*NODE id")
                xyz = tuple(
                    _f(row[k], block.line, "*NODE coordinate")
                    if k < len(row) and row[k]
                    else 0.0
                    for k in (1, 2, 3)
                )
                if nid in nodes:
                    raise InpError(f"duplicate node id {nid}", line=block.line)
                nodes[nid] = xyz  # type: ignore[assignment]
                if nset_key is not None:
                    nsets[nset_key].append(nid)
        elif kw == "ELEMENT":
            etype = block.params.get("TYPE", "").upper()
            if not etype:
                raise InpError("*ELEMENT requires a TYPE parameter", line=block.line)
            mapped = _ELEMENT_TYPES.get(etype)
            if mapped is None:
                n_rows = sum(1 for r in block.data if any(r))
                skipped_types[etype] = skipped_types.get(etype, 0) + n_rows
                continue
            ftype, n_nodes = mapped
            elset_key = (
                _set_key(block.params["ELSET"], elsets) if block.params.get("ELSET") else None
            )
            for row in block.data:
                toks = list(row)
                while toks and not toks[-1]:
                    toks.pop()
                if not toks:
                    continue
                eid = _i(toks[0], block.line, "*ELEMENT id")
                conn = tuple(_i(t, block.line, "*ELEMENT node") for t in toks[1:])
                if len(conn) != n_nodes:
                    raise InpError(
                        f"element {eid} ({etype}): expected {n_nodes} nodes, "
                        f"got {len(conn)}",
                        line=block.line,
                    )
                if eid in elements:
                    raise InpError(f"duplicate element id {eid}", line=block.line)
                elements[eid] = (ftype, conn)
                if elset_key is not None:
                    elsets[elset_key].append(eid)
        elif kw == "NSET":
            name = block.params.get("NSET")
            if not name:
                raise InpError("*NSET requires an NSET parameter", line=block.line)
            key = _set_key(name, nsets)
            nsets[key].extend(_expand_ids(block, nsets, "*NSET"))
        elif kw == "ELSET":
            name = block.params.get("ELSET")
            if not name:
                raise InpError("*ELSET requires an ELSET parameter", line=block.line)
            key = _set_key(name, elsets)
            elsets[key].extend(_expand_ids(block, elsets, "*ELSET"))
        elif kw in ("SOLID SECTION", "SHELL SECTION", "BEAM SECTION", "BEAM GENERAL SECTION"):
            elset = block.params.get("ELSET")
            if not elset:
                raise InpError(f"*{kw} requires an ELSET parameter", line=block.line)
            kind = {"SOLID SECTION": "solid", "SHELL SECTION": "shell"}.get(kw, "beam")
            sections.append(
                _SectionRec(
                    kind=kind,
                    elset=elset.upper(),
                    material=block.params.get("MATERIAL"),
                    block=block,
                )
            )
        elif kw == "BOUNDARY":
            btype = block.params.get("TYPE", "DISPLACEMENT").upper()
            if btype not in ("DISPLACEMENT", ""):
                notes.append(f"*BOUNDARY, TYPE={btype} ignored (DISPLACEMENT only)")
                current_material = None
                continue
            for row in block.data:
                toks = list(row)
                while toks and not toks[-1]:
                    toks.pop()
                if not toks:
                    continue
                target = toks[0]
                named = toks[1].upper() if len(toks) > 1 else ""
                if named in _BOUNDARY_TYPES:
                    boundary_rows.append((target, _BOUNDARY_TYPES[named], 0.0, block.line))
                    continue
                if len(toks) < 2:
                    raise InpError(
                        f"*BOUNDARY row {toks} needs a DOF number or a named type",
                        line=block.line,
                    )
                first_dof = _i(toks[1], block.line, "*BOUNDARY first dof")
                last_dof = (
                    _i(toks[2], block.line, "*BOUNDARY last dof")
                    if len(toks) > 2 and toks[2]
                    else first_dof
                )
                value = (
                    _f(toks[3], block.line, "*BOUNDARY value")
                    if len(toks) > 3 and toks[3]
                    else 0.0
                )
                if not 1 <= first_dof <= 6 or not first_dof <= last_dof <= 6:
                    raise InpError(
                        f"*BOUNDARY dofs {first_dof}..{last_dof} outside 1..6",
                        line=block.line,
                    )
                boundary_rows.append(
                    (target, tuple(range(first_dof - 1, last_dof)), value, block.line)
                )
        else:
            # Unknown keywords do NOT close the current *MATERIAL definition:
            # unsupported material options (*PLASTIC, *EXPANSION, ...) must not
            # detach the *DENSITY that follows them.
            unsupported[kw] = unsupported.get(kw, 0) + 1
            continue
        current_material = None

    # -- build the model in dependency order -----------------------------------
    model = FEModel(name=heading or Path(path).stem)
    for nid, xyz in nodes.items():
        model.add_node(id=nid, xyz=xyz)

    mat_ids: dict[str, int] = {}
    for key, rec in materials.items():
        if rec.E is None:
            notes.append(f"material {rec.name!r} has no isotropic *ELASTIC data; skipped")
            continue
        mid = len(mat_ids) + 1
        mat_ids[key] = mid
        model.add_material(
            id=mid, type="isotropic", E=rec.E, nu=rec.nu, rho=rec.rho, name=rec.name
        )

    def _material_id(sec: _SectionRec) -> int:
        block = sec.block
        if not sec.material:
            raise InpError(f"*{block.keyword} requires a MATERIAL parameter", line=block.line)
        mid = mat_ids.get(sec.material.upper())
        if mid is None:
            raise InpError(
                f"*{block.keyword}: material {sec.material!r} is not defined "
                "(or has no isotropic *ELASTIC data)",
                line=block.line,
            )
        return mid

    element_pid: dict[int, int] = {}
    element_orient: dict[int, np.ndarray] = {}
    next_mat_id = len(mat_ids) + 1

    def _direction_row(row: list[str] | None, block: _Block, what: str) -> np.ndarray:
        if row is None or not any(row):
            return np.array(_DEFAULT_N1)
        vals = [_f(t, block.line, what) if t else 0.0 for t in row[:3]]
        vals += [0.0] * (3 - len(vals))
        return np.array(vals)

    for pid, sec in enumerate(sections, start=1):
        block = sec.block
        eids = elsets.get(sec.elset)
        if eids is None:
            raise InpError(
                f"*{block.keyword}: element set "
                f"{set_case.get(sec.elset, sec.elset)!r} is not defined",
                line=block.line,
            )
        member_types = {elements[e][0] for e in eids if e in elements}
        rows = [[t for t in row if t] for row in block.data]
        rows = [r for r in rows if r]
        orientation: np.ndarray | None = None

        if sec.kind == "solid":
            if member_types == {"BAR2"}:
                # For trusses the *SOLID SECTION data line holds the area.
                area = _f(rows[0][0], block.line, "*SOLID SECTION area") if rows else 1.0
                model.add_property(id=pid, type="bar", material_id=_material_id(sec), A=area)
            elif member_types <= {"HEX8", "TET4", "TET10"}:
                model.add_property(id=pid, type="solid", material_id=_material_id(sec))
            else:
                raise InpError(
                    f"*SOLID SECTION on elset "
                    f"{set_case.get(sec.elset, sec.elset)!r}: unsupported element "
                    f"type mix {sorted(member_types)}",
                    line=block.line,
                )
        elif sec.kind == "shell":
            if not rows:
                raise InpError("*SHELL SECTION has no thickness data line", line=block.line)
            t = _f(rows[0][0], block.line, "*SHELL SECTION thickness")
            model.add_property(id=pid, type="shell", material_id=_material_id(sec), t=t)
        elif block.keyword == "BEAM GENERAL SECTION":
            shape = block.params.get("SECTION", "GENERAL").upper()
            if shape != "GENERAL":
                raise InpError(
                    f"*BEAM GENERAL SECTION, SECTION={shape} is not supported "
                    "(GENERAL only)",
                    line=block.line,
                )
            data = block.data  # positional: blank direction line is meaningful
            if not data or not any(data[0]):
                raise InpError(
                    "*BEAM GENERAL SECTION has no section data line", line=block.line
                )
            vals = [
                _f(t, block.line, "*BEAM GENERAL SECTION constants") if t else 0.0
                for t in data[0]
            ]
            vals += [0.0] * (5 - len(vals))
            area, i11, i12, i22, j = vals[:5]
            if i12:
                notes.append(
                    f"*BEAM GENERAL SECTION on elset {sec.elset!r}: product of "
                    f"inertia I12={i12} ignored"
                )
            if j == 0.0:
                j = i11 + i22
                notes.append(
                    f"*BEAM GENERAL SECTION on elset {sec.elset!r}: torsion constant "
                    f"missing; J = I11 + I22 = {j} assumed"
                )
            orientation = _direction_row(
                data[1] if len(data) > 1 else None, block, "*BEAM GENERAL SECTION n1"
            )
            if len(data) < 3 or not any(data[2]):
                raise InpError(
                    "*BEAM GENERAL SECTION needs an E, G data line (third data line)",
                    line=block.line,
                )
            emod = _f(data[2][0], block.line, "*BEAM GENERAL SECTION E")
            gmod = (
                _f(data[2][1], block.line, "*BEAM GENERAL SECTION G")
                if len(data[2]) > 1 and data[2][1]
                else None
            )
            nu = emod / (2.0 * gmod) - 1.0 if gmod else 0.0
            if sec.material:
                notes.append(
                    f"*BEAM GENERAL SECTION on elset {sec.elset!r}: "
                    f"MATERIAL={sec.material} ignored (E, G come from the section)"
                )
            rho_s = block.params.get("DENSITY")
            mid = next_mat_id
            next_mat_id += 1
            model.add_material(
                id=mid,
                type="isotropic",
                E=emod,
                nu=nu,
                rho=_f(rho_s, block.line, "*BEAM GENERAL SECTION DENSITY") if rho_s else 0.0,
                name=f"beam-section-{set_case.get(sec.elset, sec.elset)}",
            )
            model.add_property(id=pid, type="beam", material_id=mid, A=area, Iy=i11, Iz=i22, J=j)
        else:  # *BEAM SECTION, SECTION=RECT|CIRC
            shape = block.params.get("SECTION", "").upper()
            data = block.data
            if not data or not any(data[0]):
                raise InpError("*BEAM SECTION has no dimension data line", line=block.line)
            dims = [
                _f(t, block.line, "*BEAM SECTION dimensions") for t in data[0] if t
            ]
            if shape == "RECT":
                if len(dims) < 2:
                    raise InpError(
                        "*BEAM SECTION, SECTION=RECT needs a, b dimensions", line=block.line
                    )
                a, b = dims[0], dims[1]
                area, i11, i22 = a * b, a * b**3 / 12.0, b * a**3 / 12.0
                j = _rect_torsion(a, b)
            elif shape == "CIRC":
                if not dims:
                    raise InpError(
                        "*BEAM SECTION, SECTION=CIRC needs a radius", line=block.line
                    )
                r = dims[0]
                area = math.pi * r**2
                i11 = i22 = math.pi * r**4 / 4.0
                j = math.pi * r**4 / 2.0
            else:
                raise InpError(
                    f"*BEAM SECTION, SECTION={shape or '(none)'} is not supported "
                    "(RECT or CIRC)",
                    line=block.line,
                )
            orientation = _direction_row(
                data[1] if len(data) > 1 else None, block, "*BEAM SECTION n1"
            )
            model.add_property(
                id=pid,
                type="beam",
                material_id=_material_id(sec),
                A=area,
                Iy=i11,
                Iz=i22,
                J=j,
            )

        for eid in eids:
            if eid not in elements:
                raise InpError(
                    f"*{block.keyword}: elset "
                    f"{set_case.get(sec.elset, sec.elset)!r} references unknown "
                    f"element {eid}",
                    line=block.line,
                )
            if eid in element_pid:
                raise InpError(
                    f"element {eid} is assigned to more than one section", line=block.line
                )
            element_pid[eid] = pid
            if orientation is not None:
                element_orient[eid] = orientation

    missing = [eid for eid in elements if eid not in element_pid]
    if missing:
        shown = ", ".join(map(str, missing[:5])) + (", ..." if len(missing) > 5 else "")
        raise InpError(f"{len(missing)} element(s) have no section assignment [{shown}]")

    for eid, (ftype, conn) in elements.items():
        model.add_element(
            id=eid,
            type=ftype,
            nodes=conn,
            property_id=element_pid[eid],
            orientation=element_orient.get(eid),
        )

    # -- boundaries (resolved against the node sets) ----------------------------
    for target, dofs, value, lineno in boundary_rows:
        try:
            nids = [int(target)]
        except ValueError:
            key = target.upper()
            if key not in nsets:
                raise InpError(
                    f"*BOUNDARY: unknown node or node set {target!r}", line=lineno
                ) from None
            nids = nsets[key]
        mask = [d in dofs for d in range(6)]
        for nid in nids:
            if nid not in model.nodes:
                raise InpError(f"*BOUNDARY references undefined node {nid}", line=lineno)
            model.add_spc(node_id=nid, mask=mask, value=value)

    # -- named sets --------------------------------------------------------------
    for key, ids in nsets.items():
        model.add_set(NodeSet(set_case[key], frozenset(ids)))
    for key, ids in elsets.items():
        name = set_case[key]
        if name in model.sets:  # Abaqus allows an NSET and an ELSET of one name
            notes.append(
                f"element set {name!r} renamed to {name + ':ELSET'!r} (a node set "
                "of the same name exists)"
            )
            name += ":ELSET"
        model.add_set(ElementSet(name, frozenset(ids)))

    # -- aggregated warnings -------------------------------------------------------
    fname = Path(path).name
    for etype, count in sorted(skipped_types.items()):
        warnings.warn(
            f"read_inp({fname}): skipped {count} element(s) of unsupported TYPE={etype}",
            UserWarning,
            stacklevel=2,
        )
    for kw, count in sorted(unsupported.items()):
        warnings.warn(
            f"read_inp({fname}): skipped unsupported keyword *{kw} (x{count})",
            UserWarning,
            stacklevel=2,
        )
    for note in notes:
        warnings.warn(f"read_inp({fname}): {note}", UserWarning, stacklevel=2)
    return model


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

#: femtools type -> Abaqus TYPE (writer side).
_WRITE_TYPES: dict[str, str] = {
    "HEX8": "C3D8",
    "TET4": "C3D4",
    "TET10": "C3D10",
    "QUAD4": "S4",
    "TRIA3": "S3",
    "BEAM2": "B31",
    "BAR2": "T3D2",
    "TRUSS2D": "T3D2",
}


def _g(v: float | None) -> str:
    return format(0.0 if v is None else float(v), ".12g")


def write_inp(path: str | Path | FEModel, model: FEModel | str | Path | None = None) -> None:
    """Write an :class:`FEModel` as an Abaqus input-file TEXT subset.

    Accepts ``write_inp(path, model)`` or ``write_inp(model, path)``.

    Elements are grouped per (property, type) under an auto element set
    ``EL_P<pid>``; each property is emitted as the matching section
    keyword bound to that set.  Beam properties are written as ``*BEAM
    GENERAL SECTION`` (``A, I11=Iy, 0, I22=Iz, J``, the orientation of
    the first element of the group, and the material's ``E, G`` with the
    density on the keyword line).  Lumped MASS/SPRING/DAMPER elements
    and nodal loads have no mapping in this subset and are skipped with
    one aggregated warning each.
    """
    from ._compat import coerce_path_model

    path, model = coerce_path_model(path, model)
    lines: list[str] = ["*HEADING", f"{model.name}", "** femtools .inp export"]

    if model.nodes:
        lines.append("*NODE")
        for nid in model.node_ids():
            x, y, z = model.nodes[nid].xyz
            lines.append(f"{nid}, {_g(x)}, {_g(y)}, {_g(z)}")

    groups: dict[tuple[int, str], list[int]] = {}
    skipped_elements: dict[str, int] = {}
    for eid in sorted(model.elements):
        el = model.elements[eid]
        if el.type not in _WRITE_TYPES or el.property_id is None:
            skipped_elements[el.type] = skipped_elements.get(el.type, 0) + 1
            continue
        groups.setdefault((el.property_id, _WRITE_TYPES[el.type]), []).append(eid)
    for (pid, atype), eids in sorted(groups.items()):
        lines.append(f"*ELEMENT, TYPE={atype}, ELSET=EL_P{pid}")
        for eid in eids:
            conn = ", ".join(str(n) for n in model.elements[eid].nodes)
            lines.append(f"{eid}, {conn}")

    written_pids = {pid for pid, _ in groups}
    # Materials referenced only by beam properties are embedded in their
    # *BEAM GENERAL SECTION (E, G + DENSITY=) and need no *MATERIAL block.
    beam_only_mids = {
        p.material_id for p in model.properties.values() if p.type == "beam"
    } - {
        p.material_id for p in model.properties.values() if p.type != "beam"
    }

    for mid in sorted(model.materials):
        if mid in beam_only_mids:
            continue
        mat = model.materials[mid]
        if mat.type != "isotropic":
            warnings.warn(
                f"write_inp: material {mid} is {mat.type}; written with E1/nu12",
                UserWarning,
                stacklevel=2,
            )
        emod = mat.E if mat.type == "isotropic" else mat.E1
        nu = mat.nu if mat.type == "isotropic" else mat.nu12
        lines.append(f"*MATERIAL, NAME={mat.name or f'MAT{mid}'}")
        lines.append("*ELASTIC")
        lines.append(f"{_g(emod)}, {_g(nu)}")
        if mat.rho is not None:
            lines.append("*DENSITY")
            lines.append(f"{_g(mat.rho)}")

    def _mat_name(mid: int | None) -> str:
        if mid is None or mid not in model.materials:
            return "MAT0"
        return model.materials[mid].name or f"MAT{mid}"

    skipped_props: list[int] = []
    for pid in sorted(model.properties):
        prop = model.properties[pid]
        if pid not in written_pids:
            continue  # not referenced by any writable element
        if prop.type in ("solid", "bar"):
            lines.append(
                f"*SOLID SECTION, ELSET=EL_P{pid}, MATERIAL={_mat_name(prop.material_id)}"
            )
            if prop.type == "bar":
                lines.append(f"{_g(prop.A)}")
        elif prop.type == "shell":
            lines.append(
                f"*SHELL SECTION, ELSET=EL_P{pid}, MATERIAL={_mat_name(prop.material_id)}"
            )
            lines.append(f"{_g(prop.t)}")
        elif prop.type == "beam":
            bmat = model.materials.get(prop.material_id) if prop.material_id else None
            rho = bmat.rho if bmat is not None and bmat.rho is not None else 0.0
            lines.append(
                f"*BEAM GENERAL SECTION, ELSET=EL_P{pid}, SECTION=GENERAL, "
                f"DENSITY={_g(rho)}"
            )
            lines.append(f"{_g(prop.A)}, {_g(prop.Iy)}, 0., {_g(prop.Iz)}, {_g(prop.J)}")
            first_eid = min(
                eid for (gpid, _), eids in groups.items() if gpid == pid for eid in eids
            )
            v = model.elements[first_eid].orientation
            if v is None:
                v = np.array(_DEFAULT_N1)
            lines.append(f"{_g(v[0])}, {_g(v[1])}, {_g(v[2])}")
            emod = bmat.E if bmat is not None and bmat.E is not None else 0.0
            gmod = bmat.G if bmat is not None and bmat.G is not None else 0.0
            lines.append(f"{_g(emod)}, {_g(gmod)}")
        else:
            skipped_props.append(pid)

    for spc in model.spcs:
        runs: list[list[int]] = []  # [first_dof, last_dof], 1-based
        for d, on in enumerate(spc.mask):
            if not on:
                continue
            if runs and runs[-1][1] == d:  # previous dof (1-based d) is masked
                runs[-1][1] = d + 1
            else:
                runs.append([d + 1, d + 1])
        if not runs:
            continue
        lines.append("*BOUNDARY")
        for first_dof, last_dof in runs:
            if spc.value == 0.0:
                lines.append(f"{spc.node_id}, {first_dof}, {last_dof}")
            else:
                lines.append(f"{spc.node_id}, {first_dof}, {last_dof}, {_g(spc.value)}")

    for name in sorted(model.sets):
        s = model.sets[name]
        kw, param = ("*NSET", "NSET") if isinstance(s, NodeSet) else ("*ELSET", "ELSET")
        lines.append(f"{kw}, {param}={name}")
        ids = s.sorted_ids()
        for k in range(0, len(ids), 16):
            lines.append(", ".join(str(i) for i in ids[k : k + 16]))

    for etype, count in sorted(skipped_elements.items()):
        warnings.warn(
            f"write_inp: {count} {etype} element(s) have no .inp mapping; skipped",
            UserWarning,
            stacklevel=2,
        )
    if skipped_props:
        warnings.warn(
            f"write_inp: lumped propert{'ies' if len(skipped_props) > 1 else 'y'} "
            f"{skipped_props} skipped (no .inp mapping)",
            UserWarning,
            stacklevel=2,
        )
    if model.loads:
        warnings.warn(
            f"write_inp: {len(model.loads)} nodal load(s) skipped (*CLOAD is outside "
            "the supported subset)",
            UserWarning,
            stacklevel=2,
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

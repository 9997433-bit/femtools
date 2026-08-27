"""Nastran-like bulk data (BDF) translator.

Supported cards (Round 1 subset):

* geometry: ``GRID`` (CP/CD/PS honoured), ``CORD2R``, ``CORD2C``, ``CORD2S``
* elements: ``CROD``, ``CBAR``, ``CBEAM``, ``CTRIA3``, ``CQUAD4``,
  ``CTETRA`` (4-node), ``CHEXA`` (8-node), ``CONM2``, ``CELAS2``, ``CDAMP2``.
  10-node ``CTETRA`` / 20-node ``CHEXA`` are accepted with a ``UserWarning``
  (one aggregated warning per card type, never an error): midside nodes are
  dropped and the element degrades to its linear corner form.
* properties: ``PROD``, ``PBAR``, ``PBEAM`` (first station), ``PSHELL``,
  ``PSOLID``
* materials: ``MAT1``
* constraints/loads: ``SPC1`` (with ``THRU``), ``SPC``, ``FORCE``, ``MOMENT``

Element type mapping (femtools <-> Nastran):

=========  ==================  =========================================
femtools   read from           written as
=========  ==================  =========================================
BAR2       CROD                CROD (+ PROD)
TRUSS2D    --                  CROD (+ PROD); planar nature is metadata
BEAM2      CBAR, CBEAM         CBAR (+ PBAR)
TRIA3      CTRIA3              CTRIA3 (+ PSHELL)
QUAD4      CQUAD4              CQUAD4 (+ PSHELL)
TET4       CTETRA              CTETRA (+ PSOLID)
HEX8       CHEXA               CHEXA (+ PSOLID)
MASS       CONM2               CONM2 (mass value from lumped property)
SPRING     CELAS2              CELAS2 (k from lumped property)
DAMPER     CDAMP2              CDAMP2 (c from lumped property)
=========  ==================  =========================================

Small-field, large-field and free-field (comma) input formats are all
accepted, including ``+``/``*``/blank continuations and Nastran embedded
exponents (``2.1+11`` == ``2.1e11``).  Grids are written in large-field
format for full coordinate precision.  Unsupported cards are skipped with
one ``UserWarning`` per card name.  BDF files carry no unit information:
the model's :class:`~femtools.core.units.UnitSystem` is left at its
default (SI) and it is the caller's responsibility that the deck is
consistent.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from ..core.coords import CoordSys
from ..core.errors import FileFormatError
from ..core.model import FEModel, comps_to_mask

__all__ = ["read_bdf", "write_bdf", "BdfError"]


class BdfError(FileFormatError):
    """Raised for malformed bulk data (a :class:`ValueError` via
    :class:`~femtools.core.errors.FileFormatError`)."""


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

_SMALL_SLOTS = [(i * 8, (i + 1) * 8) for i in range(1, 9)]  # fields 2..9
_LARGE_SLOTS = [(8, 24), (24, 40), (40, 56), (56, 72)]  # fields 2..5


def _strip_comment(line: str) -> str:
    i = line.find("$")
    return line if i < 0 else line[:i]


def _split_line(line: str) -> tuple[str, list[str]]:
    """One physical line -> (field1, data_fields) with positional padding."""
    if "," in line:
        toks = [t.strip() for t in line.split(",")]
        head, data = toks[0], toks[1:]
        data += [""] * (8 - len(data)) if len(data) < 8 else []
        return head, data[:9]
    head = line[:8].strip()
    slots = _LARGE_SLOTS if head.endswith("*") or head.startswith("*") else _SMALL_SLOTS
    padded = line.ljust(80)
    return head, [padded[a:b].strip() for a, b in slots]


def _logical_cards(text: str) -> list[list[str]]:
    """Physical lines -> logical cards ``[name, field2, field3, ...]``."""
    lines = text.splitlines()
    # skip executive/case control when present
    start = 0
    for i, raw in enumerate(lines):
        if raw.upper().lstrip().startswith("BEGIN") and "BULK" in raw.upper():
            start = i + 1
            break
    cards: list[list[str]] = []
    for raw in lines[start:]:
        line = _strip_comment(raw.rstrip("\n"))
        if not line.strip():
            continue
        upper = line.upper().lstrip()
        if upper.startswith("ENDDATA"):
            break
        if upper.startswith("INCLUDE"):
            warnings.warn(
                "read_bdf: INCLUDE statements are not followed", UserWarning, stacklevel=3
            )
            continue
        head, data = _split_line(line)
        if head == "" or head.startswith("+") or head.startswith("*"):
            if not cards:
                continue  # stray continuation
            cards[-1].extend(data)
        else:
            name = head.rstrip("*").upper()
            cards.append([name, *data])
    return cards


_NASTRAN_FLOAT = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)")


def _f(card: list[str], i: int, default: float | None = None) -> float | None:
    """Field ``i`` (1-based like Nastran docs, name = field 1) as float."""
    if i >= len(card) or card[i] == "":
        return default
    s = card[i].strip()
    try:
        return float(s)
    except ValueError:
        pass
    s2 = s.replace("D", "E").replace("d", "e")
    try:
        return float(s2)
    except ValueError:
        pass
    m = _NASTRAN_FLOAT.fullmatch(s)
    if m:
        return float(m.group(1) + "e" + m.group(2))
    raise BdfError(f"cannot parse {s!r} as a real number (card {card[0]})")


def _i(card: list[str], i: int, default: int | None = None) -> int | None:
    if i >= len(card) or card[i] == "":
        return default
    s = card[i].strip()
    try:
        return int(s)
    except ValueError as exc:
        raise BdfError(f"cannot parse {s!r} as an integer (card {card[0]})") from exc


def _s(card: list[str], i: int, default: str = "") -> str:
    return card[i].strip() if i < len(card) and card[i] else default


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _grid_list(card: list[str], start: int) -> list[int]:
    """Expand an id list with THRU (used by SPC1)."""
    ids: list[int] = []
    j = start
    while j < len(card):
        tok = _s(card, j)
        if not tok:
            j += 1
            continue
        if tok.upper() == "THRU":
            last = _i(card, j + 1)
            if not ids or last is None:
                raise BdfError(f"malformed THRU in {card[0]}")
            ids.extend(range(ids[-1] + 1, last + 1))
            j += 2
        else:
            ids.append(int(tok))
            j += 1
    return ids


def read_bdf(path: str | Path) -> FEModel:
    """Read a Nastran-like bulk data file into an :class:`FEModel`.

    Cards are collected first and the model is built in dependency order
    (coordinate systems, grids, materials, properties, elements,
    constraints, loads), so card order in the file does not matter.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cards = _logical_cards(text)
    by_name: dict[str, list[list[str]]] = {}
    for c in cards:
        by_name.setdefault(c[0], []).append(c)

    model = FEModel(name=Path(path).stem)
    unsupported: dict[str, int] = {}
    notes: list[str] = []

    # -- coordinate systems (CORD2R/C/S reference only cid 0 in Round 1) ----
    cord_types = {"CORD2R": "cartesian", "CORD2C": "cylindrical", "CORD2S": "spherical"}
    for name, cstype in cord_types.items():
        for c in by_name.pop(name, []):
            cid = _i(c, 1)
            rid = _i(c, 2, 0)
            if rid not in (0, None):
                notes.append(f"{name} {cid}: reference system RID={rid} not supported; assumed 0")
            a = [_f(c, i, 0.0) for i in (3, 4, 5)]
            b = [_f(c, i, 0.0) for i in (6, 7, 8)]
            cc = [_f(c, i, 0.0) for i in (9, 10, 11)]
            cs = CoordSys.from_points(cid, a, b, cc, type=cstype)  # type: ignore[arg-type]
            model.add_coord_system(cs)

    # -- grids ----------------------------------------------------------------
    pending_ps: list[tuple[int, str]] = []
    for c in by_name.pop("GRID", []):
        nid = _i(c, 1)
        cp = _i(c, 2, 0) or 0
        xyz = (_f(c, 3, 0.0), _f(c, 4, 0.0), _f(c, 5, 0.0))
        cd = _i(c, 6, 0) or 0
        ps = _s(c, 7)
        model.add_node(id=nid, xyz=xyz, cp=cp, cd=cd)
        if ps:
            pending_ps.append((nid, ps))

    # -- materials -------------------------------------------------------------
    for c in by_name.pop("MAT1", []):
        mid = _i(c, 1)
        e = _f(c, 2)
        g = _f(c, 3)
        nu = _f(c, 4)
        rho = _f(c, 5, 0.0)
        alpha = _f(c, 6)
        ge = _f(c, 8)
        # Nastran completion rules for the E/G/NU triplet
        if nu is None and e is not None and g is not None and g != 0.0:
            nu = e / (2.0 * g) - 1.0
        model.add_material(
            id=mid, type="isotropic", E=e, G=g, nu=nu, rho=rho, alpha=alpha, damping=ge
        )

    # -- properties ---------------------------------------------------------------
    for c in by_name.pop("PROD", []):
        model.add_property(
            id=_i(c, 1), type="bar", material_id=_i(c, 2), A=_f(c, 3), J=_f(c, 4), nsm=_f(c, 6)
        )
    for c in by_name.pop("PBAR", []):
        # PBAR: PID MID A I1 I2 J NSM / (stress recovery) / K1 K2 I12
        kappa = _f(c, 17)  # K1 = logical field 18 -> card index 17
        model.add_property(
            id=_i(c, 1),
            type="beam",
            material_id=_i(c, 2),
            A=_f(c, 3, 0.0),
            Iz=_f(c, 4, 0.0),
            Iy=_f(c, 5, 0.0),
            J=_f(c, 6, 0.0),
            nsm=_f(c, 7),
            kappa=kappa,
        )
    for c in by_name.pop("PBEAM", []):
        # first station only: PID MID A(A) I1(A) I2(A) I12(A) J(A) NSM(A)
        notes_extra = " (only end-A section used)"
        model.add_property(
            id=_i(c, 1),
            type="beam",
            material_id=_i(c, 2),
            A=_f(c, 3, 0.0),
            Iz=_f(c, 4, 0.0),
            Iy=_f(c, 5, 0.0),
            J=_f(c, 7, 0.0),
            nsm=_f(c, 8),
        )
        notes.append(f"PBEAM {_i(c, 1)}: tapered/composite data ignored{notes_extra}")
    for c in by_name.pop("PSHELL", []):
        mid2 = _i(c, 4)
        mid1 = _i(c, 2)
        if mid2 is not None and mid2 != mid1:
            notes.append(f"PSHELL {_i(c, 1)}: bending material MID2={mid2} ignored (MID1 used)")
        model.add_property(id=_i(c, 1), type="shell", material_id=mid1, t=_f(c, 3), nsm=_f(c, 8))
    for c in by_name.pop("PSOLID", []):
        model.add_property(id=_i(c, 1), type="solid", material_id=_i(c, 2))

    # -- structural elements ---------------------------------------------------------
    def _orientation(c: list[str], ga: int) -> np.ndarray | None:
        """CBAR/CBEAM logical fields 6-8 (card indices 5-7): G0 or the X1,X2,X3 vector."""
        f5, f6, f7 = _s(c, 5), _s(c, 6), _s(c, 7)
        if not f5:
            return None
        if "." not in f5 and not f6 and not f7:
            g0 = int(f5)
            n0 = model.nodes.get(g0)
            na = model.nodes.get(ga)
            if n0 is None or na is None:
                raise BdfError(f"{c[0]}: orientation node G0={g0} or GA={ga} undefined")
            return n0.xyz - na.xyz
        return np.array([_f(c, 5, 0.0), _f(c, 6, 0.0), _f(c, 7, 0.0)])

    for c in by_name.pop("CROD", []):
        model.add_element(
            id=_i(c, 1), type="BAR2", nodes=(_i(c, 3), _i(c, 4)), property_id=_i(c, 2)
        )
    for name in ("CBAR", "CBEAM"):
        for c in by_name.pop(name, []):
            ga, gb = _i(c, 3), _i(c, 4)
            model.add_element(
                id=_i(c, 1),
                type="BEAM2",
                nodes=(ga, gb),
                property_id=_i(c, 2),
                orientation=_orientation(c, ga),
            )
    for c in by_name.pop("CTRIA3", []):
        model.add_element(
            id=_i(c, 1), type="TRIA3", nodes=(_i(c, 3), _i(c, 4), _i(c, 5)), property_id=_i(c, 2)
        )
    for c in by_name.pop("CQUAD4", []):
        model.add_element(
            id=_i(c, 1),
            type="QUAD4",
            nodes=(_i(c, 3), _i(c, 4), _i(c, 5), _i(c, 6)),
            property_id=_i(c, 2),
        )
    # Quadratic solids are accepted but degraded to their linear corner
    # elements: this is a warning, never an error.  One aggregated note per
    # card type (a real TET10/HEX20 mesh has thousands of such elements).
    midside_drops: dict[str, list[int]] = {}
    for c in by_name.pop("CTETRA", []):
        nodes = [_i(c, j) for j in range(3, 13) if _s(c, j)]
        if len(nodes) > 4:
            midside_drops.setdefault("CTETRA (TET10 -> TET4)", []).append(_i(c, 1))
        model.add_element(id=_i(c, 1), type="TET4", nodes=tuple(nodes[:4]), property_id=_i(c, 2))
    for c in by_name.pop("CHEXA", []):
        nodes = [_i(c, j) for j in range(3, 23) if _s(c, j)]
        if len(nodes) > 8:
            midside_drops.setdefault("CHEXA (HEX20 -> HEX8)", []).append(_i(c, 1))
        model.add_element(id=_i(c, 1), type="HEX8", nodes=tuple(nodes[:8]), property_id=_i(c, 2))
    for what, eids in midside_drops.items():
        shown = ", ".join(map(str, eids[:5])) + (", ..." if len(eids) > 5 else "")
        notes.append(
            f"{what}: midside nodes dropped on {len(eids)} element(s) [{shown}]; "
            "quadratic accuracy is lost and the midside grids remain as "
            "unconnected nodes (harmless: the assembler eliminates unattached DOFs)"
        )

    # -- lumped elements (values live on the card -> auto lumped properties) ---------
    auto_pid = (max(model.properties, default=0) // 1000 + 1) * 1000 + 1

    def _new_lumped(**fields: float) -> int:
        nonlocal auto_pid
        while auto_pid in model.properties:
            auto_pid += 1
        model.add_property(id=auto_pid, type="lumped", **fields)
        pid = auto_pid
        auto_pid += 1
        return pid

    for c in by_name.pop("CONM2", []):
        eid, g = _i(c, 1), _i(c, 2)
        mass = _f(c, 4, 0.0)
        offs = [(_f(c, j, 0.0) or 0.0) for j in (5, 6, 7)]
        if any(offs):
            notes.append(f"CONM2 {eid}: mass offset {offs} ignored")
        if any((_f(c, j, 0.0) or 0.0) != 0.0 for j in range(9, 15)):
            notes.append(f"CONM2 {eid}: rotary inertia terms ignored")
        model.add_element(id=eid, type="MASS", nodes=(g,), property_id=_new_lumped(m=mass))
    for c in by_name.pop("CELAS2", []):
        eid, k = _i(c, 1), _f(c, 2, 0.0)
        g1, c1, g2, c2 = _i(c, 3), _i(c, 4, 1), _i(c, 5), _i(c, 6, 1)
        nodes = (g1,) if g2 in (None, 0) else (g1, g2)
        dofs = ((c1 or 1) - 1,) if len(nodes) == 1 else ((c1 or 1) - 1, (c2 or 1) - 1)
        model.add_element(
            id=eid, type="SPRING", nodes=nodes, dofs=dofs, property_id=_new_lumped(k=k)
        )
    for c in by_name.pop("CDAMP2", []):
        eid, b = _i(c, 1), _f(c, 2, 0.0)
        g1, c1, g2, c2 = _i(c, 3), _i(c, 4, 1), _i(c, 5), _i(c, 6, 1)
        nodes = (g1,) if g2 in (None, 0) else (g1, g2)
        dofs = ((c1 or 1) - 1,) if len(nodes) == 1 else ((c1 or 1) - 1, (c2 or 1) - 1)
        model.add_element(
            id=eid, type="DAMPER", nodes=nodes, dofs=dofs, property_id=_new_lumped(c=b)
        )

    # -- constraints -------------------------------------------------------------------
    for nid, ps in pending_ps:
        model.add_spc(node_id=nid, mask=comps_to_mask(ps), sid=0)
    for c in by_name.pop("SPC1", []):
        sid = _i(c, 1, 1)
        mask = comps_to_mask(_s(c, 2, "0"))
        for nid in _grid_list(c, 3):
            model.add_spc(node_id=nid, mask=mask, sid=sid)
    for c in by_name.pop("SPC", []):
        sid = _i(c, 1, 1)
        for base in (2, 5):
            g = _i(c, base)
            if g is None:
                continue
            comp = _s(c, base + 1, "0")
            d = _f(c, base + 2, 0.0) or 0.0
            model.add_spc(node_id=g, mask=comps_to_mask(comp), value=d, sid=sid)

    # -- loads ------------------------------------------------------------------------
    def _load_vector(c: list[str]) -> tuple[int, int, np.ndarray]:
        sid, g, cid = _i(c, 1, 1), _i(c, 2), _i(c, 3, 0) or 0
        scale = _f(c, 4, 0.0) or 0.0
        vec = np.array([_f(c, j, 0.0) or 0.0 for j in (5, 6, 7)]) * scale
        if cid != 0:
            cs = model.coord_systems.get(cid)
            if cs is None:
                raise BdfError(f"{c[0]}: coordinate system {cid} undefined")
            at = model.nodes[g].xyz if g in model.nodes else (0.0, 0.0, 0.0)
            vec = cs.transform_vector_to_global(vec, at=at)
        return sid, g, vec

    for c in by_name.pop("FORCE", []):
        sid, g, vec = _load_vector(c)
        model.add_load(node_id=g, force=vec, sid=sid)
    for c in by_name.pop("MOMENT", []):
        sid, g, vec = _load_vector(c)
        model.add_load(node_id=g, moment=vec, sid=sid)

    # -- leftovers ---------------------------------------------------------------------
    for name, group in by_name.items():
        unsupported[name] = len(group)
    for name, count in sorted(unsupported.items()):
        warnings.warn(
            f"read_bdf({Path(path).name}): skipped unsupported card {name} (x{count})",
            UserWarning,
            stacklevel=2,
        )
    for note in notes:
        warnings.warn(f"read_bdf({Path(path).name}): {note}", UserWarning, stacklevel=2)
    return model


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _fmt_float(v: float, width: int) -> str:
    """Most precise Nastran real that fits ``width`` chars.

    Tries, per precision level (highest first), both plain notation and the
    Nastran embedded-exponent form (``1.234-4`` == 1.234e-4).
    """
    v = float(v)
    if v == 0.0:
        return "0."
    if not np.isfinite(v):
        raise BdfError(f"cannot write non-finite real {v!r}")
    for p in range(min(width, 17) - 1, -1, -1):
        candidates: list[str] = []
        s = f"{v:.{p}g}"
        if "e" in s or "E" in s:
            mant, _, exp = s.lower().partition("e")
            candidates.append(mant + f"{int(exp):+d}")
        else:
            candidates.append(s if "." in s else s + ".")
        se = f"{v:.{p}e}".lower()
        mant, _, exp = se.partition("e")
        if "." in mant:
            mant = mant.rstrip("0")
        if "." not in mant:
            mant += "."
        candidates.append(mant + f"{int(exp):+d}")
        for cand in candidates:
            if len(cand) <= width:
                return cand
    raise BdfError(f"cannot format {v!r} in {width} characters")


def _field8(v: float | int | str | None) -> str:
    if v is None or v == "":
        return " " * 8
    if isinstance(v, str):
        return f"{v:<8s}"[:8]
    if isinstance(v, (int, np.integer)):
        s = str(int(v))
        if len(s) > 8:
            raise BdfError(f"integer {v} does not fit in a small field")
        return f"{s:>8s}"
    return f"{_fmt_float(float(v), 8):>8s}"


def _card8(name: str, *fields: float | int | str | None) -> list[str]:
    """Small-field card with automatic +C continuations (max 8 data fields/line)."""
    vals = list(fields)
    while vals and (vals[-1] is None or vals[-1] == ""):
        vals.pop()
    lines: list[str] = []
    first = True
    while True:
        chunk, vals = vals[:8], vals[8:]
        head = f"{name:<8s}" if first else "+       "
        body = "".join(_field8(v) for v in chunk)
        if vals:
            body = f"{body:<64s}+"
        lines.append((head + body).rstrip())
        first = False
        if not vals:
            break
    return lines


def _grid_large(node_id: int, cp: int, xyz: np.ndarray, cd: int) -> list[str]:
    """Large-field GRID for full double precision."""
    line1 = (
        f"{'GRID*':<8s}{node_id:>16d}{cp:>16d}"
        f"{_fmt_float(xyz[0], 16):>16s}{_fmt_float(xyz[1], 16):>16s}*"
    )
    line2 = f"{'*':<8s}{_fmt_float(xyz[2], 16):>16s}{cd:>16d}"
    return [line1, line2.rstrip()]


_CORD_NAME = {"cartesian": "CORD2R", "cylindrical": "CORD2C", "spherical": "CORD2S"}


def write_bdf(path: str | Path | FEModel, model: FEModel | str | Path | None = None) -> None:
    """Write an :class:`FEModel` as a bulk-data deck (small field, GRID* large).

    Accepts ``write_bdf(path, model)`` or ``write_bdf(model, path)``.

    Lumped MASS/SPRING/DAMPER elements are written as CONM2/CELAS2/CDAMP2
    with values taken from their lumped property; the auto property card
    itself is not emitted (Nastran keeps these values on the element).
    """
    from ._compat import coerce_path_model

    path, model = coerce_path_model(path, model)
    lines: list[str] = [
        f"$ femtools bulk data export -- model: {model.name}",
        f"$ units: {model.units.length}-{model.units.force}-{model.units.mass}-{model.units.time}"
        + ("" if model.units.is_consistent else "  (WARNING: inconsistent system)"),
        "BEGIN BULK",
    ]

    for cid in sorted(model.coord_systems):
        cs = model.coord_systems[cid]
        a = cs.origin
        b = cs.origin + cs.rotation[:, 2]
        c = cs.origin + cs.rotation[:, 0]
        lines += _card8(_CORD_NAME[cs.type], cid, 0, *a, *b, *c)

    for nid in model.node_ids():
        node = model.nodes[nid]
        # coordinates are stored global; cp is metadata only, so write CP=0
        lines += _grid_large(nid, 0, node.xyz, node.cd)

    for mid in sorted(model.materials):
        mat = model.materials[mid]
        if mat.type != "isotropic":
            warnings.warn(
                f"write_bdf: material {mid} is {mat.type}; written as MAT1 with E1/nu12",
                UserWarning,
                stacklevel=2,
            )
            lines += _card8("MAT1", mid, mat.E1, None, mat.nu12, mat.rho)
            continue
        lines += _card8("MAT1", mid, mat.E, mat.G, mat.nu, mat.rho, mat.alpha, None, mat.damping)

    lumped_pids: set[int] = set()
    for pid in sorted(model.properties):
        prop = model.properties[pid]
        if prop.type == "bar":
            lines += _card8("PROD", pid, prop.material_id, prop.A, prop.J, None, prop.nsm)
        elif prop.type == "beam":
            fields: list = [pid, prop.material_id, prop.A, prop.Iz, prop.Iy, prop.J, prop.nsm]
            if prop.kappa is not None:
                # pad through the stress-recovery continuation to reach K1/K2
                fields += [None] * 9 + [prop.kappa, prop.kappa]
            lines += _card8("PBAR", *fields)
        elif prop.type == "shell":
            lines += _card8(
                "PSHELL", pid, prop.material_id, prop.t, None, None, None, None, prop.nsm
            )
        elif prop.type == "solid":
            lines += _card8("PSOLID", pid, prop.material_id)
        else:  # lumped -> values are written on the element cards
            lumped_pids.add(pid)

    def _lumped_value(el_pid: int | None, attr: str) -> float:
        if el_pid is None or el_pid not in model.properties:
            return 0.0
        v = getattr(model.properties[el_pid], attr)
        return 0.0 if v is None else float(v)

    for eid in sorted(model.elements):
        el = model.elements[eid]
        t = el.type
        if t in ("BAR2", "TRUSS2D"):
            lines += _card8("CROD", eid, el.property_id, *el.nodes)
        elif t == "BEAM2":
            v = el.orientation
            if v is None:
                axis = model.nodes[el.nodes[1]].xyz - model.nodes[el.nodes[0]].xyz
                n = np.linalg.norm(axis)
                axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
                v = np.array([0.0, 0.0, 1.0])
                if abs(float(axis @ v)) > 0.999:
                    v = np.array([0.0, 1.0, 0.0])
            lines += _card8("CBAR", eid, el.property_id, *el.nodes, *v)
        elif t == "TRIA3":
            lines += _card8("CTRIA3", eid, el.property_id, *el.nodes)
        elif t == "QUAD4":
            lines += _card8("CQUAD4", eid, el.property_id, *el.nodes)
        elif t == "TET4":
            lines += _card8("CTETRA", eid, el.property_id, *el.nodes)
        elif t == "HEX8":
            lines += _card8("CHEXA", eid, el.property_id, *el.nodes)
        elif t == "MASS":
            lines += _card8("CONM2", eid, el.nodes[0], 0, _lumped_value(el.property_id, "m"))
        elif t == "SPRING":
            d = el.dofs or (0,) * el.n_nodes
            g2 = el.nodes[1] if el.n_nodes == 2 else None
            c2 = d[1] + 1 if el.n_nodes == 2 else None
            lines += _card8(
                "CELAS2", eid, _lumped_value(el.property_id, "k"), el.nodes[0], d[0] + 1, g2, c2
            )
        elif t == "DAMPER":
            d = el.dofs or (0,) * el.n_nodes
            g2 = el.nodes[1] if el.n_nodes == 2 else None
            c2 = d[1] + 1 if el.n_nodes == 2 else None
            lines += _card8(
                "CDAMP2", eid, _lumped_value(el.property_id, "c"), el.nodes[0], d[0] + 1, g2, c2
            )
        else:  # pragma: no cover - catalogue is closed
            raise BdfError(f"element type {t} has no BDF mapping")

    for spc in model.spcs:
        comps = spc.comps
        if not comps:
            continue
        sid = spc.sid if spc.sid > 0 else 1
        if spc.value == 0.0:
            lines += _card8("SPC1", sid, int(comps), spc.node_id)
        else:
            lines += _card8("SPC", sid, spc.node_id, int(comps), spc.value)

    for load in model.loads:
        if load.force is not None:
            lines += _card8("FORCE", load.sid, load.node_id, 0, 1.0, *load.force)
        if load.moment is not None:
            lines += _card8("MOMENT", load.sid, load.node_id, 0, 1.0, *load.moment)

    lines.append("ENDDATA")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

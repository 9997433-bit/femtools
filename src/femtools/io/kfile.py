"""LS-DYNA keyword file (.k) translator -- keyword TEXT subset.

An original parser built from the publicly documented keyword card
layouts (same policy as the BDF/CDB translators: **no** vendor code,
**no** binary d3plot/d3hsp support, by design).  Supported keywords:

======================  =====================================================
keyword                 cards (standard fixed-width or comma free format)
======================  =====================================================
``*KEYWORD``            file marker (options ignored)
``*TITLE``              one card: model name
``*NODE``               ``nid(I8) x y z (E16 each) tc(I8) rc(I8)``;
                        tc/rc constraint codes 1-7 become SPCs
``*ELEMENT_SOLID``      ``eid pid n1..n8`` (10I8); the documented
                        two-line form (``eid pid`` / ``n1..n10``) is
                        also read.  A complete ten-node record (10
                        distinct nodes: 4 corners + 6 midsides) is a
                        first-class TET10 since Round 10.  ANSYS-style
                        degeneracy collapse otherwise: 8 unique corners
                        -> HEX8, 4 -> TET4 (degenerate ten-node records
                        drop their midsides with a warning);
                        wedges/pyramids are skipped with a warning
``*ELEMENT_SHELL``      ``eid pid n1..n4 [n5..n8]`` (10I8); ``n4`` blank
                        or duplicated -> TRIA3; midside nodes n5..n8 of
                        an 8-node shell are dropped with a warning
``*ELEMENT_BEAM``       ``eid pid n1 n2 n3 ...`` (10I8); the third node
                        is the orientation node
``*MAT_ELASTIC``        ``mid ro e pr`` (I10/E10); ``_TITLE`` variant read
``*SECTION_SOLID``      ``secid elform``
``*SECTION_SHELL``      card 1 ``secid elform ...``; card 2 ``t1..t4``
                        (thickness = T1; differing T2..T4 are warned)
``*SECTION_BEAM``       card 1 ``secid elform shrf qr/irid cst``; card 2
                        by ELFORM: 1 = Hughes-Liu (CST 0 rectangular
                        ``ts1 ts2 tt1 tt2``, CST 1 tubular outer/inner
                        diameters -> A/I/J closed forms), 2 = resultant
                        ``A Iss Itt J SA``, 3 = truss ``A`` (elements
                        with a truss section become BAR2)
``*PART``               title card + ``pid secid mid`` card; a femtools
                        property per part (``_TITLE`` form == plain form)
``*BOUNDARY_SPC_NODE``  ``nid cid dofx dofy dofz dofrx dofry dofrz``;
                        ``cid != 0`` is warned and treated as global
``*END``                stops parsing
======================  =====================================================

Element type mapping (LS-DYNA -> femtools): ``*ELEMENT_SOLID`` ->
HEX8/TET4/TET10, ``*ELEMENT_SHELL`` -> QUAD4/TRIA3, ``*ELEMENT_BEAM`` ->
BEAM2 (BAR2 when its section has ELFORM 3).  Beam section inertias map
as ``Iss -> Iy``, ``Itt -> Iz`` with the orientation node providing the
femtools ``orientation`` vector (local x-y plane).

Card layout: data lines containing a comma are split as free format;
otherwise they are sliced at the documented fixed column widths (8-char
ids on nodes/elements, 10-char fields elsewhere; ``*NODE`` coordinates
are 16 chars).  ``$`` lines are comments.  The long/I10 format markers
(``*NODE +`` / ``*NODE %``) are not supported and raise.  Unknown
keywords are ignored with **one aggregated** ``UserWarning`` per
keyword name (like the BDF midside drop); malformed cards raise
:class:`KFileError`, a :class:`~femtools.core.errors.FileFormatError`.

A .k file carries no unit information: the returned model keeps the
default SI :class:`~femtools.core.units.UnitSystem` and deck consistency
is the caller's responsibility.

:func:`write_k` (Round 7) emits the same subset in comma free format
(full double precision; the reader accepts it on every card):
``*TITLE`` = model name, ``*NODE``, ``*ELEMENT_SOLID`` (TET4 with the
last corner repeated to 8; TET10 in the documented two-line ten-node
form), ``*ELEMENT_SHELL`` (TRIA3 with n3 repeated),
``*ELEMENT_BEAM`` (orientation vectors become extra third nodes appended
after the real nodes), ``*MAT_ELASTIC``/``_TITLE``, one ``*SECTION_*`` +
``*PART`` pair per femtools property (secid = part id = property id;
beam -> ELFORM 2 resultant, bar/TRUSS2D -> ELFORM 3 truss) and
``*BOUNDARY_SPC_NODE`` rows for the SPCs.  Documented losses (aggregated
``UserWarning`` s, never silent): lumped MASS/SPRING/DAMPER elements and
properties, nodal loads, RBE2 tables, enforced SPC values (written as
fixed-at-zero), beam ``kappa`` and TRUSS2D planarity (reads back as
BAR2).  Named sets are metadata without a subset keyword and are
dropped silently.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import FileFormatError
from ..core.model import FEModel

__all__ = ["read_k", "write_k", "KFileError"]


class KFileError(FileFormatError):
    """Raised for malformed LS-DYNA keyword content (a :class:`ValueError`
    via :class:`~femtools.core.errors.FileFormatError`)."""


# ---------------------------------------------------------------------------
# card slicing
# ---------------------------------------------------------------------------


def _tokens(line: str, widths: tuple[int, ...]) -> list[str]:
    """Free format (commas) or fixed columns -> stripped token list."""
    if "," in line:
        return [t.strip() for t in line.split(",")]
    out: list[str] = []
    pos = 0
    for w in widths:
        out.append(line[pos : pos + w].strip())
        pos += w
        if pos >= len(line):
            break
    return out


def _int_at(toks: list[str], i: int, lineno: int, what: str, default: int = 0) -> int:
    if i >= len(toks) or not toks[i]:
        return default
    try:
        return int(float(toks[i])) if "." in toks[i] else int(toks[i])
    except ValueError as exc:
        raise KFileError(
            f"cannot parse {toks[i]!r} as an integer ({what})", line=lineno
        ) from exc


def _float_at(toks: list[str], i: int, lineno: int, what: str, default: float = 0.0) -> float:
    if i >= len(toks) or not toks[i]:
        return default
    try:
        return float(toks[i].replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise KFileError(
            f"cannot parse {toks[i]!r} as a real number ({what})", line=lineno
        ) from exc


#: Column layouts (standard keyword format).
_W_NODE = (8, 16, 16, 16, 8, 8)
_W_I8 = (8,) * 10
_W_I10 = (10,) * 8

#: *NODE / *ELEMENT tc-rc constraint codes -> constrained axes (0-based offsets).
_SPC_CODE = {
    0: (),
    1: (0,),
    2: (1,),
    3: (2,),
    4: (0, 1),
    5: (1, 2),
    6: (2, 0),
    7: (0, 1, 2),
}


@dataclass
class _Keyword:
    name: str  # normalised, without a trailing _TITLE
    titled: bool
    cards: list[tuple[int, str]]  # (lineno, text)
    line: int


def _keywords(text: str) -> list[_Keyword]:
    out: list[_Keyword] = []
    ended = False
    for i, raw in enumerate(text.splitlines()):
        lineno = i + 1
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("$"):
            continue
        if line.lstrip().startswith("*"):
            head = line.strip().upper()
            name = head.split()[0].lstrip("*")
            rest = head[len(head.split()[0]) :].strip()
            if rest in ("+", "%") or name.endswith(("+", "%")):
                raise KFileError(
                    f"long/I10 keyword format marker on *{name} is not supported "
                    "(standard fixed-width or comma-separated cards only)",
                    line=lineno,
                )
            if name == "END":
                ended = True
                break
            titled = name.endswith("_TITLE")
            if titled:
                name = name[: -len("_TITLE")]
            out.append(_Keyword(name=name, titled=titled, cards=[], line=lineno))
        else:
            if not out:
                raise KFileError(
                    f"data card before any keyword: {line.strip()!r}", line=lineno
                )
            out[-1].cards.append((lineno, line))
    if not out and not ended:
        raise KFileError("no LS-DYNA keywords found (not a keyword deck?)")
    return out


# ---------------------------------------------------------------------------
# section records
# ---------------------------------------------------------------------------


@dataclass
class _SectionRec:
    secid: int
    kind: str  # "solid" | "shell" | "beam" | "bar"
    fields: dict[str, float]


def _tube_section(do: float, di: float) -> dict[str, float]:
    pi = float(np.pi)
    return {
        "A": pi * (do**2 - di**2) / 4.0,
        "Iy": pi * (do**4 - di**4) / 64.0,
        "Iz": pi * (do**4 - di**4) / 64.0,
        "J": pi * (do**4 - di**4) / 32.0,
    }


def _rect_section(ts: float, tt: float) -> dict[str, float]:
    """Rectangle ``ts x tt`` (s- and t-direction thicknesses): standard
    closed forms; torsion via the Roark/Timoshenko rectangular formula."""
    length, s = (ts, tt) if ts >= tt else (tt, ts)
    j = 0.0
    if s > 0.0 and length > 0.0:
        j = length * s**3 * (1.0 / 3.0 - 0.21 * (s / length) * (1.0 - s**4 / (12.0 * length**4)))
    return {
        "A": ts * tt,
        "Iy": ts * tt**3 / 12.0,  # about s (deflection in t) -> local y
        "Iz": tt * ts**3 / 12.0,
        "J": j,
    }


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_k(path: str | Path) -> FEModel:
    """Read an LS-DYNA keyword TEXT subset into an :class:`FEModel`.

    Keywords are collected first and the model is built in dependency
    order (nodes, materials, sections, parts/properties, elements,
    constraints), so keyword order in the file does not matter.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    keywords = _keywords(text)

    model_name: str | None = None
    nodes: dict[int, tuple[float, float, float]] = {}
    node_spc: list[tuple[int, int, int, int]] = []  # nid, tc, rc, lineno
    # eid -> (pid, femtools type or "BEAM", nodes tuple, orientation node)
    solids: list[tuple[int, int, tuple[int, ...], bool, int]] = []
    shells: list[tuple[int, int, tuple[int, ...], int]] = []
    beams: list[tuple[int, int, int, int, int, int]] = []  # eid pid n1 n2 n3 lineno
    materials: dict[int, tuple[float, float, float, str, int]] = {}  # mid -> ro,e,pr,title,line
    sections: dict[int, _SectionRec] = {}
    parts: dict[int, tuple[int, int, str, int]] = {}  # pid -> secid, mid, title, lineno
    spc_rows: list[tuple[int, int, tuple[int, ...], int]] = []  # nid, cid, dofs, lineno
    unsupported: dict[str, int] = {}
    notes: list[str] = []
    warned_local_spc = False

    for kw in keywords:
        name = kw.name
        cards = kw.cards
        if name == "KEYWORD":
            continue
        if name == "TITLE":
            if cards:
                model_name = cards[0][1].strip()
        elif name == "NODE":
            for lineno, line in cards:
                toks = _tokens(line, _W_NODE)
                if not any(toks):
                    continue
                nid = _int_at(toks, 0, lineno, "*NODE nid")
                if nid <= 0:
                    raise KFileError(f"invalid node id {nid}", line=lineno)
                if nid in nodes:
                    raise KFileError(f"duplicate node id {nid}", line=lineno)
                nodes[nid] = (
                    _float_at(toks, 1, lineno, "*NODE x"),
                    _float_at(toks, 2, lineno, "*NODE y"),
                    _float_at(toks, 3, lineno, "*NODE z"),
                )
                tc = _int_at(toks, 4, lineno, "*NODE tc")
                rc = _int_at(toks, 5, lineno, "*NODE rc")
                if tc or rc:
                    node_spc.append((nid, tc, rc, lineno))
        elif name == "ELEMENT_SOLID":
            pending: tuple[int, int, int] | None = None  # eid, pid, lineno
            for lineno, line in cards:
                toks = _tokens(line, _W_I8)
                if not any(toks):
                    continue
                if pending is not None:
                    eid, pid, first_line = pending
                    pending = None
                    conn = tuple(
                        _int_at(toks, k, lineno, "*ELEMENT_SOLID node") for k in range(10)
                    )
                    solids.append((eid, pid, conn, True, first_line))
                    continue
                vals = [
                    _int_at(toks, k, lineno, "*ELEMENT_SOLID field") for k in range(len(toks))
                ]
                nonzero = [k for k, v in enumerate(vals) if v]
                if max(nonzero, default=0) <= 1 and len(vals) >= 2:
                    # documented two-line form: "eid pid" then "n1..n10"
                    pending = (vals[0], vals[1], lineno)
                    continue
                eid, pid = vals[0], vals[1] if len(vals) > 1 else 0
                conn = tuple(vals[2:10]) + (0,) * (8 - len(vals[2:10]))
                solids.append((eid, pid, conn, False, lineno))
            if pending is not None:
                raise KFileError(
                    "*ELEMENT_SOLID two-line record is missing its node card",
                    line=pending[2],
                )
        elif name == "ELEMENT_SHELL":
            for lineno, line in cards:
                toks = _tokens(line, _W_I8)
                if not any(toks):
                    continue
                eid = _int_at(toks, 0, lineno, "*ELEMENT_SHELL eid")
                pid = _int_at(toks, 1, lineno, "*ELEMENT_SHELL pid")
                conn = tuple(
                    _int_at(toks, k, lineno, "*ELEMENT_SHELL node") for k in range(2, 10)
                )
                shells.append((eid, pid, conn, lineno))
        elif name == "ELEMENT_BEAM":
            for lineno, line in cards:
                toks = _tokens(line, _W_I8)
                if not any(toks):
                    continue
                eid = _int_at(toks, 0, lineno, "*ELEMENT_BEAM eid")
                pid = _int_at(toks, 1, lineno, "*ELEMENT_BEAM pid")
                n1 = _int_at(toks, 2, lineno, "*ELEMENT_BEAM n1")
                n2 = _int_at(toks, 3, lineno, "*ELEMENT_BEAM n2")
                n3 = _int_at(toks, 4, lineno, "*ELEMENT_BEAM n3")
                beams.append((eid, pid, n1, n2, n3, lineno))
        elif name == "MAT_ELASTIC":
            idx = 0
            title = ""
            if kw.titled:
                if not cards:
                    raise KFileError("*MAT_ELASTIC_TITLE has no title card", line=kw.line)
                title = cards[0][1].strip()
                idx = 1
            if len(cards) <= idx:
                raise KFileError("*MAT_ELASTIC has no data card", line=kw.line)
            lineno, line = cards[idx]
            toks = _tokens(line, _W_I10)
            mid = _int_at(toks, 0, lineno, "*MAT_ELASTIC mid")
            if mid <= 0:
                raise KFileError(f"invalid material id {mid}", line=lineno)
            if mid in materials:
                raise KFileError(f"duplicate material id {mid}", line=lineno)
            materials[mid] = (
                _float_at(toks, 1, lineno, "*MAT_ELASTIC ro"),
                _float_at(toks, 2, lineno, "*MAT_ELASTIC e"),
                _float_at(toks, 3, lineno, "*MAT_ELASTIC pr"),
                title,
                lineno,
            )
            if len(cards) > idx + 1:
                notes.append(
                    f"*MAT_ELASTIC {mid}: {len(cards) - idx - 1} extra card(s) ignored"
                )
        elif name in ("SECTION_SOLID", "SECTION_SHELL", "SECTION_BEAM"):
            idx = 1 if kw.titled else 0
            if kw.titled and not cards:
                raise KFileError(f"*{name}_TITLE has no title card", line=kw.line)
            if len(cards) <= idx:
                raise KFileError(f"*{name} has no data card", line=kw.line)
            lineno, line = cards[idx]
            toks = _tokens(line, _W_I10)
            secid = _int_at(toks, 0, lineno, f"*{name} secid")
            if secid <= 0:
                raise KFileError(f"invalid section id {secid}", line=lineno)
            if secid in sections:
                raise KFileError(f"duplicate section id {secid}", line=lineno)
            elform = _int_at(toks, 1, lineno, f"*{name} elform", default=1)
            if name == "SECTION_SOLID":
                sections[secid] = _SectionRec(secid, "solid", {})
            elif name == "SECTION_SHELL":
                if len(cards) <= idx + 1:
                    raise KFileError(
                        "*SECTION_SHELL is missing its thickness card", line=lineno
                    )
                lineno2, line2 = cards[idx + 1]
                toks2 = _tokens(line2, _W_I10)
                t_vals = [
                    _float_at(toks2, k, lineno2, "*SECTION_SHELL thickness")
                    for k in range(4)
                ]
                t1 = t_vals[0]
                if t1 <= 0.0:
                    raise KFileError(
                        f"*SECTION_SHELL {secid}: thickness T1 must be > 0, got {t1}",
                        line=lineno2,
                    )
                others = [t for t in t_vals[1:] if t]
                if any(abs(t - t1) > 1.0e-12 * max(abs(t1), 1.0) for t in others):
                    notes.append(
                        f"*SECTION_SHELL {secid}: non-uniform thicknesses "
                        f"{t_vals} reduced to T1={t1}"
                    )
                sections[secid] = _SectionRec(secid, "shell", {"t": t1})
            else:  # SECTION_BEAM
                if len(cards) <= idx + 1:
                    raise KFileError(
                        "*SECTION_BEAM is missing its second card", line=lineno
                    )
                cst = _int_at(toks, 4, lineno, "*SECTION_BEAM cst")
                lineno2, line2 = cards[idx + 1]
                toks2 = _tokens(line2, _W_I10)
                bvals = [
                    _float_at(toks2, k, lineno2, "*SECTION_BEAM card 2") for k in range(6)
                ]
                fields: dict[str, float]
                if elform == 3:  # truss: A, RAMPT, STRESS
                    if bvals[0] <= 0.0:
                        raise KFileError(
                            f"*SECTION_BEAM {secid} (truss): area must be > 0",
                            line=lineno2,
                        )
                    sections[secid] = _SectionRec(secid, "bar", {"A": bvals[0]})
                elif elform == 2:  # resultant: A, ISS, ITT, IRR(J), SA
                    a, iss, itt, irr = bvals[0], bvals[1], bvals[2], bvals[3]
                    if irr == 0.0:
                        irr = iss + itt
                        notes.append(
                            f"*SECTION_BEAM {secid}: torsion constant IRR missing; "
                            f"J = ISS + ITT = {irr} assumed"
                        )
                    fields = {"A": a, "Iy": iss, "Iz": itt, "J": irr}
                    if bvals[4]:
                        fields["Asy"] = fields["Asz"] = bvals[4]
                    sections[secid] = _SectionRec(secid, "beam", fields)
                elif elform == 1:  # Hughes-Liu: TS1 TS2 TT1 TT2 (dims)
                    ts = 0.5 * (bvals[0] + (bvals[1] or bvals[0]))
                    tt = 0.5 * (bvals[2] + (bvals[3] or bvals[2]))
                    if cst == 1:  # tubular: outer/inner diameters
                        fields = _tube_section(ts, tt)
                    elif cst == 0:  # rectangular
                        fields = _rect_section(ts, tt)
                    else:
                        raise KFileError(
                            f"*SECTION_BEAM {secid}: CST={cst} is not supported "
                            "(0 rectangular or 1 tubular)",
                            line=lineno,
                        )
                    if fields["A"] <= 0.0:
                        raise KFileError(
                            f"*SECTION_BEAM {secid}: degenerate cross section "
                            f"(TS={ts}, TT={tt})",
                            line=lineno2,
                        )
                    sections[secid] = _SectionRec(secid, "beam", fields)
                else:
                    raise KFileError(
                        f"*SECTION_BEAM {secid}: ELFORM={elform} is not supported "
                        "(1 Hughes-Liu, 2 resultant, 3 truss)",
                        line=lineno,
                    )
        elif name == "PART":
            # card 1: heading (text), card 2: pid secid mid ...; the pair may
            # repeat inside one *PART keyword.  *PART_TITLE uses the same
            # two-card layout.
            i = 0
            while i < len(cards):
                title = cards[i][1].strip()
                if i + 1 >= len(cards):
                    raise KFileError(
                        "*PART heading card without a data card", line=cards[i][0]
                    )
                lineno, line = cards[i + 1]
                toks = _tokens(line, _W_I10)
                pid = _int_at(toks, 0, lineno, "*PART pid")
                if pid <= 0:
                    raise KFileError(f"invalid part id {pid}", line=lineno)
                if pid in parts:
                    raise KFileError(f"duplicate part id {pid}", line=lineno)
                secid = _int_at(toks, 1, lineno, "*PART secid")
                mid = _int_at(toks, 2, lineno, "*PART mid")
                parts[pid] = (secid, mid, title, lineno)
                i += 2
        elif name == "BOUNDARY_SPC_NODE":
            for lineno, line in cards:
                toks = _tokens(line, _W_I10)
                if not any(toks):
                    continue
                nid = _int_at(toks, 0, lineno, "*BOUNDARY_SPC_NODE nid")
                cid = _int_at(toks, 1, lineno, "*BOUNDARY_SPC_NODE cid")
                if cid and not warned_local_spc:
                    notes.append(
                        "*BOUNDARY_SPC_NODE: local coordinate system CID ignored "
                        "(constraints applied in global axes)"
                    )
                    warned_local_spc = True
                dofs = tuple(
                    d
                    for d in range(6)
                    if _int_at(toks, 2 + d, lineno, "*BOUNDARY_SPC_NODE dof flag")
                )
                if dofs:
                    spc_rows.append((nid, cid, dofs, lineno))
        else:
            unsupported[name] = unsupported.get(name, 0) + 1

    # -- build the model in dependency order -----------------------------------
    model = FEModel(name=model_name or Path(path).stem)
    for nid, xyz in nodes.items():
        model.add_node(id=nid, xyz=xyz)

    for mid, (ro, e, pr, title, lineno) in materials.items():
        if e <= 0.0:
            raise KFileError(
                f"*MAT_ELASTIC {mid}: Young's modulus must be > 0, got {e}", line=lineno
            )
        model.add_material(id=mid, type="isotropic", E=e, nu=pr, rho=ro, name=title)

    part_kind: dict[int, str] = {}
    for pid, (secid, mid, title, lineno) in parts.items():
        sec = sections.get(secid)
        if sec is None:
            raise KFileError(f"*PART {pid}: section {secid} is not defined", line=lineno)
        if mid not in model.materials:
            raise KFileError(f"*PART {pid}: material {mid} is not defined", line=lineno)
        model.add_property(
            id=pid, type=sec.kind, material_id=mid, name=title, **sec.fields
        )
        part_kind[pid] = sec.kind

    def _check_part(eid: int, pid: int, kinds: tuple[str, ...], what: str, lineno: int) -> None:
        if pid not in parts:
            raise KFileError(f"{what} {eid}: part {pid} is not defined", line=lineno)
        if part_kind[pid] not in kinds:
            raise KFileError(
                f"{what} {eid}: part {pid} has a {part_kind[pid]!r} section "
                f"(expected {' or '.join(kinds)})",
                line=lineno,
            )

    midside_drops: dict[str, list[int]] = {}
    degenerate_skips: list[int] = []

    for eid, pid, conn, ten_node, lineno in solids:
        _check_part(eid, pid, ("solid",), "*ELEMENT_SOLID", lineno)
        nonzero = [n for n in conn if n]
        if ten_node and len(nonzero) == 10:
            unique = list(dict.fromkeys(nonzero))
            if len(unique) == 10:  # complete ten-node tet: first-class TET10
                model.add_element(id=eid, type="TET10", nodes=unique, property_id=pid)
                continue
            midside_drops.setdefault("*ELEMENT_SOLID (degenerate TET10 -> TET4)", []).append(eid)
            unique = list(dict.fromkeys(nonzero[:4]))
        else:
            unique = list(dict.fromkeys(nonzero))
        if len(unique) == 8:
            model.add_element(id=eid, type="HEX8", nodes=unique, property_id=pid)
        elif len(unique) == 4:
            model.add_element(id=eid, type="TET4", nodes=unique, property_id=pid)
        else:
            degenerate_skips.append(eid)
    for eid, pid, conn, lineno in shells:
        _check_part(eid, pid, ("shell",), "*ELEMENT_SHELL", lineno)
        if any(conn[4:]):
            midside_drops.setdefault("*ELEMENT_SHELL (8-node -> 4-node)", []).append(eid)
        corners = list(dict.fromkeys(n for n in conn[:4] if n))
        if len(corners) == 4:
            model.add_element(id=eid, type="QUAD4", nodes=corners, property_id=pid)
        elif len(corners) == 3:
            model.add_element(id=eid, type="TRIA3", nodes=corners, property_id=pid)
        else:
            raise KFileError(
                f"*ELEMENT_SHELL {eid}: needs 3 or 4 distinct nodes, got {corners}",
                line=lineno,
            )
    for eid, pid, n1, n2, n3, lineno in beams:
        _check_part(eid, pid, ("beam", "bar"), "*ELEMENT_BEAM", lineno)
        ftype = "BEAM2" if part_kind[pid] == "beam" else "BAR2"
        orientation = None
        if n3 and ftype == "BEAM2":
            if n3 not in nodes or n1 not in nodes:
                raise KFileError(
                    f"*ELEMENT_BEAM {eid}: orientation node {n3} (or {n1}) undefined",
                    line=lineno,
                )
            orientation = np.asarray(nodes[n3], dtype=float) - np.asarray(
                nodes[n1], dtype=float
            )
        model.add_element(
            id=eid, type=ftype, nodes=(n1, n2), property_id=pid, orientation=orientation
        )

    # -- constraints -------------------------------------------------------------
    for nid, tc, rc, lineno in node_spc:
        if tc not in _SPC_CODE or rc not in _SPC_CODE:
            raise KFileError(
                f"*NODE {nid}: invalid constraint code tc={tc}, rc={rc}", line=lineno
            )
        mask = [False] * 6
        for d in _SPC_CODE[tc]:
            mask[d] = True
        for d in _SPC_CODE[rc]:
            mask[3 + d] = True
        model.add_spc(node_id=nid, mask=mask, sid=0)
    for nid, _cid, dofs, lineno in spc_rows:
        if nid not in model.nodes:
            raise KFileError(
                f"*BOUNDARY_SPC_NODE references undefined node {nid}", line=lineno
            )
        mask = [d in dofs for d in range(6)]
        model.add_spc(node_id=nid, mask=mask, sid=1)

    # -- aggregated warnings -------------------------------------------------------
    fname = Path(path).name
    for what, eids in midside_drops.items():
        shown = ", ".join(map(str, eids[:5])) + (", ..." if len(eids) > 5 else "")
        notes.append(
            f"{what}: midside nodes dropped on {len(eids)} element(s) [{shown}]; "
            "quadratic accuracy is lost"
        )
    if degenerate_skips:
        shown = ", ".join(map(str, degenerate_skips[:5])) + (
            ", ..." if len(degenerate_skips) > 5 else ""
        )
        notes.append(
            f"*ELEMENT_SOLID: skipped {len(degenerate_skips)} wedge/pyramid "
            f"element(s) [{shown}] (only HEX8/TET4 degeneracies are supported)"
        )
    for name, count in sorted(unsupported.items()):
        warnings.warn(
            f"read_k({fname}): skipped unsupported keyword *{name} (x{count})",
            UserWarning,
            stacklevel=2,
        )
    for note in notes:
        warnings.warn(f"read_k({fname}): {note}", UserWarning, stacklevel=2)
    return model


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _gk(v: float | None) -> str:
    """Full-precision comma-free-format value."""
    return format(0.0 if v is None else float(v), ".17g")


def write_k(path: str | Path | FEModel, model: FEModel | str | Path | None = None) -> None:
    """Write an :class:`FEModel` as an LS-DYNA keyword TEXT subset.

    Accepts ``write_k(path, model)`` or ``write_k(model, path)``.

    Emits exactly the keyword subset :func:`read_k` parses, in comma
    free format (see the module docstring for the layout and the
    documented losses).  BEAM2 orientation vectors are materialized as
    ``*ELEMENT_BEAM`` third nodes: one extra ``*NODE`` per distinct
    ``(end A, orientation)`` position, ids continuing after the real
    nodes (they read back as unattached nodes; the assembler drops
    their DOFs).  Every degradation raises one aggregated
    ``UserWarning``; a material that cannot express ``E > 0`` raises
    :class:`KFileError` (``*MAT_ELASTIC`` requires it).
    """
    from ._compat import coerce_path_model

    out_path, model = coerce_path_model(path, model)
    notes: list[str] = []

    # -- plan elements (and the beam orientation nodes they need) -----------
    next_nid = max(model.nodes, default=0) + 1
    orient_ids: dict[tuple[float, float, float], int] = {}
    extra_nodes: list[tuple[int, float, float, float]] = []
    solid_rows: list[str] = []
    shell_rows: list[str] = []
    beam_rows: list[str] = []
    dropped_lumped_elems: dict[str, int] = {}
    n_truss2d = 0

    for eid in sorted(model.elements):
        el = model.elements[eid]
        pid = el.property_id or 0
        if el.type in ("MASS", "SPRING", "DAMPER"):
            dropped_lumped_elems[el.type] = dropped_lumped_elems.get(el.type, 0) + 1
            continue
        if el.type == "TET10":
            # documented two-line form: "eid, pid" then the 10 nodes
            solid_rows.append(f"{eid}, {pid}")
            solid_rows.append(", ".join(str(n) for n in el.nodes))
        elif el.type in ("HEX8", "TET4"):
            conn = list(el.nodes)
            conn += [conn[-1]] * (8 - len(conn))  # TET4: repeat the last corner
            solid_rows.append(f"{eid}, {pid}, " + ", ".join(str(n) for n in conn))
        elif el.type in ("QUAD4", "TRIA3"):
            conn = list(el.nodes)
            if el.type == "TRIA3":
                conn.append(conn[2])
            shell_rows.append(f"{eid}, {pid}, " + ", ".join(str(n) for n in conn))
        else:  # BEAM2 / BAR2 / TRUSS2D
            if el.type == "TRUSS2D":
                n_truss2d += 1
            n3 = 0
            if el.type == "BEAM2" and el.orientation is not None:
                xyz = model.nodes[el.nodes[0]].xyz + np.asarray(el.orientation, dtype=float)
                key = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                n3 = orient_ids.get(key, 0)
                if not n3:
                    n3 = next_nid
                    next_nid += 1
                    orient_ids[key] = n3
                    extra_nodes.append((n3, *key))
            beam_rows.append(f"{eid}, {pid}, {el.nodes[0]}, {el.nodes[1]}, {n3}")

    lines: list[str] = [
        "*KEYWORD",
        "$ femtools LS-DYNA keyword export",
        "*TITLE",
        model.name or "model",
    ]

    # -- nodes -----------------------------------------------------------------
    all_nodes = [(nid, *model.nodes[nid].xyz) for nid in model.node_ids()]
    if all_nodes or extra_nodes:
        lines.append("*NODE")
        for nid, x, y, z in [*all_nodes, *extra_nodes]:
            lines.append(f"{nid}, {_gk(x)}, {_gk(y)}, {_gk(z)}")

    # -- elements -----------------------------------------------------------------
    if solid_rows:
        lines += ["*ELEMENT_SOLID", *solid_rows]
    if shell_rows:
        lines += ["*ELEMENT_SHELL", *shell_rows]
    if beam_rows:
        lines += ["*ELEMENT_BEAM", *beam_rows]

    # -- materials -------------------------------------------------------------------
    for mid in sorted(model.materials):
        mtl = model.materials[mid]
        if mtl.type != "isotropic":
            warnings.warn(
                f"write_k: material {mid} is {mtl.type}; written as *MAT_ELASTIC "
                "with E1/nu12",
                UserWarning,
                stacklevel=2,
            )
            e, nu, g = mtl.E1, mtl.nu12, None
        else:
            e, nu, g = mtl.E, mtl.nu, mtl.G
        if e is None and g is not None and nu is not None:
            e = 2.0 * g * (1.0 + nu)
        if e is None or e <= 0.0:
            raise KFileError(
                f"material {mid}: *MAT_ELASTIC requires E > 0 and neither E nor "
                "G+nu are available"
            )
        if nu is None:
            notes.append(f"material {mid}: Poisson ratio missing; PR written as 0")
        if mtl.damping:
            notes.append(f"material {mid}: structural damping GE has no card; dropped")
        if mtl.name:
            lines += ["*MAT_ELASTIC_TITLE", mtl.name]
        else:
            lines.append("*MAT_ELASTIC")
        lines.append(f"{mid}, {_gk(mtl.rho)}, {_gk(e)}, {_gk(nu)}")

    # -- sections + parts (secid = part id = femtools property id) ---------------------
    dropped_lumped_props: list[int] = []
    for pid in sorted(model.properties):
        prop = model.properties[pid]
        if prop.type == "solid":
            lines += ["*SECTION_SOLID", f"{pid}, 1"]
        elif prop.type == "shell":
            t = _gk(prop.t)
            lines += ["*SECTION_SHELL", f"{pid}, 2", f"{t}, {t}, {t}, {t}"]
        elif prop.type == "beam":
            row = f"{_gk(prop.A)}, {_gk(prop.Iy)}, {_gk(prop.Iz)}, {_gk(prop.J)}"
            asy, asz = prop.attrs.get("Asy"), prop.attrs.get("Asz")
            sa = asy if asy is not None else asz
            if sa is not None:
                if asy is not None and asz is not None and asy != asz:
                    notes.append(
                        f"beam property {pid}: Asy != Asz; SA written as Asy "
                        "(the format has one shear area)"
                    )
                row += f", {_gk(sa)}"
            if prop.kappa is not None:
                notes.append(f"beam property {pid}: shear factor kappa not written")
            lines += ["*SECTION_BEAM", f"{pid}, 2", row]
        elif prop.type == "bar":
            # trailing comma keeps the single-value card in free format
            lines += ["*SECTION_BEAM", f"{pid}, 3", f"{_gk(prop.A)},"]
        else:  # lumped: no keyword mapping in this subset
            dropped_lumped_props.append(pid)
            continue
        if prop.nsm:
            notes.append(f"property {pid}: non-structural mass has no card; dropped")
        lines += ["*PART", prop.name or f"P{pid}", f"{pid}, {pid}, {prop.material_id or 0}"]

    # -- constraints ---------------------------------------------------------------------
    spc_rows: list[str] = []
    n_enforced = 0
    for spc in model.spcs:
        if not any(spc.mask):
            continue
        if spc.value != 0.0:
            n_enforced += 1
        flags = ", ".join("1" if m else "0" for m in spc.mask)
        spc_rows.append(f"{spc.node_id}, 0, {flags}")
    if spc_rows:
        lines += ["*BOUNDARY_SPC_NODE", *spc_rows]

    # -- aggregated warnings ----------------------------------------------------------------
    for etype, count in sorted(dropped_lumped_elems.items()):
        notes.append(f"{count} {etype} element(s) dropped (no keyword mapping in this subset)")
    if dropped_lumped_props:
        notes.append(
            f"lumped propert{'ies' if len(dropped_lumped_props) > 1 else 'y'} "
            f"{dropped_lumped_props} dropped (no keyword mapping in this subset)"
        )
    if n_truss2d:
        notes.append(
            f"{n_truss2d} TRUSS2D element(s) written as truss-section beams; "
            "they read back as BAR2 (planar nature is femtools metadata)"
        )
    if n_enforced:
        notes.append(
            f"{n_enforced} enforced SPC value(s) dropped (*BOUNDARY_SPC_NODE has no "
            "value field; written as fixed at zero)"
        )
    if model.loads:
        notes.append(
            f"{len(model.loads)} nodal load(s) dropped (*LOAD_NODE is outside the "
            "supported subset)"
        )
    if model.rbe2:
        notes.append(
            f"{len(model.rbe2)} RBE2 table(s) dropped (*CONSTRAINED_NODAL_RIGID_BODY "
            "is outside the supported subset)"
        )

    lines.append("*END")
    for note in dict.fromkeys(notes):
        warnings.warn(f"write_k: {note}", UserWarning, stacklevel=2)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

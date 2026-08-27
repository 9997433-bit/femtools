"""Universal File (UNV / UFF) translator.

Supported datasets (Round 1 subset, documented in the report):

=======  ====================================================  ==========
dataset  content                                               read/write
=======  ====================================================  ==========
151      file header                                           r/w
164      unit system (length/force/temperature factors)        r/w
15       nodes, single precision                               r/w
2411     nodes, double precision (default on write)            r/w
2412     elements (rod/beam/tria/quad/tet/hex/spring/mass)     r/w
82       display tracelines                                    r/w
55       data at nodes -- normal (2) and complex (3) modes     r/w
58       function at nodal DOF -- FRFs and general functions   r/w (ascii)
30000    femtools material/property cards (private, JSON)      r/w
=======  ====================================================  ==========

Unknown datasets are skipped with a warning (never an error).  When a 164
dataset is present, node coordinates are converted to SI on read (the UFF
convention is ``value_si = value_file / factor``) and the returned model is
tagged with SI units.  Mode shapes / FRF ordinates are *not* rescaled
(their physical unit depends on the data characteristic); this is recorded
as a warning when factors differ from 1.

Exception: femtools-authored files carry the writing model's exact unit
system inside private dataset 30000 (see below).  Every value in such a
file -- coordinates *and* material/property tables -- is stored verbatim
in those units, so on read the original unit system is restored and
nothing is rescaled: the round trip is exact and the model stays
internally consistent.  (Rescaling only the coordinates, as for foreign
files, would silently mix unit systems: metre coordinates against MPa
moduli.)  Third-party readers still convert correctly via dataset 164.

Material/property gap
---------------------
The classic UFF catalogue has no simple material/property cards for this
model subset (datasets 1710/2437/... are I-DEAS database dumps far beyond
Round-2 scope), so *standard* universal files carry geometry, connectivity
and property *ids* only -- materials and section values are lost when a
model is exchanged with third-party tools.  To keep femtools round trips
lossless, :func:`write_unv` appends one **private dataset 30000** (dataset
numbers 1..32767 are legal; unassigned numbers are skipped by conforming
readers) holding the material and property tables plus the model's unit
system as line-wrapped JSON behind a ``FEMTOOLSCARDS`` marker.
:func:`read_unv` restores it; every
other UFF reader ignores it with at most an "unknown dataset" note.
``read_unv(...).model`` is always a fully-formed :class:`FEModel` either
way -- when dataset 30000 is absent the tables are simply empty.

DOF conventions: UNV direction codes ``+-1..3`` (X, Y, Z translations) and
``+-4..6`` (rotations) map to femtools local DOFs ``0..5``.
"""

from __future__ import annotations

import datetime as _dt
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.model import FEModel, Material, Property
from ..core.results import FRFResult, ModalResult
from ..core.units import UnitError, UnitSystem

__all__ = [
    "UnvData",
    "UnvFunction",
    "Traceline",
    "read_unv",
    "write_unv",
]

# -- FE descriptor id <-> femtools element type ------------------------------

_FE_TO_FEMTOOLS: dict[int, str] = {
    11: "BAR2",  # rod
    21: "BEAM2",  # linear beam
    22: "BEAM2",  # tapered beam (read as linear)
    23: "BEAM2",  # curved beam (read as linear)
    24: "BEAM2",  # parabolic beam (read as linear)
    41: "TRIA3",  # plane stress linear triangle
    44: "QUAD4",  # plane stress linear quadrilateral
    81: "TRIA3",  # axisymmetric linear triangle (approximate)
    84: "QUAD4",  # axisymmetric linear quad (approximate)
    91: "TRIA3",  # thin shell linear triangle
    94: "QUAD4",  # thin shell linear quadrilateral
    111: "TET4",  # solid linear tetrahedron
    115: "HEX8",  # solid linear brick
    136: "SPRING",  # node-to-node translational spring
    137: "SPRING",  # node-to-node rotational spring
    138: "SPRING",  # node-to-ground translational spring
    139: "SPRING",  # node-to-ground rotational spring
    141: "DAMPER",  # node-to-node damper
    142: "DAMPER",  # node-to-ground damper
    161: "MASS",  # lumped mass
}

_FEMTOOLS_TO_FE: dict[str, int] = {
    "BAR2": 11,
    "TRUSS2D": 11,  # written as rod; 2-D nature is a femtools attribute
    "BEAM2": 21,
    "TRIA3": 91,
    "QUAD4": 94,
    "TET4": 111,
    "HEX8": 115,
    "SPRING": 136,
    "DAMPER": 141,
    "MASS": 161,
}

#: descriptors whose dataset-2412 record carries the extra 3I10 beam record
_BEAM_LIKE: frozenset[int] = frozenset({11, 21, 22, 23, 24})

#: femtools private dataset carrying material/property cards as JSON.
#: 1..32767 is the legal dataset-number range; 30000 is far above every
#: catalogued UFF dataset and conforming readers skip unknown numbers.
_FEMTOOLS_CARDS_DS = 30000
_FEMTOOLS_CARDS_MARKER = "FEMTOOLSCARDS"
_FEMTOOLS_CARDS_VERSION = 1

_DELIM = "    -1"


# -- small containers ----------------------------------------------------------


@dataclass
class Traceline:
    """Dataset 82: display sequence of node ids; 0 means pen-up (move)."""

    id: int
    nodes: list[int]
    name: str = "NONE"
    color: int = 0


@dataclass
class UnvFunction:
    """One dataset 58 (function at nodal DOF), kept verbatim."""

    func_type: int  # 4 = FRF, 1 = time response, 2/3 = spectra, ...
    func_id: int
    load_case: int
    rsp_entity: str
    rsp_node: int
    rsp_dir: int  # +-1..6
    ref_entity: str
    ref_node: int
    ref_dir: int
    x: NDArray[np.float64]
    y: NDArray  # float or complex
    abscissa_spec: int = 18  # 18 = frequency, 17 = time
    ordinate_num_spec: int = 8  # 8 disp, 11 vel, 12 acc, 13 force
    ordinate_den_spec: int = 13
    id_lines: list[str] = field(default_factory=list)

    @property
    def is_frf(self) -> bool:
        return self.func_type == 4


@dataclass
class _ModeShape55:
    """One dataset 55 in modal form (analysis type 2 or 3)."""

    analysis_type: int
    mode_number: int
    freq_hz: float
    modal_mass: float
    damping_visc: float
    damping_hyst: float
    eigenvalue: complex
    ndv: int
    is_complex: bool
    data: dict[int, NDArray]  # node id -> (ndv,) real or complex
    id_lines: list[str] = field(default_factory=list)


@dataclass
class UnvData:
    """Everything read from a universal file."""

    model: FEModel
    modal: ModalResult | None = None
    frf: FRFResult | None = None
    functions: list[UnvFunction] = field(default_factory=list)
    tracelines: list[Traceline] = field(default_factory=list)
    header: dict[str, str] = field(default_factory=dict)
    units_factors: tuple[float, float, float] | None = None  # length, force, temperature


# -- low-level helpers -----------------------------------------------------------


def _to_float(tok: str) -> float:
    """Parse a Fortran real ('1.5D+03' and friends)."""
    return float(tok.replace("D", "E").replace("d", "e"))


def _line_floats(line: str) -> list[float]:
    return [_to_float(t) for t in line.split()]


def _line_ints(line: str) -> list[int]:
    return [int(t) for t in line.split()]


def _split_datasets(text: str) -> list[tuple[int, list[str]]]:
    """Split file text into (dataset_number, body_lines) blocks."""
    lines = text.splitlines()
    out: list[tuple[int, list[str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "-1":
            i += 1
            continue
        i += 1
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            dsnum = int(lines[i].strip())
        except ValueError:
            # malformed block: resynchronise on the next delimiter
            continue
        i += 1
        body: list[str] = []
        while i < n and lines[i].strip() != "-1":
            body.append(lines[i].rstrip())
            i += 1
        i += 1  # closing -1
        out.append((dsnum, body))
    return out


def _fixed_or_split(line: str, cols: Sequence[tuple[int, int]], kinds: str) -> list:
    """Parse fixed columns; fall back to whitespace splitting.

    ``kinds`` is one char per field: 'i' int, 'f' float, 's' string.
    """
    padded = line.ljust(cols[-1][1])
    try:
        vals: list = []
        for (a, b), kind in zip(cols, kinds, strict=True):
            raw = padded[a:b].strip()
            if kind == "i":
                vals.append(int(raw) if raw else 0)
            elif kind == "f":
                vals.append(_to_float(raw) if raw else 0.0)
            else:
                vals.append(raw)
        return vals
    except ValueError:
        toks = line.split()
        vals = []
        for j, kind in enumerate(kinds):
            raw = toks[j] if j < len(toks) else ""
            if kind == "i":
                vals.append(int(raw) if raw else 0)
            elif kind == "f":
                vals.append(_to_float(raw) if raw else 0.0)
            else:
                vals.append(raw)
        return vals


def _i10(v: int) -> str:
    return f"{int(v):10d}"


def _e13(v: float) -> str:
    return f"{float(v):13.5E}"


def _e20(v: float) -> str:
    return f"{float(v):20.12E}"


def _e25(v: float) -> str:
    return f"{float(v):25.16E}"


def _chunk(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _dir_to_dof(direction: int) -> tuple[int, float]:
    """UNV direction code (+-1..6) -> (local dof 0..5, sign)."""
    d = int(direction)
    if d == 0 or abs(d) > 6:
        raise ValueError(f"invalid UNV direction code {direction}")
    return abs(d) - 1, (1.0 if d > 0 else -1.0)


# -- dataset readers ---------------------------------------------------------


def _read_151(body: list[str]) -> dict[str, str]:
    keys = (
        "model_name",
        "description",
        "db_program",
        "db_created",
        "db_saved",
        "uf_program",
        "uf_written",
    )
    return {k: body[i].strip() for i, k in enumerate(keys) if i < len(body)}


def _read_164(body: list[str]) -> tuple[int, str, int, tuple[float, float, float]]:
    code, desc, tmode = 0, "", 1
    if body:
        code, desc, tmode = _fixed_or_split(body[0], [(0, 10), (10, 30), (30, 40)], "isi")
    vals: list[float] = []
    for line in body[1:]:
        vals.extend(_line_floats(line))
        if len(vals) >= 3:
            break
    while len(vals) < 3:
        vals.append(1.0)
    return int(code), str(desc).strip(), int(tmode), (vals[0], vals[1], vals[2])


def _read_nodes_15(body: list[str], model: FEModel) -> None:
    for line in body:
        toks = line.split()
        if len(toks) < 7:
            raise ValueError(f"malformed dataset 15 record: {line!r}")
        nid, cp, cd = int(toks[0]), int(toks[1]), int(toks[2])
        x, y, z = (_to_float(t) for t in toks[4:7])
        model.add_node(id=nid, xyz=(x, y, z), cp=0, cd=cd)
        model.nodes[nid].cp = cp  # keep label without re-resolving coordinates


def _read_nodes_2411(body: list[str], model: FEModel) -> None:
    i = 0
    while i + 1 < len(body):
        ints = _line_ints(body[i])
        xyz = _line_floats(body[i + 1])
        if len(ints) < 4 or len(xyz) < 3:
            raise ValueError(f"malformed dataset 2411 record near: {body[i]!r}")
        nid, cp, cd = ints[0], ints[1], ints[2]
        model.add_node(id=nid, xyz=xyz[:3], cp=0, cd=cd)
        model.nodes[nid].cp = cp
        i += 2


def _read_2412(body: list[str], model: FEModel, notes: list[str]) -> None:
    i = 0
    pending_orient: list[tuple[int, int]] = []  # (element id, orientation node)
    while i < len(body):
        rec1 = _line_ints(body[i])
        if len(rec1) < 6:
            raise ValueError(f"malformed dataset 2412 record: {body[i]!r}")
        eid, fe_id, pid, _mat, _color, n_nodes = rec1[:6]
        i += 1
        orient_node = 0
        if fe_id in _BEAM_LIKE:
            beam_rec = _line_ints(body[i])
            orient_node = beam_rec[0] if beam_rec else 0
            i += 1
        conn: list[int] = []
        while len(conn) < n_nodes and i < len(body):
            conn.extend(_line_ints(body[i]))
            i += 1
        if len(conn) != n_nodes:
            raise ValueError(f"element {eid}: expected {n_nodes} nodes, got {len(conn)}")
        etype = _FE_TO_FEMTOOLS.get(fe_id)
        if etype is None:
            notes.append(
                f"dataset 2412: skipped element {eid} with unsupported FE descriptor {fe_id}"
            )
            continue
        # grounded spring/damper descriptors connect a single node
        if fe_id in (138, 139, 142):
            conn = conn[:1]
        model.add_element(
            id=eid,
            type=etype,
            nodes=tuple(conn),
            property_id=pid if pid > 0 else None,
            check_refs=False,
        )
        if orient_node > 0:
            pending_orient.append((eid, orient_node))
    for eid, onode in pending_orient:
        el = model.elements.get(eid)
        node = model.nodes.get(onode)
        if el is not None and node is not None and el.nodes[0] in model.nodes:
            el.orientation = node.xyz - model.nodes[el.nodes[0]].xyz


def _json_scalar(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _read_cards_30000(body: list[str], model: FEModel, notes: list[str]) -> UnitSystem | None:
    """Restore materials/properties from the femtools private dataset.

    Returns the writing model's :class:`UnitSystem` when the payload carries
    one (files written since the units fix), else ``None``.
    """
    head = body[0].split() if body else []
    if not head or head[0] != _FEMTOOLS_CARDS_MARKER:
        raise ValueError(
            f"dataset {_FEMTOOLS_CARDS_DS} lacks the {_FEMTOOLS_CARDS_MARKER} marker"
        )
    version = int(head[1]) if len(head) > 1 else 1
    if version > _FEMTOOLS_CARDS_VERSION:
        notes.append(
            f"dataset {_FEMTOOLS_CARDS_DS}: version {version} newer than supported "
            f"{_FEMTOOLS_CARDS_VERSION}; material/property cards skipped"
        )
        return None
    payload = json.loads("".join(line.strip() for line in body[1:]))
    for md in payload.get("materials", ()):
        mat = Material(**md)
        if mat.id in model.materials:
            notes.append(f"dataset {_FEMTOOLS_CARDS_DS}: duplicate material {mat.id} ignored")
        else:
            model.materials[mat.id] = mat
    for pd in payload.get("properties", ()):
        prop = Property(**pd)
        if prop.id in model.properties:
            notes.append(f"dataset {_FEMTOOLS_CARDS_DS}: duplicate property {prop.id} ignored")
        else:
            model.properties[prop.id] = prop
    units_dict = payload.get("units")
    if units_dict is not None:
        try:
            return UnitSystem.from_dict(units_dict)
        except (UnitError, TypeError) as exc:
            notes.append(f"dataset {_FEMTOOLS_CARDS_DS}: unreadable unit system ({exc}); ignored")
    return None


def _read_82(body: list[str]) -> Traceline:
    rec1 = _line_ints(body[0])
    tl_id = rec1[0] if rec1 else 0
    n = rec1[1] if len(rec1) > 1 else 0
    color = rec1[2] if len(rec1) > 2 else 0
    name = body[1].strip() if len(body) > 1 else "NONE"
    nodes: list[int] = []
    for line in body[2:]:
        nodes.extend(_line_ints(line))
        if n and len(nodes) >= n:
            break
    return Traceline(id=tl_id, nodes=nodes[:n] if n else nodes, name=name, color=color)


def _read_55(body: list[str], notes: list[str]) -> _ModeShape55 | None:
    if len(body) < 7:
        raise ValueError("dataset 55 too short")
    id_lines = [ln.strip() for ln in body[:4]]
    rec5 = _line_ints(body[4])
    while len(rec5) < 6:
        rec5.append(0)
    _model_type, analysis_type, _data_char, _spec_type, data_type, ndv = rec5[:6]
    is_complex = data_type in (5, 6)

    # record 6: n_ints, n_reals, then n_ints integers (may continue on following lines)
    cursor = 5
    ints: list[int] = _line_ints(body[cursor])
    cursor += 1
    if len(ints) < 2:
        raise ValueError("dataset 55 record 6 malformed")
    n_ints, n_reals = ints[0], ints[1]
    tail = ints[2:]
    while len(tail) < n_ints and cursor < len(body):
        tail.extend(_line_ints(body[cursor]))
        cursor += 1
    reals: list[float] = []
    while len(reals) < n_reals and cursor < len(body):
        reals.extend(_line_floats(body[cursor]))
        cursor += 1

    if analysis_type == 2:
        mode_number = tail[1] if len(tail) > 1 else (tail[0] if tail else 0)
        freq = reals[0] if reals else 0.0
        modal_mass = reals[1] if len(reals) > 1 else 0.0
        d_visc = reals[2] if len(reals) > 2 else 0.0
        d_hyst = reals[3] if len(reals) > 3 else 0.0
        eigenvalue = complex((2.0 * np.pi * freq) ** 2)
    elif analysis_type == 3:
        mode_number = tail[1] if len(tail) > 1 else 0
        lam = complex(reals[0], reals[1]) if len(reals) > 1 else 0j
        freq = abs(lam) / (2.0 * np.pi)
        modal_mass = reals[2] if len(reals) > 2 else 0.0
        d_visc = -lam.real / abs(lam) if abs(lam) > 0 else 0.0
        d_hyst = 0.0
        eigenvalue = lam
    else:
        notes.append(f"dataset 55: skipped analysis type {analysis_type} (not a mode shape)")
        return None

    per_node = ndv * (2 if is_complex else 1)
    data: dict[int, NDArray] = {}
    while cursor < len(body):
        node_toks = _line_ints(body[cursor])
        cursor += 1
        if not node_toks:
            continue
        nid = node_toks[0]
        vals: list[float] = []
        while len(vals) < per_node and cursor < len(body):
            vals.extend(_line_floats(body[cursor]))
            cursor += 1
        arr = np.asarray(vals[:per_node], dtype=float)
        if is_complex:
            arr = arr[0::2] + 1j * arr[1::2]
        data[nid] = arr

    return _ModeShape55(
        analysis_type=analysis_type,
        mode_number=mode_number,
        freq_hz=freq,
        modal_mass=modal_mass,
        damping_visc=d_visc,
        damping_hyst=d_hyst,
        eigenvalue=eigenvalue,
        ndv=ndv,
        is_complex=is_complex,
        data=data,
        id_lines=id_lines,
    )


_REC6_COLS = [
    (0, 5), (5, 15), (15, 20), (20, 30), (31, 41),
    (41, 51), (51, 55), (56, 66), (66, 76), (76, 80),
]


def _read_58(body: list[str], notes: list[str]) -> UnvFunction:
    if len(body) < 12:
        raise ValueError("dataset 58 too short")
    id_lines = [ln.strip() for ln in body[:5]]
    (
        func_type,
        func_id,
        _version,
        load_case,
        rsp_entity,
        rsp_node,
        rsp_dir,
        ref_entity,
        ref_node,
        ref_dir,
    ) = _fixed_or_split(body[5], _REC6_COLS, "iiiisiisii")

    rec7 = body[6].split()
    ordinate_type = int(rec7[0])
    n_values = int(rec7[1])
    spacing = int(rec7[2]) if len(rec7) > 2 else 0
    xmin = _to_float(rec7[3]) if len(rec7) > 3 else 0.0
    dx = _to_float(rec7[4]) if len(rec7) > 4 else 0.0

    absc_spec = _fixed_or_split(body[7], [(0, 10)], "i")[0]
    ord_num_spec = _fixed_or_split(body[8], [(0, 10)], "i")[0]
    ord_den_spec = _fixed_or_split(body[9], [(0, 10)], "i")[0]
    # body[10] is the z-axis characteristics record

    vals: list[float] = []
    for line in body[11:]:
        vals.extend(_line_floats(line))

    is_complex = ordinate_type in (5, 6)
    even = spacing == 1
    if even:
        if is_complex:
            arr = np.asarray(vals[: 2 * n_values], dtype=float)
            y = arr[0::2] + 1j * arr[1::2]
        else:
            y = np.asarray(vals[:n_values], dtype=float)
        x = xmin + dx * np.arange(n_values, dtype=float)
    else:
        stride = 3 if is_complex else 2
        arr = np.asarray(vals[: stride * n_values], dtype=float).reshape(-1, stride)
        x = arr[:, 0]
        y = (arr[:, 1] + 1j * arr[:, 2]) if is_complex else arr[:, 1]
    if len(y) != n_values:
        notes.append(
            f"dataset 58 (function {func_id}): expected {n_values} values, parsed {len(y)}"
        )

    return UnvFunction(
        func_type=int(func_type),
        func_id=int(func_id),
        load_case=int(load_case),
        rsp_entity=str(rsp_entity) or "NONE",
        rsp_node=int(rsp_node),
        rsp_dir=int(rsp_dir),
        ref_entity=str(ref_entity) or "NONE",
        ref_node=int(ref_node),
        ref_dir=int(ref_dir),
        x=np.asarray(x, dtype=float),
        y=y,
        abscissa_spec=int(absc_spec),
        ordinate_num_spec=int(ord_num_spec),
        ordinate_den_spec=int(ord_den_spec),
        id_lines=id_lines,
    )


# -- result assembly ----------------------------------------------------------


def _assemble_modal(shapes: list[_ModeShape55], notes: list[str]) -> ModalResult | None:
    if not shapes:
        return None
    shapes = sorted(shapes, key=lambda s: (s.freq_hz, s.mode_number))
    ndv = max(s.ndv for s in shapes)
    node_ids = sorted({nid for s in shapes for nid in s.data})
    if not node_ids:
        return None
    dof_index = [(nid, d) for nid in node_ids for d in range(ndv)]
    row = {pair: i for i, pair in enumerate(dof_index)}
    any_complex = any(s.is_complex for s in shapes)
    modes = np.zeros((len(dof_index), len(shapes)), dtype=complex if any_complex else float)
    for j, s in enumerate(shapes):
        for nid, vec in s.data.items():
            for d in range(min(s.ndv, len(vec))):
                modes[row[(nid, d)], j] = vec[d]
    freq = np.asarray([s.freq_hz for s in shapes], dtype=float)
    eig = np.asarray([s.eigenvalue for s in shapes])
    if not any_complex:
        eig = eig.real
    gm = np.asarray([s.modal_mass for s in shapes], dtype=float)
    damping = np.asarray([s.damping_visc for s in shapes], dtype=float)
    if np.all(gm == 0.0):
        gm = np.ones_like(gm)
        notes.append("dataset 55: modal masses were 0/absent; generalized_mass set to 1.0")
    return ModalResult(
        freq_hz=freq,
        eigenvalues=eig,
        modes=modes,
        generalized_mass=gm,
        dof_index=tuple(dof_index),
        damping=None if np.all(damping == 0.0) else damping,
    )


_FRF_KIND_BY_SPEC = {8: "receptance", 11: "mobility", 12: "accelerance"}


def _assemble_frf(functions: list[UnvFunction], notes: list[str]) -> FRFResult | None:
    frfs = [f for f in functions if f.is_frf and f.rsp_node > 0 and f.ref_node > 0]
    if not frfs:
        return None
    x0 = frfs[0].x
    usable = []
    for f in frfs:
        if len(f.x) == len(x0) and np.allclose(f.x, x0, rtol=1e-10, atol=1e-12):
            usable.append(f)
        else:
            notes.append(
                f"dataset 58 (function {f.func_id}): frequency axis differs; "
                "excluded from the assembled FRFResult (still in .functions)"
            )
    if not usable:
        return None

    def pair(node: int, direction: int) -> tuple[int, int]:
        dof, _sign = _dir_to_dof(direction)
        return (node, dof)

    outputs = sorted({pair(f.rsp_node, f.rsp_dir) for f in usable})
    inputs = sorted({pair(f.ref_node, f.ref_dir) for f in usable})
    h = np.zeros((len(outputs), len(inputs), len(x0)), dtype=complex)
    filled = np.zeros((len(outputs), len(inputs)), dtype=bool)
    for f in usable:
        _, s_out = _dir_to_dof(f.rsp_dir)
        _, s_in = _dir_to_dof(f.ref_dir)
        i = outputs.index(pair(f.rsp_node, f.rsp_dir))
        j = inputs.index(pair(f.ref_node, f.ref_dir))
        h[i, j, :] = f.y * s_out * s_in
        filled[i, j] = True
    if not filled.all():
        notes.append(
            f"dataset 58: FRF grid incomplete ({int(filled.sum())}/{filled.size} "
            "output x input pairs present); missing entries are zero"
        )
    kind = _FRF_KIND_BY_SPEC.get(usable[0].ordinate_num_spec, "receptance")
    return FRFResult(
        freq_hz=x0, h_complex=h, inputs=tuple(inputs), outputs=tuple(outputs), kind=kind
    )


# -- public read --------------------------------------------------------------


def read_unv(path: str | Path) -> UnvData:
    """Read a universal file.

    Returns an :class:`UnvData` bundle: the FE model (nodes/elements/
    tracelines), an assembled :class:`ModalResult` (datasets 55) and
    :class:`FRFResult` (datasets 58) when present, plus all raw functions.
    Unsupported datasets are skipped with a ``UserWarning``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    model = FEModel(name=Path(path).stem)
    notes: list[str] = []
    header: dict[str, str] = {}
    tracelines: list[Traceline] = []
    functions: list[UnvFunction] = []
    shapes: list[_ModeShape55] = []
    units_factors: tuple[float, float, float] | None = None
    cards_units: UnitSystem | None = None

    for dsnum, body in _split_datasets(text):
        try:
            if dsnum == 151:
                header.update(_read_151(body))
            elif dsnum == 164:
                code, desc, _tmode, factors = _read_164(body)
                header["units_code"] = str(code)
                header["units_description"] = desc
                units_factors = factors
            elif dsnum == 15:
                _read_nodes_15(body, model)
            elif dsnum == 2411:
                _read_nodes_2411(body, model)
            elif dsnum == 2412:
                _read_2412(body, model, notes)
            elif dsnum == _FEMTOOLS_CARDS_DS:
                units = _read_cards_30000(body, model, notes)
                if cards_units is None:
                    cards_units = units
            elif dsnum == 82:
                tracelines.append(_read_82(body))
            elif dsnum == 55:
                shape = _read_55(body, notes)
                if shape is not None:
                    shapes.append(shape)
            elif dsnum == 58:
                functions.append(_read_58(body, notes))
            else:
                notes.append(f"skipped unsupported dataset {dsnum} ({len(body)} lines)")
        except Exception as exc:  # noqa: BLE001 - robust reader: keep going per dataset
            notes.append(f"dataset {dsnum}: parse error: {exc}")

    if header.get("model_name"):
        model.name = header["model_name"]

    # unit handling.  femtools-authored files record the writing model's unit
    # system in dataset 30000 and store every value verbatim in those units,
    # so restore the system as-is (rescaling only the coordinates would leave
    # them inconsistent with the material/property tables).  For foreign
    # files, convert coordinates to SI when a non-trivial 164 is present.
    if cards_units is not None:
        model.units = cards_units
    elif units_factors is not None:
        lf = units_factors[0]
        if lf not in (0.0, 1.0):
            for node in model.nodes.values():
                node.xyz = node.xyz / lf
            notes.append(
                f"dataset 164: coordinates divided by length factor {lf:g} (converted to SI); "
                "mode shape / FRF ordinates were NOT rescaled"
            )
        model.units = UnitSystem.si()

    modal = _assemble_modal(shapes, notes)
    frf = _assemble_frf(functions, notes)
    for note in notes:
        warnings.warn(f"read_unv({Path(path).name}): {note}", UserWarning, stacklevel=2)

    return UnvData(
        model=model,
        modal=modal,
        frf=frf,
        functions=functions,
        tracelines=tracelines,
        header=header,
        units_factors=units_factors,
    )


# -- dataset writers ------------------------------------------------------------


def _ds(lines: list[str], dsnum: int, body: list[str]) -> None:
    lines.append(_DELIM)
    lines.append(f"{dsnum:6d}")
    lines.extend(body)
    lines.append(_DELIM)


def _write_151(model_name: str, header: dict[str, str] | None) -> list[str]:
    h = header or {}
    now = _dt.datetime.now().strftime("%d-%b-%y  %H:%M:%S")
    return [
        h.get("model_name", model_name)[:80],
        h.get("description", "NONE")[:80],
        h.get("db_program", "femtools")[:80],
        h.get("db_created", now)[:80],
        h.get("db_saved", now)[:80],
        h.get("uf_program", "femtools")[:80],
        h.get("uf_written", now)[:80],
    ]


def _write_164(units: UnitSystem) -> list[str]:
    # UFF convention: value_si = value_file / factor  ->  factor = units per SI unit
    lf = 1.0 / units.si_factor("length")
    ff = 1.0 / units.si_factor("force")
    code = 1 if units.is_si else 0
    desc = f"{units.length}-{units.force}-{units.mass}-{units.time}"
    return [
        f"{code:10d}{desc:<20s}{2:10d}",
        f"{lf:25.16E}{ff:25.16E}{1.0:25.16E}",
    ]


def _write_2411(model: FEModel) -> list[str]:
    body: list[str] = []
    for nid in model.node_ids():
        node = model.nodes[nid]
        body.append(_i10(nid) + _i10(node.cp) + _i10(node.cd) + _i10(0))
        body.append("".join(_e25(v) for v in node.xyz))
    return body


def _write_15(model: FEModel) -> list[str]:
    body: list[str] = []
    for nid in model.node_ids():
        node = model.nodes[nid]
        body.append(
            _i10(nid) + _i10(node.cp) + _i10(node.cd) + _i10(0) + "".join(_e13(v) for v in node.xyz)
        )
    return body


def _write_2412(model: FEModel) -> list[str]:
    body: list[str] = []
    for eid in sorted(model.elements):
        el = model.elements[eid]
        fe_id = _FEMTOOLS_TO_FE[el.type]
        if el.type in ("SPRING", "DAMPER") and el.n_nodes == 1:
            fe_id = {"SPRING": 138, "DAMPER": 142}[el.type]
        prop = model.properties.get(el.property_id) if el.property_id is not None else None
        mat_id = prop.material_id if prop is not None and prop.material_id is not None else 0
        pid = el.property_id if el.property_id is not None else 0
        body.append(
            _i10(eid) + _i10(fe_id) + _i10(pid) + _i10(mat_id) + _i10(0) + _i10(el.n_nodes)
        )
        if fe_id in _BEAM_LIKE:
            body.append(_i10(0) + _i10(0) + _i10(0))
        for chunk in _chunk(el.nodes, 8):
            body.append("".join(_i10(n) for n in chunk))
    return body


def _write_cards_30000(model: FEModel) -> list[str]:
    """Material/property tables as line-wrapped JSON (femtools private).

    The JSON is compact (no separator whitespace) and literal spaces inside
    string values are re-escaped as ``\\u0020``, so the payload contains no
    space characters at all: it can be split at any 78-column boundary and
    re-joined after readers strip trailing blanks.
    """
    payload = {
        "materials": [
            {k: v for k, v in vars(mat).items() if v is not None or k in ("E", "nu", "rho")}
            for mat in model.materials.values()
        ],
        "properties": [
            {
                k: v
                for k, v in vars(prop).items()
                if (v is not None and (k != "attrs" or v)) or k == "material_id"
            }
            for prop in model.properties.values()
        ],
        # exact unit system of every value in this file; read_unv restores it
        # instead of rescaling coordinates to SI (additive key, still version 1:
        # older readers ignore it and behave as before)
        "units": model.units.to_dict(),
    }
    text = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, default=_json_scalar
    ).replace(" ", "\\u0020")
    body = [f"{_FEMTOOLS_CARDS_MARKER} {_FEMTOOLS_CARDS_VERSION} JSON materials/properties"]
    body.extend(text[i : i + 78] for i in range(0, len(text), 78))
    return body


def _write_82(tl: Traceline) -> list[str]:
    body = [_i10(tl.id) + _i10(len(tl.nodes)) + _i10(tl.color), (tl.name or "NONE")[:80]]
    for chunk in _chunk(tl.nodes, 8):
        body.append("".join(_i10(n) for n in chunk))
    return body


def _write_55_mode(
    modal: ModalResult,
    j: int,
    node_ids: list[int],
    row_of: dict[tuple[int, int], int],
    ndv: int,
    model_name: str,
) -> list[str]:
    is_complex = np.iscomplexobj(modal.modes)
    data_type = 5 if is_complex else 2
    data_char = 3 if ndv == 6 else 2
    analysis_type = 3 if is_complex else 2
    zeta = float(modal.damping[j]) if modal.damping is not None else 0.0
    if analysis_type == 2:
        rec6 = _i10(2) + _i10(4) + _i10(1) + _i10(j + 1)
        rec7 = (
            _e13(float(modal.freq_hz[j]))
            + _e13(float(modal.generalized_mass[j]))
            + _e13(zeta)
            + _e13(0.0)
        )
    else:
        lam = complex(modal.eigenvalues[j])
        rec6 = _i10(2) + _i10(6) + _i10(1) + _i10(j + 1)
        rec7 = (
            _e13(lam.real)
            + _e13(lam.imag)
            + _e13(float(modal.generalized_mass[j]))
            + _e13(0.0)
            + _e13(0.0)
            + _e13(0.0)
        )
    body = [
        f"{model_name} - normal modes"[:80],
        "NONE",
        "NONE",
        f"Mode {j + 1}",
        _i10(1) + _i10(analysis_type) + _i10(data_char) + _i10(8) + _i10(data_type) + _i10(ndv),
        rec6,
        rec7,
    ]
    for nid in node_ids:
        body.append(_i10(nid))
        vals: list[float] = []
        for d in range(ndv):
            r = row_of.get((nid, d))
            v = modal.modes[r, j] if r is not None else 0.0
            if is_complex:
                vals.extend([float(np.real(v)), float(np.imag(v))])
            else:
                vals.append(float(np.real(v)))
        for chunk in _chunk(vals, 6):
            body.append("".join(_e13(v) for v in chunk))
    return body


_FRF_SPEC_BY_KIND = {"receptance": 8, "mobility": 11, "accelerance": 12}


def _write_58_frf(frf: FRFResult, i_out: int, i_in: int, func_id: int) -> list[str]:
    out_node, out_dof = frf.outputs[i_out]
    in_node, in_dof = frf.inputs[i_in]
    x = frf.freq_hz
    y = frf.h_complex[i_out, i_in, :]
    n = len(x)
    even = n > 1 and bool(np.allclose(np.diff(x), x[1] - x[0], rtol=1e-8, atol=1e-12))
    xmin = float(x[0]) if n else 0.0
    dx = float(x[1] - x[0]) if even and n > 1 else 0.0
    num_spec = _FRF_SPEC_BY_KIND.get(frf.kind, 8)

    rec6 = (
        f"{4:5d}{func_id:10d}{0:5d}{0:10d} "
        f"{'NONE':<10s}{out_node:10d}{out_dof + 1:4d} "
        f"{'NONE':<10s}{in_node:10d}{in_dof + 1:4d}"
    )
    body = [
        f"FRF H({out_node},{out_dof + 1})/({in_node},{in_dof + 1})"[:80],
        "NONE",
        "NONE",
        "NONE",
        "NONE",
        rec6,
        _i10(6) + _i10(n) + _i10(1 if even else 0) + _e13(xmin) + _e13(dx) + _e13(0.0),
        _i10(18) + f"{0:5d}{0:5d}{0:5d} " + f"{'Frequency':<20s} " + f"{'Hz':<20s}",
        _i10(num_spec) + f"{0:5d}{0:5d}{0:5d} " + f"{'Response':<20s} " + f"{'NONE':<20s}",
        _i10(13) + f"{0:5d}{0:5d}{0:5d} " + f"{'Excitation':<20s} " + f"{'NONE':<20s}",
        _i10(0) + f"{0:5d}{0:5d}{0:5d} " + f"{'NONE':<20s} " + f"{'NONE':<20s}",
    ]
    if even:
        vals: list[float] = []
        for v in y:
            vals.extend([float(v.real), float(v.imag)])
        for chunk in _chunk(vals, 4):
            body.append("".join(_e20(v) for v in chunk))
    else:
        for xi, yi in zip(x, y, strict=True):
            body.append(_e13(float(xi)) + _e20(float(yi.real)) + _e20(float(yi.imag)))
    return body


# -- public write ---------------------------------------------------------------


def write_unv(
    path: str | Path | FEModel,
    model: FEModel | str | Path | None = None,
    modal: ModalResult | None = None,
    frf: FRFResult | None = None,
    tracelines: Sequence[Traceline] | None = None,
    header: dict[str, str] | None = None,
    node_dataset: int = 2411,
) -> None:
    """Write a universal file with the given content.

    Accepts ``write_unv(path, model=...)`` or ``write_unv(model, path)``.

    * ``model`` -> datasets 151/164 + nodes (2411 by default, or 15) + 2412
      (+ private dataset 30000 with material/property cards when the model
      has any -- see the module docstring; other readers skip it)
    * ``modal`` -> one dataset 55 per mode (normal or complex modes)
    * ``frf``   -> one dataset 58 per (output, input) pair
    * ``tracelines`` -> datasets 82

    A ``modal`` result without ``dof_index`` requires ``model`` (the
    canonical model DOF ordering is assumed).
    """
    from ._compat import coerce_path_model

    if not isinstance(path, (str, Path)) or (
        model is not None and not isinstance(model, FEModel)
    ):
        path, model = coerce_path_model(path, model)
    if node_dataset not in (15, 2411):
        raise ValueError("node_dataset must be 15 or 2411")
    lines: list[str] = []
    name = model.name if model is not None else "femtools"

    _ds(lines, 151, _write_151(name, header))
    units = model.units if model is not None else UnitSystem.si()
    _ds(lines, 164, _write_164(units))

    if model is not None:
        if model.nodes:
            body = _write_2411(model) if node_dataset == 2411 else _write_15(model)
            _ds(lines, node_dataset, body)
        if model.elements:
            _ds(lines, 2412, _write_2412(model))
        if model.materials or model.properties:
            _ds(lines, _FEMTOOLS_CARDS_DS, _write_cards_30000(model))

    for tl in tracelines or ():
        _ds(lines, 82, _write_82(tl))

    if modal is not None:
        dof_index = modal.dof_index
        if dof_index is None:
            if model is None:
                raise ValueError("write_unv: modal has no dof_index and no model was given")
            dof_index = model.dof_index()
            if len(dof_index) != modal.n_dof:
                raise ValueError(
                    f"write_unv: modal has {modal.n_dof} rows but the model has "
                    f"{len(dof_index)} DOFs and modal.dof_index is missing"
                )
        row_of = {pair: i for i, pair in enumerate(dof_index)}
        node_ids = sorted({nid for nid, _ in dof_index})
        ndv = 6 if any(d >= 3 for _, d in dof_index) else 3
        for j in range(modal.n_modes):
            _ds(lines, 55, _write_55_mode(modal, j, node_ids, row_of, ndv, name))

    if frf is not None:
        func_id = 1
        for i_out in range(frf.n_out):
            for i_in in range(frf.n_in):
                _ds(lines, 58, _write_58_frf(frf, i_out, i_in, func_id))
                func_id += 1

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

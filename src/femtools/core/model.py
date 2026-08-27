"""In-memory relational FE/test database.

DOF convention (used consistently across femtools)
---------------------------------------------------
Every 3-D node carries 6 DOFs.  Local DOFs are **0-based**:

===  =====  =========================
dof  label  meaning
===  =====  =========================
0    UX     translation along global X
1    UY     translation along global Y
2    UZ     translation along global Z
3    RX     rotation about global X
4    RY     rotation about global Y
5    RZ     rotation about global Z
===  =====  =========================

Global equation numbering: nodes sorted ascending by id, 6 consecutive DOFs
per node (see :meth:`FEModel.dof_map`).  Translators (Nastran components
``1..6``, UNV directions) convert at the boundary via
:func:`comps_to_mask` / :func:`mask_to_comps`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coords import CoordSys
from .units import UnitSystem

if TYPE_CHECKING:
    from .sets import ElementSet, NodeSet

__all__ = [
    "NDOF_PER_NODE",
    "DOF_LABELS",
    "ELEMENT_NODE_COUNTS",
    "ELEMENT_TYPES",
    "PROPERTY_TYPES",
    "MATERIAL_TYPES",
    "ModelError",
    "Node",
    "Element",
    "Material",
    "Property",
    "SPC",
    "Load",
    "DOFSet",
    "FEModel",
    "comps_to_mask",
    "mask_to_comps",
]

NDOF_PER_NODE: int = 6
DOF_LABELS: tuple[str, ...] = ("UX", "UY", "UZ", "RX", "RY", "RZ")

#: Allowed node counts per element type (Round 1 element catalogue).
ELEMENT_NODE_COUNTS: dict[str, tuple[int, ...]] = {
    "BAR2": (2,),  # 2-node axial bar (rod)
    "BEAM2": (2,),  # 2-node Euler/Timoshenko beam
    "TRUSS2D": (2,),  # 2-node planar truss (x-y plane)
    "QUAD4": (4,),  # 4-node shell
    "TRIA3": (3,),  # 3-node shell
    "HEX8": (8,),  # 8-node solid brick
    "TET4": (4,),  # 4-node solid tetrahedron
    "MASS": (1,),  # lumped mass at a node
    "SPRING": (1, 2),  # node-to-node (2) or grounded (1) spring
    "DAMPER": (1, 2),  # node-to-node (2) or grounded (1) viscous damper
}

ELEMENT_TYPES: tuple[str, ...] = tuple(ELEMENT_NODE_COUNTS)

PROPERTY_TYPES: tuple[str, ...] = ("bar", "beam", "shell", "solid", "lumped")
MATERIAL_TYPES: tuple[str, ...] = ("isotropic", "orthotropic")

#: Property types that require a material reference.
_PROPERTY_NEEDS_MATERIAL: frozenset[str] = frozenset({"bar", "beam", "shell", "solid"})

#: Element types that require a property reference.
_ELEMENT_NEEDS_PROPERTY: frozenset[str] = frozenset(
    {"BAR2", "BEAM2", "TRUSS2D", "QUAD4", "TRIA3", "HEX8", "TET4"}
)


class ModelError(ValueError):
    """Raised for invalid model construction (duplicate ids, bad references, ...)."""


def comps_to_mask(comps: int | str) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Nastran-style component string/int (e.g. ``123456``, ``"35"``) -> 6-bool mask."""
    s = str(comps).strip()
    if s in ("", "0"):
        return (False,) * 6
    mask = [False] * NDOF_PER_NODE
    for ch in s:
        if ch not in "123456":
            raise ModelError(f"invalid DOF component {ch!r} in {comps!r}; expected digits 1-6")
        mask[int(ch) - 1] = True
    return tuple(mask)  # type: ignore[return-value]


def mask_to_comps(mask: Sequence[bool]) -> str:
    """6-bool mask -> Nastran-style component string (``""`` when empty)."""
    if len(mask) != NDOF_PER_NODE:
        raise ModelError(f"mask must have 6 entries, got {len(mask)}")
    return "".join(str(i + 1) for i, m in enumerate(mask) if m)


def _mask6(mask: Sequence[bool]) -> tuple[bool, bool, bool, bool, bool, bool]:
    if len(mask) != NDOF_PER_NODE:
        raise ModelError(f"DOF mask must have 6 entries, got {len(mask)}")
    return tuple(bool(m) for m in mask)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A 3-D node with 6 DOFs.

    ``xyz`` is always stored in **global cartesian** coordinates (in the
    model's unit system).  ``cp`` records the id of the coordinate system the
    position was defined in (0 = global); ``cd`` is the output/displacement
    coordinate system id (kept as metadata, Round 1 solvers work globally).
    """

    id: int
    xyz: NDArray[np.float64]
    cp: int = 0
    cd: int = 0

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)


@dataclass
class Element:
    """A finite element.

    Attributes
    ----------
    id, type, nodes, property_id:
        Core connectivity.  ``type`` is one of :data:`ELEMENT_TYPES`.
    orientation:
        Optional ``(3,)`` beam orientation vector (defines the local x-y
        plane, like Nastran CBAR ``X1 X2 X3``).
    dofs:
        For SPRING/DAMPER: the connected local DOF (0-based) at each node,
        e.g. ``(0, 0)`` couples UX of node A to UX of node B.
    """

    id: int
    type: str
    nodes: tuple[int, ...]
    property_id: int | None = None
    orientation: NDArray[np.float64] | None = None
    dofs: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.type = str(self.type).upper()
        if self.type not in ELEMENT_NODE_COUNTS:
            raise ModelError(
                f"unknown element type {self.type!r}; expected one of {ELEMENT_TYPES}"
            )
        self.nodes = tuple(int(n) for n in self.nodes)
        allowed = ELEMENT_NODE_COUNTS[self.type]
        if len(self.nodes) not in allowed:
            raise ModelError(
                f"element {self.id} ({self.type}) has {len(self.nodes)} nodes; "
                f"expected {' or '.join(map(str, allowed))}"
            )
        if self.property_id is not None:
            self.property_id = int(self.property_id)
        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=float).reshape(3)
        if self.dofs is not None:
            self.dofs = tuple(int(d) for d in self.dofs)
            if len(self.dofs) != len(self.nodes):
                raise ModelError(
                    f"element {self.id}: dofs must give one local DOF per node "
                    f"({len(self.nodes)}), got {len(self.dofs)}"
                )
            for d in self.dofs:
                if not 0 <= d < NDOF_PER_NODE:
                    raise ModelError(f"element {self.id}: local dof {d} outside 0..5")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


@dataclass
class Material:
    """Material definition.

    Isotropic: ``E``, ``nu``, ``rho`` required; ``G`` derived as
    ``E / (2 (1 + nu))`` when omitted.  Orthotropic fields (``E1..E3``,
    ``nu12/nu23/nu13``, ``G12/G23/G13``) are stored for shell/solid
    formulations; Round-1 solvers use the isotropic subset.
    """

    id: int
    type: Literal["isotropic", "orthotropic"] = "isotropic"
    # isotropic
    E: float | None = None
    nu: float | None = None
    rho: float | None = None
    G: float | None = None
    alpha: float | None = None  # thermal expansion
    damping: float | None = None  # structural damping coefficient (GE)
    # orthotropic
    E1: float | None = None
    E2: float | None = None
    E3: float | None = None
    nu12: float | None = None
    nu23: float | None = None
    nu13: float | None = None
    G12: float | None = None
    G23: float | None = None
    G13: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.id = int(self.id)
        if self.type not in MATERIAL_TYPES:
            raise ModelError(
                f"unknown material type {self.type!r}; expected one of {MATERIAL_TYPES}"
            )
        if self.type == "isotropic":
            if self.E is None and self.G is None:
                raise ModelError(f"isotropic material {self.id}: E (or G) is required")
            if self.E is not None and self.E <= 0.0:
                raise ModelError(f"material {self.id}: E must be > 0, got {self.E}")
            if self.nu is not None and not -1.0 < self.nu < 0.5:
                raise ModelError(f"material {self.id}: nu must be in (-1, 0.5), got {self.nu}")
            if self.rho is not None and self.rho < 0.0:
                raise ModelError(f"material {self.id}: rho must be >= 0, got {self.rho}")
            if self.G is None and self.E is not None and self.nu is not None:
                self.G = self.E / (2.0 * (1.0 + self.nu))
        else:
            if self.E1 is None or self.E2 is None:
                raise ModelError(f"orthotropic material {self.id}: E1 and E2 are required")


#: Fields required per property type.
_PROPERTY_REQUIRED: dict[str, tuple[str, ...]] = {
    "bar": ("A",),
    "beam": ("A", "Iy", "Iz", "J"),
    "shell": ("t",),
    "solid": (),
    "lumped": (),
}


@dataclass
class Property:
    """Element property (section) definition.

    ========  ==========================================================
    type      fields
    ========  ==========================================================
    bar       ``A`` (axial area)
    beam      ``A``, ``Iy``, ``Iz``, ``J`` (+ optional ``kappa`` shear
              correction factor, 0 = no shear flexibility)
    shell     ``t`` (thickness)
    solid     material only
    lumped    ``m`` (mass), ``k`` (stiffness), ``c`` (viscous damping) --
              at least one; used by MASS/SPRING/DAMPER elements
    ========  ==========================================================
    """

    id: int
    type: str
    material_id: int | None = None
    A: float | None = None
    Iy: float | None = None
    Iz: float | None = None
    J: float | None = None
    kappa: float | None = None
    t: float | None = None
    m: float | None = None
    k: float | None = None
    c: float | None = None
    nsm: float | None = None  # non-structural mass (per length / per area)
    name: str = ""

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.type = str(self.type).lower()
        if self.type not in PROPERTY_TYPES:
            raise ModelError(
                f"unknown property type {self.type!r}; expected one of {PROPERTY_TYPES}"
            )
        missing = [f for f in _PROPERTY_REQUIRED[self.type] if getattr(self, f) is None]
        if missing:
            raise ModelError(
                f"property {self.id} ({self.type}): missing required field(s) {missing}"
            )
        if self.type == "lumped" and self.m is None and self.k is None and self.c is None:
            raise ModelError(
                f"property {self.id} (lumped): at least one of m, k, c is required"
            )
        if self.type in _PROPERTY_NEEDS_MATERIAL and self.material_id is None:
            raise ModelError(f"property {self.id} ({self.type}): material_id is required")
        for fname in ("A", "Iy", "Iz", "J", "t", "m"):
            v = getattr(self, fname)
            if v is not None and v < 0.0:
                raise ModelError(f"property {self.id}: {fname} must be >= 0, got {v}")


@dataclass
class SPC:
    """Single-point constraint: fixed (or enforced) DOFs at one node.

    ``mask[i] is True`` -> local DOF ``i`` is constrained to ``value``
    (default 0.0).
    """

    node_id: int
    mask: tuple[bool, bool, bool, bool, bool, bool]
    value: float = 0.0
    sid: int = 1  # constraint set id (Nastran SID)

    def __post_init__(self) -> None:
        self.node_id = int(self.node_id)
        self.mask = _mask6(self.mask)
        self.value = float(self.value)
        self.sid = int(self.sid)

    @property
    def comps(self) -> str:
        """Nastran-style component string, e.g. ``"123456"``."""
        return mask_to_comps(self.mask)


@dataclass
class Load:
    """Nodal load (force and/or moment vector) in global coordinates."""

    sid: int
    node_id: int
    force: NDArray[np.float64] | None = None  # (3,) force vector
    moment: NDArray[np.float64] | None = None  # (3,) moment vector

    def __post_init__(self) -> None:
        self.sid = int(self.sid)
        self.node_id = int(self.node_id)
        if self.force is not None:
            self.force = np.asarray(self.force, dtype=float).reshape(3)
        if self.moment is not None:
            self.moment = np.asarray(self.moment, dtype=float).reshape(3)
        if self.force is None and self.moment is None:
            raise ModelError(f"load at node {self.node_id}: force or moment required")

    def as_dof_values(self) -> Iterator[tuple[int, float]]:
        """Yield ``(local_dof, value)`` pairs for the non-zero components."""
        if self.force is not None:
            for i in range(3):
                if self.force[i] != 0.0:
                    yield i, float(self.force[i])
        if self.moment is not None:
            for i in range(3):
                if self.moment[i] != 0.0:
                    yield 3 + i, float(self.moment[i])


class DOFSet:
    """A named set of ``(node_id, dof)`` pairs (dof 0-based, 0..5).

    Used for sensor/target/master DOF selections.  Supports boolean algebra
    like :class:`~femtools.core.sets.NodeSet`.
    """

    __slots__ = ("name", "entries")

    def __init__(self, name: str = "", entries: Iterable[Sequence[int]] = ()) -> None:
        self.name = name
        self.entries: frozenset[tuple[int, int]] = frozenset(
            (int(p[0]), int(p[1])) for p in entries
        )
        for _, dof in self.entries:
            if not 0 <= dof < NDOF_PER_NODE:
                raise ModelError(f"DOFSet {name!r}: local dof {dof} outside 0..5")

    @classmethod
    def from_nodes(cls, name: str, node_ids: Iterable[int], dofs: Iterable[int] = (0, 1, 2)) -> DOFSet:
        """Cartesian product of node ids and local dofs (default translations)."""
        dofs_t = tuple(int(d) for d in dofs)
        return cls(name, [(n, d) for n in node_ids for d in dofs_t])

    def __contains__(self, pair: Sequence[int]) -> bool:
        return (int(pair[0]), int(pair[1])) in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(sorted(self.entries))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DOFSet):
            return NotImplemented
        return self.entries == other.entries

    def __hash__(self) -> int:
        return hash(self.entries)

    def __repr__(self) -> str:
        return f"DOFSet(name={self.name!r}, n={len(self.entries)})"

    def union(self, other: DOFSet, name: str | None = None) -> DOFSet:
        return DOFSet(name or f"({self.name}|{other.name})", self.entries | other.entries)

    def intersect(self, other: DOFSet, name: str | None = None) -> DOFSet:
        return DOFSet(name or f"({self.name}&{other.name})", self.entries & other.entries)

    def difference(self, other: DOFSet, name: str | None = None) -> DOFSet:
        return DOFSet(name or f"({self.name}-{other.name})", self.entries - other.entries)

    __or__ = union
    __and__ = intersect
    __sub__ = difference

    def rows(self, dof_map: dict[tuple[int, int], int]) -> NDArray[np.intp]:
        """Sorted global equation numbers of this set under ``dof_map``."""
        return np.asarray(sorted(dof_map[p] for p in self.entries), dtype=np.intp)


# ---------------------------------------------------------------------------
# the database
# ---------------------------------------------------------------------------


@dataclass
class FEModel:
    """In-memory FE/test database (see module docstring for DOF conventions)."""

    name: str = "model"
    nodes: dict[int, Node] = field(default_factory=dict)
    elements: dict[int, Element] = field(default_factory=dict)
    materials: dict[int, Material] = field(default_factory=dict)
    properties: dict[int, Property] = field(default_factory=dict)
    spcs: list[SPC] = field(default_factory=list)
    sets: dict[str, NodeSet | ElementSet] = field(default_factory=dict)
    coord_systems: dict[int, CoordSys] = field(default_factory=dict)
    units: UnitSystem = field(default_factory=UnitSystem)
    loads: list[Load] = field(default_factory=list)

    # -- builders ----------------------------------------------------------
    def add_node(self, id: int, xyz: ArrayLike, cp: int = 0, cd: int = 0) -> Node:
        """Add a node.  ``xyz`` is given in coordinate system ``cp``
        (0 = global) and stored in global cartesian coordinates."""
        id = int(id)
        if id in self.nodes:
            raise ModelError(f"duplicate node id {id}")
        xyz_arr = np.asarray(xyz, dtype=float).reshape(3)
        if cp != 0:
            cs = self.coord_systems.get(int(cp))
            if cs is None:
                raise ModelError(f"node {id}: coordinate system cp={cp} is not defined")
            xyz_arr = cs.to_global(xyz_arr)
        node = Node(id=id, xyz=xyz_arr, cp=int(cp), cd=int(cd))
        self.nodes[id] = node
        return node

    def add_element(
        self,
        id: int,
        type: str,
        nodes: Sequence[int],
        property_id: int | None = None,
        orientation: ArrayLike | None = None,
        dofs: Sequence[int] | None = None,
        check_refs: bool = True,
    ) -> Element:
        """Add an element.  With ``check_refs`` (default) the referenced nodes
        and property must already exist."""
        id = int(id)
        if id in self.elements:
            raise ModelError(f"duplicate element id {id}")
        el = Element(
            id=id,
            type=type,
            nodes=tuple(nodes),
            property_id=property_id,
            orientation=None if orientation is None else np.asarray(orientation, dtype=float),
            dofs=None if dofs is None else tuple(dofs),
        )
        if check_refs:
            missing = [n for n in el.nodes if n not in self.nodes]
            if missing:
                raise ModelError(f"element {id}: undefined node(s) {missing}")
            if el.property_id is not None and el.property_id not in self.properties:
                raise ModelError(f"element {id}: undefined property {el.property_id}")
            if el.property_id is None and el.type in _ELEMENT_NEEDS_PROPERTY:
                raise ModelError(f"element {id} ({el.type}): property_id is required")
        self.elements[id] = el
        return el

    def add_material(self, id: int, type: str = "isotropic", **fields: float | str) -> Material:
        """Add a material (keyword fields per :class:`Material`)."""
        id = int(id)
        if id in self.materials:
            raise ModelError(f"duplicate material id {id}")
        mat = Material(id=id, type=type, **fields)  # type: ignore[arg-type]
        self.materials[id] = mat
        return mat

    def add_property(
        self,
        id: int,
        type: str,
        material_id: int | None = None,
        check_refs: bool = True,
        **fields: float | str,
    ) -> Property:
        """Add a property (keyword fields per :class:`Property`)."""
        id = int(id)
        if id in self.properties:
            raise ModelError(f"duplicate property id {id}")
        prop = Property(id=id, type=type, material_id=material_id, **fields)  # type: ignore[arg-type]
        if check_refs and prop.material_id is not None and prop.material_id not in self.materials:
            raise ModelError(f"property {id}: undefined material {prop.material_id}")
        self.properties[id] = prop
        return prop

    def add_spc(
        self,
        node_id: int,
        mask: Sequence[bool],
        value: float = 0.0,
        sid: int = 1,
        check_refs: bool = True,
    ) -> SPC:
        """Constrain DOFs of a node.  ``mask`` has 6 booleans (UX..RZ)."""
        spc = SPC(node_id=node_id, mask=_mask6(mask), value=value, sid=sid)
        if check_refs and spc.node_id not in self.nodes:
            raise ModelError(f"SPC references undefined node {spc.node_id}")
        self.spcs.append(spc)
        return spc

    def add_load(
        self,
        node_id: int,
        force: ArrayLike | None = None,
        moment: ArrayLike | None = None,
        sid: int = 1,
        check_refs: bool = True,
    ) -> Load:
        """Apply a nodal force/moment (global components) in load set ``sid``."""
        load = Load(sid=sid, node_id=node_id, force=force, moment=moment)  # type: ignore[arg-type]
        if check_refs and load.node_id not in self.nodes:
            raise ModelError(f"load references undefined node {load.node_id}")
        self.loads.append(load)
        return load

    def add_coord_system(self, cs: CoordSys) -> CoordSys:
        if cs.id == 0:
            raise ModelError("coordinate system id 0 is reserved for the global system")
        if cs.id in self.coord_systems:
            raise ModelError(f"duplicate coordinate system id {cs.id}")
        self.coord_systems[cs.id] = cs
        return cs

    def add_set(self, s: NodeSet | ElementSet) -> NodeSet | ElementSet:
        if not s.name:
            raise ModelError("sets stored on the model must be named")
        if s.name in self.sets:
            raise ModelError(f"duplicate set name {s.name!r}")
        self.sets[s.name] = s
        return s

    # -- DOF bookkeeping ---------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def ndof(self) -> int:
        """Total number of global DOFs (6 per node)."""
        return NDOF_PER_NODE * len(self.nodes)

    def node_ids(self) -> list[int]:
        """Node ids in canonical (ascending) order."""
        return sorted(self.nodes)

    def node_index(self) -> dict[int, int]:
        """node id -> position in the canonical node ordering."""
        return {nid: i for i, nid in enumerate(self.node_ids())}

    def dof_map(self) -> dict[tuple[int, int], int]:
        """``(node_id, local_dof)`` -> global equation number.

        Nodes ascending by id, 6 consecutive DOFs per node:
        ``eq = 6 * node_position + local_dof``.
        """
        out: dict[tuple[int, int], int] = {}
        for i, nid in enumerate(self.node_ids()):
            base = NDOF_PER_NODE * i
            for d in range(NDOF_PER_NODE):
                out[(nid, d)] = base + d
        return out

    def dof_index(self) -> tuple[tuple[int, int], ...]:
        """Row labels matching :meth:`dof_map`: ``dof_index()[eq] == (node, dof)``."""
        return tuple(
            (nid, d) for nid in self.node_ids() for d in range(NDOF_PER_NODE)
        )

    def active_dof_mask(self, sid: int | None = None) -> NDArray[np.bool_]:
        """Boolean ``(ndof,)`` mask, ``True`` for free (unconstrained) DOFs.

        ``sid`` restricts to one SPC set; default applies all SPC entries.
        """
        mask = np.ones(self.ndof, dtype=bool)
        dof_map = self.dof_map()
        for spc in self.spcs:
            if sid is not None and spc.sid != sid:
                continue
            if spc.node_id not in self.nodes:
                raise ModelError(f"SPC references undefined node {spc.node_id}")
            for d, constrained in enumerate(spc.mask):
                if constrained:
                    mask[dof_map[(spc.node_id, d)]] = False
        return mask

    # -- geometry helpers ----------------------------------------------------
    def xyz_array(self) -> NDArray[np.float64]:
        """``(n_nodes, 3)`` coordinates in canonical node order."""
        ids = self.node_ids()
        out = np.empty((len(ids), 3), dtype=float)
        for i, nid in enumerate(ids):
            out[i] = self.nodes[nid].xyz
        return out

    def element_xyz(self, element_id: int) -> NDArray[np.float64]:
        """``(n_elem_nodes, 3)`` coordinates of an element's nodes."""
        el = self.elements[element_id]
        return np.array([self.nodes[n].xyz for n in el.nodes], dtype=float)

    def element_property(self, element_id: int) -> Property | None:
        el = self.elements[element_id]
        return None if el.property_id is None else self.properties.get(el.property_id)

    def element_material(self, element_id: int) -> Material | None:
        prop = self.element_property(element_id)
        if prop is None or prop.material_id is None:
            return None
        return self.materials.get(prop.material_id)

    def bounding_box(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(min_xyz, max_xyz) over all nodes."""
        if not self.nodes:
            raise ModelError("model has no nodes")
        xyz = self.xyz_array()
        return xyz.min(axis=0), xyz.max(axis=0)

    def __repr__(self) -> str:
        return (
            f"FEModel(name={self.name!r}, nodes={len(self.nodes)}, "
            f"elements={len(self.elements)}, materials={len(self.materials)}, "
            f"properties={len(self.properties)}, spcs={len(self.spcs)}, "
            f"loads={len(self.loads)})"
        )

"""Element context, result containers and the element type registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..materials import MaterialData, material_from_record
from ..protocols import (
    DOF_LABELS,
    as_float,
    get_any,
    iter_records,
    node_xyz,
    normalize_dof_list,
)

__all__ = [
    "DOF_LABELS",
    "ElementContext",
    "ElementMatrices",
    "ElementSpec",
    "ModelIndex",
    "REGISTRY",
    "available_elements",
    "element_spec",
    "register",
]


@dataclass
class ElementMatrices:
    """Element contribution expressed on an explicit list of global DOFs.

    ``dofs`` holds ``(node_id, component)`` pairs, one per row/column of the
    matrices.  Elements therefore never need to know the global DOF numbering.
    """

    dofs: list[tuple[Any, int]]
    k: np.ndarray | None = None
    m: np.ndarray | None = None
    c: np.ndarray | None = None
    #: Drilling-penalty part of ``k``; tracked separately so the assembler can
    #: recognise DOFs whose only stiffness is fictitious.
    k_drill: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.dofs)
        for name in ("k", "m", "c", "k_drill"):
            mat = getattr(self, name)
            if mat is None:
                continue
            arr = np.asarray(mat, dtype=float)
            if arr.shape != (n, n):
                raise ValueError(
                    f"element matrix '{name}' has shape {arr.shape}, expected {(n, n)}"
                )
            setattr(self, name, arr)


@dataclass
class ElementContext:
    """Everything an element builder needs, resolved from the model database."""

    element: Any
    model: Any
    element_id: Any
    etype: str
    node_ids: tuple[Any, ...]
    coords: np.ndarray
    prop: Any = None
    mat: MaterialData = field(default_factory=MaterialData)
    raw_material: Any = None
    lumped_mass: bool = False
    drill_factor: float = 1.0e-3
    options: dict[str, Any] = field(default_factory=dict)

    def value(self, names: Sequence[str] | str, default: Any = None) -> Any:
        """Look a quantity up on the element first, then on the property.

        Element level overrides mirror how most solvers let a connectivity card
        shadow its property card (thickness on ``CTRIA3``, offsets on ``CBAR``).
        The material record is deliberately *not* searched: ``MAT1`` reuses
        field names such as ``A`` (thermal expansion) that would collide with
        section properties.
        """
        found = get_any(self.element, names, None)
        if found is None:
            found = get_any(self.prop, names, None)
        return default if found is None else found

    def prop_value(self, names: Sequence[str] | str, default: Any = None) -> Any:
        """Look a quantity up on the property only."""
        found = get_any(self.prop, names, None)
        return default if found is None else found

    def number(self, names: Sequence[str] | str, default: float | None = None) -> float | None:
        return as_float(self.value(names, None), default)

    def require(self, names: Sequence[str] | str, what: str) -> float:
        value = self.number(names, None)
        if value is None:
            raise ValueError(
                f"element {self.element_id} ({self.etype}): missing {what} "
                f"(looked for {tuple(names) if not isinstance(names, str) else (names,)})"
            )
        return float(value)

    def dof_spec(self, names: Sequence[str] | str, default: Any = None) -> list[int]:
        raw = self.value(names, None)
        if raw is None:
            return normalize_dof_list(default) if default is not None else []
        return normalize_dof_list(raw)

    def length(self) -> float:
        return float(np.linalg.norm(self.coords[1] - self.coords[0]))


#: Builder signature: ``(ctx) -> ElementMatrices``.
Builder = Callable[[ElementContext], ElementMatrices]


@dataclass(frozen=True)
class ElementSpec:
    """Registry entry describing one element type."""

    name: str
    n_nodes: tuple[int, ...]
    dofs_per_node: tuple[int, ...]
    family: str
    description: str
    builder: Builder

    def accepts(self, n: int) -> bool:
        return n in self.n_nodes


REGISTRY: dict[str, ElementSpec] = {}


def register(
    name: str,
    *,
    n_nodes: int | Sequence[int],
    dofs_per_node: Sequence[int],
    family: str,
    description: str,
    aliases: Sequence[str] = (),
) -> Callable[[Builder], Builder]:
    """Decorator registering an element builder under *name* (and aliases)."""

    counts = (int(n_nodes),) if isinstance(n_nodes, int) else tuple(int(v) for v in n_nodes)

    def decorate(builder: Builder) -> Builder:
        spec = ElementSpec(
            name=name.upper(),
            n_nodes=counts,
            dofs_per_node=tuple(int(v) for v in dofs_per_node),
            family=family,
            description=description,
            builder=builder,
        )
        REGISTRY[spec.name] = spec
        for alias in aliases:
            REGISTRY.setdefault(alias.upper(), spec)
        return builder

    return decorate


def element_spec(etype: str) -> ElementSpec:
    """Look up a registered element type, raising a helpful error otherwise."""
    key = str(etype).strip().upper()
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(
        f"unknown element type {etype!r}; available: {', '.join(available_elements())}"
    )


def available_elements(*, canonical_only: bool = True) -> list[str]:
    """Return the sorted list of supported element type names.

    With ``canonical_only`` (the default) aliases such as ``ROD`` or ``CQUAD4``
    are omitted and only the contract names are reported.
    """
    if canonical_only:
        names = {spec.name for spec in REGISTRY.values()}
    else:
        names = set(REGISTRY)
    return sorted(names)


@dataclass
class ModelIndex:
    """Cached node / property / material tables for one model.

    Rebuilding these dictionaries per element turns assembly into an
    ``O(n_elements * n_nodes)`` operation, so the assembler builds the index
    once and hands it to every element.
    """

    nodes: dict[Any, Any] = field(default_factory=dict)
    properties: dict[Any, Any] = field(default_factory=dict)
    materials: dict[Any, Any] = field(default_factory=dict)
    coords: dict[Any, np.ndarray] = field(default_factory=dict)
    _mat_cache: dict[int, MaterialData] = field(default_factory=dict, repr=False)

    def material_data(self, record: Any) -> MaterialData:
        key = id(record)
        cached = self._mat_cache.get(key)
        if cached is None:
            cached = material_from_record(record)
            self._mat_cache[key] = cached
        return cached

    @classmethod
    def build(cls, model: Any) -> ModelIndex:
        nodes = dict(iter_records(get_any(model, ("nodes", "grids", "points"), None)))
        return cls(
            nodes=nodes,
            properties=dict(
                iter_records(get_any(model, ("properties", "props", "property"), None))
            ),
            materials=dict(iter_records(get_any(model, ("materials", "mats", "material"), None))),
            coords={nid: node_xyz(node) for nid, node in nodes.items()},
        )

    def xyz(self, node_id: Any) -> np.ndarray:
        try:
            return self.coords[node_id]
        except KeyError:
            pass
        return self.coords[int(node_id)]


def resolve_material(
    model: Any, prop: Any, element: Any, index: ModelIndex | None = None
) -> tuple[MaterialData, Any]:
    """Find the material record referenced by an element or its property."""
    if index is not None:
        table = index.materials
    else:
        table = dict(iter_records(get_any(model, ("materials", "mats", "material"), None)))
    mid = get_any(element, ("material_id", "mid", "material"), None)
    if mid is None:
        mid = get_any(prop, ("material_id", "mid", "material", "mat_id"), None)
    record = None
    if mid is not None:
        if mid in table:
            record = table[mid]
        elif not isinstance(mid, (int, str)):
            record = mid  # already a material object
        else:
            try:
                record = table[int(mid)]
            except (KeyError, TypeError, ValueError):
                record = None
    if record is None and mid is not None and not isinstance(mid, (int, np.integer, str)):
        record = mid
    if record is None and len(table) == 1 and mid is None:
        # Single-material models are common in tests; use it implicitly.
        record = next(iter(table.values()))
    if index is not None:
        return index.material_data(record), record
    return material_from_record(record), record


def build_context(
    model: Any,
    element_id: Any,
    element: Any,
    *,
    lumped_mass: bool = False,
    drill_factor: float = 1.0e-3,
    options: dict[str, Any] | None = None,
    index: ModelIndex | None = None,
) -> ElementContext:
    """Resolve nodes, property and material for one element."""
    if index is None:
        index = ModelIndex.build(model)
    etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).strip().upper()
    raw_nodes = get_any(element, ("nodes", "node_ids", "connectivity", "conn", "grids"), None)
    if raw_nodes is None:
        raise ValueError(f"element {element_id}: no connectivity found")
    if isinstance(raw_nodes, (int, np.integer, str)):
        raw_nodes = (raw_nodes,)

    nodes = index.nodes
    # A blank/None/zero connection slot is the classic "grounded" marker for
    # scalar elements; drop it rather than failing the lookup.
    node_ids = tuple(
        nid
        for nid in raw_nodes
        if nid is not None and not (isinstance(nid, (int, np.integer)) and int(nid) == 0
                                    and nid not in nodes and 0 not in nodes)
    )
    if not node_ids:
        raise ValueError(f"element {element_id}: connectivity is empty")

    coords = np.empty((len(node_ids), 3), dtype=float)
    for i, nid in enumerate(node_ids):
        try:
            coords[i] = index.xyz(nid)
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"element {element_id}: node {nid!r} not in model") from exc

    properties = index.properties
    pid = get_any(element, ("property_id", "pid", "property"), None)
    prop = None
    if pid is not None:
        prop = properties.get(pid)
        if prop is None:
            try:
                prop = properties[int(pid)]
            except (KeyError, TypeError, ValueError):
                prop = pid if not isinstance(pid, (int, np.integer, str)) else None
    if prop is None and len(properties) == 1 and pid is None:
        prop = next(iter(properties.values()))

    mat, raw_mat = resolve_material(model, prop, element, index)
    return ElementContext(
        element=element,
        model=model,
        element_id=element_id,
        etype=etype,
        node_ids=node_ids,
        coords=coords,
        prop=prop,
        mat=mat,
        raw_material=raw_mat,
        lumped_mass=lumped_mass,
        drill_factor=drill_factor,
        options=dict(options or {}),
    )

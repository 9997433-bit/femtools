"""Named node / element sets with boolean algebra.

:class:`NodeSet` and :class:`ElementSet` are thin, typed wrappers around a
set of integer ids.  Boolean operations (union ``|``, intersection ``&``,
difference ``-``) are only allowed between sets of the same kind so a node
set is never silently combined with an element set.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .model import FEModel

__all__ = ["NodeSet", "ElementSet"]

_S = TypeVar("_S", bound="_IdSet")


@dataclass
class _IdSet:
    """Base class: a named set of integer ids."""

    name: str
    ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.ids = frozenset(int(i) for i in self.ids)

    # -- container protocol ----------------------------------------------
    def __contains__(self, id: int) -> bool:
        return int(id) in self.ids

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.sorted_ids())

    def __bool__(self) -> bool:
        return bool(self.ids)

    def sorted_ids(self) -> list[int]:
        return sorted(self.ids)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.sorted_ids(), dtype=np.int64)

    # -- boolean algebra ---------------------------------------------------
    def _check_kind(self, other: _IdSet, op: str) -> None:
        if type(other) is not type(self):
            raise TypeError(
                f"cannot {op} {type(self).__name__} with {type(other).__name__}; "
                "boolean set operations require the same set kind"
            )

    def union(self: _S, other: _S, name: str | None = None) -> _S:
        self._check_kind(other, "union")
        return type(self)(name or f"({self.name}|{other.name})", self.ids | other.ids)

    def intersect(self: _S, other: _S, name: str | None = None) -> _S:
        self._check_kind(other, "intersect")
        return type(self)(name or f"({self.name}&{other.name})", self.ids & other.ids)

    def difference(self: _S, other: _S, name: str | None = None) -> _S:
        self._check_kind(other, "difference")
        return type(self)(name or f"({self.name}-{other.name})", self.ids - other.ids)

    def __or__(self: _S, other: _S) -> _S:
        return self.union(other)

    def __and__(self: _S, other: _S) -> _S:
        return self.intersect(other)

    def __sub__(self: _S, other: _S) -> _S:
        return self.difference(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IdSet):
            return NotImplemented
        return type(self) is type(other) and self.ids == other.ids

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.ids))


class NodeSet(_IdSet):
    """A named set of node ids."""

    @classmethod
    def from_ids(cls, name: str, ids: Iterable[int]) -> NodeSet:
        return cls(name, frozenset(int(i) for i in ids))

    @classmethod
    def all_nodes(cls, model: FEModel, name: str = "all_nodes") -> NodeSet:
        return cls(name, frozenset(model.nodes))

    @classmethod
    def from_bbox(
        cls,
        model: FEModel,
        min_xyz: ArrayLike,
        max_xyz: ArrayLike,
        name: str = "bbox",
        tol: float = 0.0,
    ) -> NodeSet:
        """Nodes with coordinates inside an axis-aligned box (inclusive, +/- tol)."""
        lo = np.asarray(min_xyz, dtype=float).reshape(3) - tol
        hi = np.asarray(max_xyz, dtype=float).reshape(3) + tol
        ids = frozenset(
            nid
            for nid, node in model.nodes.items()
            if bool(np.all(node.xyz >= lo) and np.all(node.xyz <= hi))
        )
        return cls(name, ids)

    @classmethod
    def of_elements(cls, model: FEModel, elements: ElementSet, name: str | None = None) -> NodeSet:
        """All nodes referenced by the elements of ``elements``."""
        ids: set[int] = set()
        for eid in elements.ids:
            el = model.elements.get(eid)
            if el is not None:
                ids.update(el.nodes)
        return cls(name or f"nodes({elements.name})", frozenset(ids))


class ElementSet(_IdSet):
    """A named set of element ids."""

    @classmethod
    def from_ids(cls, name: str, ids: Iterable[int]) -> ElementSet:
        return cls(name, frozenset(int(i) for i in ids))

    @classmethod
    def all_elements(cls, model: FEModel, name: str = "all_elements") -> ElementSet:
        return cls(name, frozenset(model.elements))

    @classmethod
    def by_type(cls, model: FEModel, etype: str, name: str | None = None) -> ElementSet:
        """Elements of a given type (e.g. ``"QUAD4"``)."""
        ids = frozenset(eid for eid, el in model.elements.items() if el.type == etype)
        return cls(name or f"type:{etype}", ids)

    @classmethod
    def by_property(cls, model: FEModel, property_id: int, name: str | None = None) -> ElementSet:
        """Elements referencing a given property id."""
        ids = frozenset(eid for eid, el in model.elements.items() if el.property_id == property_id)
        return cls(name or f"prop:{property_id}", ids)

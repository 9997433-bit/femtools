"""Global degree-of-freedom bookkeeping."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from .protocols import DOF_LABELS, normalize_dof

__all__ = ["DofMap"]


class DofMap:
    """Maps ``(node_id, component)`` to a global equation number.

    The object behaves like a read-only mapping ``node_id -> ndarray`` of the
    node's global DOF numbers, and additionally accepts ``(node_id, component)``
    tuples for a single index::

        dm[7]            # array([36, 37, 38, 39, 40, 41])
        dm[7, "uz"]      # 38
        dm.index(7, 2)   # 38
    """

    __slots__ = ("_node_ids", "_pos", "dofs_per_node")

    def __init__(self, node_ids: Sequence[Any], dofs_per_node: int = 6) -> None:
        self._node_ids: list[Any] = list(node_ids)
        self.dofs_per_node = int(dofs_per_node)
        self._pos: dict[Any, int] = {}
        for i, nid in enumerate(self._node_ids):
            if nid in self._pos:
                raise ValueError(f"duplicate node id {nid!r} in DOF map")
            self._pos[nid] = i

    # -- construction -------------------------------------------------
    @classmethod
    def from_nodes(cls, nodes: dict[Any, Any], dofs_per_node: int = 6) -> DofMap:
        ids = list(nodes)
        try:
            ids = sorted(ids)
        except TypeError:
            pass
        return cls(ids, dofs_per_node)

    # -- basics -------------------------------------------------------
    @property
    def node_ids(self) -> list[Any]:
        return list(self._node_ids)

    @property
    def n_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def n_dof(self) -> int:
        return len(self._node_ids) * self.dofs_per_node

    def __len__(self) -> int:
        return len(self._node_ids)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._node_ids)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, tuple):
            return key[0] in self._pos
        return key in self._pos

    def keys(self):
        return list(self._node_ids)

    def values(self):
        return [self.node_dofs(nid) for nid in self._node_ids]

    def items(self):
        return [(nid, self.node_dofs(nid)) for nid in self._node_ids]

    # -- lookups ------------------------------------------------------
    def position(self, node_id: Any) -> int:
        try:
            return self._pos[node_id]
        except KeyError:
            pass
        try:
            return self._pos[int(node_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"node {node_id!r} is not in the DOF map") from exc

    def index(self, node_id: Any, component: Any = 0) -> int:
        comp = normalize_dof(component, dofs_per_node=self.dofs_per_node)
        return self.position(node_id) * self.dofs_per_node + comp

    def node_dofs(self, node_id: Any) -> np.ndarray:
        start = self.position(node_id) * self.dofs_per_node
        return np.arange(start, start + self.dofs_per_node, dtype=int)

    def __getitem__(self, key: Any):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise KeyError(f"expected (node_id, component), got {key!r}")
            return self.index(key[0], key[1])
        return self.node_dofs(key)

    def __call__(self, node_id: Any, component: Any = 0) -> int:
        return self.index(node_id, component)

    def get(self, key: Any, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default

    def indices(self, pairs) -> np.ndarray:
        """Vectorised lookup of an iterable of ``(node_id, component)`` pairs."""
        return np.fromiter(
            (self.index(nid, comp) for nid, comp in pairs), dtype=int, count=-1
        )

    # -- inverse ------------------------------------------------------
    def dof_node(self, dof: int) -> Any:
        return self._node_ids[int(dof) // self.dofs_per_node]

    def dof_component(self, dof: int) -> int:
        return int(dof) % self.dofs_per_node

    def dof_nodes(self) -> np.ndarray:
        """Node id of every global DOF (object array to allow non-int ids)."""
        return np.repeat(np.array(self._node_ids, dtype=object), self.dofs_per_node)

    def dof_components(self) -> np.ndarray:
        return np.tile(np.arange(self.dofs_per_node), len(self._node_ids))

    def labels(self) -> list[str]:
        names = DOF_LABELS[: self.dofs_per_node]
        return [f"{nid}:{name}" for nid in self._node_ids for name in names]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"DofMap(n_nodes={self.n_nodes}, dofs_per_node={self.dofs_per_node})"

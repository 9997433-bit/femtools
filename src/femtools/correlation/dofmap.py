"""DOF mapping between a test data set and an analysis model.

Correlation is only meaningful once both mode sets are expressed on the same
ordered list of ``(node, component)`` DOFs.  Test data typically covers a
handful of translational channels while the FE model carries every DOF, and
the two orderings never coincide.  :class:`DOFMap` describes such an ordered
DOF list and :func:`align_modes` produces the common subset, including any
sensor sign/orientation factor.

Components use the usual FE convention ``1..6 = X, Y, Z, RX, RY, RZ`` and are
accepted as integers or labels (``"x"``, ``"-Z"``, ``"TX"``, ``"RY"``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DOFMap",
    "parse_component",
    "parse_dof_label",
    "match_dofs",
    "align_modes",
    "restrict",
]

COMPONENT_NAMES: tuple[str, ...] = ("X", "Y", "Z", "RX", "RY", "RZ")

# fmt: off
_ALIASES: dict[str, int] = {
    "X": 1, "TX": 1, "UX": 1, "DX": 1, "T1": 1, "1": 1,
    "Y": 2, "TY": 2, "UY": 2, "DY": 2, "T2": 2, "2": 2,
    "Z": 3, "TZ": 3, "UZ": 3, "DZ": 3, "T3": 3, "3": 3,
    "RX": 4, "ROTX": 4, "R1": 4, "4": 4,
    "RY": 5, "ROTY": 5, "R2": 5, "5": 5,
    "RZ": 6, "ROTZ": 6, "R3": 6, "6": 6,
}
# fmt: on


def parse_component(component: Any) -> tuple[int, float]:
    """Return ``(component_id, sign)`` for an int or label such as ``"-Z"``."""
    if isinstance(component, (int, np.integer)) and not isinstance(component, bool):
        c = int(component)
        sign = -1.0 if c < 0 else 1.0
        c = abs(c)
        if not 1 <= c <= 6:
            raise ValueError(f"component {component!r} out of range 1..6")
        return c, sign
    text = str(component).strip().upper().replace("_", "")
    sign = 1.0
    if text.startswith("-"):
        sign, text = -1.0, text[1:]
    elif text.startswith("+"):
        text = text[1:]
    try:
        return _ALIASES[text], sign
    except KeyError:
        raise ValueError(f"unknown DOF component {component!r}") from None


def parse_dof_label(label: Any) -> tuple[int, int, float]:
    """Parse ``"12Z"``, ``"12:-Z"``, ``"12 RY"`` or ``(12, "z")`` into
    ``(node, component, sign)``."""
    if isinstance(label, (tuple, list, np.ndarray)) and len(label) == 2:
        comp, sign = parse_component(label[1])
        return int(label[0]), comp, sign

    text = str(label).strip()
    cut = 0
    while cut < len(text) and text[cut].isdigit():
        cut += 1
    if cut == 0:
        raise ValueError(f"cannot parse DOF label {label!r}: no node id")
    node = int(text[:cut])
    rest = text[cut:].strip()
    while rest[:1] in (":", ".", "_", "/", ","):
        rest = rest[1:].strip()
    if not rest:
        raise ValueError(f"cannot parse DOF label {label!r}: no component")
    comp, sign = parse_component(rest)
    return node, comp, sign


class DOFMap:
    """Ordered list of ``(node, component)`` DOFs with optional sign factors.

    Parameters
    ----------
    nodes:
        Node (or channel) identifiers, one per DOF.
    components:
        Component per DOF: ints ``1..6``, or labels (``"z"``, ``"-RY"``).
        A negative component or a leading ``-`` records a reversed sensor
        orientation and is stored in :attr:`scale`.
    scale:
        Additional multiplicative factor per DOF (sensor calibration or unit
        conversion).  Combined with the sign from ``components``.
    """

    __slots__ = ("_nodes", "_components", "_scale", "_lookup")

    def __init__(
        self,
        nodes: ArrayLike,
        components: Iterable[Any],
        scale: ArrayLike | None = None,
    ) -> None:
        node_arr = np.asarray(nodes, dtype=np.int64).reshape(-1)
        comps: list[int] = []
        signs: list[float] = []
        for c in components:
            cid, sgn = parse_component(c)
            comps.append(cid)
            signs.append(sgn)
        comp_arr = np.asarray(comps, dtype=np.int8)
        if node_arr.size != comp_arr.size:
            raise ValueError(f"{node_arr.size} nodes but {comp_arr.size} components")
        sign_arr = np.asarray(signs, dtype=float)
        if scale is not None:
            extra = np.asarray(scale, dtype=float).reshape(-1)
            if extra.size == 1:
                extra = np.full(node_arr.size, float(extra[0]))
            if extra.size != node_arr.size:
                raise ValueError(f"scale has {extra.size} entries, expected {node_arr.size}")
            sign_arr = sign_arr * extra

        self._nodes = node_arr
        self._components = comp_arr
        self._scale = sign_arr
        self._lookup: dict[tuple[int, int], int] = {}
        for i, key in enumerate(zip(node_arr.tolist(), comp_arr.tolist(), strict=True)):
            if key in self._lookup:
                raise ValueError(f"duplicate DOF {key} at positions {self._lookup[key]} and {i}")
            self._lookup[key] = i

    # -- construction ----------------------------------------------------
    @classmethod
    def from_keys(cls, keys: Iterable[Any], scale: ArrayLike | None = None) -> DOFMap:
        """Build from ``(node, component)`` tuples or labels like ``"12Z"``."""
        nodes: list[int] = []
        comps: list[int] = []
        signs: list[float] = []
        for key in keys:
            node, comp, sgn = parse_dof_label(key)
            nodes.append(node)
            comps.append(comp)
            signs.append(sgn)
        base = np.asarray(signs, dtype=float)
        if scale is not None:
            base = base * np.asarray(scale, dtype=float).reshape(-1)
        return cls(nodes, comps, base)

    @classmethod
    def from_nodes(cls, nodes: ArrayLike, components: Iterable[Any] = (1, 2, 3)) -> DOFMap:
        """Cartesian product of ``nodes`` with ``components`` (node-major)."""
        node_arr = np.asarray(nodes, dtype=np.int64).reshape(-1)
        comps = list(components)
        rep_nodes = np.repeat(node_arr, len(comps))
        rep_comps = comps * node_arr.size
        return cls(rep_nodes, rep_comps)

    @classmethod
    def from_mapping(cls, mapping: Any) -> DOFMap:
        """Build from an assembly DOF map.

        Accepts a :class:`DOFMap`, a ``{(node, component): index}`` dict, a
        ``{node: [index, ...]}`` / ``{node: {component: index}}`` dict, or any
        sequence of DOF keys.  Dict inputs are ordered by their global index.
        """
        if isinstance(mapping, DOFMap):
            return mapping
        if isinstance(mapping, Mapping):
            items: list[tuple[int, tuple[int, int]]] = []
            for key, value in mapping.items():
                if isinstance(key, tuple):
                    node, comp, _ = parse_dof_label(key)
                    items.append((int(value), (node, comp)))
                    continue
                node = int(key)
                if isinstance(value, Mapping):
                    for comp, idx in value.items():
                        items.append((int(idx), (node, parse_component(comp)[0])))
                elif isinstance(value, (Sequence, np.ndarray)):
                    for offset, idx in enumerate(np.asarray(value).reshape(-1).tolist()):
                        if int(idx) < 0:
                            continue
                        items.append((int(idx), (node, offset + 1)))
                else:
                    items.append((int(value), (node, 1)))
            items.sort(key=lambda kv: kv[0])
            expected = list(range(len(items)))
            if [k for k, _ in items] != expected:
                raise ValueError("DOF indices in the mapping are not a contiguous 0-based range")
            return cls.from_keys([key for _, key in items])
        return cls.from_keys(mapping)

    # -- basics ----------------------------------------------------------
    @property
    def nodes(self) -> NDArray[np.int64]:
        return self._nodes

    @property
    def components(self) -> NDArray[np.int8]:
        return self._components

    @property
    def scale(self) -> NDArray[np.float64]:
        """Per-DOF sign / calibration factor (``+1`` unless specified)."""
        return self._scale

    @property
    def keys(self) -> list[tuple[int, int]]:
        return list(zip(self._nodes.tolist(), self._components.tolist(), strict=True))

    @property
    def labels(self) -> list[str]:
        return [f"{n}{COMPONENT_NAMES[c - 1]}" for n, c in self.keys]

    def __len__(self) -> int:
        return int(self._nodes.size)

    def __iter__(self):
        return iter(self.keys)

    def __getitem__(self, item: int | slice | ArrayLike) -> tuple[int, int] | DOFMap:
        if isinstance(item, (int, np.integer)):
            return (int(self._nodes[item]), int(self._components[item]))
        idx = np.arange(len(self))[item] if isinstance(item, slice) else np.asarray(item)
        return self.take(idx)

    def __contains__(self, key: Any) -> bool:
        try:
            node, comp, _ = parse_dof_label(key)
        except (ValueError, TypeError):
            return False
        return (node, comp) in self._lookup

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DOFMap):
            return NotImplemented
        return (
            np.array_equal(self._nodes, other._nodes)
            and np.array_equal(self._components, other._components)
            and np.array_equal(self._scale, other._scale)
        )

    def __repr__(self) -> str:
        preview = ", ".join(self.labels[:6])
        more = ", ..." if len(self) > 6 else ""
        return f"DOFMap({len(self)} dof: {preview}{more})"

    # -- queries ---------------------------------------------------------
    def index_of(self, key: Any, default: int = -1) -> int:
        """Position of a DOF, or ``default`` when absent."""
        try:
            node, comp, _ = parse_dof_label(key)
        except (ValueError, TypeError):
            return default
        return self._lookup.get((node, comp), default)

    def indices(self, keys: Iterable[Any], *, missing: str = "raise") -> NDArray[np.intp]:
        """Positions of ``keys``.

        ``missing='raise'`` (default) errors on unknown DOFs, ``'skip'`` drops
        them and ``'mark'`` returns ``-1`` for them.
        """
        out: list[int] = []
        for key in keys:
            idx = self.index_of(key)
            if idx < 0:
                if missing == "raise":
                    raise KeyError(f"DOF {key!r} is not in this map")
                if missing == "skip":
                    continue
                if missing != "mark":
                    raise ValueError(f"unknown missing policy {missing!r}")
            out.append(idx)
        return np.asarray(out, dtype=np.intp)

    def take(self, index: ArrayLike) -> DOFMap:
        """Sub-map for the given positions."""
        idx = np.asarray(index, dtype=np.intp).reshape(-1)
        return DOFMap(self._nodes[idx], self._components[idx].tolist(), self._scale[idx])

    def subset(
        self, keys: Iterable[Any], *, missing: str = "raise"
    ) -> tuple[DOFMap, NDArray[np.intp]]:
        """Return ``(sub_map, positions)`` for the requested DOFs."""
        idx = self.indices(keys, missing=missing)
        return self.take(idx), idx

    def select(
        self, components: Iterable[Any] | None = None, nodes: ArrayLike | None = None
    ) -> NDArray[np.intp]:
        """Positions of the DOFs matching the given components and/or nodes."""
        mask = np.ones(len(self), dtype=bool)
        if components is not None:
            wanted = {parse_component(c)[0] for c in components}
            mask &= np.isin(self._components, sorted(wanted))
        if nodes is not None:
            mask &= np.isin(self._nodes, np.asarray(nodes, dtype=np.int64).reshape(-1))
        return np.flatnonzero(mask).astype(np.intp)

    def common(self, other: DOFMap) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Positions of the DOFs present in both maps, in *this* map's order."""
        here: list[int] = []
        there: list[int] = []
        for i, key in enumerate(self.keys):
            j = other._lookup.get(key, -1)
            if j >= 0:
                here.append(i)
                there.append(j)
        return np.asarray(here, dtype=np.intp), np.asarray(there, dtype=np.intp)


def match_dofs(map_a: Any, map_b: Any) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Common DOF positions of two maps, ordered as in ``map_a``.

    Returns ``(index_a, index_b)`` such that row ``index_a[k]`` of the first
    model and row ``index_b[k]`` of the second refer to the same physical DOF.
    """
    a = DOFMap.from_mapping(map_a)
    b = DOFMap.from_mapping(map_b)
    return a.common(b)


def align_modes(
    phi_a: ArrayLike,
    map_a: Any,
    phi_b: ArrayLike,
    map_b: Any,
    *,
    apply_scale: bool = True,
) -> tuple[NDArray[Any], NDArray[Any], DOFMap]:
    """Restrict two mode sets to their common DOFs.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shapes ``(n_dof, n_mode)`` in the ordering of ``map_a``/``map_b``.
    map_a, map_b:
        Anything :meth:`DOFMap.from_mapping` understands.
    apply_scale:
        Multiply each row by the DOF scale factor of its map, so reversed or
        differently calibrated sensors become directly comparable.

    Returns
    -------
    (phi_a_common, phi_b_common, common_map)
        Both matrices share the row ordering of ``common_map``.
    """
    a_map = DOFMap.from_mapping(map_a)
    b_map = DOFMap.from_mapping(map_b)
    a = np.asarray(phi_a)
    b = np.asarray(phi_b)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if a.shape[0] != len(a_map):
        raise ValueError(f"phi_a has {a.shape[0]} rows but map_a has {len(a_map)} DOF")
    if b.shape[0] != len(b_map):
        raise ValueError(f"phi_b has {b.shape[0]} rows but map_b has {len(b_map)} DOF")

    ia, ib = a_map.common(b_map)
    if ia.size == 0:
        raise ValueError("the two DOF maps have no DOF in common")
    a_sel = a[ia, :]
    b_sel = b[ib, :]
    if apply_scale:
        a_sel = a_sel * a_map.scale[ia][:, None]
        b_sel = b_sel * b_map.scale[ib][:, None]
    common = a_map.take(ia)
    return a_sel, b_sel, common


def restrict(matrix: Any, index: ArrayLike, index_col: ArrayLike | None = None) -> Any:
    """Extract the sub-matrix ``matrix[index, index]`` (dense or sparse).

    Used to build a test-DOF mass matrix for a cross-orthogonality check.
    """
    rows = np.asarray(index, dtype=np.intp).reshape(-1)
    cols = rows if index_col is None else np.asarray(index_col, dtype=np.intp).reshape(-1)
    if hasattr(matrix, "tocsr") and not isinstance(matrix, np.ndarray):
        return matrix.tocsr()[rows, :][:, cols]
    dense = np.asarray(matrix)
    if dense.ndim == 1:
        return dense[rows]
    return dense[np.ix_(rows, cols)]

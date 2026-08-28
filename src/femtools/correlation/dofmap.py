"""DOF mapping between a test data set and an analysis model.

Correlation is only meaningful once both mode sets are expressed on the same
ordered list of ``(node, component)`` DOFs.  Test data typically covers a
handful of translational channels while the FE model carries every DOF, and
the two orderings never coincide.  :class:`DOFMap` describes such an ordered
DOF list and :func:`align_modes` produces the common subset, including any
sensor sign/orientation factor.

Components use the usual FE convention ``1..6 = X, Y, Z, RX, RY, RZ`` and are
accepted as integers or labels (``"x"``, ``"-Z"``, ``"TX"``, ``"RY"``).  The
FEA kernel numbers the same components from zero (``femtools.fea.protocols``);
:meth:`DOFMap.from_mapping` recognizes both conventions, so the DOF map of a
solved model — ``ModalResult.dof_map``, ``AssemblyResult.dof_map``,
``FEModel.dof_map()`` — can be handed over directly::

    modal = solve_modes(model, n_modes=12)
    dofs = DOFMap.from_mapping(modal)          # 6 * n_node (node, component)
    phi_t = modal.modes[dofs.translational()]  # measurable rows only

Which node of the model a test channel belongs to is a *geometric* question,
answered before any of the above: :func:`map_nearest_nodes` matches a digitized
test point cloud against the mesh and returns the FE node id nearest to each
measurement point together with its distance.  :func:`mapped_mode_matrix` then
does the bookkeeping those ids exist for — it pulls the FE mode rows of the
matched nodes, in the order of the measurement points, so the analysis shapes
land on the test grid and can go straight into :func:`~femtools.correlation.mac_matrix`.
:func:`mapped_mac` is those three steps in one call, keeping the node match
alongside the MAC it produced::

    result = mapped_mac(phi_test, test_xyz, modal, model, dofs=("x", "y", "z"))
    print(result.table())          # MAC diagonal, and the distances behind it
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import isroutine
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._assignment import linear_sum_assignment
from ._linalg import as_mode_matrix, coordinate_table, mode_source, row_index
from .mac import mac_matrix

__all__ = [
    "DOFMap",
    "MappedMACResult",
    "NearestNodeMap",
    "as_dofmap",
    "parse_component",
    "parse_components",
    "parse_dof_label",
    "match_dofs",
    "map_nearest_nodes",
    "mapped_mac",
    "mapped_mode_matrix",
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


def _as_item_list(items: Any) -> Any:
    """Wrap a bare string so it is one label, not a sequence of characters.

    ``components="RZ"`` and ``keys="12Z"`` must mean a single item; iterating
    them would parse ``"R"`` and ``"Z"`` (or ``"1"``, ``"2"``, ``"Z"``).
    """
    return [items] if isinstance(items, str) else items


def parse_components(components: Iterable[Any]) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Vectorized :func:`parse_component` over an iterable.

    Returns ``(component_ids, signs)``.  An integer array (the common case
    when a map is built from an FE kernel DOF map) is converted with pure
    NumPy; anything else falls back to the per-item parser.  A single label
    may be passed as a bare string.
    """
    components = _as_item_list(components)
    arr = components if isinstance(components, np.ndarray) else np.asarray(list(components))
    if arr.ndim > 1:
        raise ValueError(f"components must be 1-D, got shape {arr.shape}")
    if arr.size and np.issubdtype(arr.dtype, np.integer):
        comp = np.abs(arr).astype(np.int8)
        if int(comp.min()) < 1 or int(comp.max()) > 6:
            bad = arr[(comp < 1) | (comp > 6)][0]
            raise ValueError(f"component {bad!r} out of range 1..6")
        return comp, np.where(arr < 0, -1.0, 1.0)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int8), np.zeros(0)
    parsed = [parse_component(c) for c in arr.tolist()]
    return (
        np.fromiter((c for c, _ in parsed), dtype=np.int8, count=len(parsed)),
        np.fromiter((s for _, s in parsed), dtype=float, count=len(parsed)),
    )


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


def _parse_key_raw(key: Any) -> tuple[int, int, float, bool]:
    """``(node, component, sign, from_int)`` with integer components unchecked.

    Integer components are returned as given so that the caller can still
    decide whether they are zero based (FE kernel) or one based (Nastran);
    labels are always one based and are validated immediately.
    """
    if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 2:
        node, comp = int(key[0]), key[1]
        if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
            c = int(comp)
            return node, abs(c), (-1.0 if c < 0 else 1.0), True
        cid, sign = parse_component(comp)
        return node, cid, sign, False
    node, cid, sign = parse_dof_label(key)
    return node, cid, sign, False


def _shift_to_one_based(comps: list[int], from_int: list[bool], base: int | None) -> list[int]:
    """Normalize integer components to the ``1..6`` convention.

    ``base=None`` infers it: a component ``0`` can only come from the zero
    based kernel convention, since ``1..6`` has no zero.  Pass ``base``
    explicitly for a partial zero based map that happens to omit ``ux``.
    """
    if base is None:
        base = 0 if any(c == 0 and is_int for c, is_int in zip(comps, from_int, strict=True)) else 1
    if base not in (0, 1):
        raise ValueError(f"component base must be 0 or 1, got {base!r}")
    if base == 0:
        return [c + 1 if is_int else c for c, is_int in zip(comps, from_int, strict=True)]
    return comps


def _node_array(nodes: Any) -> NDArray[np.int64]:
    """Node ids as ``int64``, with an explanatory error for exotic ids."""
    try:
        return np.asarray(nodes, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "DOFMap requires integer node/channel ids; "
            f"got {type(np.asarray(nodes).flat[0]).__name__ if np.size(nodes) else 'none'}"
        ) from exc


#: Node ids above this magnitude would overflow the packed ``(node, component)``
#: code used by the vectorized set operations; those fall back to a dict.
_CODE_LIMIT = 1 << 59


def _codes(nodes: NDArray[np.int64], comps: NDArray[np.int8]) -> NDArray[np.int64] | None:
    """Pack ``(node, component)`` into one int64 per DOF, or ``None`` if unsafe."""
    if nodes.size and int(np.abs(nodes).max()) >= _CODE_LIMIT:
        return None
    return nodes * 8 + comps


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
    base:
        Convention of *integer* components: ``1`` for ``1..6`` (default when
        no zero appears), ``0`` for the kernel's ``0..5``, ``None`` to infer.
    """

    __slots__ = ("_nodes", "_components", "_scale", "_lookup")

    def __init__(
        self,
        nodes: ArrayLike,
        components: Iterable[Any],
        scale: ArrayLike | None = None,
        *,
        base: int | None = None,
    ) -> None:
        node_arr = _node_array(nodes)
        components = _as_item_list(components)
        comp_in = components if isinstance(components, np.ndarray) else np.asarray(list(components))
        if base != 1 and comp_in.size and np.issubdtype(comp_in.dtype, np.integer):
            if base == 0 or bool(np.any(comp_in == 0)):
                comp_in = comp_in + np.where(comp_in < 0, -1, 1)
        comp_arr, sign_arr = parse_components(comp_in)
        if node_arr.size != comp_arr.size:
            raise ValueError(f"{node_arr.size} nodes but {comp_arr.size} components")
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
        self._lookup: dict[tuple[int, int], int] | None = None
        self._check_unique()

    def _check_unique(self) -> None:
        """Reject a duplicated ``(node, component)`` without building the index."""
        code = _codes(self._nodes, self._components)
        if code is None:
            self._index()  # the dict build reports the duplicate itself
            return
        order = np.argsort(code, kind="stable")
        same = np.flatnonzero(code[order[1:]] == code[order[:-1]])
        if same.size:
            first, second = sorted(order[same[0] : same[0] + 2].tolist())
            raise ValueError(f"duplicate DOF {self[first]} at positions {first} and {second}")

    def _index(self) -> dict[tuple[int, int], int]:
        """``(node, component) -> position``, built on first use."""
        if self._lookup is None:
            lookup: dict[tuple[int, int], int] = {}
            keys = zip(self._nodes.tolist(), self._components.tolist(), strict=True)
            for i, key in enumerate(keys):
                if key in lookup:
                    raise ValueError(f"duplicate DOF {key} at positions {lookup[key]} and {i}")
                lookup[key] = i
            self._lookup = lookup
        return self._lookup

    # -- construction ----------------------------------------------------
    @classmethod
    def from_keys(
        cls, keys: Iterable[Any], scale: ArrayLike | None = None, *, base: int | None = None
    ) -> DOFMap:
        """Build from ``(node, component)`` tuples or labels like ``"12Z"``.

        A bare string is a single key, not a sequence of characters.
        """
        nodes: list[int] = []
        comps: list[int] = []
        signs: list[float] = []
        from_int: list[bool] = []
        for key in _as_item_list(keys):
            node, comp, sgn, is_int = _parse_key_raw(key)
            nodes.append(node)
            comps.append(comp)
            signs.append(sgn)
            from_int.append(is_int)
        factor = np.asarray(signs, dtype=float)
        if scale is not None:
            factor = factor * np.asarray(scale, dtype=float).reshape(-1)
        return cls(nodes, _shift_to_one_based(comps, from_int, base), factor, base=1)

    @classmethod
    def from_nodes(
        cls,
        nodes: ArrayLike,
        components: Iterable[Any] = (1, 2, 3),
        *,
        base: int | None = None,
    ) -> DOFMap:
        """Cartesian product of ``nodes`` with ``components`` (node-major)."""
        node_arr = _node_array(nodes)
        comps = list(_as_item_list(components))
        rep_nodes = np.repeat(node_arr, len(comps))
        rep_comps = comps * node_arr.size
        return cls(rep_nodes, rep_comps, base=base)

    @classmethod
    def from_kernel(cls, dof_map: Any) -> DOFMap:
        """Build from a :class:`femtools.fea.dofmap.DofMap`.

        The kernel numbers DOFs node-major with ``dofs_per_node`` consecutive
        zero based components per node, which maps one to one onto this
        class's ``1..dofs_per_node`` ordering.
        """
        dofs_per_node = int(dof_map.dofs_per_node)
        if not 1 <= dofs_per_node <= 6:
            raise ValueError(f"dofs_per_node must be within 1..6, got {dofs_per_node}")
        ids = dof_map.node_ids
        node_arr = _node_array(ids() if isroutine(ids) else ids)
        comps = np.tile(np.arange(1, dofs_per_node + 1, dtype=np.int8), node_arr.size)
        return cls(np.repeat(node_arr, dofs_per_node), comps, base=1)

    @classmethod
    def from_mapping(cls, mapping: Any, *, base: int | None = None) -> DOFMap:
        """Build from anything that describes a global DOF ordering.

        Accepts a :class:`DOFMap`, an FE kernel
        :class:`~femtools.fea.dofmap.DofMap`, any object carrying one as
        ``dof_map`` (``ModalResult``, ``AssemblyResult``, ``FEModel``), a
        ``{(node, component): index}`` dict, a ``{node: [index, ...]}`` /
        ``{node: {component: index}}`` dict, or a sequence of DOF keys.  Dict
        inputs are ordered by their global index.

        Integer components may be zero based (kernel) or one based
        (Nastran); ``base`` forces the interpretation when the map is too
        sparse for the automatic detection.
        """
        if isinstance(mapping, DOFMap):
            return mapping
        if mapping is None:
            raise ValueError("no DOF map given")
        if isinstance(mapping, str):
            raise TypeError(f"expected a DOF map, got the string {mapping!r}")
        if not isinstance(mapping, Mapping):
            nested = getattr(mapping, "dof_map", None)
            if nested is not None and nested is not mapping:
                # ``FEModel.dof_map`` is a method, ``ModalResult.dof_map`` an object.
                return cls.from_mapping(nested() if isroutine(nested) else nested, base=base)
            if hasattr(mapping, "dofs_per_node") and hasattr(mapping, "node_ids"):
                return cls.from_kernel(mapping)
        if isinstance(mapping, Mapping):
            items: list[tuple[int, Any]] = []
            keyed = False
            for key, value in mapping.items():
                if isinstance(key, tuple):
                    keyed = True
                    items.append((int(value), key))
                    continue
                node = int(key)
                if isinstance(value, Mapping):
                    keyed = True
                    for comp, idx in value.items():
                        items.append((int(idx), (node, comp)))
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
            # ``{node: [index, ...]}`` carries no component of its own: the
            # positions were already expanded one based just above.
            return cls.from_keys([key for _, key in items], base=base if keyed else 1)
        return cls.from_keys(mapping, base=base)

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
        idx = np.arange(len(self))[item] if isinstance(item, slice) else item
        return self.take(idx)

    def __contains__(self, key: Any) -> bool:
        try:
            node, comp, _ = parse_dof_label(key)
        except (ValueError, TypeError):
            return False
        return (node, comp) in self._index()

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
        return self._index().get((node, comp), default)

    def indices(self, keys: Iterable[Any], *, missing: str = "raise") -> NDArray[np.intp]:
        """Positions of ``keys``.

        ``missing='raise'`` (default) errors on unknown DOFs, ``'skip'`` drops
        them and ``'mark'`` returns ``-1`` for them.  A bare string is a
        single key.
        """
        out: list[int] = []
        for key in _as_item_list(keys):
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
        """Sub-map for the given positions, or for a boolean mask over them."""
        idx = row_index(index, len(self))
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
        """Positions of the DOFs matching the given components and/or nodes.

        ``components`` may be a single label (``"RZ"``) or an iterable of them.
        """
        mask = np.ones(len(self), dtype=bool)
        if components is not None:
            wanted = {parse_component(c)[0] for c in _as_item_list(components)}
            mask &= np.isin(self._components, sorted(wanted))
        if nodes is not None:
            mask &= np.isin(self._nodes, _node_array(nodes))
        return np.flatnonzero(mask).astype(np.intp)

    def translational(self, nodes: ArrayLike | None = None) -> NDArray[np.intp]:
        """Positions of the translational DOFs (``X``, ``Y``, ``Z``).

        The measurable set of a modal test: accelerometers see translations,
        so this is the slice a pretest works on::

            rows = dofs.translational()
            efi = effective_independence(modal.modes[rows], candidate_dofs=rows)
        """
        mask = self._components <= 3
        if nodes is not None:
            mask &= np.isin(self._nodes, _node_array(nodes))
        return np.flatnonzero(mask).astype(np.intp)

    def rotational(self, nodes: ArrayLike | None = None) -> NDArray[np.intp]:
        """Positions of the rotational DOFs (``RX``, ``RY``, ``RZ``)."""
        mask = self._components > 3
        if nodes is not None:
            mask &= np.isin(self._nodes, _node_array(nodes))
        return np.flatnonzero(mask).astype(np.intp)

    def common(self, other: DOFMap) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Positions of the DOFs present in both maps, in *this* map's order.

        Vectorized through a sorted search on the packed ``(node, component)``
        codes, so aligning a 30-channel test map against a 100 000 DOF model
        costs one sort rather than one dict lookup per DOF.
        """
        mine = _codes(self._nodes, self._components)
        theirs = _codes(other._nodes, other._components)
        if mine is None or theirs is None:  # pragma: no cover - astronomic node ids
            lookup = other._index()
            found = [(i, lookup[k]) for i, k in enumerate(self.keys) if k in lookup]
            pairs = np.asarray(found, dtype=np.intp).reshape(-1, 2)
            return pairs[:, 0], pairs[:, 1]
        if mine.size == 0 or theirs.size == 0:
            return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
        order = np.argsort(theirs, kind="stable")
        sorted_theirs = theirs[order]
        pos = np.clip(np.searchsorted(sorted_theirs, mine), 0, sorted_theirs.size - 1)
        hit = sorted_theirs[pos] == mine
        return np.flatnonzero(hit).astype(np.intp), order[pos[hit]].astype(np.intp)


def as_dofmap(source: Any, *, base: int | None = None) -> DOFMap:
    """Coerce ``source`` to a :class:`DOFMap` (alias of :meth:`DOFMap.from_mapping`)."""
    return DOFMap.from_mapping(source, base=base)


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
    map_a: Any = None,
    phi_b: ArrayLike | None = None,
    map_b: Any = None,
    *,
    apply_scale: bool = True,
) -> tuple[NDArray[Any], NDArray[Any], DOFMap]:
    """Restrict two mode sets to their common DOFs.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shapes ``(n_dof, n_mode)`` in the ordering of ``map_a``/``map_b``.
        A modal result object is accepted and unwrapped to its modes.
    map_a, map_b:
        Anything :meth:`DOFMap.from_mapping` understands.  ``None`` takes the
        map from the corresponding mode object, so a solved model needs no
        second argument::

            phi_t, phi_a, dofs = align_modes(phi_test, test_map, modal)

    apply_scale:
        Multiply each row by the DOF scale factor of its map, so reversed or
        differently calibrated sensors become directly comparable.

    Returns
    -------
    (phi_a_common, phi_b_common, common_map)
        Both matrices share the row ordering of ``common_map``.
    """
    if phi_b is None:
        raise ValueError("align_modes needs two mode sets")
    a_map = DOFMap.from_mapping(map_a if map_a is not None else phi_a)
    b_map = DOFMap.from_mapping(map_b if map_b is not None else phi_b)
    a = np.asarray(mode_source(phi_a))
    b = np.asarray(mode_source(phi_b))
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

    ``index`` holds positions or a boolean mask over the rows.  Used to build
    a test-DOF mass matrix for a cross-orthogonality check.

    A 1-D ``matrix`` is a lumped diagonal and yields the sub-vector
    ``matrix[index]``.  It has no off-diagonal block, so an ``index_col`` that
    selects one is rejected rather than answered with the sub-vector.
    """
    shape = np.shape(matrix)
    rows = row_index(index, shape[0] if shape else None)
    cols = rows if index_col is None else row_index(index_col, shape[-1] if shape else None)
    if hasattr(matrix, "tocsr") and not isinstance(matrix, np.ndarray):
        return matrix.tocsr()[rows, :][:, cols]
    dense = np.asarray(matrix)
    if dense.ndim == 1:
        if index_col is not None:
            raise ValueError(
                "a 1-D (lumped diagonal) operator has no row/column block; "
                "expand it to a diagonal matrix to restrict rows and columns separately"
            )
        return dense[rows]
    return dense[np.ix_(rows, cols)]


# -- geometric test-to-model node mapping ---------------------------------

#: Point pairs held in memory at once by the exact chunked search.
_CHUNK_PAIRS = 2_000_000

#: Point pairs handled by the exact chunked search before a KD-tree is used.
_KDTREE_PAIRS = 2_000_000

#: Largest cost matrix built for the ``unique=True`` assignment.
_ASSIGN_ENTRY_LIMIT = 4_000_000


class NearestNodeMap(NamedTuple):
    """Result of :func:`map_nearest_nodes`, one entry per test point.

    A plain ``(ids, distance)`` tuple — ``fe_ids, dist = map_nearest_nodes(...)``
    unpacks it — with the usual quality figures reachable as attributes.
    """

    #: FE node id nearest to each test point, ``-1`` where none was accepted.
    ids: NDArray[np.int64]
    #: Distance to that node, in the units of the coordinates.  Kept for a
    #: rejected point too, so ``tol`` can be judged from the result itself.
    distance: NDArray[np.float64]

    @property
    def matched(self) -> NDArray[np.bool_]:
        """Boolean mask of the test points that got a node."""
        return np.asarray(self.ids) >= 0

    @property
    def unmatched(self) -> NDArray[np.intp]:
        """Positions of the test points left without a node."""
        return np.flatnonzero(~self.matched).astype(np.intp)

    @property
    def n_matched(self) -> int:
        return int(np.count_nonzero(self.matched))

    @property
    def is_one_to_one(self) -> bool:
        """True when every test point got a node and no node was used twice."""
        ids = np.asarray(self.ids)
        return bool(ids.size) and bool(self.matched.all()) and np.unique(ids).size == ids.size

    @property
    def max_distance(self) -> float:
        """Largest matched distance (0 for an empty match)."""
        d = np.asarray(self.distance)[self.matched]
        return float(d.max()) if d.size else 0.0

    @property
    def rms_distance(self) -> float:
        """RMS of the matched distances — the scalar geometry-fit figure."""
        d = np.asarray(self.distance)[self.matched]
        return float(np.sqrt(np.mean(d**2))) if d.size else 0.0

    def table(self) -> str:
        """Plain-text listing, one line per test point."""
        head = f"{'test':>5} {'fe node':>9} {'distance':>13}"
        lines = [head, "-" * len(head)]
        ids = np.asarray(self.ids)
        dist = np.asarray(self.distance)
        for k in range(ids.size):
            node = "-" if ids[k] < 0 else str(int(ids[k]))
            lines.append(f"{k:>5} {node:>9} {dist[k]:>13.6g}")
        lines.append(
            f"matched {self.n_matched}/{ids.size}, max {self.max_distance:.6g}, "
            f"rms {self.rms_distance:.6g}" + ("" if self.is_one_to_one else ", not one-to-one")
        )
        return "\n".join(lines)


def _cloud(source: Any, name: str) -> tuple[NDArray[np.int64] | None, NDArray[np.float64]]:
    """``(ids, xyz)`` of a point cloud; ``ids`` is ``None`` for a bare array."""
    try:
        table = coordinate_table(source)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    if table is None:
        raise ValueError(
            f"{name} must be an (n_point, 3) array, a {{node: xyz}} mapping, an "
            f"(ids, xyz) pair or a model carrying nodal coordinates, got "
            f"{type(source).__name__}"
        )
    ids, xyz = table
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
        raise ValueError(f"{name} must hold 2-D or 3-D points, got shape {xyz.shape}")
    if xyz.shape[1] == 2:  # a planar test geometry lies in z = 0
        xyz = np.column_stack((xyz, np.zeros(xyz.shape[0])))
    return ids, np.ascontiguousarray(xyz)


def _distance_block(query: NDArray[np.float64], ref: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact ``(n_query, n_ref)`` Euclidean distances, filled in row chunks.

    The straight difference is used rather than the ``|q|^2 + |r|^2 - 2 q.r``
    expansion: the expansion loses half the significant digits near zero, and
    coincident points must come back as an exact 0.0, not as 1e-8.
    """
    n, m = query.shape[0], ref.shape[0]
    out = np.empty((n, m), dtype=float)
    rows = max(1, int(_CHUNK_PAIRS // max(m, 1)))
    for start in range(0, n, rows):
        block = query[start : start + rows]
        diff = block[:, None, :] - ref[None, :, :]
        np.sqrt(np.einsum("ijk,ijk->ij", diff, diff), out=out[start : start + rows])
    return out


def _brute_nearest(
    query: NDArray[np.float64], ref: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Nearest reference point per query row; ties go to the lowest position."""
    n, m = query.shape[0], ref.shape[0]
    pos = np.empty(n, dtype=np.intp)
    dist = np.empty(n, dtype=float)
    rows = max(1, int(_CHUNK_PAIRS // max(m, 1)))
    for start in range(0, n, rows):
        block = query[start : start + rows]
        diff = block[:, None, :] - ref[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        best = np.argmin(d2, axis=1)  # first minimum: deterministic on a tie
        pos[start : start + rows] = best
        dist[start : start + rows] = np.sqrt(d2[np.arange(block.shape[0]), best])
    return pos, dist


def _tree_nearest(
    query: NDArray[np.float64], ref: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """KD-tree nearest neighbour, with the same lowest-position tie rule."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - SciPy is a hard dependency
        return _brute_nearest(query, ref)
    tree = cKDTree(ref)
    dist, pos = tree.query(query, k=2)
    dist = np.asarray(dist, dtype=float)
    out = np.ascontiguousarray(np.asarray(pos, dtype=np.intp)[:, 0])
    # A tie is visible as an equal second distance, so the ball query that
    # resolves it deterministically runs only for the few points that need it.
    for i in np.flatnonzero(dist[:, 1] <= dist[:, 0]):
        hits = tree.query_ball_point(query[i], float(dist[i, 0]))
        if hits:
            out[i] = min(hits)
    # Recomputed rather than taken from the tree, so that the two search
    # paths return the same floating-point distance to the same node.
    diff = query - ref[out]
    return out, np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _nearest(
    query: NDArray[np.float64], ref: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    n, m = query.shape[0], ref.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    if m < 2 or n * m <= _KDTREE_PAIRS:
        return _brute_nearest(query, ref)
    return _tree_nearest(query, ref)


def _candidate_columns(
    query: NDArray[np.float64], ref: NDArray[np.float64], k: int
) -> NDArray[np.intp]:
    """Reference points that an optimal one-to-one assignment can use.

    In an optimal assignment of ``n`` query rows, the point matched to a row
    is always among that row's ``n`` nearest: at most ``n - 1`` of them are
    taken by other rows, so a row sent further away could always be moved
    back onto a free one at a lower cost.  Keeping the union of the ``n``
    nearest per row therefore leaves the optimum untouched while shrinking a
    whole-mesh cost matrix to the neighbourhood of the test points.
    """
    m = ref.shape[0]
    if k >= m:
        return np.arange(m, dtype=np.intp)
    if query.shape[0] * m > _KDTREE_PAIRS:
        try:
            from scipy.spatial import cKDTree
        except ImportError:  # pragma: no cover - SciPy is a hard dependency
            pass
        else:
            _, idx = cKDTree(ref).query(query, k=k)
            return np.unique(np.asarray(idx, dtype=np.intp))
    picked: list[NDArray[np.intp]] = []
    rows = max(1, int(_CHUNK_PAIRS // max(m, 1)))
    for start in range(0, query.shape[0], rows):
        block = query[start : start + rows]
        diff = block[:, None, :] - ref[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        picked.append(np.argpartition(d2, k - 1, axis=1)[:, :k].astype(np.intp))
    return np.unique(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.intp)


def _unique_match(
    query: NDArray[np.float64],
    ref: NDArray[np.float64],
    pos: NDArray[np.intp],
    dist: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Turn a nearest-neighbour map into a minimum-total-distance matching."""
    n, m = query.shape[0], ref.shape[0]
    if n == 0:
        return pos, dist
    if np.unique(pos).size == pos.size:
        # Every query row already holds its own minimum and no row collides,
        # so this matching attains the lower bound: it is the optimal one.
        return pos, dist
    cols = _candidate_columns(query, ref, min(n, m))
    if n * cols.size > _ASSIGN_ENTRY_LIMIT:
        raise ValueError(
            f"a one-to-one match of {n} test points against {m} FE nodes needs a "
            f"{n} x {cols.size} cost matrix; restrict the model to the test region "
            f"(e.g. an exterior node set) or use unique=False"
        )
    cost = _distance_block(query, ref[cols])
    rows, taken = linear_sum_assignment(cost)
    out_pos = np.full(n, -1, dtype=np.intp)
    out_dist = dist.copy()
    out_pos[rows] = cols[taken]
    out_dist[rows] = cost[rows, taken]
    return out_pos, out_dist


def map_nearest_nodes(
    xyz_test: Any,
    xyz_fe: Any,
    *,
    tol: float | None = None,
    unique: bool = False,
) -> NearestNodeMap:
    """Match a test point cloud onto the nodes of an analysis model.

    Before any DOF can be aligned, each measurement point has to be told
    which node of the mesh it sits on.  Test geometry is digitized, not
    numbered like the model, so the correspondence is geometric: for every
    row of ``xyz_test`` this returns the id of the closest FE node and how
    far away it is.  The distance is the diagnostic — a sensor that lands
    10 mm from the nearest node is either mis-digitized or measuring
    something the model does not resolve, and correlating it silently would
    blame the model for a bookkeeping error.

    Parameters
    ----------
    xyz_test:
        Measurement points: an ``(n, 3)`` (or planar ``(n, 2)``) array, a
        ``{id: xyz}`` mapping, an ``(ids, xyz)`` pair, or any model-like
        object carrying nodal coordinates.  The result follows this order.
    xyz_fe:
        The analysis side, in the same forms — typically an
        :class:`~femtools.core.model.FEModel`, whose node ids are what comes
        back (a solved result carries matrices and a DOF map, not geometry).
        A bare coordinate array carries no ids of its own, so the returned
        "ids" are then its row positions.
    tol:
        Largest accepted distance.  A test point whose nearest node lies
        beyond it is reported as ``-1`` (its true distance is kept), instead
        of being attached to a node it does not belong to.
    unique:
        Force a one-to-one match: no FE node is used by two test points.
        The plain nearest-neighbour map is returned unchanged whenever it is
        already injective (it is then optimal); otherwise the pairing that
        minimizes the *total* distance is solved with the same rectangular
        assignment used by ``pair_modes(method="hungarian")``.  With more
        test points than nodes the surplus points come back unmatched.

    Returns
    -------
    NearestNodeMap
        The ``(ids, distance)`` pair, one entry per test point;
        ``result.is_one_to_one``, ``result.max_distance`` and
        ``result.table()`` summarize the fit.

    Notes
    -----
    The search is exact (no bucketing, no rounding), computed in row chunks
    for a small cloud and through a ``scipy.spatial.cKDTree`` once the
    brute-force cost would grow past a few million point pairs; both paths
    resolve an exact tie — a point equidistant from two nodes, common on a
    symmetric mesh — to the lower node position, so the answer does not
    depend on the size of the model.

    Geometry that is not already in the model frame must be aligned first;
    the two steps compose directly::

        fit = align_geometry(test_xyz, fe_xyz)      # test frame -> FE frame
        ids, dist = map_nearest_nodes(fit.apply(test_xyz), model)

    and the ids then relabel a test DOF map onto the model::

        lookup = dict(zip(test_ids, ids))
        fe_map = DOFMap([lookup[n] for n in test_map.nodes],
                        test_map.components, test_map.scale)
    """
    _, q = _cloud(xyz_test, "xyz_test")
    fe_ids, r = _cloud(xyz_fe, "xyz_fe")
    if r.shape[0] == 0:
        raise ValueError("xyz_fe holds no points to match against")
    if tol is not None:
        tol = float(tol)
        if not tol >= 0.0:
            raise ValueError(f"tol must be non-negative, got {tol}")
    ids = np.arange(r.shape[0], dtype=np.int64) if fe_ids is None else fe_ids.astype(np.int64)

    pos, dist = _nearest(q, r)
    if unique:
        pos, dist = _unique_match(q, r, pos, dist)

    matched = pos >= 0
    if tol is not None:
        matched &= dist <= tol
    out = np.where(matched, ids[np.where(pos >= 0, pos, 0)], -1).astype(np.int64)
    return NearestNodeMap(out, dist.astype(float, copy=False))


# -- mode shapes on the mapped node order ---------------------------------

#: Policies for a test point that :func:`map_nearest_nodes` left unmatched.
_MISSING_POLICIES = ("raise", "zero", "drop")


def _mapped_components(
    dofs: Iterable[Any] | None, dof_map: DOFMap | None, dofs_per_node: int
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """``(components, signs)`` pulled at every mapped node, in the given order."""
    if dofs is None:
        if dof_map is None:
            comps = np.arange(1, dofs_per_node + 1, dtype=np.int8)
        else:
            comps = np.unique(dof_map.components).astype(np.int8)
        return comps, np.ones(comps.size)
    comps, signs = parse_components(dofs)
    if comps.size == 0:
        raise ValueError("dofs is empty: no component to pull at each node")
    seen, counts = np.unique(comps, return_counts=True)
    if seen.size != comps.size:
        dup = int(seen[counts > 1][0])
        raise ValueError(f"dofs asks for component {COMPONENT_NAMES[dup - 1]} twice")
    return comps, signs


def _mapped_rows(
    dof_map: DOFMap, nodes: NDArray[np.int64], comps: NDArray[np.int8]
) -> NDArray[np.intp]:
    """Rows of ``dof_map`` holding ``comps`` at each of ``nodes``, node-major.

    :meth:`DOFMap.common` cannot serve here: a node repeated in ``nodes`` is
    legitimate — two measurement points may share their nearest FE node when
    the match is not one-to-one — and a DOF that is absent has to be named
    rather than dropped.
    """
    want_nodes = np.repeat(nodes, comps.size)
    want_comps = np.tile(comps, nodes.size)
    have = _codes(dof_map.nodes, dof_map.components)
    want = _codes(want_nodes, want_comps)
    if have is None or want is None:  # pragma: no cover - astronomic node ids
        lookup = dof_map._index()
        keys = zip(want_nodes.tolist(), want_comps.tolist(), strict=True)
        found = [(lookup.get(key, -1), key) for key in keys]
        for row, key in found:
            if row < 0:
                raise ValueError(f"the FE DOF map has no DOF {key}")
        return np.asarray([row for row, _ in found], dtype=np.intp)
    order = np.argsort(have, kind="stable")
    pos = np.clip(np.searchsorted(have[order], want), 0, have.size - 1)
    rows = order[pos]
    bad = np.flatnonzero(have[rows] != want)
    if bad.size:
        node, comp = int(want_nodes[bad[0]]), int(want_comps[bad[0]])
        raise ValueError(
            f"the FE DOF map has no DOF ({node}, {COMPONENT_NAMES[comp - 1]}); it is "
            f"missing a requested component at {np.unique(want_nodes[bad]).size} of "
            f"the {nodes.size} mapped nodes"
        )
    return rows.astype(np.intp)


def _positional_rows(
    nodes: NDArray[np.int64], comps: NDArray[np.int8], dofs_per_node: int, n_row: int
) -> NDArray[np.intp]:
    """Rows of a node-major matrix with ``dofs_per_node`` rows per node."""
    top = int(comps.max())
    if top > dofs_per_node:
        raise ValueError(
            f"component {COMPONENT_NAMES[top - 1]} is beyond the {dofs_per_node} "
            f"DOFs per node of the mode matrix"
        )
    rows = (nodes[:, None] * dofs_per_node + (comps[None, :] - 1)).reshape(-1)
    if int(rows.max()) >= n_row:
        raise ValueError(
            f"node id {int(nodes.max())} needs row {int(rows.max())} of a mode matrix "
            f"with {n_row} rows; without a DOF map the ids are read as 0-based node "
            f"positions with {dofs_per_node} DOFs per node, so pass dof_map= (or a "
            f"ModalResult, which carries its own) when they are model node ids"
        )
    return rows.astype(np.intp)


class _GatherPlan(NamedTuple):
    """Everything the row gather needs, resolved once from the analysis side."""

    phi: NDArray[Any]
    dof_map: DOFMap | None
    components: NDArray[np.int8]
    signs: NDArray[np.float64]
    dofs_per_node: int


def _gather_plan(
    modes: Any,
    dof_map: Any,
    dofs: Iterable[Any] | None,
    dofs_per_node: int | None,
) -> _GatherPlan:
    """Resolve the mode matrix, its DOF ordering and the components to pull."""
    phi = as_mode_matrix(modes, "modes")
    if phi.shape[0] == 0:
        raise ValueError("modes has no rows to gather from")
    source = dof_map
    if source is None and not isinstance(modes, (np.ndarray, list, tuple)):
        source = getattr(modes, "dof_map", None)
    if isroutine(source):  # ``FEModel.dof_map`` is a method, not an attribute
        source = source()
    if source is not None and dofs_per_node is not None:
        raise ValueError(
            "dofs_per_node describes the row layout used when no DOF map is given; "
            "drop it, or select components with dofs= instead"
        )
    dmap = DOFMap.from_mapping(source) if source is not None else None
    if dmap is not None and len(dmap) != phi.shape[0]:
        raise ValueError(f"modes has {phi.shape[0]} rows but the DOF map has {len(dmap)} DOF")

    per_node = 6 if dofs_per_node is None else int(dofs_per_node)
    if dmap is None and not 1 <= per_node <= 6:
        raise ValueError(f"dofs_per_node must be within 1..6, got {per_node}")
    comps, signs = _mapped_components(dofs, dmap, per_node)
    return _GatherPlan(phi, dmap, comps, signs, per_node)


def _gather_mapped(plan: _GatherPlan, fe_ids: Any, missing: str) -> NDArray[Any]:
    """Copy the rows of the matched nodes, in the order of the test points."""
    phi, dmap, comps, signs = plan.phi, plan.dof_map, plan.components, plan.signs
    ids = _node_array(getattr(fe_ids, "ids", fe_ids))
    keep = ids >= 0
    if missing == "raise" and not keep.all():
        lost = np.flatnonzero(~keep)
        raise ValueError(
            f"{lost.size} of {ids.size} test points have no FE node (first at "
            f"position {int(lost[0])}); raise tol, or pass missing='zero' / 'drop'"
        )
    kept = ids[keep]
    n_point = kept.size if missing == "drop" else ids.size
    if kept.size == 0:
        return np.zeros((n_point * comps.size, phi.shape[1]), dtype=phi.dtype)

    rows = (
        _mapped_rows(dmap, kept, comps)
        if dmap is not None
        else _positional_rows(kept, comps, plan.dofs_per_node, phi.shape[0])
    )
    gathered = phi[rows, :]
    if not np.all(signs == 1.0):
        gathered = gathered * np.tile(signs, kept.size)[:, None]
    if missing == "drop" or keep.all():
        return gathered
    out = np.zeros((ids.size * comps.size, phi.shape[1]), dtype=gathered.dtype)
    out[np.repeat(keep, comps.size), :] = gathered
    return out


def mapped_mode_matrix(
    modes: Any,
    fe_ids: Any,
    *,
    dof_map: Any = None,
    dofs: Iterable[Any] | None = None,
    dofs_per_node: int | None = None,
    missing: str = "raise",
) -> NDArray[Any]:
    """Analysis mode shapes gathered on the nodes matched to the test points.

    :func:`map_nearest_nodes` answers *which* FE node each measurement point
    sits on; this performs the gather that answer is for.  For every id it
    returns, the rows of ``modes`` belonging to that node are copied into the
    output in the order of the test points, so the result is a
    ``(n_point * n_component, n_mode)`` matrix on the *test* grid.  A mode set
    written on the model can then be compared with one written on the test
    geometry without either side being renumbered::

        fit = align_geometry(test_xyz, model)        # test frame -> FE frame
        ids, dist = map_nearest_nodes(fit.apply(test_xyz), model)
        phi_a = mapped_mode_matrix(modal, ids, dofs=("x", "y", "z"))
        mac = mac_matrix(phi_test, phi_a)

    Parameters
    ----------
    modes:
        FE mode shapes ``(n_dof, n_mode)`` over the model's own DOF ordering,
        or a modal result carrying them (its ``dof_map`` is then used).
    fe_ids:
        The node ids the test points were matched to: a
        :class:`NearestNodeMap`, or the ``ids`` array itself.  A ``-1``
        (unmatched, or beyond ``tol``) is handled per ``missing``.
    dof_map:
        DOF ordering of ``modes`` — anything :meth:`DOFMap.from_mapping`
        understands, including the kernel DOF map of a solved model.  When it
        is omitted and ``modes`` carries none, the ids are read as 0-based
        node positions of a node-major matrix instead.
    dofs:
        Components to pull at each node, in output order; ints ``1..6`` or
        labels, and a leading ``-`` flips that row's sign the way a reversed
        sensor does.  The default takes all six, or the components the DOF map
        actually holds.
    dofs_per_node:
        Row block size of the positional fallback (default 6).  Rejected
        together with ``dof_map``, which already fixes the layout; to pull
        fewer components from a 6-DOF model use ``dofs``.
    missing:
        What to do with an unmatched test point: ``'raise'`` (default),
        ``'zero'`` to keep its rows as zeros, or ``'drop'`` to leave them out
        — ``NearestNodeMap.matched`` then selects the surviving test rows.

    Returns
    -------
    ndarray
        ``(n_kept * n_component, n_mode)``, node-major: all components of the
        first test point, then of the second, and so on.  The dtype of
        ``modes`` is preserved, so complex modes stay complex.

    Notes
    -----
    Only rows are gathered; no value is interpolated and no shape is scaled,
    so a mapped mode is the FE mode itself and correlating a model against a
    relabelled copy of itself returns an exact unit MAC diagonal.  Whether the
    gather is *meaningful* is what ``NearestNodeMap.distance`` reports: a
    point sitting 10 mm from its node gets that node's motion regardless.
    """
    if missing not in _MISSING_POLICIES:
        raise ValueError(f"unknown missing policy {missing!r}, expected one of {_MISSING_POLICIES}")
    plan = _gather_plan(modes, dof_map, dofs, dofs_per_node)
    return _gather_mapped(plan, fe_ids, missing)


# -- the whole test-to-model correlation in one call ----------------------


def _component_labels(comps: NDArray[np.int8], signs: NDArray[np.float64]) -> tuple[str, ...]:
    """``('X', 'Y', '-Z')`` for the components pulled at each node."""
    return tuple(
        ("-" if s < 0 else "") + COMPONENT_NAMES[c - 1]
        for c, s in zip(comps.tolist(), signs.tolist(), strict=True)
    )


def _fe_geometry(xyz_fe: Any, modes: Any) -> Any:
    """The analysis point cloud, falling back to one carried by ``modes``."""
    if xyz_fe is not None:
        return xyz_fe
    if coordinate_table(modes) is not None:
        return modes
    raise ValueError(
        f"xyz_fe is required: {type(modes).__name__} carries no nodal coordinates "
        "(a solved result holds matrices and a DOF map, not geometry); pass the "
        "FEModel, a {node: xyz} mapping or an (ids, xyz) pair"
    )


def _test_matrix(
    phi_test: Any,
    test_ids: NDArray[np.int64] | None,
    test_map: Any,
    comps: NDArray[np.int8],
    keep: NDArray[np.bool_],
    missing: str,
) -> NDArray[Any]:
    """Test mode shapes as ``(n_point * n_component, n_mode)`` on the test grid.

    The rows are gathered by ``(node, component)`` when the test side carries
    a DOF map *and* the test geometry carries node ids, and read positionally
    — node-major, in the order of ``xyz_test`` — otherwise.
    """
    phi = as_mode_matrix(phi_test, "phi_test")
    source = test_map
    explicit = source is not None
    if source is None and not isinstance(phi_test, (np.ndarray, list, tuple)):
        source = getattr(phi_test, "dof_map", None)
    if isroutine(source):
        source = source()
    if source is not None and test_ids is None:
        if explicit:
            raise ValueError(
                "test_map orders phi_test by (node, component), but xyz_test is a bare "
                "coordinate array carrying no node ids; give the test geometry as a "
                "{node: xyz} mapping, an (ids, xyz) pair or a model"
            )
        source = None  # nothing to match the ids of: fall back to the row order

    n_comp = int(comps.size)
    if source is not None and test_ids is not None:
        tmap = DOFMap.from_mapping(source)
        if len(tmap) != phi.shape[0]:
            raise ValueError(
                f"phi_test has {phi.shape[0]} rows but its DOF map has {len(tmap)} DOF"
            )
        wanted = test_ids[keep] if missing == "drop" else test_ids
        if wanted.size == 0:
            return np.zeros((0, phi.shape[1]), dtype=phi.dtype)
        return phi[_mapped_rows(tmap, wanted, comps), :]

    n_point = int(keep.size)
    n_kept = int(np.count_nonzero(keep))
    if phi.shape[0] == n_point * n_comp:
        if missing == "drop" and n_kept != n_point:
            return phi[np.repeat(keep, n_comp), :]
        return phi
    if missing == "drop" and phi.shape[0] == n_kept * n_comp:
        return phi  # already restricted to the matched points by the caller

    names = ", ".join(COMPONENT_NAMES[c - 1] for c in comps.tolist())
    hint = ""
    if n_point and phi.shape[0] % n_point == 0 and 1 <= phi.shape[0] // n_point <= 6:
        per = phi.shape[0] // n_point
        suggest = '("x", "y", "z")' if per == 3 else f"a {per}-component list"
        hint = f"; {per} rows per test point would fit, so pass dofs={suggest}"
    raise ValueError(
        f"phi_test has {phi.shape[0]} rows but the mapped analysis matrix has "
        f"{n_point * n_comp}: {n_point} test points x {n_comp} components ({names}). "
        f"phi_test must be node-major in the order of xyz_test{hint}"
    )


@dataclass
class MappedMACResult:
    """Result of :func:`mapped_mac`: the MAC plus the node match behind it.

    ``np.asarray(result)`` and ``result[i, j]`` give the MAC matrix itself, so
    the object can be used wherever :func:`~femtools.correlation.mac_matrix`
    would be; the geometry figures stay reachable next to it because they are
    what says whether the number means anything.
    """

    #: ``(n_mode_test, n_mode_fe)`` MAC of the test modes against the mapped ones.
    mac: NDArray[np.float64]
    #: Node match produced by :func:`map_nearest_nodes`, one entry per test point.
    nodes: NearestNodeMap
    #: Test mode shapes as correlated, ``(n_point * n_component, n_mode_test)``.
    phi_test: NDArray[Any]
    #: Analysis mode shapes gathered on the test grid, same row set.
    phi_fe: NDArray[Any]
    #: Components pulled at each node, in row order (``('X', 'Y', '-Z')``).
    dofs: tuple[str, ...] = ()

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        arr = np.asarray(self.mac, dtype=dtype)
        return arr.copy() if copy else arr

    def __getitem__(self, item: Any) -> Any:
        return self.mac[item]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mac.shape

    @property
    def ids(self) -> NDArray[np.int64]:
        """FE node id matched to each test point (``-1`` where none was)."""
        return np.asarray(self.nodes.ids)

    @property
    def distance(self) -> NDArray[np.float64]:
        """Distance from each test point to its FE node."""
        return np.asarray(self.nodes.distance)

    @property
    def max_distance(self) -> float:
        return self.nodes.max_distance

    @property
    def rms_distance(self) -> float:
        return self.nodes.rms_distance

    @property
    def is_one_to_one(self) -> bool:
        return self.nodes.is_one_to_one

    @property
    def n_matched(self) -> int:
        return self.nodes.n_matched

    @property
    def diagonal(self) -> NDArray[np.float64]:
        """``mac[i, i]`` — the paired-mode values the correlation is judged on."""
        return np.diagonal(self.mac).copy()

    @property
    def min_diagonal(self) -> float:
        """Worst paired value (0.0 when there is no pair)."""
        diag = self.diagonal
        return float(diag.min()) if diag.size else 0.0

    @property
    def n_dof(self) -> int:
        """Rows correlated: test points times components."""
        return int(self.phi_fe.shape[0])

    def table(self) -> str:
        """Plain-text listing: the MAC diagonal and the geometry it rests on."""
        mac = np.asarray(self.mac, dtype=float)
        head = f"{'mode':>5} {'MAC':>8} {'best':>5} {'MAC(best)':>10}"
        lines = [head, "-" * len(head)]
        for i in range(min(mac.shape)):
            best = int(np.argmax(mac[i, :])) if mac.shape[1] else -1
            lines.append(f"{i:>5} {mac[i, i]:>8.4f} {best:>5} {mac[i, best]:>10.4f}")
        comps = ", ".join(self.dofs) if self.dofs else "-"
        lines.append(
            f"{self.ids.size} test points, {self.n_matched} matched on ({comps}), "
            f"max {self.max_distance:.6g}, rms {self.rms_distance:.6g}"
            + ("" if self.is_one_to_one else ", not one-to-one")
        )
        return "\n".join(lines)


def mapped_mac(
    phi_test: Any,
    xyz_test: Any,
    modes: Any,
    xyz_fe: Any = None,
    *,
    test_map: Any = None,
    tol: float | None = None,
    unique: bool = False,
    dof_map: Any = None,
    dofs: Iterable[Any] | None = None,
    dofs_per_node: int | None = None,
    missing: str = "raise",
    weights: Any = None,
) -> MappedMACResult:
    """MAC between a test mode set and an FE one, through the node match.

    The three steps that correlate a digitized test against a mesh —
    :func:`map_nearest_nodes` to find which FE node each measurement point
    sits on, :func:`mapped_mode_matrix` to gather the analysis rows of those
    nodes in the order of the test points, and
    :func:`~femtools.correlation.mac_matrix` on the result — are one call::

        result = mapped_mac(phi_test, test_xyz, modal, model, dofs=("x", "y", "z"))
        print(result.table())
        assert result.min_diagonal > 0.9

    The intermediate objects are kept, not thrown away: ``result.mac`` is the
    matrix, ``result.nodes`` is the :class:`NearestNodeMap` whose distances
    say whether the match is trustworthy, and ``result.phi_fe`` is the mapped
    analysis matrix.  ``np.asarray(result)`` is the MAC itself.

    Nothing here is a new criterion; it is the bookkeeping of the three calls
    above, done once and consistently.  Geometry that is not already in the
    model frame still has to be aligned first — compose with
    :func:`~femtools.correlation.align_geometry`::

        fit = align_geometry(test_xyz, model)
        result = mapped_mac(phi_test, fit.apply(test_xyz), modal, model, tol=0.01)

    Parameters
    ----------
    phi_test:
        Test mode shapes, ``(n_point * n_component, n_mode)`` node-major in
        the order of ``xyz_test`` — all components of the first measurement
        point, then of the second.  A modal result is unwrapped; when it
        carries a DOF map and ``xyz_test`` carries node ids, the rows are
        gathered by ``(node, component)`` instead of by position, so a test
        model with its own node numbering needs no reordering.
    xyz_test:
        Measurement points: an ``(n, 3)`` (or planar ``(n, 2)``) array, a
        ``{id: xyz}`` mapping, an ``(ids, xyz)`` pair or a model.  Everything
        is reported in this order.
    modes:
        FE mode shapes ``(n_dof, n_mode)`` or a modal result carrying them
        (and its own ``dof_map``).
    xyz_fe:
        The analysis geometry.  May be omitted only when ``modes`` itself
        carries nodal coordinates; a solved result usually does not.
    test_map:
        DOF ordering of ``phi_test``, when it is neither positional nor
        carried by the mode object.  Requires ``xyz_test`` to carry node ids.
    tol, unique:
        Passed to :func:`map_nearest_nodes`: largest accepted distance, and
        whether to force a one-to-one match.
    dof_map, dofs, dofs_per_node, missing:
        Passed to :func:`mapped_mode_matrix`.  ``dofs`` fixes the components
        pulled at each node **and** the rows expected of ``phi_test``; a
        leading ``-`` (``dofs=("x", "y", "-z")``) flips the analysis row, as a
        reversed sensor does, and is therefore applied to one side only.
        ``missing='drop'`` drops the unmatched points from *both* matrices,
        ``'zero'`` keeps their test rows against zeroed analysis rows — which
        lowers the MAC honestly rather than hiding the gap.
    weights:
        MAC weighting operator on the mapped row set, as in
        :func:`~femtools.correlation.mac_matrix`.

    Returns
    -------
    MappedMACResult
        ``result.mac`` is ``(n_mode_test, n_mode_fe)``; ``result.diagonal``,
        ``result.min_diagonal`` and ``result.table()`` summarize it.

    Notes
    -----
    Only rows are gathered — no interpolation, no scaling — so a model
    correlated against a translated, renumbered copy of itself returns a unit
    MAC diagonal to round-off.  A poor match is a geometry problem and shows
    up in ``result.nodes``, never as a silently massaged MAC.
    """
    if missing not in _MISSING_POLICIES:
        raise ValueError(f"unknown missing policy {missing!r}, expected one of {_MISSING_POLICIES}")

    test_ids, q = _cloud(xyz_test, "xyz_test")
    nodes = map_nearest_nodes(q, _fe_geometry(xyz_fe, modes), tol=tol, unique=unique)

    plan = _gather_plan(modes, dof_map, dofs, dofs_per_node)
    phi_fe = _gather_mapped(plan, nodes, missing)
    phi_t = _test_matrix(phi_test, test_ids, test_map, plan.components, nodes.matched, missing)

    return MappedMACResult(
        mac=mac_matrix(phi_t, phi_fe, weights=weights),
        nodes=nodes,
        phi_test=phi_t,
        phi_fe=phi_fe,
        dofs=_component_labels(plan.components, plan.signs),
    )

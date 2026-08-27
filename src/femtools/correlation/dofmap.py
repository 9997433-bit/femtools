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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from inspect import isroutine
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import mode_source, row_index

__all__ = [
    "DOFMap",
    "as_dofmap",
    "parse_component",
    "parse_components",
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

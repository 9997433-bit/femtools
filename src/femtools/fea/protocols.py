"""Structural typing layer between the FEA kernel and the model database.

The FEA kernel never imports :mod:`femtools.core.model` directly.  It only
requires objects that *behave* like the contract objects (``FEModel``, ``Node``,
``Element`` ...).  This keeps the solver usable with the core database, with
plain dataclasses, with dictionaries and with any third party container that
exposes the same names.

Attribute lookup is deliberately forgiving: names are tried in order, mappings
are supported and a handful of common "bag of extra parameters" containers
(``attrs``, ``params``, ``extra``, ``data``, ``fields``, ``options``) are
searched as a fallback.  That tolerance is what makes the duck typing usable
against readers (BDF/UNV) that spell the same quantity differently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DOF_LABELS",
    "DOF_ALIASES",
    "ElementLike",
    "MaterialLike",
    "ModelLike",
    "NodeLike",
    "PropertyLike",
    "SPCLike",
    "as_bool",
    "as_float",
    "get_any",
    "iter_records",
    "node_xyz",
    "normalize_dof",
    "normalize_dof_list",
    "record_id",
    "spc_entries",
]


#: Canonical global DOF ordering used everywhere in the kernel.
DOF_LABELS: tuple[str, ...] = ("ux", "uy", "uz", "rx", "ry", "rz")

#: Accepted spellings for a single DOF component (case insensitive).
DOF_ALIASES: dict[str, int] = {
    "ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5,
    "u": 0, "v": 1, "w": 2,
    "tx": 0, "ty": 1, "tz": 2,
    "t1": 0, "t2": 1, "t3": 2, "r1": 3, "r2": 4, "r3": 5,
    "dx": 0, "dy": 1, "dz": 2,
    "x": 0, "y": 1, "z": 2,
    "fx": 0, "fy": 1, "fz": 2, "mx": 3, "my": 4, "mz": 5,
    "f1": 0, "f2": 1, "f3": 2, "m1": 3, "m2": 4, "m3": 5,
    "thx": 3, "thy": 4, "thz": 5,
    "rotx": 3, "roty": 4, "rotz": 5,
    "theta_x": 3, "theta_y": 4, "theta_z": 5,
}

_MISSING = object()

_NESTED_CONTAINERS = ("attrs", "params", "extra", "data", "fields", "options", "kwargs", "props")


@runtime_checkable
class NodeLike(Protocol):
    """A grid point.  Only ``xyz`` is mandatory."""

    xyz: Any


@runtime_checkable
class ElementLike(Protocol):
    """A finite element connectivity record."""

    type: Any
    nodes: Any


@runtime_checkable
class MaterialLike(Protocol):
    """An isotropic (or orthotropic) material record."""

    E: Any


@runtime_checkable
class PropertyLike(Protocol):
    """A physical property record (section / thickness / scalar constants)."""

    type: Any


@runtime_checkable
class SPCLike(Protocol):
    """A single point constraint record."""

    node_id: Any


@runtime_checkable
class ModelLike(Protocol):
    """Minimal finite element model database required by the kernel."""

    nodes: Any
    elements: Any


def _lookup(obj: Any, name: str) -> Any:
    if obj is None:
        return _MISSING
    # ``dict`` first: it covers almost every mapping in practice and avoids the
    # abc ``Mapping`` check, which dominates assembly profiles otherwise.
    if type(obj) is dict:
        return obj.get(name, _MISSING)
    if isinstance(obj, Mapping):
        return obj[name] if name in obj else _MISSING
    return getattr(obj, name, _MISSING)


def get_any(obj: Any, names: str | Sequence[str], default: Any = _MISSING) -> Any:
    """Return the first non-``None`` attribute/key of *obj* among *names*.

    Nested parameter bags are searched after the direct lookup fails, so a
    reader may either promote ``A`` to a real attribute or stash it in
    ``prop.attrs["A"]`` without the kernel caring.
    """
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = _lookup(obj, name)
        if value is not _MISSING and value is not None:
            return value
    for container in _NESTED_CONTAINERS:
        bag = _lookup(obj, container)
        if bag is _MISSING or not isinstance(bag, Mapping):
            continue
        for name in names:
            value = bag.get(name, _MISSING)
            if value is not _MISSING and value is not None:
                return value
    if default is _MISSING:
        raise KeyError(
            f"{type(obj).__name__} object provides none of {tuple(names)!r}"
        )
    return default


def as_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float conversion that tolerates ``None`` and strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Interpret Nastran-ish truthiness (``1``, ``"Y"``, ``"true"``, ...)."""
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "y", "yes", "true", "t", "on"}:
            return True
        if token in {"0", "n", "no", "false", "f", "off", ""}:
            return False
        return default
    return bool(value)


def record_id(record: Any, fallback: Any = None) -> Any:
    """Extract an ``id``-like field from a record, falling back if absent."""
    for name in ("id", "eid", "nid", "pid", "mid", "sid", "ident", "label"):
        value = _lookup(record, name)
        if value is not _MISSING and value is not None:
            return value
    return fallback


def iter_records(container: Any) -> Iterable[tuple[Any, Any]]:
    """Yield ``(id, record)`` pairs from a dict, list, tuple or ``None``."""
    if container is None:
        return
    if isinstance(container, Mapping):
        for key, value in container.items():
            yield key, value
        return
    if isinstance(container, (list, tuple, set, frozenset)):
        for index, value in enumerate(container):
            yield record_id(value, index), value
        return
    # Some databases expose ``.values()`` without being a Mapping.
    values = getattr(container, "values", None)
    if callable(values):
        for value in values():
            yield record_id(value), value
        return
    if isinstance(container, Iterable):
        for index, value in enumerate(container):
            yield record_id(value, index), value


def node_xyz(node: Any) -> np.ndarray:
    """Return the ``(3,)`` position of a node-like object."""
    if isinstance(node, (list, tuple, np.ndarray)):
        xyz = np.asarray(node, dtype=float).ravel()
    else:
        raw = get_any(node, ("xyz", "coords", "coordinates", "position", "xyz_global"), None)
        if raw is None:
            x = as_float(get_any(node, ("x", "x1"), 0.0), 0.0)
            y = as_float(get_any(node, ("y", "x2"), 0.0), 0.0)
            z = as_float(get_any(node, ("z", "x3"), 0.0), 0.0)
            raw = (x, y, z)
        xyz = np.asarray(raw, dtype=float).ravel()
    if xyz.size == 2:
        xyz = np.array([xyz[0], xyz[1], 0.0])
    if xyz.size != 3:
        raise ValueError(f"node coordinates must have 3 components, got {xyz.size}")
    return xyz


def normalize_dof(token: Any, *, dofs_per_node: int = 6) -> int:
    """Map a DOF designator to a zero based component index.

    Integers are interpreted as **zero based** component indices (``0..5``).
    Strings accept the aliases in :data:`DOF_ALIASES` as well as the Nastran
    one based digit form (``"1".."6"``).
    """
    if isinstance(token, (bool, np.bool_)):
        raise TypeError("boolean is not a valid DOF designator")
    if isinstance(token, (int, np.integer)):
        index = int(token)
    elif isinstance(token, (float, np.floating)):
        if float(token) != int(token):
            raise ValueError(f"non integer DOF designator {token!r}")
        index = int(token)
    else:
        text = str(token).strip().lower()
        if text in DOF_ALIASES:
            index = DOF_ALIASES[text]
        elif text.isdigit():
            # Nastran component digits are one based.
            index = int(text) - 1
        else:
            raise ValueError(f"unrecognised DOF designator {token!r}")
    if not 0 <= index < dofs_per_node:
        raise ValueError(f"DOF index {index} out of range 0..{dofs_per_node - 1}")
    return index


def normalize_dof_list(spec: Any, *, dofs_per_node: int = 6) -> list[int]:
    """Expand a DOF specification into a sorted list of component indices.

    Accepts masks (``(True, False, ...)``), component strings (``"123"``),
    iterables of designators and single designators.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text in DOF_ALIASES:
            return [DOF_ALIASES[text]]
        if text.isdigit():
            return sorted({int(ch) - 1 for ch in text if ch != "0"})
        parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
        if len(parts) > 1:
            out: set[int] = set()
            for part in parts:
                out.update(normalize_dof_list(part, dofs_per_node=dofs_per_node))
            return sorted(out)
        return [normalize_dof(text, dofs_per_node=dofs_per_node)]
    if isinstance(spec, (int, np.integer, float, np.floating)) and not isinstance(
        spec, (bool, np.bool_)
    ):
        return [normalize_dof(spec, dofs_per_node=dofs_per_node)]
    if isinstance(spec, (bool, np.bool_)):
        raise ValueError("a bare boolean is not a DOF specification")
    if isinstance(spec, Iterable):
        items = list(spec)
        if items and all(isinstance(v, (bool, np.bool_)) for v in items):
            return [i for i, flag in enumerate(items) if bool(flag)][:dofs_per_node]
        out_set: set[int] = set()
        for item in items:
            out_set.update(normalize_dof_list(item, dofs_per_node=dofs_per_node))
        return sorted(out_set)
    raise ValueError(f"unrecognised DOF specification {spec!r}")


def spc_entries(model: Any, *, dofs_per_node: int = 6) -> list[tuple[Any, int, float]]:
    """Collect ``(node_id, dof_component, enforced_value)`` triples.

    Reads ``model.spcs`` (also ``constraints`` / ``spc`` / ``bcs``).  Each
    record may express its constrained components as a 6 entry boolean
    ``mask``, a ``dofs``/``components`` list or a Nastran component string.
    """
    container = None
    for name in ("spcs", "spc", "constraints", "boundary_conditions", "bcs"):
        container = get_any(model, name, None)
        if container is not None:
            break
    out: list[tuple[Any, int, float]] = []
    for _, record in iter_records(container):
        if record is None:
            continue
        node_id = get_any(record, ("node_id", "nid", "node", "grid", "gid", "g"), None)
        if node_id is None:
            continue
        mask = get_any(record, ("mask",), None)
        dof_spec = get_any(record, ("dofs", "dof", "components", "component", "c"), None)
        if mask is not None:
            comps = normalize_dof_list(mask, dofs_per_node=dofs_per_node)
        elif dof_spec is not None:
            comps = normalize_dof_list(dof_spec, dofs_per_node=dofs_per_node)
        else:
            comps = list(range(dofs_per_node))
        raw_value = get_any(record, ("value", "enforced", "d", "displacement", "val"), 0.0)
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            values = np.asarray(raw_value, dtype=float).ravel()
            for comp in comps:
                out.append((node_id, comp, float(values[comp]) if comp < values.size else 0.0))
        else:
            value = as_float(raw_value, 0.0) or 0.0
            for comp in comps:
                out.append((node_id, comp, value))
    return out

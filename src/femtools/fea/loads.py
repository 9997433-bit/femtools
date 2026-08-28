"""Building global load vectors from assorted load descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .dofmap import DofMap
from .protocols import as_float, get_any, iter_records, normalize_dof, normalize_dof_list

__all__ = ["build_load_vector", "model_loads"]

_FORCE_KEYS = ("fx", "fy", "fz", "mx", "my", "mz")


def model_loads(model: Any) -> Any:
    """Return the model's load container, if it has one."""
    for name in ("loads", "load", "forces", "force", "point_loads", "nodal_loads"):
        container = get_any(model, name, None)
        if container is not None:
            return container
    return None


def _add(vector: np.ndarray, dof_map: DofMap, node_id: Any, comp: int, value: float) -> None:
    try:
        vector[dof_map.index(node_id, comp)] += float(value)
    except KeyError:
        pass


def _apply_record(vector: np.ndarray, dof_map: DofMap, record: Any) -> bool:
    node_id = get_any(record, ("node_id", "nid", "node", "grid", "gid", "g"), None)
    if node_id is None:
        return False
    scale = as_float(get_any(record, ("scale", "magnitude", "mag", "f", "factor"), None), None)

    expand = getattr(record, "as_dof_values", None)
    if callable(expand):
        factor = 1.0 if scale is None else float(scale)
        for comp, value in expand():
            _add(vector, dof_map, node_id, int(comp), factor * float(value))
        return True

    force = get_any(record, ("force", "fxyz"), None)
    moment = get_any(record, ("moment", "mxyz"), None)
    if force is not None or moment is not None:
        factor = 1.0 if scale is None else float(scale)
        if force is not None:
            for i, value in enumerate(np.asarray(force, dtype=float).ravel()[:3]):
                _add(vector, dof_map, node_id, i, factor * float(value))
        if moment is not None:
            for i, value in enumerate(np.asarray(moment, dtype=float).ravel()[:3]):
                _add(vector, dof_map, node_id, 3 + i, factor * float(value))
        return True

    vec = get_any(record, ("vector", "components", "xyz", "direction", "n", "comp_vector"), None)
    if vec is not None and not isinstance(vec, (str, int, float)):
        arr = np.asarray(vec, dtype=float).ravel()
        if scale is not None:
            arr = arr * scale
        kind = str(get_any(record, ("type", "kind"), "force") or "force").lower()
        offset = 3 if ("moment" in kind or kind in {"m", "mom"}) and arr.size == 3 else 0
        for i, value in enumerate(arr[:6]):
            _add(vector, dof_map, node_id, i + offset, value)
        return True

    dof_spec = get_any(record, ("dof", "component", "components", "c"), None)
    value = get_any(record, ("value", "magnitude", "mag", "f", "load"), None)
    if dof_spec is not None and value is not None:
        comps = normalize_dof_list(dof_spec)
        val = float(as_float(value, 0.0) or 0.0)
        for comp in comps:
            _add(vector, dof_map, node_id, comp, val)
        return True

    applied = False
    for i, key in enumerate(_FORCE_KEYS):
        raw = get_any(record, (key, key.upper()), None)
        if raw is None:
            continue
        val = float(as_float(raw, 0.0) or 0.0)
        if scale is not None:
            val *= scale
        _add(vector, dof_map, node_id, i, val)
        applied = True
    return applied


def build_load_vector(
    loads: Any,
    dof_map: DofMap,
    *,
    model: Any = None,
) -> np.ndarray:
    """Convert *loads* into a dense global load vector (or matrix of columns).

    Supported forms:

    * ``None`` -- read ``model.loads`` (or ``forces``);
    * an ``ndarray`` of length ``n_dof`` (or ``(n_dof, n_cases)``);
    * ``{(node_id, dof): value}`` where ``dof`` is a zero based index or a name
      such as ``"uz"``/``"fz"``;
    * ``{node_id: [fx, fy, fz, mx, my, mz]}`` (shorter sequences are padded);
    * a sequence of load records exposing ``node_id`` plus either
      ``as_dof_values()`` (the ``FEModel.Load`` protocol), ``force``/``moment``
      vectors, ``dof``/``value``, ``fx..mz`` or a ``vector``.
    """
    n = dof_map.n_dof
    if loads is None:
        loads = model_loads(model)
    if loads is None:
        return np.zeros(n)

    if isinstance(loads, np.ndarray):
        arr = np.asarray(loads, dtype=float)
        if arr.ndim == 1 and arr.size == n:
            return arr.copy()
        if arr.ndim == 2 and arr.shape[0] == n:
            return arr.copy()
        raise ValueError(f"load array of shape {arr.shape} does not match n_dof={n}")

    vector = np.zeros(n)
    if isinstance(loads, Mapping):
        for key, value in loads.items():
            if isinstance(key, tuple) and len(key) == 2:
                comp = normalize_dof(key[1], dofs_per_node=dof_map.dofs_per_node)
                _add(vector, dof_map, key[0], comp, float(value))
                continue
            if np.isscalar(value):
                raise ValueError(
                    f"ambiguous load entry {key!r}: give (node_id, dof) keys or a "
                    "per-node sequence of components"
                )
            arr = np.asarray(value, dtype=float).ravel()
            for i, val in enumerate(arr[: dof_map.dofs_per_node]):
                _add(vector, dof_map, key, i, val)
        return vector

    if isinstance(loads, (list, tuple, set)) and loads and all(
        isinstance(v, (int, float, np.number)) for v in loads
    ):
        arr = np.asarray(list(loads), dtype=float)
        if arr.size != n:
            raise ValueError(f"load list of length {arr.size} does not match n_dof={n}")
        return arr

    applied_any = False
    for _, record in iter_records(loads):
        if record is None:
            continue
        applied_any |= _apply_record(vector, dof_map, record)
    if not applied_any and not isinstance(loads, (list, tuple)):
        raise ValueError(f"could not interpret loads of type {type(loads).__name__}")
    return vector

"""Updating parameters: declaration, bounds handling and model application.

A :class:`Parameter` describes one scalar design/updating variable together with
how it maps onto an :class:`femtools.core.model.FEModel`.  The mapping is kept
deliberately duck-typed so that this package does not hard-depend on the core
database layout: a parameter either names a well known physical quantity
(``E``, ``rho``, ``nu``, ``thickness``, ``spring_k``, ``area``, ...) plus the ids
of the entities it acts on, or it carries an explicit ``setter`` callback.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Parameter",
    "ParameterSet",
    "as_parameters",
    "apply_parameters",
    "parameter_bounds",
    "clip_to_bounds",
    "unwrap_model",
]

# Containers an FE model database is expected to expose.
_MODEL_CONTAINERS = ("materials", "properties", "elements", "nodes")


def unwrap_model(obj: Any) -> Any:
    """Return the FE model carried by a project-style wrapper.

    ``femtools.io.project.load_project`` hands back a ``Project`` holding the
    database in ``.model``; readers and GUI state objects do the same.  Anything
    that already exposes the model containers is returned unchanged.
    """
    current = obj
    for _ in range(4):
        if current is None or any(hasattr(current, a) for a in _MODEL_CONTAINERS):
            return current
        nxt = getattr(current, "model", None)
        if nxt is None or nxt is current:
            return obj
        current = nxt
    return obj

# Canonical attribute names probed on materials/properties/elements for each kind.
_KIND_ATTRS: dict[str, tuple[str, ...]] = {
    "e": ("E", "e", "youngs_modulus", "young", "modulus"),
    "g": ("G", "g", "shear_modulus"),
    "nu": ("nu", "poisson", "poissons_ratio"),
    "rho": ("rho", "density"),
    "thickness": ("t", "thickness", "h"),
    "area": ("A", "area"),
    "spring_k": ("k", "K", "stiffness", "spring_k"),
    "damper_c": ("c", "C", "damping", "damper_c"),
    "mass": ("m", "mass"),
    "i": ("I", "Iyy", "Iy", "inertia"),
    "iy": ("Iy", "Iyy", "I1"),
    "iz": ("Iz", "Izz", "I2"),
    "j": ("J", "Jx", "torsion"),
}

# Which FEModel container a kind lives in, in probing order.
_KIND_CONTAINERS: dict[str, tuple[str, ...]] = {
    "e": ("materials",),
    "g": ("materials",),
    "nu": ("materials",),
    "rho": ("materials",),
    "thickness": ("properties",),
    "area": ("properties",),
    "spring_k": ("properties", "elements"),
    "damper_c": ("properties", "elements"),
    "mass": ("properties", "elements"),
    "i": ("properties",),
    "iy": ("properties",),
    "iz": ("properties",),
    "j": ("properties",),
}

_ALIASES = {
    "youngs_modulus": "e",
    "young": "e",
    "modulus": "e",
    "density": "rho",
    "t": "thickness",
    "h": "thickness",
    "a": "area",
    "k": "spring_k",
    "spring": "spring_k",
    "stiffness": "spring_k",
    "c": "damper_c",
    "damper": "damper_c",
}

# Entity-class names accepted by the ``container`` field / the ``type`` key of a
# parameter descriptor mapping.
_CONTAINER_ALIASES = {
    "material": "materials",
    "materials": "materials",
    "mat": "materials",
    "mats": "materials",
    "property": "properties",
    "properties": "properties",
    "prop": "properties",
    "props": "properties",
    "element": "elements",
    "elements": "elements",
    "elem": "elements",
    "elems": "elements",
    "node": "nodes",
    "nodes": "nodes",
}


def _normalize_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    return _ALIASES.get(key, key)


def _normalize_container(name: Any) -> str:
    key = str(name).strip().lower()
    try:
        return _CONTAINER_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown model container {name!r}; expected one of "
            f"{sorted(set(_CONTAINER_ALIASES.values()))}"
        ) from None


@dataclass
class Parameter:
    """One scalar updating parameter.

    Parameters
    ----------
    name:
        Human readable identifier (used as dictionary key in results).
    kind:
        Physical quantity: ``E``, ``rho``, ``nu``, ``thickness``, ``spring_k``,
        ``area``, ``mass``, ... or ``generic`` when a ``setter`` is supplied.
    value:
        Current / initial value.  For ``relative=True`` parameters this is a
        multiplier on the model's baseline value (starting at 1.0).
    lower, upper:
        Inclusive bounds.  ``-inf`` / ``+inf`` mean unbounded.
    target:
        Entity id or iterable of entity ids the parameter acts on.  ``None``
        means *every* entity of the relevant container.
    relative:
        When ``True`` the parameter multiplies the baseline model value instead
        of replacing it.  This is the numerically better conditioned form and is
        the default for design-variable style updating.
    setter:
        Optional ``setter(model, value, parameter)`` callback.  When given it
        fully overrides the ``kind``/``target`` machinery.
    scale:
        Optional scaling used to non-dimensionalise the parameter in the solver.
        Defaults to ``abs(value)`` (or 1.0).
    container:
        Restricts the search to one model container (``"materials"``,
        ``"properties"``, ``"elements"``, ``"nodes"``).  ``None`` probes the
        containers that normally carry ``kind``.
    """

    name: str
    kind: str = "generic"
    value: float = 1.0
    lower: float = -math.inf
    upper: float = math.inf
    target: Any = None
    relative: bool = False
    setter: Callable[[Any, float, Parameter], None] | None = None
    scale: float | None = None
    unit: str | None = None
    container: str | None = None

    def __post_init__(self) -> None:
        self.kind = _normalize_kind(self.kind)
        if self.container is not None:
            self.container = _normalize_container(self.container)
        self.value = float(self.value)
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        if self.lower > self.upper:
            raise ValueError(f"parameter {self.name!r}: lower > upper")

    # ------------------------------------------------------------------
    @property
    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    @property
    def effective_scale(self) -> float:
        if self.scale is not None and self.scale != 0.0:
            return float(abs(self.scale))
        if self.value != 0.0:
            return float(abs(self.value))
        return 1.0

    def clip(self, value: float) -> float:
        return float(min(max(float(value), self.lower), self.upper))

    def target_ids(self) -> list[Any] | None:
        if self.target is None:
            return None
        if isinstance(self.target, (str, bytes)):
            return [self.target]
        if isinstance(self.target, Iterable):
            return list(self.target)
        return [self.target]

    # ------------------------------------------------------------------
    def apply(self, model: Any, value: float, baseline: Mapping[int, float] | None = None) -> None:
        """Write ``value`` into ``model`` (in place)."""
        if self.setter is not None:
            self.setter(model, float(value), self)
            return
        entities = _resolve_entities(model, self)
        if not entities:
            raise KeyError(
                f"parameter {self.name!r}: no model entity matched "
                f"kind={self.kind!r} target={self.target!r}"
            )
        for key, obj, attr in entities:
            if self.relative:
                base = None if baseline is None else baseline.get(key)
                if base is None:
                    base = float(getattr(obj, attr))
                setattr(obj, attr, float(base) * float(value))
            else:
                setattr(obj, attr, float(value))

    def baseline(self, model: Any) -> dict[int, float]:
        """Snapshot of the model values this parameter controls."""
        if self.setter is not None:
            return {}
        out: dict[int, float] = {}
        for key, obj, attr in _resolve_entities(model, self):
            out[key] = float(getattr(obj, attr))
        return out


def _iter_container(model: Any, name: str) -> Iterator[tuple[Any, Any]]:
    container = getattr(model, name, None)
    if container is None:
        return
    if isinstance(container, Mapping):
        yield from container.items()
    elif isinstance(container, Iterable):
        for obj in container:
            yield getattr(obj, "id", id(obj)), obj


def _resolve_entities(model: Any, param: Parameter) -> list[tuple[Any, Any, str]]:
    """Return ``(key, object, attribute)`` triples the parameter writes to."""
    kind = param.kind
    attrs = _KIND_ATTRS.get(kind)
    if attrs is None:
        # Fall back to using the kind itself as the attribute name.
        attrs = (param.kind,)
    if param.container is not None:
        containers: tuple[str, ...] = (param.container,)
    else:
        containers = _KIND_CONTAINERS.get(kind, ("materials", "properties", "elements"))
    wanted = param.target_ids()
    found: list[tuple[Any, Any, str]] = []
    for cname in containers:
        for key, obj in _iter_container(model, cname):
            if wanted is not None and key not in wanted:
                continue
            for attr in attrs:
                val = getattr(obj, attr, None)
                if isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(
                    val, bool
                ):
                    found.append(((cname, key), obj, attr))
                    break
        if found:
            break
    return found


class ParameterSet(Sequence[Parameter]):
    """An ordered, list-like collection of :class:`Parameter`."""

    def __init__(self, parameters: Iterable[Parameter]):
        self._items: list[Parameter] = list(parameters)
        names = [p.name for p in self._items]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")

    # Sequence protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, str):
            for p in self._items:
                if p.name == index:
                    return p
            raise KeyError(index)
        if isinstance(index, slice):
            return ParameterSet(self._items[index])
        return self._items[index]

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParameterSet({[p.name for p in self._items]!r})"

    # Convenience -------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [p.name for p in self._items]

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self._items], dtype=float)

    @property
    def lower(self) -> np.ndarray:
        return np.array([p.lower for p in self._items], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([p.upper for p in self._items], dtype=float)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [p.bounds for p in self._items]

    @property
    def scales(self) -> np.ndarray:
        return np.array([p.effective_scale for p in self._items], dtype=float)

    def to_dict(self, values: ArrayLike | None = None) -> dict[str, float]:
        vals = self.values if values is None else np.asarray(values, dtype=float)
        return {p.name: float(v) for p, v in zip(self._items, vals, strict=True)}

    def clip(self, values: ArrayLike) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return np.clip(v, self.lower, self.upper)


# Descriptor-mapping key aliases, mapped onto ``Parameter`` field names.
_DESCRIPTOR_KEYS: dict[str, str] = {
    "name": "name",
    "kind": "kind",
    "quantity": "kind",
    "parameter": "kind",
    "attribute": "kind",
    "attr": "kind",
    "value": "value",
    "initial": "value",
    "start": "value",
    "x0": "value",
    "lower": "lower",
    "low": "lower",
    "min": "lower",
    "lb": "lower",
    "upper": "upper",
    "high": "upper",
    "max": "upper",
    "ub": "upper",
    "target": "target",
    "targets": "target",
    "id": "target",
    "ids": "target",
    "entity_id": "target",
    "entity_ids": "target",
    "mid": "target",
    "pid": "target",
    "eid": "target",
    "material_id": "target",
    "property_id": "target",
    "element_id": "target",
    "relative": "relative",
    "setter": "setter",
    "scale": "scale",
    "unit": "unit",
    "units": "unit",
    "container": "container",
    "type": "container",
    "entity": "container",
}


def _parameter_from_mapping(item: Mapping[str, Any], default_name: str) -> Parameter:
    """Build a :class:`Parameter` from a descriptor mapping.

    Beyond the plain ``Parameter`` field names this understands the entity
    descriptor style used throughout the examples and the CLI JSON files::

        {"type": "material", "id": 1, "name": "E", "lower": 0.5, "upper": 2.0}

    Here ``type`` names the model container, ``id`` the entity and ``name`` the
    physical quantity.  A descriptor that does not state an explicit ``value``
    is interpreted as a *multiplier* on the model's own value, starting at 1.0 —
    which is what bounds such as ``(0.5, 2.0)`` imply.
    """
    kwargs: dict[str, Any] = {}
    for key, val in item.items():
        field_name = _DESCRIPTOR_KEYS.get(str(key).strip().lower())
        if field_name is None:
            raise TypeError(
                f"unknown parameter descriptor key {key!r}; expected one of "
                f"{sorted(set(_DESCRIPTOR_KEYS))}"
            )
        if field_name in kwargs and kwargs[field_name] != val:
            raise TypeError(f"parameter descriptor sets {field_name!r} twice")
        kwargs[field_name] = val
    kwargs.setdefault("name", default_name)
    if "kind" not in kwargs:
        kwargs["kind"] = _guess_kind(str(kwargs["name"]))
    if "relative" not in kwargs and "value" not in kwargs and kwargs.get("setter") is None:
        # No absolute starting value was supplied, so the parameter can only be
        # meant as a multiplier on whatever the model already carries.
        kwargs["relative"] = True
    return Parameter(**kwargs)


def _deduplicate_names(params: list[Parameter]) -> list[Parameter]:
    """Disambiguate repeated names by appending the entity id, then an index."""
    counts: dict[str, int] = {}
    for p in params:
        counts[p.name] = counts.get(p.name, 0) + 1
    if all(n == 1 for n in counts.values()):
        return params
    seen: set[str] = set()
    for i, p in enumerate(params):
        if counts[p.name] == 1:
            seen.add(p.name)
            continue
        base = p.name
        candidate = f"{base}_{p.target}" if p.target is not None else f"{base}_{i + 1}"
        if candidate in seen:
            candidate = f"{base}_{i + 1}"
        p.name = candidate
        seen.add(candidate)
    return params


def as_parameters(spec: Any) -> ParameterSet:
    """Coerce a user-supplied parameter specification into a :class:`ParameterSet`.

    Accepted forms::

        ParameterSet(...)                       # passthrough
        [Parameter(...), ...]
        {"E1": 210e9, "E2": 200e9}              # name -> initial value
        {"E1": {"kind": "E", "value": 2.1e11, "lower": ..., "target": 1}}
        [{"type": "material", "id": 1, "name": "E", "lower": 0.5, "upper": 2.0}]
        [("E", 1, 210e9), ...]                  # (kind, target, value)
        ["E", "rho"]                            # names only, value 1.0 (relative)
        3                                       # 3 anonymous unit-valued params
    """
    if isinstance(spec, ParameterSet):
        return spec
    if isinstance(spec, Parameter):
        return ParameterSet([spec])
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        n = int(spec)
        return ParameterSet([Parameter(name=f"p{i + 1}") for i in range(n)])
    if isinstance(spec, Mapping):
        out: list[Parameter] = []
        for name, val in spec.items():
            if isinstance(val, Parameter):
                p = copy.copy(val)
                p.name = p.name or str(name)
                out.append(p)
            elif isinstance(val, Mapping):
                p = _parameter_from_mapping(val, str(name))
                p.name = str(name)
                out.append(p)
            else:
                out.append(
                    Parameter(name=str(name), kind=_guess_kind(str(name)), value=float(val))
                )
        return ParameterSet(out)
    if isinstance(spec, Iterable):
        out = []
        for i, item in enumerate(spec):
            if isinstance(item, Parameter):
                out.append(item)
            elif isinstance(item, Mapping):
                out.append(_parameter_from_mapping(item, f"p{i + 1}"))
            elif isinstance(item, str):
                out.append(Parameter(name=item, kind=_guess_kind(item), relative=True))
            elif isinstance(item, (tuple, list)):
                kind = str(item[0])
                target = item[1] if len(item) > 1 else None
                value = float(item[2]) if len(item) > 2 else 1.0
                out.append(
                    Parameter(
                        name=f"{kind}_{target}" if target is not None else f"{kind}",
                        kind=kind,
                        target=target,
                        value=value,
                    )
                )
            elif isinstance(item, (int, float, np.floating)):
                out.append(Parameter(name=f"p{i + 1}", value=float(item)))
            else:
                raise TypeError(f"cannot interpret parameter spec element {item!r}")
        return ParameterSet(_deduplicate_names(out))
    raise TypeError(f"cannot interpret parameter specification {spec!r}")


def _guess_kind(name: str) -> str:
    low = name.strip().lower()
    for kind in ("thickness", "spring_k", "damper_c", "rho", "area", "mass", "nu"):
        if low.startswith(kind) or kind in low:
            return kind
    if low.startswith("e") or low.startswith("young"):
        return "e"
    if low.startswith("t"):
        return "thickness"
    if low.startswith("k"):
        return "spring_k"
    if low.startswith("a"):
        return "area"
    return "generic"


def apply_parameters(
    model: Any,
    parameters: ParameterSet | Iterable[Parameter],
    values: ArrayLike | None = None,
    *,
    copy_model: bool = True,
    baseline: Mapping[str, Mapping[int, float]] | None = None,
) -> Any:
    """Return a model with ``values`` written into it.

    ``copy_model=True`` (default) deep-copies so the caller's model is untouched.
    """
    pset = parameters if isinstance(parameters, ParameterSet) else ParameterSet(parameters)
    vals = pset.values if values is None else np.asarray(values, dtype=float)
    if len(vals) != len(pset):
        raise ValueError(f"expected {len(pset)} values, got {len(vals)}")
    target = copy.deepcopy(model) if copy_model else model
    for p, v in zip(pset, vals, strict=True):
        base = None if baseline is None else baseline.get(p.name)
        p.apply(target, float(v), baseline=base)
    return target


def snapshot_baseline(model: Any, parameters: ParameterSet) -> dict[str, dict[int, float]]:
    """Capture the pre-update model values controlled by each parameter."""
    return {p.name: p.baseline(model) for p in parameters}


def parameter_bounds(
    parameters: ParameterSet | Iterable[Parameter],
    bounds: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve effective (lower, upper) arrays.

    ``bounds`` may be ``None`` (use the parameters' own bounds), a single
    ``(lo, hi)`` pair applied to all, a sequence of pairs, or a mapping
    ``{name: (lo, hi)}``.
    """
    pset = parameters if isinstance(parameters, ParameterSet) else ParameterSet(parameters)
    lo = pset.lower.copy()
    hi = pset.upper.copy()
    if bounds is None:
        return lo, hi
    if isinstance(bounds, Mapping):
        for i, p in enumerate(pset):
            if p.name in bounds:
                b = bounds[p.name]
                lo[i], hi[i] = _pair(b)
        return lo, hi
    arr = list(bounds)
    if len(arr) == 2 and all(np.isscalar(x) or x is None for x in arr):
        lo_v, hi_v = _pair(arr)
        return np.full(len(pset), lo_v), np.full(len(pset), hi_v)
    if len(arr) != len(pset):
        raise ValueError(f"bounds length {len(arr)} != number of parameters {len(pset)}")
    for i, b in enumerate(arr):
        lo[i], hi[i] = _pair(b)
    return lo, hi


def _pair(b: Any) -> tuple[float, float]:
    if b is None:
        return (-math.inf, math.inf)
    lo, hi = b
    lo = -math.inf if lo is None else float(lo)
    hi = math.inf if hi is None else float(hi)
    return lo, hi


def clip_to_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lower, upper)


@dataclass
class ParameterReport:
    """Per-parameter before/after summary produced by :func:`update_model`."""

    name: str
    initial: float
    final: float
    lower: float
    upper: float
    unit: str | None = None
    reference: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def change(self) -> float:
        return self.final - self.initial

    @property
    def relative_change(self) -> float:
        return self.change / self.initial if self.initial else math.inf

    @property
    def error_vs_reference(self) -> float | None:
        if self.reference in (None, 0.0):
            return None
        return (self.final - self.reference) / self.reference

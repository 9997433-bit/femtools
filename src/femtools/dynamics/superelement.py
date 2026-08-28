"""Persist a reduced component (superelement) to a single ``.npz`` file.

A component mode synthesis run is the expensive half of a substructured analysis: the
parent model may be millions of DOFs, the reduced one is a few dozen. Handing the reduced
component to the next analysis — a system assembly, a coupled eigen solve, a load case
run somewhere else entirely — should not mean re-solving the parent, and that is all this
module does. :func:`dump_cms` writes the reduced matrices and the reduction basis,
:func:`load_cms` reads them back into an object of the same class, so a
:class:`~femtools.dynamics.craig_bampton.CraigBamptonResult` that has been through disk
still answers :meth:`~femtools.dynamics.craig_bampton.CraigBamptonResult.solve_modes` and
expands its modes to the parent's DOF space.

What is stored is exactly what the reduced component *is*: ``K``, ``M``, the basis ``T``
and the boundary/interior DOF identifiers, plus whatever else the source class carries
(fixed-interface frequencies and constraint modes for Craig-Bampton, the free-interface
and residual sets for MacNeal/Rubin) and the ``meta`` dictionary as JSON. Everything is
written as raw ``float64``, so ``K`` and ``M`` come back **bit-identical** — a reduced
model that changed in the last bits between the run that produced it and the run that
consumes it is a reduced model whose frequencies cannot be compared with anyone else's.

The format is deliberately plain: an ``npz`` archive with one array per field and a
``format`` tag. Anything that can read ``numpy`` archives can read a femtools
superelement without importing femtools, which is the point of writing it out at all.

    dump_cms(craig_bampton(K, M, boundary, 12), "wing.npz")
    wing = load_cms("wing.npz")          # a CraigBamptonResult again
    wing.solve_modes().freq_hz[:6]

A duck-typed reduction that is neither of the two known classes but exposes ``K``, ``M``
and ``T`` is written too, and comes back as a :class:`Superelement` — the minimum object
that can still be expanded and eigen-solved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI, as_dense, dumps_meta, get_field, json_meta, npz_path, npz_text
from .cms_free import FreeCMSResult, _semidefinite_modes
from .craig_bampton import CraigBamptonResult
from .modal import ModalModel

__all__ = ["Superelement", "dump_cms", "load_cms"]

#: Tag written into every archive; ``load_cms`` refuses anything else.
FORMAT = "femtools.dynamics.superelement/1"

_MATRICES = ("K", "M", "T")
_DOF_SETS = ("boundary_dofs", "interior_dofs")
#: Extra fields carried by each known reduction, restored so the class round-trips whole.
_EXTRA: dict[str, tuple[str, ...]] = {
    "craig_bampton": ("fixed_freq_hz", "constraint_modes", "fixed_modes"),
    "free_cms": ("free_freq_hz", "residual_freq_hz", "residual_flexibility"),
    "superelement": (),
}


@dataclass
class Superelement:
    """A reduced component with no class of its own: matrices, basis and DOF sets.

    This is what :func:`load_cms` returns for an archive that did not come from
    :func:`~femtools.dynamics.craig_bampton.craig_bampton`, :func:`~femtools.dynamics.
    cms_free.rubin` or :func:`~femtools.dynamics.cms_free.macneal`, and it is the smallest
    object that is still useful: it can be expanded to the parent DOF space and it can be
    eigen-solved. The solve tolerates a singular ``M`` the way MacNeal's reduction needs.

    Attributes
    ----------
    K, M:
        Reduced stiffness and mass, shape ``(n_reduced, n_reduced)``.
    T:
        Reduction basis in the parent DOF ordering, shape ``(ndof, n_reduced)``.
    boundary_dofs, interior_dofs:
        Index arrays into the parent DOF numbering.
    method:
        Free-text provenance tag, e.g. ``"rubin"``.
    """

    K: np.ndarray
    M: np.ndarray
    T: np.ndarray
    boundary_dofs: np.ndarray
    interior_dofs: np.ndarray
    method: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_boundary(self) -> int:
        """Number of retained boundary DOFs."""
        return int(self.boundary_dofs.size)

    @property
    def n_reduced(self) -> int:
        """Size of the reduced model."""
        return int(self.K.shape[0])

    @property
    def ndof(self) -> int:
        """Size of the parent (full) model."""
        return int(self.T.shape[0])

    def expand(self, y: np.ndarray) -> np.ndarray:
        """Expand reduced coordinates back to full physical DOFs (``u = T y``)."""
        return self.T @ np.asarray(y)

    def solve_modes(self, n_modes: int | None = None) -> ModalModel:
        """Eigen-solve the reduced model and expand the modes to the parent DOF space."""
        lam, Q = _semidefinite_modes(self.K, self.M)
        if n_modes is not None:
            lam, Q = lam[: int(n_modes)], Q[:, : int(n_modes)]
        return ModalModel(
            freq_hz=np.sqrt(lam) / TWO_PI,
            modes=self.T @ Q,
            generalized_mass=np.ones(lam.size),
            eigenvalues=lam,
            meta={"source": f"superelement.{self.method}" if self.method else "superelement"},
        )


def _classify(result: Any) -> str:
    if isinstance(result, CraigBamptonResult):
        return "craig_bampton"
    if isinstance(result, FreeCMSResult):
        return "free_cms"
    return "superelement"


def dump_cms(result: Any, path: Any, *, compress: bool = False, meta: Any = None) -> Any:
    """Write a reduced component to an ``.npz`` archive.

    Parameters
    ----------
    result:
        A :class:`~femtools.dynamics.craig_bampton.CraigBamptonResult`, a
        :class:`~femtools.dynamics.cms_free.FreeCMSResult` (Rubin or MacNeal), or any
        object or mapping exposing at least ``K``, ``M`` and ``T``. ``boundary_dofs`` and
        ``interior_dofs`` are stored when present and default to empty otherwise.
    path:
        Destination. A ``str`` or path-like without an ``.npz`` suffix gets one, and the
        resolved :class:`~pathlib.Path` is returned; an open binary file object is written
        to as-is and returned unchanged.
    compress:
        Use ``np.savez_compressed``. The basis ``T`` of a real reduction is mostly zeros,
        so this is usually worth it; it is off by default because it is not free and the
        bits that come back are identical either way.
    meta:
        Extra entries merged into the stored ``meta`` mapping, overriding the source's own.
        Anything JSON cannot represent is stored as its ``str``, and tuples come back as
        lists — ``meta`` is provenance, not data.

    Returns
    -------
    pathlib.Path or file object
        Where the archive was written.

    Raises
    ------
    TypeError
        If ``result`` carries no ``K``, ``M`` and ``T``.
    """
    kind = _classify(result)
    fields: dict[str, np.ndarray] = {}
    for name in _MATRICES:
        value = get_field(result, name)
        if value is None:
            raise TypeError(
                f"{type(result).__name__} has no {name!r}; a reduced component must "
                "carry the reduced stiffness K, the reduced mass M and the reduction "
                "basis T to be worth storing"
            )
        fields[name] = as_dense(value)
    if fields["K"].shape != fields["M"].shape or fields["K"].ndim != 2:
        raise ValueError(
            f"K and M must be square and equally sized, got {fields['K'].shape} and "
            f"{fields['M'].shape}"
        )
    if fields["T"].ndim != 2 or fields["T"].shape[1] != fields["K"].shape[0]:
        raise ValueError(
            f"T must have one column per reduced coordinate ({fields['K'].shape[0]}), "
            f"got shape {fields['T'].shape}"
        )

    for name in _DOF_SETS:
        value = get_field(result, name)
        fields[name] = (
            np.zeros(0, dtype=np.int64)
            if value is None
            else np.asarray(value, dtype=np.int64).reshape(-1)
        )
    for name in _EXTRA[kind]:
        value = get_field(result, name)
        if value is not None:
            fields[name] = as_dense(value)

    stored_meta = json_meta(result, meta)
    method = str(get_field(result, "method") or "")

    payload: dict[str, Any] = dict(fields)
    payload["format"] = np.array(FORMAT)
    payload["kind"] = np.array(kind)
    payload["method"] = np.array(method)
    payload["source_class"] = np.array(type(result).__name__)
    payload["meta_json"] = np.array(dumps_meta(stored_meta))

    target = npz_path(path)
    save = np.savez_compressed if compress else np.savez
    save(target if target is not None else path, **payload)
    return target if target is not None else path


def load_cms(path: Any) -> Any:
    """Read a reduced component back from an ``.npz`` archive written by :func:`dump_cms`.

    The class the archive came from is restored, so a Craig-Bampton component comes back
    as a :class:`~femtools.dynamics.craig_bampton.CraigBamptonResult` and a Rubin or
    MacNeal component as a :class:`~femtools.dynamics.cms_free.FreeCMSResult`, both with
    their ``solve_modes`` / ``expand`` behaviour intact. Anything else comes back as a
    :class:`Superelement`.

    ``K``, ``M`` and ``T`` are bit-identical to what was written; ``meta`` round-trips
    through JSON, so its tuples arrive as lists.

    Parameters
    ----------
    path:
        Source archive: a path, a path without its ``.npz`` suffix, or an open binary
        file object.

    Returns
    -------
    CraigBamptonResult, FreeCMSResult or Superelement

    Raises
    ------
    ValueError
        If the archive was not written by :func:`dump_cms`, or has lost a field the
        class it claims to be needs.
    """
    target = npz_path(path)
    with np.load(target if target is not None else path, allow_pickle=False) as data:
        tag = npz_text(data, "format")
        if tag != FORMAT:
            raise ValueError(
                f"{path!r} is not a femtools superelement archive (format tag "
                f"{tag or 'absent'!r}, expected {FORMAT!r})"
            )
        kind = npz_text(data, "kind", "superelement")
        if kind not in _EXTRA:
            raise ValueError(f"unknown superelement kind {kind!r} in {path!r}")
        arrays = {name: np.array(data[name]) for name in (*_MATRICES, *_DOF_SETS)}
        missing = [name for name in _EXTRA[kind] if name not in data.files]
        if missing:
            raise ValueError(
                f"{path!r} claims to be a {kind} component but is missing "
                f"{', '.join(missing)}"
            )
        arrays.update({name: np.array(data[name]) for name in _EXTRA[kind]})
        method = npz_text(data, "method")
        meta_text = npz_text(data, "meta_json", "{}")

    meta = dict(json.loads(meta_text or "{}"))
    meta["loaded_from"] = str(target) if target is not None else repr(path)
    boundary = arrays.pop("boundary_dofs")
    interior = arrays.pop("interior_dofs")

    if kind == "craig_bampton":
        return CraigBamptonResult(
            K=arrays["K"],
            M=arrays["M"],
            T=arrays["T"],
            boundary_dofs=boundary,
            interior_dofs=interior,
            fixed_freq_hz=arrays["fixed_freq_hz"],
            constraint_modes=arrays["constraint_modes"],
            fixed_modes=arrays["fixed_modes"],
            meta=meta,
        )
    if kind == "free_cms":
        return FreeCMSResult(
            K=arrays["K"],
            M=arrays["M"],
            T=arrays["T"],
            boundary_dofs=boundary,
            interior_dofs=interior,
            free_freq_hz=arrays["free_freq_hz"],
            residual_freq_hz=arrays["residual_freq_hz"],
            residual_flexibility=arrays["residual_flexibility"],
            method=method or "rubin",
            meta=meta,
        )
    return Superelement(
        K=arrays["K"],
        M=arrays["M"],
        T=arrays["T"],
        boundary_dofs=boundary,
        interior_dofs=interior,
        method=method,
        meta=meta,
    )

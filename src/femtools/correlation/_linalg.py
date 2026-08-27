"""Internal array helpers shared by the correlation and pretest packages.

Kept private (no public re-export) so the frozen public contract stays small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_mode_matrix",
    "mode_source",
    "mode_frequencies",
    "row_index",
    "same_array",
    "weighted",
    "column_norms_sq",
    "safe_divide",
]

#: Attributes searched when unwrapping a modal result object.
_MODE_ATTRS = ("modes", "phi", "mode_shapes", "shapes", "eigenvectors")

#: Attributes searched when unwrapping the frequencies of a modal result.
_FREQ_ATTRS = ("freq_hz", "frequencies", "freq")


def mode_source(obj: Any) -> Any:
    """Unwrap a modal result to its mode shape matrix.

    Anything array-like passes through unchanged; an object exposing
    ``modes`` / ``phi`` (:class:`femtools.fea.eigen.ModalResult` and the
    equivalents produced by the readers) yields that array, so the whole
    correlation and pretest API accepts a solver result where it documents a
    ``(n_dof, n_mode)`` array.
    """
    if obj is None or isinstance(obj, (np.ndarray, list, tuple)):
        return obj
    for name in _MODE_ATTRS:
        value = getattr(obj, name, None)
        if value is not None and not callable(value):
            return value
    return obj


def mode_frequencies(obj: Any) -> NDArray[np.float64] | None:
    """Natural frequencies [Hz] carried by a modal result, if any."""
    if obj is None or isinstance(obj, (np.ndarray, list, tuple)):
        return None
    for name in _FREQ_ATTRS:
        value = getattr(obj, name, None)
        if value is not None and not callable(value):
            return np.asarray(value, dtype=float).reshape(-1)
    return None


def as_mode_matrix(phi: ArrayLike, name: str = "phi") -> NDArray[Any]:
    """Return ``phi`` as a 2-D ``(n_dof, n_mode)`` array.

    A 1-D input is interpreted as a single mode shape (one column); a modal
    result object is unwrapped by :func:`mode_source` first.
    """
    arr = np.asarray(mode_source(phi))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        arr = arr.astype(float)
    return arr


def row_index(index: Any, size: int | None = None) -> NDArray[np.intp]:
    """Normalize a row selector to integer positions.

    A boolean array is a *mask* and is expanded with :func:`numpy.flatnonzero`;
    casting it to positions would silently turn ``[False, True]`` into rows
    ``0`` and ``1``.  Anything else is taken as integer positions, negative
    values included.
    """
    arr = np.asarray(index)
    if arr.dtype == bool:
        if size is not None and arr.size != size:
            raise ValueError(f"boolean mask has {arr.size} entries, expected {size}")
        return np.flatnonzero(arr).astype(np.intp)
    return arr.reshape(-1).astype(np.intp)


def same_array(a: NDArray[Any], b: NDArray[Any] | None) -> bool:
    """True when ``b`` is (or numerically equals) ``a``.

    Used to trigger the exact-symmetry branch of the assurance criteria so a
    self-comparison has a bit-exact unit diagonal instead of ``1 - 1e-16``.
    """
    if b is None or a is b:
        return True
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def weighted(w: Any, x: NDArray[Any]) -> NDArray[Any]:
    """Apply a weighting operator ``w`` to the columns of ``x``.

    ``w`` may be ``None`` (identity), a scalar, a 1-D array of diagonal
    weights, a dense 2-D matrix, or any object exposing ``__matmul__``
    (e.g. a ``scipy.sparse`` matrix).
    """
    if w is None:
        return x
    if np.isscalar(w):
        return np.asarray(w) * x
    if isinstance(w, np.ndarray):
        if w.ndim == 1:
            if w.shape[0] != x.shape[0]:
                raise ValueError(
                    f"diagonal weight length {w.shape[0]} does not match {x.shape[0]} rows"
                )
            return w[:, None] * x
        if w.ndim != 2:
            raise ValueError(f"weight must be 1-D or 2-D, got shape {w.shape}")
        if w.shape[1] != x.shape[0]:
            raise ValueError(f"weight shape {w.shape} does not match {x.shape[0]} rows")
        return w @ x
    result = w @ x
    return np.asarray(result)


def column_norms_sq(x: NDArray[Any], wx: NDArray[Any] | None = None) -> NDArray[Any]:
    """Return ``diag(x^H w x)`` as a real 1-D array."""
    if wx is None:
        wx = x
    val = np.einsum("ij,ij->j", x.conj(), wx)
    return np.real(val)


def safe_divide(num: NDArray[Any], den: NDArray[Any]) -> NDArray[Any]:
    """Element-wise ``num / den`` with ``0`` wherever ``den`` is non-positive.

    Null mode shapes (all-zero columns) would otherwise yield NaN.
    """
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    good = den > 0.0
    np.divide(num, den, out=out, where=good)
    return out

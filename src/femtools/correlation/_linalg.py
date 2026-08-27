"""Internal array helpers shared by the correlation and pretest packages.

Kept private (no public re-export) so the frozen public contract stays small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_mode_matrix",
    "same_array",
    "weighted",
    "column_norms_sq",
    "safe_divide",
]


def as_mode_matrix(phi: ArrayLike, name: str = "phi") -> NDArray[Any]:
    """Return ``phi`` as a 2-D ``(n_dof, n_mode)`` array.

    A 1-D input is interpreted as a single mode shape (one column).
    """
    arr = np.asarray(phi)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        arr = arr.astype(float)
    return arr


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

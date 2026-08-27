"""Internal helpers shared by the :mod:`femtools.dynamics` modules.

Nothing in here is part of the public contract; the public surface is re-exported
from :mod:`femtools.dynamics`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "as_dense",
    "broadcast_scalar",
    "factorized_solver",
    "is_sparse",
    "resolve_dofs",
    "symmetrize",
]

TWO_PI = 2.0 * np.pi


def is_sparse(a: Any) -> bool:
    """True for any scipy sparse array/matrix."""
    return sp.issparse(a)


def as_dense(a: Any, dtype: Any = float) -> np.ndarray:
    """Return ``a`` as a dense C-contiguous ndarray of ``dtype``."""
    if a is None:
        raise TypeError("expected an array, got None")
    if sp.issparse(a):
        a = a.toarray()
    return np.ascontiguousarray(np.asarray(a, dtype=dtype))


def symmetrize(a: np.ndarray) -> np.ndarray:
    """Return the symmetric part ``(A + A.T) / 2``, killing round-off asymmetry."""
    return 0.5 * (a + a.T)


def broadcast_scalar(value: Any, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or length-``n`` sequence to a float array of length ``n``."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(n, float(arr.reshape(-1)[0]))
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or length {n}, got size {arr.size}")
    return arr.astype(float, copy=False).reshape(n)


def resolve_dofs(
    spec: Any,
    ndof: int,
    name: str = "dofs",
    dof_ids: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Normalise a DOF selection to a 1-D array of integer indices in ``[0, ndof)``.

    Accepted forms:

    * ``None``     -> all DOFs, ``arange(ndof)``
    * ``int``      -> single DOF
    * ``slice``    -> sliced range
    * sequence of ints -> used directly
    * boolean mask of length ``ndof``
    * sequence of DOF *labels* when ``dof_ids`` is supplied
    """
    if spec is None:
        return np.arange(ndof, dtype=int)
    if isinstance(spec, slice):
        return np.arange(ndof, dtype=int)[spec]
    arr = np.asarray(spec)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype == bool:
        if arr.size != ndof:
            raise ValueError(f"{name} boolean mask must have length {ndof}, got {arr.size}")
        return np.flatnonzero(arr).astype(int)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.asarray(arr, dtype=int)
    idx = arr.astype(int).reshape(-1)
    if dof_ids is not None:
        labels = np.asarray(dof_ids, dtype=int)
        lookup = {int(label): i for i, label in enumerate(labels)}
        if all(int(v) in lookup for v in idx) and not np.array_equal(
            labels, np.arange(labels.size)
        ):
            idx = np.array([lookup[int(v)] for v in idx], dtype=int)
    if idx.size and (idx.min() < -ndof or idx.max() >= ndof):
        raise IndexError(f"{name} out of range for {ndof} DOFs: {idx}")
    return np.where(idx < 0, idx + ndof, idx)


def factorized_solver(A: Any):  # noqa: N803 - matrix name follows the maths
    """Return a callable solving ``A x = b`` for dense or sparse ``A``.

    Falls back to a least-squares / pseudo-inverse solve when ``A`` is singular so
    that free-free models do not blow up the caller.
    """
    if sp.issparse(A):
        Acsc = sp.csc_matrix(A)
        try:
            lu = spla.splu(Acsc)
        except (RuntimeError, ValueError):
            dense = Acsc.toarray()
            pinv = np.linalg.pinv(dense)
            return lambda b: pinv @ np.asarray(b)
        return lambda b: lu.solve(np.asarray(b, dtype=float))

    dense = as_dense(A)
    try:
        from scipy.linalg import lu_factor, lu_solve

        piv = lu_factor(dense)
        cond_ok = np.isfinite(piv[0]).all() and np.abs(np.diag(piv[0])).min() > 0.0
        if not cond_ok:
            raise np.linalg.LinAlgError("singular")
    except (np.linalg.LinAlgError, ValueError):
        pinv = np.linalg.pinv(dense)
        return lambda b: pinv @ np.asarray(b)
    return lambda b: lu_solve(piv, np.asarray(b, dtype=float))

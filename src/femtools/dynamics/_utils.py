"""Internal helpers shared by the :mod:`femtools.dynamics` modules.

Nothing in here is part of the public contract; the public surface is re-exported
from :mod:`femtools.dynamics`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "SINGULAR_RCOND",
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


#: A matrix whose reciprocal condition number falls to this level is singular to working
#: precision: the LU solve still returns *a* vector, but none of its digits are the
#: caller's, and the null-space content it picks up is arbitrary.
SINGULAR_RCOND = float(np.finfo(float).eps)


def _pinv_solver(dense: np.ndarray, why: str):
    """Minimum-norm solver for a matrix that LU cannot be trusted on."""
    warnings.warn(
        f"the matrix is {why}, so the solve falls back to the minimum-norm "
        "pseudo-inverse solution; a component floating on a mechanism (an unrestrained "
        "interior partition, a free-free stiffness) is the usual cause",
        RuntimeWarning,
        stacklevel=3,
    )
    pinv = np.linalg.pinv(dense)
    return lambda b: pinv @ np.asarray(b)


def _reciprocal_condition(A: np.ndarray, lu: np.ndarray) -> float:  # noqa: N803
    """LAPACK 1-norm reciprocal condition estimate of ``A`` from its LU factor."""
    diag = np.abs(np.diag(lu))
    if not np.isfinite(lu).all() or diag.size == 0 or diag.min() == 0.0:
        return 0.0
    anorm = float(np.linalg.norm(A, 1))
    if not np.isfinite(anorm) or anorm == 0.0:
        return 0.0
    from scipy.linalg.lapack import get_lapack_funcs

    (gecon,) = get_lapack_funcs(("gecon",), (lu,))
    rcond, info = gecon(lu, anorm)
    return float(rcond) if info == 0 and np.isfinite(rcond) else 0.0


def factorized_solver(A: Any, *, rcond: float = SINGULAR_RCOND):  # noqa: N803
    """Return a callable solving ``A x = b`` for dense or sparse ``A``.

    Falls back to a minimum-norm pseudo-inverse solve when ``A`` is singular, so that a
    free-free model does not blow up the caller. That promise needs the singularity to be
    *detected*, which an exactly-zero pivot does not do: LU on a numerically singular
    matrix normally produces a pivot around ``1e-16`` rather than ``0``, returns a vector
    that satisfies ``A x = b`` to round-off, and hides an arbitrary multiple of the null
    space inside it — arbitrary enough that the dense and the sparse path of this very
    function used to disagree by a factor of three on the same free-free chain. The
    factorisation is therefore accepted only when LAPACK's reciprocal condition estimate
    (its sparse proxy: the spread of the ``U`` diagonal) stays above ``rcond``.
    """
    if sp.issparse(A):
        Acsc = sp.csc_matrix(A)
        try:
            lu = spla.splu(Acsc)
        except (RuntimeError, ValueError):
            return _pinv_solver(Acsc.toarray(), "exactly singular")
        pivots = np.abs(lu.U.diagonal())
        if pivots.size and (
            not np.isfinite(pivots).all()
            or pivots.max() <= 0.0
            or pivots.min() <= rcond * pivots.max()
        ):
            return _pinv_solver(Acsc.toarray(), "singular to working precision")
        return lambda b: lu.solve(np.asarray(b, dtype=float))

    dense = as_dense(A)
    from scipy.linalg import lu_factor, lu_solve

    try:
        with warnings.catch_warnings():
            # LinAlgWarning duplicates the check below, which reports it properly.
            warnings.simplefilter("ignore")
            piv = lu_factor(dense)
        if _reciprocal_condition(dense, piv[0]) <= rcond:
            raise np.linalg.LinAlgError("singular to working precision")
    except (np.linalg.LinAlgError, ValueError):
        return _pinv_solver(dense, "singular to working precision")
    return lambda b: lu_solve(piv, np.asarray(b, dtype=float))

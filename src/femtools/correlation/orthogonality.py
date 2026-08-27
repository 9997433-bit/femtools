"""(Pseudo-)orthogonality checks between two sets of mode shapes.

The cross-orthogonality matrix ``Phi_a^H M Phi_b`` is the mass-weighted
counterpart of the MAC.  For a correlated pair of mass-normalized bases it
tends to the identity matrix; the off-diagonal magnitude is the usual
acceptance metric (< 0.1 for a well correlated test/analysis pair).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import as_mode_matrix, column_norms_sq, same_array, weighted

__all__ = [
    "cross_orthogonality",
    "auto_orthogonality",
    "off_diagonal_max",
    "orthogonality_error",
]


def cross_orthogonality(
    phi_a: ArrayLike,
    phi_b: ArrayLike | None = None,
    mass: Any = None,
    *,
    normalize: bool = True,
    absolute: bool = False,
) -> NDArray[Any]:
    """Cross-orthogonality (pseudo-orthogonality) matrix ``Phi_a^H M Phi_b``.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shape matrices ``(n_dof, n_mode)``.  Both must be expressed on the
        same (test) DOF set; use :mod:`femtools.correlation.dofmap` to align
        them first.  ``phi_b=None`` compares ``phi_a`` with itself.
    mass:
        Mass matrix (or reduced/TAM mass matrix) of shape ``(n_dof, n_dof)``.
        A 1-D array is treated as a lumped/diagonal mass, ``None`` as identity.
        Any object supporting ``@`` (e.g. ``scipy.sparse``) is accepted.
    normalize:
        Divide by ``sqrt((a^H M a)(b^H M b))`` so the result is independent of
        the modal scaling.  With ``False`` the raw generalized-mass matrix is
        returned (identity for mass-normalized, correlated modes).
    absolute:
        Return magnitudes instead of signed values.

    Returns
    -------
    ndarray
        ``(n_mode_a, n_mode_b)`` matrix.  Real for real inputs.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = a if phi_b is None else as_mode_matrix(phi_b, "phi_b")
    self_case = phi_b is None or same_array(a, b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"phi_a has {a.shape[0]} DOF but phi_b has {b.shape[0]}; align the DOF sets first"
        )

    mb = weighted(mass, b)
    if mb.shape != b.shape:
        raise ValueError(f"mass operator produced shape {mb.shape}, expected {b.shape}")
    cross = a.conj().T @ mb

    if normalize:
        ma = mb if self_case else weighted(mass, a)
        na = column_norms_sq(a, ma)
        nb = column_norms_sq(b, mb)
        scale = np.sqrt(np.abs(np.outer(na, nb)))
        scaled = np.zeros_like(cross)
        np.divide(cross, scale, out=scaled, where=scale > 0.0)
        cross = scaled
        if self_case:
            cross = 0.5 * (cross + cross.conj().T)
            np.fill_diagonal(cross, np.where(na > 0.0, 1.0, 0.0))

    return np.abs(cross) if absolute else cross


def auto_orthogonality(
    phi: ArrayLike, mass: Any = None, *, normalize: bool = True, absolute: bool = False
) -> NDArray[Any]:
    """Orthogonality of a mode set against itself (``Phi^H M Phi``)."""
    return cross_orthogonality(phi, None, mass, normalize=normalize, absolute=absolute)


def off_diagonal_max(matrix: ArrayLike) -> float:
    """Largest off-diagonal magnitude of a (possibly rectangular) matrix."""
    m = np.abs(np.asarray(matrix))
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {m.shape}")
    if m.size == 0:
        return 0.0
    mask = ~np.eye(m.shape[0], m.shape[1], dtype=bool)
    return float(m[mask].max()) if mask.any() else 0.0


def orthogonality_error(phi: ArrayLike, mass: Any = None) -> float:
    """``max |Phi^H M Phi - I|`` — mass-normalization quality of a mode set."""
    gram = auto_orthogonality(phi, mass, normalize=False)
    if gram.size == 0:
        return 0.0
    return float(np.abs(gram - np.eye(gram.shape[0], gram.shape[1])).max())

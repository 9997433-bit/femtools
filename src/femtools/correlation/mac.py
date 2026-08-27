"""Modal Assurance Criterion and its relatives (MAC, COMAC, POC).

``mac[i, j] = |phi_a[:, i]^H phi_b[:, j]|^2
              / ((phi_a[:, i]^H phi_a[:, i]) (phi_b[:, j]^H phi_b[:, j]))``

All routines are vectorized: a full MAC matrix costs a single BLAS ``gemm``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import as_mode_matrix, column_norms_sq, safe_divide, same_array, weighted
from .orthogonality import cross_orthogonality

__all__ = [
    "mac",
    "mac_value",
    "mac_matrix",
    "comac",
    "ecomac",
    "poc",
    "modal_scale_factor",
    "mac_pairs",
]


def mac_matrix(
    phi_a: ArrayLike,
    phi_b: ArrayLike | None = None,
    *,
    weights: Any = None,
) -> NDArray[np.float64]:
    """Modal Assurance Criterion matrix between two mode sets.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shapes as ``(n_dof, n_mode)`` arrays (a 1-D array is one mode).
        ``phi_b=None`` computes the auto-MAC of ``phi_a``.  Real or complex.
    weights:
        Optional weighting operator ``W`` giving the generalized MAC
        ``|a^H W b|^2 / ((a^H W a)(b^H W b))``.  Scalar, 1-D diagonal, dense
        matrix or sparse operator.  With a mass matrix this is the (squared)
        normalized cross-orthogonality.

    Returns
    -------
    ndarray
        Real ``(n_mode_a, n_mode_b)`` matrix with entries in ``[0, 1]``.
        A self-comparison is exactly symmetric with a unit diagonal
        (zero for null mode shapes).

    Notes
    -----
    Complex (damped) modes are handled with the Hermitian product, which is
    the classical MAC for complex modes; see :func:`mac_pairs` for the
    diagonal-only variant.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    self_case = same_array(a, None if phi_b is None else np.asarray(phi_b))
    b = a if phi_b is None else as_mode_matrix(phi_b, "phi_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"phi_a has {a.shape[0]} DOF but phi_b has {b.shape[0]}; align the DOF sets first"
        )

    wb = weighted(weights, b)
    cross = a.conj().T @ wb
    num = np.real(cross * cross.conj())

    wa = wb if self_case else weighted(weights, a)
    na = column_norms_sq(a, wa)
    nb = column_norms_sq(b, wb)
    out = safe_divide(num, np.outer(na, nb))

    if self_case:
        out = 0.5 * (out + out.T)
        np.fill_diagonal(out, np.where(na > 0.0, 1.0, 0.0))
    if weights is None:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def mac_value(phi_a: ArrayLike, phi_b: ArrayLike, *, weights: Any = None) -> float:
    """Scalar MAC between two individual mode shape vectors."""
    a = np.asarray(phi_a).reshape(-1)
    b = np.asarray(phi_b).reshape(-1)
    return float(mac_matrix(a, b, weights=weights)[0, 0])


#: Alias of :func:`mac_value`.  Not re-exported from ``femtools.correlation``
#: so that the attribute ``femtools.correlation.mac`` stays this module.
mac = mac_value


def mac_pairs(phi_a: ArrayLike, phi_b: ArrayLike, *, weights: Any = None) -> NDArray[np.float64]:
    """MAC of column ``i`` of ``phi_a`` against column ``i`` of ``phi_b``.

    Cheaper than building the full matrix when the modes are already paired.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = as_mode_matrix(phi_b, "phi_b")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for paired MAC: {a.shape} vs {b.shape}")
    wb = weighted(weights, b)
    wa = weighted(weights, a)
    cross = np.einsum("ij,ij->j", a.conj(), wb)
    num = np.real(cross * cross.conj())
    den = column_norms_sq(a, wa) * column_norms_sq(b, wb)
    return safe_divide(num, den)


def modal_scale_factor(phi_a: ArrayLike, phi_b: ArrayLike, *, weights: Any = None) -> NDArray[Any]:
    """Modal Scale Factor per column pair, ``msf[i] = (b_i^H a_i)/(b_i^H b_i)``.

    ``phi_b[:, i] * msf[i]`` is the least-squares best fit to ``phi_a[:, i]``,
    which is how mode shapes are rescaled before a COMAC or a shape difference.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = as_mode_matrix(phi_b, "phi_b")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for MSF: {a.shape} vs {b.shape}")
    wb = weighted(weights, b)
    num = np.einsum("ij,ij->j", b.conj(), weighted(weights, a))
    den = column_norms_sq(b, wb)
    out = np.zeros(num.shape, dtype=num.dtype)
    np.divide(num, den, out=out, where=den > 0.0)
    return out


def _is_pair_objects(items: list[Any]) -> bool:
    return bool(items) and hasattr(items[0], "index_a")


def _paired_columns(
    a: NDArray[Any], b: NDArray[Any], pairs: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any]]:
    if pairs is None:
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"{a.shape[1]} vs {b.shape[1]} modes: pass `pairs` when the sets differ in size"
            )
        return a, b
    raw: Any = pairs
    if hasattr(raw, "as_pairs"):  # a PairingResult
        raw = raw.as_pairs()
    else:
        listed = list(raw)
        raw = [(p.index_a, p.index_b) for p in listed] if _is_pair_objects(listed) else listed
    idx = np.atleast_2d(np.asarray(raw, dtype=int))
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError("pairs must be a sequence of (index_a, index_b) tuples")
    return a[:, idx[:, 0]], b[:, idx[:, 1]]


def comac(
    phi_a: ArrayLike,
    phi_b: ArrayLike,
    pairs: ArrayLike | None = None,
    *,
    use_abs: bool = True,
) -> NDArray[np.float64]:
    """Coordinate MAC — per-DOF correlation across a set of mode pairs.

    ``comac[q] = (sum_i |a[q, i] b[q, i]|)^2
                 / (sum_i |a[q, i]|^2 * sum_i |b[q, i]|^2)``

    A value near 1 means the DOF behaves consistently in both models; low
    values localize the mismatch (Lieven & Ewins).  ``pairs`` selects the
    matched columns, e.g. ``pair_modes(...).as_pairs()``; without it the two
    mode sets are assumed to be already in matching column order.

    The modal scaling cancels out per mode only when ``use_abs`` is ``True``
    (default).  Set it to ``False`` for the signed variant, which requires
    consistently scaled shapes.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = as_mode_matrix(phi_b, "phi_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"phi_a has {a.shape[0]} DOF but phi_b has {b.shape[0]}")
    a, b = _paired_columns(a, b, pairs)

    prod = a * b.conj()
    num = np.abs(prod).sum(axis=1) if use_abs else np.abs(prod.sum(axis=1))
    den = np.einsum("ij,ij->i", a.conj(), a).real * np.einsum("ij,ij->i", b.conj(), b).real
    return safe_divide(num**2, den)


def ecomac(
    phi_a: ArrayLike, phi_b: ArrayLike, pairs: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Enhanced COMAC — mean per-DOF shape difference of unity-scaled modes.

    ``ecomac[q] = sum_i |a_hat[q, i] - b_hat[q, i]| / (2 n_pair)`` with each
    mode scaled to unit norm and sign-aligned by its modal scale factor.
    Unlike :func:`comac`, 0 means perfect agreement and it stays meaningful
    for DOFs with small modal amplitude.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = as_mode_matrix(phi_b, "phi_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"phi_a has {a.shape[0]} DOF but phi_b has {b.shape[0]}")
    a, b = _paired_columns(a, b, pairs)
    if a.shape[1] == 0:
        return np.zeros(a.shape[0])

    na = np.sqrt(np.einsum("ij,ij->j", a.conj(), a).real)
    nb = np.sqrt(np.einsum("ij,ij->j", b.conj(), b).real)
    ah = np.zeros_like(a, dtype=complex if np.iscomplexobj(a) else float)
    bh = np.zeros_like(b, dtype=complex if np.iscomplexobj(b) else float)
    np.divide(a, na, out=ah, where=na > 0.0)
    np.divide(b, nb, out=bh, where=nb > 0.0)
    msf = modal_scale_factor(ah, bh)
    sign = np.where(np.abs(msf) > 0.0, msf / np.where(np.abs(msf) > 0.0, np.abs(msf), 1.0), 1.0)
    return np.abs(ah - bh * sign).sum(axis=1) / (2.0 * a.shape[1])


def poc(
    phi_a: ArrayLike,
    phi_b: ArrayLike | None = None,
    mass: Any = None,
    *,
    normalize: bool = True,
    absolute: bool = True,
) -> NDArray[Any]:
    """Pseudo-Orthogonality Check ``Phi_a^H M Phi_b``.

    Thin wrapper around :func:`femtools.correlation.orthogonality.cross_orthogonality`
    that reports magnitudes by default, the usual convention for a POC table
    (diagonal near 1, off-diagonal below ~0.1 for a correlated pair).
    """
    return cross_orthogonality(phi_a, phi_b, mass, normalize=normalize, absolute=absolute)

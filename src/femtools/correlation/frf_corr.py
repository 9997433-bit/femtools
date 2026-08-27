"""FRF-based correlation metrics (FRAC, CSAC, CSF, FDAC).

These are the frequency-domain analogues of the MAC.  Two families exist,
distinguished by the axis that is contracted:

* **Response-wise** (:func:`frac`) — correlate a single response over a
  frequency band; one value per response location, so it localizes *where*
  the models disagree.
* **Signature-wise** (:func:`csac`, :func:`csf`) — correlate the deflection
  shape over all responses at one frequency line; one value per frequency, so
  it shows *in which band* the models disagree.  CSAC judges the shape, CSF
  the amplitude.

Every function works on real or complex data and never loops over frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import safe_divide

__all__ = ["frac", "csac", "csf", "fdac", "frf_difference"]

_Axis = int | tuple[int, ...]


def _pair(h_a: ArrayLike, h_b: ArrayLike) -> tuple[NDArray[Any], NDArray[Any]]:
    a = np.asarray(h_a)
    b = np.asarray(h_b)
    if a.shape != b.shape:
        raise ValueError(f"FRF arrays must have the same shape, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("FRF arrays are empty")
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(float)
    if np.issubdtype(b.dtype, np.integer):
        b = b.astype(float)
    return a, b


def _inner(a: NDArray[Any], b: NDArray[Any], axis: _Axis) -> NDArray[Any]:
    return np.sum(a.conj() * b, axis=axis)


def _energy(a: NDArray[Any], axis: _Axis) -> NDArray[np.float64]:
    return np.real(np.sum(a.conj() * a, axis=axis))


def frac(h_a: ArrayLike, h_b: ArrayLike, axis: _Axis = -1) -> NDArray[np.float64]:
    """Frequency Response Assurance Criterion.

    ``frac = |h_a^H h_b|^2 / ((h_a^H h_a)(h_b^H h_b))`` contracted over the
    frequency axis, i.e. one value in ``[0, 1]`` per response location.

    Parameters
    ----------
    h_a, h_b:
        FRF arrays of identical shape, e.g. ``(n_out, n_freq)`` or
        ``(n_out, n_in, n_freq)``.
    axis:
        Frequency axis (default the last one).

    Returns
    -------
    ndarray
        Shape of the inputs with ``axis`` removed (a 0-d array for a single
        FRF pair).  FRAC is insensitive to a common complex scale factor, so
        it measures shape agreement of the FRF over the band, not level.
    """
    a, b = _pair(h_a, h_b)
    num = np.abs(_inner(a, b, axis)) ** 2
    return safe_divide(num, _energy(a, axis) * _energy(b, axis))


def csac(h_a: ArrayLike, h_b: ArrayLike, axis: _Axis = 0) -> NDArray[np.float64]:
    """Cross Signature Assurance Criterion — one value per frequency line.

    Same expression as :func:`frac` but contracted over the *response* axis,
    so it compares the operating deflection shapes frequency by frequency.
    """
    a, b = _pair(h_a, h_b)
    num = np.abs(_inner(a, b, axis)) ** 2
    return safe_divide(num, _energy(a, axis) * _energy(b, axis))


def csf(h_a: ArrayLike, h_b: ArrayLike, axis: _Axis = 0) -> NDArray[np.float64]:
    """Cross Signature Scale Factor — amplitude agreement per frequency line.

    ``csf = 2 |h_a^H h_b| / (h_a^H h_a + h_b^H h_b)``, contracted over the
    response axis.  Unlike :func:`csac` this is *not* invariant to scaling: it
    reaches 1 only when the two signatures also have the same magnitude.
    """
    a, b = _pair(h_a, h_b)
    num = 2.0 * np.abs(_inner(a, b, axis))
    return safe_divide(num, _energy(a, axis) + _energy(b, axis))


def fdac(h_a: ArrayLike, h_b: ArrayLike, axis: int = 0) -> NDArray[np.float64]:
    """Frequency Domain Assurance Criterion matrix.

    Correlates every frequency line of ``h_a`` with every frequency line of
    ``h_b``; off-diagonal ridges reveal a frequency shift between the models.

    Parameters
    ----------
    h_a, h_b:
        2-D FRF arrays with the response axis given by ``axis``; the two may
        have different numbers of frequency lines but must share the response
        set.

    Returns
    -------
    ndarray
        ``(n_freq_a, n_freq_b)`` matrix in ``[0, 1]``.
    """
    a = np.asarray(h_a)
    b = np.asarray(h_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("fdac expects 2-D FRF arrays")
    if axis == 1:
        a, b = a.T, b.T
    elif axis != 0:
        raise ValueError("axis must be 0 or 1")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"response counts differ: {a.shape[0]} vs {b.shape[0]}")
    num = np.abs(a.conj().T @ b) ** 2
    den = np.outer(_energy(a, 0), _energy(b, 0))
    return safe_divide(num, den)


def frf_difference(
    h_a: ArrayLike, h_b: ArrayLike, axis: _Axis = -1, *, relative: bool = True
) -> NDArray[np.float64]:
    """L2 norm of the FRF difference, optionally relative to ``h_b``.

    Complements the assurance criteria, which are blind to a common scale
    (FRAC/CSAC) or to a sign flip.
    """
    a, b = _pair(h_a, h_b)
    diff = np.sqrt(_energy(a - b, axis))
    if not relative:
        return diff
    return safe_divide(diff, np.sqrt(_energy(b, axis)))

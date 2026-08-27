"""Modal Assurance Criterion and its relatives (MAC, MACX, NMD, COMAC, POC).

``mac[i, j] = |phi_a[:, i]^H phi_b[:, j]|^2
              / ((phi_a[:, i]^H phi_a[:, i]) (phi_b[:, j]^H phi_b[:, j]))``

All routines are vectorized: a full MAC matrix costs a single BLAS ``gemm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import (
    as_mode_matrix,
    column_norms_sq,
    mode_frequencies,
    safe_divide,
    same_array,
    weighted,
)
from .orthogonality import cross_orthogonality

__all__ = [
    "mac",
    "mac_value",
    "mac_matrix",
    "macx",
    "nmd",
    "comac",
    "ecomac",
    "fmac",
    "poc",
    "modal_scale_factor",
    "mac_pairs",
    "FMACResult",
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
        Mode shapes as ``(n_dof, n_mode)`` arrays (a 1-D array is one mode),
        or a modal result carrying them (``ModalResult.modes``).
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
    b = a if phi_b is None else as_mode_matrix(phi_b, "phi_b")
    self_case = phi_b is None or same_array(a, b)
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


def macx(
    phi_a: ArrayLike,
    phi_b: ArrayLike | None = None,
    *,
    weights: Any = None,
) -> NDArray[np.float64]:
    """Extended MAC (MACX) for complex modes — uses both ``phi`` and ``conj(phi)``.

    ``macx[i, j] = (|a^H W b| + |a^T W b|)^2
                   / ((a^H W a + |a^T W a|) (b^H W b + |b^T W b|))``

    Parameters
    ----------
    phi_a, phi_b, weights:
        As in :func:`mac_matrix`; ``phi_b=None`` gives the auto-MACX.

    Returns
    -------
    ndarray
        Real ``(n_mode_a, n_mode_b)`` matrix with entries in ``[0, 1]``,
        exactly symmetric with a unit diagonal for a self-comparison.

    Notes
    -----
    The classical MAC keeps only the Hermitian product ``a^H b``, which for a
    damped structure measures the two complex modes *as digitized*: a mode and
    its own complex conjugate — the same physical motion, only the other half
    of the conjugate pair, and equally likely to come out of a curve fit —
    score 0 rather than 1, and a mode whose phase lead varies across the
    structure is penalized for a difference that is not a shape difference.
    Adding the bilinear product ``a^T b`` makes the criterion insensitive to
    conjugation (swapping ``b`` for ``conj(b)`` exchanges the two terms and
    leaves the sum untouched) while keeping the complex-scaling invariance of
    the MAC, so a complex mode still correlates with its rotated, rescaled
    copy at 1 (Vacher, Jacquier & Bucharles, IMAC/ISMA 2010).

    For real mode shapes the two products coincide and MACX reduces to
    :func:`mac_matrix` — bit for bit, so it is a safe drop-in on an undamped
    model.  The reduction is what makes the difference between the two
    diagnostic: it isolates the part of a low MAC that is pure phase.
    """
    a = as_mode_matrix(phi_a, "phi_a")
    b = a if phi_b is None else as_mode_matrix(phi_b, "phi_b")
    self_case = phi_b is None or same_array(a, b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"phi_a has {a.shape[0]} DOF but phi_b has {b.shape[0]}; align the DOF sets first"
        )

    wb = weighted(weights, b)
    wa = wb if self_case else weighted(weights, a)
    num = (np.abs(a.conj().T @ wb) + np.abs(a.T @ wb)) ** 2

    da = column_norms_sq(a, wa) + np.abs(np.einsum("ij,ij->j", a, wa))
    db = da if self_case else column_norms_sq(b, wb) + np.abs(np.einsum("ij,ij->j", b, wb))
    out = safe_divide(num, np.outer(da, db))

    if self_case:
        out = 0.5 * (out + out.T)
        np.fill_diagonal(out, np.where(da > 0.0, 1.0, 0.0))
    if weights is None:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def nmd(
    phi_a: ArrayLike | None = None,
    phi_b: ArrayLike | None = None,
    *,
    weights: Any = None,
    mac: ArrayLike | None = None,
    relative: bool = False,
) -> NDArray[np.float64]:
    """Normalized Modal Difference, ``sqrt(1 - MAC)`` (Allemang).

    The MAC is a squared cosine, so it is flat near 1: two shapes that differ
    by 5 % score 0.9975, and a table of such numbers hides the ranking it is
    supposed to show.  The NMD is the corresponding sine — a *difference*
    measure, linear in the mismatch for well-correlated modes (0.05 for that
    pair) and directly readable as a fraction of the shape.

    Parameters
    ----------
    phi_a, phi_b, weights:
        As in :func:`mac_matrix`.  Ignored when ``mac`` is given.
    mac:
        Pre-computed MAC matrix, to avoid recomputing it or to take the NMD
        of a variant criterion, e.g. ``nmd(mac=macx(phi_a, phi_b))``.
    relative:
        Return Allemang's ratio form ``sqrt((1 - MAC) / MAC)`` instead — the
        tangent rather than the sine of the same angle, i.e. the difference
        measured against the *correlated* part of the shape.  It is the
        sharper of the two for closely correlated modes and the two agree to
        first order there, but it grows without bound as the MAC goes to 0
        (``inf`` for uncorrelated modes).

    Returns
    -------
    ndarray
        Non-negative array of the shape of the MAC it was built from:
        ``(n_mode_a, n_mode_b)`` for two mode sets, so ``nmd(...)[i, j]``
        belongs to the pair ``(i, j)``.  0 means identical shapes.
    """
    if mac is None:
        if phi_a is None:
            raise ValueError("pass mode shapes or a pre-computed `mac` matrix")
        values = mac_matrix(phi_a, phi_b, weights=weights)
    else:
        values = np.asarray(mac, dtype=float)

    # A weighted or user-supplied MAC is not clipped to [0, 1]; without the
    # floor a round-off overshoot of 1 would come back as NaN.
    gap = np.clip(1.0 - values, 0.0, None)
    if not relative:
        return np.sqrt(gap)
    out = np.full(gap.shape, np.inf)
    np.sqrt(gap / np.where(values > 0.0, values, 1.0), out=out, where=values > 0.0)
    return out


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


@dataclass
class FMACResult:
    """Frequency-scaled MAC data (:func:`fmac`).

    The FMAC diagram plots one point per correlated mode pair at
    ``(freq_a, freq_b)``, sized or coloured by its MAC value.  Everything the
    plot shows is available numerically here: ``np.asarray(result)`` gives the
    MAC value of each pair and :attr:`points` the ``(n_pair, 3)`` array of
    ``[f_a, f_b, mac]`` rows that the diagram draws.
    """

    mac: NDArray[np.float64]
    index_a: NDArray[np.intp]
    index_b: NDArray[np.intp]
    values: NDArray[np.float64]
    freq_a: NDArray[np.float64]
    freq_b: NDArray[np.float64]
    unpaired_a: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    unpaired_b: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    method: str = "greedy"

    @property
    def points(self) -> NDArray[np.float64]:
        """``(n_pair, 3)`` array of ``[f_a, f_b, mac]``, the diagram itself."""
        return np.column_stack((self.freq_a, self.freq_b, self.values))

    @property
    def freq_error(self) -> NDArray[np.float64]:
        """Relative frequency error ``(f_a - f_b) / f_b`` per pair (NaN if unknown)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.where(self.freq_b == 0.0, np.nan, self.freq_b)
            return np.asarray((self.freq_a - self.freq_b) / den, dtype=float)

    @property
    def freq_error_pct(self) -> NDArray[np.float64]:
        return 100.0 * self.freq_error

    @property
    def scale_factor(self) -> float:
        """MAC-weighted slope of ``f_b`` against ``f_a`` through the origin.

        The FMAC diagram is read against the 45-degree line: a slope above 1
        means the analytical frequencies are globally high (model too stiff or
        too light) rather than individually wrong, which is the distinction
        the diagram exists to make.  Well-correlated pairs dominate the fit
        because the MAC values are the weights.
        """
        w = self.values
        den = float(np.sum(w * self.freq_a**2))
        if den <= 0.0:
            return float("nan")
        return float(np.sum(w * self.freq_a * self.freq_b) / den)

    @property
    def n_pairs(self) -> int:
        return int(self.values.size)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self.values
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, item: Any) -> Any:
        return self.values[item]

    def table(self) -> str:
        """Plain-text FMAC table, one line per correlated pair."""
        head = (
            f"{'#':>3} {'A':>4} {'B':>4} {'MAC':>7} {'f_A [Hz]':>11} {'f_B [Hz]':>11} {'df [%]':>8}"
        )
        lines = [head, "-" * len(head)]
        err = self.freq_error_pct
        for k in range(self.n_pairs):
            lines.append(
                f"{k:>3} {int(self.index_a[k]):>4} {int(self.index_b[k]):>4} "
                f"{self.values[k]:>7.4f} {self.freq_a[k]:>11.4f} {self.freq_b[k]:>11.4f} "
                f"{err[k]:>8.2f}"
            )
        if self.n_pairs:
            lines.append(
                f"mean MAC = {float(np.mean(self.values)):.4f}, "
                f"frequency scale factor = {self.scale_factor:.5f}"
            )
        if self.unpaired_a.size:
            lines.append(f"unpaired A: {self.unpaired_a.tolist()}")
        if self.unpaired_b.size:
            lines.append(f"unpaired B: {self.unpaired_b.tolist()}")
        return "\n".join(lines)


def fmac(
    phi_a: ArrayLike,
    phi_b: ArrayLike,
    freq_a: ArrayLike | None = None,
    freq_b: ArrayLike | None = None,
    *,
    pairs: ArrayLike | None = None,
    method: str = "greedy",
    mac_threshold: float = 0.0,
    freq_tol: float | None = None,
    weights: Any = None,
    mac: ArrayLike | None = None,
) -> FMACResult:
    """Frequency-scaled MAC: the MAC of each pair together with its frequencies.

    A MAC matrix says which modes look alike but nothing about *where* they
    sit; a frequency table says how far apart they are but nothing about
    whether the compared modes are the same one.  The FMAC combines the two
    (Fotsch & Ewins, IMAC XVIII, 2000): each correlated pair contributes the
    point ``(f_a, f_b, mac)``, so a systematic frequency bias shows up as a
    line tilted away from the diagonal — reported here as
    :attr:`FMACResult.scale_factor` — while a single bad pair shows up as one
    outlier, and a poorly correlated pair as a low MAC.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shapes ``(n_dof, n_mode)`` on a common DOF set (use
        :func:`~femtools.correlation.dofmap.align_modes` first), or modal
        results carrying them and their frequencies.
    freq_a, freq_b:
        Frequencies [Hz] of the two mode sets; taken from the mode objects
        when they carry them.
    pairs:
        Explicit pairing: a sequence of ``(index_a, index_b)`` tuples or a
        :class:`~femtools.correlation.pairing.PairingResult`.  By default the
        modes are paired by :func:`~femtools.correlation.pairing.pair_modes`.
    method, mac_threshold, freq_tol:
        Forwarded to :func:`~femtools.correlation.pairing.pair_modes` when
        ``pairs`` is not given.
    weights:
        Optional MAC weighting (e.g. a mass matrix, giving the
        cross-orthogonality based variant).
    mac:
        Pre-computed MAC matrix ``(n_a, n_b)``.

    Returns
    -------
    FMACResult
        The pairs with their MAC values and frequencies, plus the full MAC
        matrix and the modes left unpaired.
    """
    # Local import: `pairing` imports this module, so the dependency can only
    # run in this direction at call time.
    from .pairing import PairingResult, pair_modes

    mac_mat = mac_matrix(phi_a, phi_b, weights=weights) if mac is None else np.asarray(mac, float)
    if mac_mat.ndim != 2:
        raise ValueError(f"mac must be 2-D, got shape {mac_mat.shape}")
    n_a, n_b = mac_mat.shape

    fa = _fmac_freqs(phi_a, freq_a, n_a, "freq_a")
    fb = _fmac_freqs(phi_b, freq_b, n_b, "freq_b")

    if pairs is None:
        paired = pair_modes(
            phi_a,
            phi_b,
            fa,
            fb,
            method=method,
            mac_threshold=mac_threshold,
            freq_tol=freq_tol,
            mac=mac_mat,
        )
        idx = paired.as_pairs()
        used = paired.method
        left_a = np.asarray(paired.unpaired_a, dtype=np.intp)
        left_b = np.asarray(paired.unpaired_b, dtype=np.intp)
    else:
        raw: Any = pairs
        if isinstance(raw, PairingResult):
            raw = raw.as_pairs()
        elif hasattr(raw, "as_pairs"):  # pragma: no cover - duck-typed pairing result
            raw = raw.as_pairs()
        else:
            listed = list(raw)
            raw = [(p.index_a, p.index_b) for p in listed] if _is_pair_objects(listed) else listed
        idx = np.atleast_2d(np.asarray(raw, dtype=np.intp)).reshape(-1, 2)
        if idx.size and (idx[:, 0].max() >= n_a or idx[:, 1].max() >= n_b):
            raise ValueError("pairs index outside the MAC matrix")
        used = "given"
        left_a = np.setdiff1d(np.arange(n_a, dtype=np.intp), idx[:, 0])
        left_b = np.setdiff1d(np.arange(n_b, dtype=np.intp), idx[:, 1])

    ia, ib = idx[:, 0], idx[:, 1]
    return FMACResult(
        mac=mac_mat,
        index_a=ia,
        index_b=ib,
        values=mac_mat[ia, ib],
        freq_a=fa[ia],
        freq_b=fb[ib],
        unpaired_a=left_a,
        unpaired_b=left_b,
        method=used,
    )


def _fmac_freqs(phi: Any, freq: ArrayLike | None, n: int, name: str) -> NDArray[np.float64]:
    """Frequencies of a mode set, from the argument or the mode object."""
    if freq is None:
        inherited = mode_frequencies(phi)
        if inherited is None:
            return np.full(n, np.nan)
        freq = inherited
    arr = np.asarray(freq, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} has {arr.size} entries but there are {n} modes")
    return arr


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

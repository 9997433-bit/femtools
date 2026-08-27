"""Covariance-driven Stochastic Subspace Identification (SSI-COV).

Output-only (operational) modal analysis in the time domain.  A structure
excited by broadband ambient forces is modelled as the stochastic state-space
system

.. math::
    x_{k+1} = A\\,x_k + w_k, \\qquad y_k = C\\,x_k + v_k ,

whose output covariances factorise into the observability matrix and the
"stochastic controllability" matrix (Van Overschee & De Moor, 1996; Peeters &
De Roeck, 1999):

.. math::
    R_i = E\\!\\left[y_{k+i}\\,y_k^T\\right] = C A^{i-1} G, \\qquad G = E[x_{k+1}y_k^T].

Stacking the lags into a block Toeplitz matrix therefore gives a rank-``n``
product

.. math::
    T_{1|i} = \\begin{bmatrix} R_i & R_{i-1} & \\cdots & R_1\\\\
                               R_{i+1} & R_i & \\cdots & R_2\\\\
                               \\vdots & & \\ddots & \\vdots\\\\
                               R_{2i-1} & \\cdots & & R_i\\end{bmatrix}
             = \\mathcal{O}_i\\,\\Gamma_i ,

which a truncated SVD splits into
:math:`\\mathcal{O}_i = U_1 S_1^{1/2}` and :math:`\\Gamma_i = S_1^{1/2}V_1^T`.
``C`` is the first block row of :math:`\\mathcal{O}_i` and ``A`` follows from its
shift invariance, :math:`\\mathcal{O}_i^{\\uparrow} A = \\mathcal{O}_i^{\\downarrow}`,
in the least-squares sense.  The discrete eigenvalues of ``A`` map to
continuous poles by :math:`\\lambda = \\ln(\\mu)/\\Delta t` and the mode shapes are
the observed eigenvectors :math:`\\phi = C\\,\\psi`.

Because the model order is not known in advance, the identification is repeated
over a range of orders and the physical poles are the ones that *stabilise*
(:func:`femtools.mpe.common.stabilization_diagram`).

Documented subset
-----------------
Implemented: covariance-driven SSI with unweighted (``weighting="none"``, the
classic SSI-COV / "principal component" choice) and canonical-variate
(``weighting="cva"``) weighting, reference-based reduction (``ref_channels``),
a multi-order stabilisation sweep with frequency / damping / mode-shape
criteria, and mode shapes from the observability matrix.

Not implemented here: data-driven SSI (``SSI-DATA``, which projects raw block
Hankel row spaces via an LQ decomposition instead of covariances), combined
deterministic--stochastic identification with measured inputs, and uncertainty
quantification of the identified poles.  ``SSI-COV`` is the variant that is
cheapest, most robust on long ambient records, and standard for OMA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .common import (
    ModalParameterResult,
    poles_from_roots,
    select_stable_modes,
    stabilization_diagram,
)

__all__ = ["ssi_cov", "output_covariances", "block_toeplitz"]


def _as_channels(data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"data must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        return np.ascontiguousarray(arr.T)
    return arr


def output_covariances(
    data: ArrayLike,
    max_lag: int,
    *,
    ref_channels: Sequence[int] | np.ndarray | None = None,
    unbiased: bool = True,
    detrend: bool = True,
) -> np.ndarray:
    """Output covariance sequence ``R_i = E[y_{k+i} y_ref_k^T]``.

    Parameters
    ----------
    data:
        ``(n_channels, n_samples)`` measured responses (a "tall" array is
        transposed automatically).
    max_lag:
        Largest lag computed; the returned array holds lags ``0 .. max_lag``.
    ref_channels:
        Indices of the reference (projection) channels.  ``None`` uses all
        channels, which is the plain, non-reference-based formulation.
    unbiased:
        Divide lag ``i`` by ``N - i`` (default) instead of ``N``.

    Returns
    -------
    ndarray
        ``(max_lag + 1, n_channels, n_ref)``.
    """
    y = _as_channels(data)
    if detrend:
        y = y - np.mean(y, axis=1, keepdims=True)
    n_ch, n = y.shape
    ref = np.arange(n_ch) if ref_channels is None else np.asarray(ref_channels, dtype=int)
    yr = y[ref]
    m = int(max_lag)
    if m >= n:
        raise ValueError(f"max_lag={m} exceeds the record length ({n} samples)")
    R = np.zeros((m + 1, n_ch, ref.size))
    for i in range(m + 1):
        norm = float(n - i) if unbiased else float(n)
        R[i] = (y[:, i:] @ yr[:, : n - i].T) / norm
    return R


def block_toeplitz(R: np.ndarray, block_rows: int, *, shift: int = 0) -> np.ndarray:
    """Block Toeplitz matrix of covariances, ``T[a, b] = R[i + shift + a - b]``.

    ``shift=0`` builds :math:`T_{1|i}`; ``shift=1`` builds the once-shifted
    :math:`T_{2|i+1}` used by the alternative (non-shift-invariance) estimate
    of ``A``.
    """
    i = int(block_rows)
    n_out, n_ref = R.shape[1], R.shape[2]
    T = np.zeros((i * n_out, i * n_ref))
    for a in range(i):
        for b in range(i):
            T[a * n_out : (a + 1) * n_out, b * n_ref : (b + 1) * n_ref] = R[
                i + shift + a - b
            ]
    return T


def _covariance_toeplitz(R: np.ndarray, block_rows: int) -> np.ndarray:
    """Symmetric block Toeplitz ``E[Y Y^T]``, ``T[a, b] = R[a - b]``."""
    i = int(block_rows)
    dim = R.shape[1]
    T = np.zeros((i * dim, i * dim))
    for a in range(i):
        for b in range(i):
            k = a - b
            blk = R[k] if k >= 0 else R[-k].T
            T[a * dim : (a + 1) * dim, b * dim : (b + 1) * dim] = blk
    return 0.5 * (T + T.T)


def _cva_weights(
    R_full: np.ndarray, R_ref: np.ndarray, block_rows: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Canonical-variate weighting matrices ``(W1, W2)``.

    ``W1 = L_+^{-1}`` and ``W2 = L_-^{-T}``, where ``L L^T`` are Cholesky
    factors of the block Toeplitz covariance matrices of the future
    (``R_full``) and past (``R_ref``) outputs.  Returns ``(None, None)`` when
    either factorisation fails, so the caller can fall back to the unweighted
    formulation.
    """
    i = int(block_rows)
    try:
        Tp = _covariance_toeplitz(R_full, i)
        Tm = _covariance_toeplitz(R_ref, i)
        jitter_p = 1e-12 * np.trace(Tp) / Tp.shape[0]
        jitter_m = 1e-12 * np.trace(Tm) / Tm.shape[0]
        L1 = np.linalg.cholesky(Tp + jitter_p * np.eye(Tp.shape[0]))
        L2 = np.linalg.cholesky(Tm + jitter_m * np.eye(Tm.shape[0]))
    except np.linalg.LinAlgError:
        return None, None
    return np.linalg.inv(L1), np.linalg.inv(L2).T


def _state_space(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    order: int,
    n_ch: int,
    n_ref: int,
    W1_inv: np.ndarray | None,
    W2_inv: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A, C, G)`` for one model order from a truncated SVD.

    ``G = E[x_{k+1} y_k^T]`` is the first block column of the reversed
    controllability matrix, i.e. the *last* block column of ``Gamma``.
    """
    n = int(order)
    sq = np.sqrt(s[:n])
    obs = U[:, :n] * sq[None, :]
    gamma = sq[:, None] * Vt[:n, :]
    if W1_inv is not None:
        obs = W1_inv @ obs
    if W2_inv is not None:
        gamma = gamma @ W2_inv
    C = obs[:n_ch, :]
    A, *_ = np.linalg.lstsq(obs[:-n_ch, :], obs[n_ch:, :], rcond=None)
    return A, C, gamma[:, -n_ref:]


def ssi_cov(
    data: ArrayLike,
    fs: float | None = None,
    *,
    dt: float | None = None,
    order: int | None = None,
    orders: Sequence[int] | None = None,
    order_min: int = 2,
    order_step: int = 2,
    block_rows: int | None = None,
    ref_channels: Sequence[int] | None = None,
    weighting: str = "none",
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
    max_damping: float = 0.25,
    min_damping: float = 0.0,
    stabilization: bool = True,
    stabilization_level: str = "d",
    tol_freq: float = 0.01,
    tol_damp: float = 0.05,
    tol_mac: float = 0.02,
    min_count: int = 2,
    cluster_tol: float = 0.01,
    mode_shapes: bool = True,
) -> ModalParameterResult:
    """Covariance-driven stochastic subspace identification (SSI-COV).

    Parameters
    ----------
    data:
        ``(n_channels, n_samples)`` output-only response time histories.
    fs, dt:
        Sampling frequency [Hz] or time step [s]; exactly one is required.
    order:
        Largest state-space model order (number of *states*, i.e. twice the
        number of mode pairs).  Defaults to ``2 * n_modes + 10`` when
        ``n_modes`` is given, otherwise 30.
    orders:
        Explicit list of orders for the stabilisation sweep; overrides
        ``order`` / ``order_min`` / ``order_step``.
    block_rows:
        Number of block rows ``i`` of the Toeplitz matrix.  Must satisfy
        ``i * n_ref >= order``; defaults to a comfortable
        ``ceil(2 * order / n_ref)`` (at least 10), which is the usual "at least
        twice the maximum order" rule of thumb.
    ref_channels:
        Reference channel indices.  Reference-based SSI keeps the Toeplitz
        matrix small on large sensor arrays at the cost of assuming the
        references observe every mode.
    weighting:
        ``"none"`` (default, classic SSI-COV) or ``"cva"`` (canonical variate
        analysis — the covariance sequence is pre- and post-whitened before the
        SVD, which equalises the influence of strong and weak channels).
    f_range, max_damping, min_damping:
        Physical-pole acceptance window.
    stabilization:
        Sweep the orders and keep the poles that stabilise.  With
        ``stabilization=False`` only ``order`` is identified.
    tol_freq, tol_damp, tol_mac, stabilization_level, min_count, cluster_tol:
        Stabilisation criteria, see
        :func:`femtools.mpe.common.stabilization_diagram`.
    n_modes:
        Keep at most this many modes (the most persistent clusters).

    Returns
    -------
    ModalParameterResult
        ``extras`` carries ``singular_values`` (the Toeplitz spectrum, whose
        drop indicates the physical order), ``A``, ``C``, ``orders`` and
        ``cluster_counts``.

    Notes
    -----
    SSI-COV assumes the excitation is broadband and stationary; a harmonic
    component (rotating machinery) appears as a pole with near-zero damping
    that stabilises just like a structural mode, so damping values close to 0
    should be treated as suspect rather than as very lightly damped modes.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.mpe.synthetic import synthetic_response
    >>> from femtools.mpe.ssi import ssi_cov
    >>> syn = synthetic_response([5.0, 13.0], damping=0.02, n_out=4,
    ...                          fs=256.0, duration=180.0, noise=0.01, seed=3)
    >>> res = ssi_cov(syn.data, fs=syn.fs, order=20, n_modes=2, f_range=(1.0, 60.0))
    >>> bool(np.max(np.abs(res.freq_hz - syn.freq_hz) / syn.freq_hz) < 0.02)
    True
    """
    if (fs is None) == (dt is None):
        raise ValueError("exactly one of `fs` or `dt` must be given")
    step = float(dt) if dt is not None else 1.0 / float(fs)  # type: ignore[arg-type]

    y = _as_channels(data)
    n_ch, n_samples = y.shape
    ref = np.arange(n_ch) if ref_channels is None else np.asarray(ref_channels, dtype=int)
    n_ref = int(ref.size)
    if n_ref == 0:
        raise ValueError("`ref_channels` selects no channel")

    if order is None:
        order = 2 * int(n_modes) + 10 if n_modes else 30
    order = int(order)
    if order < 2:
        raise ValueError("order must be at least 2")

    if block_rows is None:
        block_rows = max(10, int(math.ceil(2.0 * order / n_ref)))
    i_blk = int(block_rows)
    if i_blk * n_ref < order:
        raise ValueError(
            f"block_rows={i_blk} with {n_ref} references gives a Toeplitz matrix of "
            f"rank at most {i_blk * n_ref}, below the requested order {order}"
        )
    if 2 * i_blk >= n_samples:
        raise ValueError(
            f"record of {n_samples} samples is too short for {i_blk} block rows"
        )

    weight = str(weighting).strip().lower()
    want_cva = weight in ("cva", "canonical", "canonical-variate")
    # CVA additionally needs the auto-covariances of *all* channels, not just of
    # the references, so it pays for the full correlation matrix.
    R_full = output_covariances(y, 2 * i_blk) if want_cva else np.zeros(0)
    R = R_full[:, :, ref] if want_cva else output_covariances(y, 2 * i_blk, ref_channels=ref)
    T = block_toeplitz(R, i_blk)

    W1_inv: np.ndarray | None = None
    W2_inv: np.ndarray | None = None
    if want_cva:
        W1, W2 = _cva_weights(R_full, R_full[:, ref][:, :, ref], i_blk)
        if W1 is not None and W2 is not None:
            T = W1 @ T @ W2
            W1_inv = np.linalg.inv(W1)
            W2_inv = np.linalg.inv(W2)
            weight = "cva"
        else:  # pragma: no cover - only on singular covariance matrices
            weight = "none (cva factorisation failed)"
    elif weight not in ("none", "", "pc", "unweighted"):
        raise ValueError(f"unknown weighting {weighting!r}; expected 'none' or 'cva'")

    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    max_order = min(order, int(np.sum(s > s[0] * 1e-14)), U.shape[1])
    if max_order < 2:  # pragma: no cover - degenerate data
        raise RuntimeError("the block Toeplitz matrix is numerically rank deficient")

    if orders is not None:
        order_list = sorted({int(o) for o in orders if 2 <= int(o) <= max_order})
    elif stabilization:
        order_list = list(range(max(2, int(order_min)), max_order + 1, max(1, int(order_step))))
        if not order_list or order_list[-1] != max_order:
            order_list.append(max_order)
    else:
        order_list = [max_order]
    if not order_list:
        raise ValueError("no usable model order in the requested range")

    band = f_range if f_range is not None else (0.0, 0.5 / step)
    pole_sets: dict[int, np.ndarray] = {}
    vector_sets: dict[int, np.ndarray] = {}
    weight_sets: dict[int, np.ndarray] = {}
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for o in order_list:
        try:
            A, C, G = _state_space(U, s, Vt, o, n_ch, n_ref, W1_inv, W2_inv)
            mu, psi = np.linalg.eig(A)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        lam = poles_from_roots(mu, step)
        phi = C @ psi
        keep = np.isfinite(lam) & (lam.imag > 0) & (lam.real < 0)
        wn = np.abs(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            zeta = np.where(wn > 0, -lam.real / np.where(wn > 0, wn, 1.0), 1.0)
        f_hz = wn / (2.0 * math.pi)
        keep &= (zeta <= max_damping) & (zeta >= min_damping)
        keep &= (f_hz >= band[0]) & (f_hz <= band[1]) & (f_hz > 0)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(np.abs(lam[idx]))]
        pole_sets[o] = lam[idx]
        vector_sets[o] = phi[:, idx]
        weight_sets[o] = _modal_contribution(psi, phi, G)[idx]
        models[o] = (A, C)

    if not pole_sets:
        raise RuntimeError(
            "SSI-COV found no physical poles; check the frequency range, the "
            "model order, or whether the record is long enough"
        )

    diagram = None
    used_orders = sorted(pole_sets)
    if len(pole_sets) > 1:
        diagram = stabilization_diagram(
            pole_sets,
            vector_sets=vector_sets,
            tol_freq=tol_freq,
            tol_damp=tol_damp,
            tol_mac=tol_mac,
        )
        reps, counts = select_stable_modes(
            diagram,
            level=stabilization_level,
            cluster_tol=cluster_tol,
            min_count=min_count,
            n_modes=n_modes,
        )
        if reps.size == 0:  # pragma: no cover - relaxation almost always finds some
            reps = pole_sets[used_orders[-1]]
            counts = np.ones(reps.size, dtype=int)
    else:
        reps = pole_sets[used_orders[-1]]
        counts = np.ones(reps.size, dtype=int)

    if n_modes is not None and reps.size > n_modes:
        # Persistence across orders decides first; ties (and the single-order
        # case, where every count is 1) are broken by how much each pole
        # actually contributes to the measured output covariance.
        contrib = _contribution_of(reps, pole_sets, weight_sets, used_orders)
        rank = np.lexsort((-contrib, -counts))
        keep_i = np.sort(rank[: int(n_modes)])
        reps, counts = reps[keep_i], counts[keep_i]

    srt = np.argsort(np.abs(reps))
    lam = reps[srt]
    counts = counts[srt]
    wn = np.abs(lam)

    shapes = None
    if mode_shapes and lam.size:
        shapes = _shapes_for(lam, pole_sets, vector_sets, used_orders)

    top = used_orders[-1]
    return ModalParameterResult(
        freq_hz=wn / (2.0 * math.pi),
        damping=np.where(wn > 0, -lam.real / wn, 0.0),
        poles=lam,
        mode_shapes=shapes,
        order=top,
        method="SSI-COV",
        stabilization=diagram,
        extras={
            "dt": step,
            "orders": used_orders,
            "block_rows": i_blk,
            "singular_values": s,
            "cluster_counts": counts,
            "modal_contribution": _contribution_of(
                lam, pole_sets, weight_sets, used_orders
            ),
            "weighting": weight,
            "A": models[top][0],
            "C": models[top][1],
            "ref_channels": ref,
        },
    )


def _modal_contribution(psi: np.ndarray, phi: np.ndarray, G: np.ndarray) -> np.ndarray:
    """How strongly each state-space mode drives the measured covariances.

    The modal decomposition of ``R_i = C A^{i-1} G`` is
    :math:`\\sum_r (C\\psi_r)\\,\\mu_r^{i-1}\\,(\\psi^{-1}G)_r`, so the product of
    the observability ``|C psi_r|`` and the participation ``|(psi^-1 G)_r|``
    measures the contribution of mode ``r``.  Computational ("mathematical")
    poles, which only serve to fit noise, score orders of magnitude lower.
    """
    try:
        L = np.linalg.solve(psi, G)
    except np.linalg.LinAlgError:  # pragma: no cover - defective A
        L = np.linalg.pinv(psi) @ G
    return np.linalg.norm(phi, axis=0) * np.linalg.norm(L, axis=1)


def _contribution_of(
    lam: np.ndarray,
    pole_sets: dict[int, np.ndarray],
    weight_sets: dict[int, np.ndarray],
    orders: Sequence[int],
) -> np.ndarray:
    """Contribution score of each selected pole, taken from its closest match."""
    out = np.zeros(lam.size)
    for j, p in enumerate(lam):
        best = math.inf
        for o in orders:
            cand = pole_sets[o]
            if cand.size == 0:
                continue
            k = int(np.argmin(np.abs(cand - p)))
            d = float(np.abs(cand[k] - p))
            if d < best:
                best, out[j] = d, float(weight_sets[o][k])
    return out


def _shapes_for(
    lam: np.ndarray,
    pole_sets: dict[int, np.ndarray],
    vector_sets: dict[int, np.ndarray],
    orders: Sequence[int],
) -> np.ndarray:
    """Pick, for every selected pole, the mode shape of the closest candidate.

    The representative pole of a stabilisation cluster comes from one specific
    model order, so its shape is taken from that same order rather than from an
    average — averaging complex shapes identified at different orders would
    need a phase alignment that buys nothing here.
    """
    n_ch = next(iter(vector_sets.values())).shape[0]
    out = np.zeros((n_ch, lam.size), dtype=complex)
    for j, p in enumerate(lam):
        best: tuple[float, np.ndarray | None] = (math.inf, None)
        for o in orders:
            cand = pole_sets[o]
            if cand.size == 0:
                continue
            k = int(np.argmin(np.abs(cand - p)))
            d = float(np.abs(cand[k] - p))
            if d < best[0]:
                best = (d, vector_sets[o][:, k])
        vec = best[1]
        if vec is None:  # pragma: no cover
            continue
        m = int(np.argmax(np.abs(vec)))
        out[:, j] = vec / vec[m] if abs(vec[m]) > 0 else vec
    return out

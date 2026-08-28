"""Eigensystem Realization Algorithm (ERA).

Juang, J.N., Pappa, R.S., *An Eigensystem Realization Algorithm for Modal
Parameter Identification and Model Reduction*, Journal of Guidance, Control,
and Dynamics, 8(5), 1985, pp. 620-627.

ERA is the minimal-realization answer to the time-domain identification
problem.  A discrete-time state-space model

.. math::
    x_{k+1} = A\\,x_k + B\\,u_k, \\qquad y_k = C\\,x_k

reproduces the measured pulse response (the *Markov parameters*)
:math:`Y_k = C A^{k} B`.  Stacking those Markov parameters into a block Hankel
matrix

.. math::
    H(\\kappa) = \\begin{bmatrix}
        Y_{\\kappa}   & Y_{\\kappa+1} & \\cdots & Y_{\\kappa+c-1}\\\\
        Y_{\\kappa+1} & Y_{\\kappa+2} & \\cdots & Y_{\\kappa+c}\\\\
        \\vdots       &              & \\ddots & \\vdots\\\\
        Y_{\\kappa+r-1} & \\cdots     &        & Y_{\\kappa+r+c-2}
    \\end{bmatrix}

factorises it as :math:`H(0) = \\mathcal{O}\\,\\mathcal{C}` with the
observability matrix :math:`\\mathcal{O} = [C; CA; \\dots; CA^{r-1}]` and the
controllability matrix :math:`\\mathcal{C} = [B, AB, \\dots, A^{c-1}B]`, and the
once-shifted matrix as :math:`H(1) = \\mathcal{O}A\\,\\mathcal{C}`.  A truncated
SVD :math:`H(0) = U S V^T` therefore *realises* the system: with
:math:`\\mathcal{O} = U_n S_n^{1/2}` and :math:`\\mathcal{C} = S_n^{1/2}V_n^T`,

.. math::
    A = S_n^{-1/2} U_n^T H(1) V_n S_n^{-1/2}, \\qquad
    B = \\mathcal{C}[:, :n_{in}], \\qquad
    C = \\mathcal{O}[:n_{out}, :] .

The modal parameters follow from the eigenproblem
:math:`A\\psi_j = \\mu_j\\psi_j`: continuous poles are
:math:`\\lambda_j = \\ln(\\mu_j)/\\Delta t` and the mode shapes observed at the
sensors are :math:`\\phi_j = C\\psi_j`, exactly as in
:mod:`femtools.mpe.ssi`.  The two algorithms differ only in which matrix is
factorised -- Hankel matrix of *deterministic* pulse responses here, block
Toeplitz matrix of output covariances (or a projection of raw data) there.

Accuracy indicator
------------------
The 1985 paper introduces the *modal amplitude coherence* to separate physical
modes from the computational ones that only fit noise: the modal amplitude
history extracted from the realised controllability matrix,
:math:`\\bar q_j = (\\psi^{-1}\\mathcal{C})_j`, is compared with the pure
geometric sequence :math:`\\hat q_j` that a single mode would produce.  Their
correlation is 1 for a mode that behaves like a free decay and drops well below
1 for noise modes.  It is reported as ``extras["amplitude_coherence"]``.

Documented subset
-----------------
Implemented: single- and multi-reference ERA from Markov parameters / sampled
impulse responses or from an FRF matrix (inverse FFT via
:func:`femtools.mpe.lsce.irf_from_frf`), an optional multi-order stabilisation
sweep sharing :func:`femtools.mpe.common.stabilization_diagram` with the other
estimators, mode shapes from ``C psi`` and the modal amplitude coherence.

Not implemented here: ERA/DC (the data-correlation variant that first forms
correlations of the Markov parameters), the extended modal amplitude coherence
and modal phase collinearity indicators of the later Pappa papers, and the
direct feedthrough term ``D``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .common import (
    ModalParameterResult,
    as_frf_array,
    poles_from_roots,
    select_stable_modes,
    stabilization_diagram,
)
from .lsce import irf_from_frf
from .ssi import block_hankel

__all__ = ["era", "markov_hankel", "era_realization"]

#: Upper bound on the number of block columns picked automatically.  The
#: realisation only needs ``c * n_in >= order`` columns; everything beyond that
#: buys averaging, and the tail of a decayed impulse response is mostly noise,
#: so a very long record is truncated rather than turned into a huge SVD.
_MAX_AUTO_BLOCK_COLS = 2048


def _as_markov(data: ArrayLike) -> np.ndarray:
    """Coerce impulse responses / Markov parameters to ``(n_out, n_in, n_t)``."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Markov parameters must have 1-3 dimensions, got shape {arr.shape}")


def markov_hankel(
    markov: ArrayLike, block_rows: int, block_cols: int, *, offset: int = 0
) -> np.ndarray:
    """Block Hankel matrix :math:`H(\\kappa)` of Markov parameters.

    Parameters
    ----------
    markov:
        ``(n_out, n_in, n_t)`` pulse-response sequence (2-D input is read as a
        single input, 1-D as a single input and a single output).
    block_rows, block_cols:
        Number of ``n_out x n_in`` blocks down and across, ``r`` and ``c``.
    offset:
        Index :math:`\\kappa` of the leading Markov parameter; ``offset=0``
        builds ``H(0)`` and ``offset=1`` the shifted ``H(1)``.

    Returns
    -------
    ndarray
        ``(block_rows * n_out, block_cols * n_in)`` with block ``(a, b)`` equal
        to ``markov[:, :, offset + a + b]``.

    Notes
    -----
    Each input column is stacked by :func:`femtools.mpe.ssi.block_hankel` -- the
    row structure of an ERA Hankel matrix is the same shifted stacking that
    subspace identification uses -- and the results are interleaved so that a
    *block* column holds one full ``n_out x n_in`` Markov parameter.
    """
    Y = _as_markov(markov)
    n_out, n_in, n_t = Y.shape
    r, c = int(block_rows), int(block_cols)
    if r < 1 or c < 1:
        raise ValueError("block_rows and block_cols must be positive")
    if int(offset) + r + c - 1 > n_t:
        raise ValueError(
            f"{n_t} Markov parameters cannot fill a {r}x{c} block Hankel matrix at "
            f"offset {offset}: {offset + r + c - 1} are needed"
        )
    H = np.empty((r * n_out, c * n_in))
    for i in range(n_in):
        # `block_hankel` transposes "tall" records; the guard above cannot rule
        # that out on its own when there are more sensors than time samples.
        column = Y[:, i, :]
        if n_out > n_t:  # pragma: no cover - unusable record, kept explicit
            raise ValueError("need at least as many time samples as output channels")
        H[:, i::n_in] = block_hankel(column, r, n_columns=c, offset=int(offset))
    return H


def era_realization(
    markov: ArrayLike,
    order: int,
    *,
    block_rows: int | None = None,
    block_cols: int | None = None,
    first: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal discrete-time realisation ``(A, B, C)`` of a pulse response.

    Parameters
    ----------
    markov:
        ``(n_out, n_in, n_t)`` Markov parameters ``Y_k = C A^k B``.  Sampled
        impulse response functions are Markov parameters of the equivalent
        discrete system, so they can be passed directly.
    order:
        Number of retained states ``n`` (two per mode pair).
    block_rows, block_cols:
        Hankel shape; see :func:`era`.
    first:
        Index of the first Markov parameter used.  Leave at 0 for an impulse
        response; use 1 when ``markov[:, :, 0]`` is a direct feedthrough ``D``.

    Returns
    -------
    (A, B, C)
        ``A`` ``(n, n)``, ``B`` ``(n, n_in)``, ``C`` ``(n_out, n)``.
    """
    Y = _as_markov(markov)
    n_out, n_in, n_t = Y.shape
    n = int(order)
    r, c = _hankel_shape(n_out, n_in, n_t, n, block_rows, block_cols, int(first))
    H0 = markov_hankel(Y, r, c, offset=int(first))
    H1 = markov_hankel(Y, r, c, offset=int(first) + 1)
    U, s, Vt = np.linalg.svd(H0, full_matrices=False)
    if n > s.size:
        raise ValueError(f"order {n} exceeds the {s.size} singular values of H(0)")
    return _realize(U, s, Vt, H1, n, n_out, n_in)


def _realize(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    H1: np.ndarray,
    order: int,
    n_out: int,
    n_in: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A, B, C)`` for one model order from the truncated SVD of ``H(0)``."""
    n = int(order)
    sq = np.sqrt(s[:n])
    obs = U[:, :n] * sq[None, :]
    ctrl = sq[:, None] * Vt[:n, :]
    A = (U[:, :n].T @ H1 @ Vt[:n, :].T) / np.outer(sq, sq)
    return A, ctrl[:, :n_in], obs[:n_out, :]


def _hankel_shape(
    n_out: int,
    n_in: int,
    n_t: int,
    order: int,
    block_rows: int | None,
    block_cols: int | None,
    first: int,
) -> tuple[int, int]:
    """Resolve ``(r, c)``, defaulting to "twice the order" rows and a long record."""
    avail = n_t - first  # H(1) needs r + c <= n_t - first
    if avail < 4:
        raise ValueError(f"only {max(avail, 0)} usable Markov parameters")
    if block_rows is None:
        want = max(10, int(math.ceil(2.0 * order / n_out)))
        block_rows = max(2, min(want, avail // 2))
    r = int(block_rows)
    if block_cols is None:
        block_cols = min(avail - r, _MAX_AUTO_BLOCK_COLS)
    c = int(block_cols)
    if r < 2 or c < 2:
        raise ValueError("the block Hankel matrix needs at least 2x2 blocks")
    if r + c > avail:
        raise ValueError(
            f"{n_t} Markov parameters are too few for {r}x{c} blocks starting at {first}"
        )
    if min(r * n_out, c * n_in) < order:
        raise ValueError(
            f"a {r}x{c} block Hankel matrix of {n_out}x{n_in} blocks has rank at most "
            f"{min(r * n_out, c * n_in)}, below the requested order {order}"
        )
    return r, c


def _amplitude_coherence(
    psi: np.ndarray, mu: np.ndarray, ctrl: np.ndarray, n_in: int
) -> np.ndarray:
    """Juang--Pappa modal amplitude coherence, one value per state.

    ``ctrl`` is the realised controllability matrix; its modal decomposition
    ``psi^{-1} ctrl`` holds, for every mode, the amplitude history the
    realisation actually produced.  A physical mode decays as a pure geometric
    sequence in ``mu``, so the correlation between the two is 1; a mode that
    only fits noise scores much lower.
    """
    try:
        q_bar = np.linalg.solve(psi, ctrl.astype(complex))
    except np.linalg.LinAlgError:  # pragma: no cover - defective A
        q_bar = np.linalg.pinv(psi) @ ctrl.astype(complex)
    n_block = q_bar.shape[1] // n_in
    k = np.arange(n_block)
    out = np.zeros(mu.size)
    for j in range(mu.size):
        if mu[j] == 0:  # pragma: no cover - a dead state carries no amplitude
            continue
        # An unstable state would overflow ``mu**k`` long before the end of the
        # sequence, so build it as exp(k log mu) with the peak scaled to one;
        # a constant factor cancels in the normalised correlation below.
        log_mu = np.log(mu[j])
        growth = np.exp(k * log_mu - max(0.0, (n_block - 1) * log_mu.real))
        q_hat = np.repeat(growth, n_in) * np.tile(q_bar[j, :n_in], n_block)
        na = float(np.real(q_bar[j].conj() @ q_bar[j]))
        nb = float(np.real(q_hat.conj() @ q_hat))
        if na > 0 and nb > 0:
            out[j] = float(abs(q_bar[j].conj() @ q_hat) / math.sqrt(na * nb))
    return out


def _nearest(
    lam: np.ndarray,
    pole_sets: dict[int, np.ndarray],
    value_sets: dict[int, np.ndarray],
    orders: Sequence[int],
    *,
    columns: bool = False,
) -> np.ndarray:
    """Value attached to the candidate pole closest to each selected pole."""
    if columns:
        n_row = next(iter(value_sets.values())).shape[0]
        out: np.ndarray = np.zeros((n_row, lam.size), dtype=complex)
    else:
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
                best = d
                if columns:
                    vec = value_sets[o][:, k]
                    m = int(np.argmax(np.abs(vec)))
                    out[:, j] = vec / vec[m] if abs(vec[m]) > 0 else vec
                else:
                    out[j] = float(value_sets[o][k])
    return out


def era(
    data: Any,
    dt: float | None = None,
    *,
    fs: float | None = None,
    freq_hz: ArrayLike | None = None,
    order: int | None = None,
    orders: Sequence[int] | None = None,
    order_min: int = 2,
    order_step: int = 2,
    block_rows: int | None = None,
    block_cols: int | None = None,
    first: int = 0,
    n_samples: int | None = None,
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
    max_damping: float = 0.25,
    min_damping: float = 0.0,
    min_coherence: float = 0.0,
    stabilization: bool = True,
    stabilization_level: str = "d",
    tol_freq: float = 0.01,
    tol_damp: float = 0.05,
    tol_mac: float = 0.02,
    min_count: int = 2,
    cluster_tol: float = 0.01,
    mode_shapes: bool = True,
    window: str | None = "exponential",
    window_factor: float = 0.01,
) -> ModalParameterResult:
    """Eigensystem Realization Algorithm modal parameter estimation.

    Parameters
    ----------
    data:
        Either **real** Markov parameters / impulse responses shaped
        ``(n_out, n_in, n_t)``, ``(n_out, n_t)`` or ``(n_t,)`` together with
        ``dt``/``fs``, or a **complex** FRF matrix together with ``freq_hz`` --
        in that case the impulse responses are obtained by inverse FFT with
        :func:`femtools.mpe.lsce.irf_from_frf` first.  An object exposing
        ``.frf`` (and optionally ``.freq_axis``) is accepted as well.
    dt, fs:
        Time step [s] or sampling rate [Hz] of the Markov parameters; ignored
        (and derived from the spectrum) for FRF input.
    order:
        Largest number of *states* retained from the Hankel SVD, i.e. twice the
        number of mode pairs.  Defaults to ``2 * n_modes + 10`` when
        ``n_modes`` is given, otherwise 30.
    block_rows, block_cols:
        Shape of the block Hankel matrix in blocks, ``r`` and ``c``.  Defaults
        are ``r = max(10, ceil(2 * order / n_out))`` and as many columns as the
        record allows (capped at 2048), the usual "make the matrix at least
        twice as big as the order" rule.
    first:
        Index of the first Markov parameter entering ``H(0)``.  Sampled impulse
        responses have no feedthrough term, so the default 0 uses ``h(0)``;
        pass ``first=1`` for a Markov sequence whose first entry is ``D``.
    n_samples:
        Truncate the record to this many samples before realising -- useful for
        FRF-derived responses whose tail is wrap-around rather than signal.
    f_range, max_damping, min_damping:
        Physical-pole acceptance window.
    min_coherence:
        Discard poles whose modal amplitude coherence falls below this value
        (0 by default, i.e. no filtering).
    stabilization:
        Sweep the model orders and keep the poles that stabilise.  With
        ``stabilization=False`` a single order is realised, which is enough for
        clean pulse responses.
    tol_freq, tol_damp, tol_mac, stabilization_level, min_count, cluster_tol:
        Stabilisation criteria, see
        :func:`femtools.mpe.common.stabilization_diagram`.
    n_modes:
        Keep at most this many modes (the most persistent, then the most
        coherent ones).
    window, window_factor:
        Exponential window applied by :func:`irf_from_frf` to FRF input.  It
        multiplies the pulse response by ``rho**k``, which scales the realised
        ``A`` by exactly ``rho``, so its bias is removed analytically from the
        realisation rather than from the poles.

    Returns
    -------
    ModalParameterResult
        The same container LSCE and SSI return, with ``method="ERA"``.
        ``extras`` carries ``singular_values`` (the Hankel spectrum, whose drop
        indicates the true order), ``amplitude_coherence``, ``block_rows``,
        ``block_cols``, ``dt``, ``orders``, ``cluster_counts`` and the
        top-order ``A``, ``B``, ``C``.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.mpe.era import era
    >>> from femtools.mpe.synthetic import synthetic_frf
    >>> f = np.arange(0, 128.0, 0.25)
    >>> syn = synthetic_frf(f, [12.0, 41.0], damping=0.01, n_out=4)
    >>> res = era(syn.frf, freq_hz=f, n_modes=2, order=12)
    >>> bool(np.max(np.abs(res.freq_hz - syn.freq_hz)) < 0.25)
    True
    """
    if not isinstance(data, np.ndarray) and getattr(data, "frf", None) is not None:
        if freq_hz is None:
            freq_hz = getattr(data, "freq_axis", None)
        data = data.frf

    arr = np.asarray(data)
    rho = 1.0
    if arr.dtype.kind == "c":
        if freq_hz is None:
            raise ValueError("complex input is treated as an FRF and needs freq_hz")
        f_axis = np.asarray(freq_hz, dtype=float).ravel()
        H = as_frf_array(arr)
        if f_range is not None:
            sel = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            if int(sel.sum()) >= 8:
                H, f_axis = H[:, :, sel], f_axis[sel]
        Y, dt_calc = irf_from_frf(H, f_axis, window=window, window_factor=window_factor)
        dt = dt_calc if dt is None else float(dt)
        if window and window.lower().startswith("exp") and Y.shape[2] > 1:
            # irf_from_frf multiplies sample k by rho**k, so the realised system
            # is (rho*A, B, C): dividing A by rho restores the true poles
            # exactly, with no correction needed on B, C or the mode shapes.
            rho = float(max(window_factor, 1e-12) ** (1.0 / (Y.shape[2] - 1)))
    else:
        if dt is None:
            if fs is None:
                raise ValueError("dt or fs must be given for time-domain input")
            dt = 1.0 / float(fs)
        Y = _as_markov(arr)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if n_samples is not None:
        Y = Y[:, :, : int(n_samples)]
    n_out, n_in, n_t = Y.shape

    if order is None:
        order = 2 * int(n_modes) + 10 if n_modes else 30
    order = int(order)
    if order < 2:
        raise ValueError("order must be at least 2")

    r, c = _hankel_shape(n_out, n_in, n_t, order, block_rows, block_cols, int(first))
    H0 = markov_hankel(Y, r, c, offset=int(first))
    H1 = markov_hankel(Y, r, c, offset=int(first) + 1)
    U, s, Vt = np.linalg.svd(H0, full_matrices=False)

    max_order = min(order, int(np.sum(s > s[0] * 1e-14)) if s.size else 0, U.shape[1])
    if max_order < 2:
        raise RuntimeError("the block Hankel matrix of Markov parameters is rank deficient")

    if orders is not None:
        order_list = sorted({int(o) for o in orders if 2 <= int(o) <= max_order})
    elif stabilization:
        order_list = list(
            range(max(2, int(order_min)), max_order + 1, max(1, int(order_step)))
        )
        if not order_list or order_list[-1] != max_order:
            order_list.append(max_order)
    else:
        order_list = [max_order]
    if not order_list:
        raise ValueError("no usable model order in the requested range")

    band = f_range if f_range is not None else (0.0, 0.5 / dt)
    pole_sets: dict[int, np.ndarray] = {}
    vector_sets: dict[int, np.ndarray] = {}
    coherence_sets: dict[int, np.ndarray] = {}
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for o in order_list:
        try:
            A, B, C = _realize(U, s, Vt, H1, o, n_out, n_in)
            mu, psi = np.linalg.eig(A)
        except np.linalg.LinAlgError:  # pragma: no cover - numpy fallback
            continue
        # The coherence compares the realised modal amplitudes with the decay of
        # the realised system, so it is evaluated before the window is undone;
        # scaling A leaves the eigenvectors -- and the mode shapes -- untouched.
        coh = _amplitude_coherence(psi, mu, np.sqrt(s[:o])[:, None] * Vt[:o, :], n_in)
        if rho != 1.0:
            A, mu = A / rho, mu / rho
        lam = poles_from_roots(mu, dt)
        phi = C @ psi
        wn = np.abs(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            zeta = np.where(wn > 0, -lam.real / np.where(wn > 0, wn, 1.0), 1.0)
        f_id = wn / (2.0 * math.pi)
        keep = np.isfinite(lam) & (lam.imag > 0) & (lam.real < 0)
        keep &= (zeta <= max_damping) & (zeta >= min_damping)
        keep &= (f_id >= band[0]) & (f_id <= band[1]) & (f_id > 0)
        keep &= coh >= min_coherence
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(np.abs(lam[idx]))]
        pole_sets[o] = lam[idx]
        vector_sets[o] = phi[:, idx]
        coherence_sets[o] = coh[idx]
        models[o] = (A, B, C)

    if not pole_sets:
        raise RuntimeError(
            "ERA found no physical poles; check the frequency range, the model "
            "order, or whether the pulse response is long enough"
        )

    used_orders = sorted(pole_sets)
    diagram = None
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
        # case) are broken by the Juang-Pappa amplitude coherence, which is
        # exactly the indicator the paper proposes for that job.
        coh = _nearest(reps, pole_sets, coherence_sets, used_orders)
        rank = np.lexsort((-coh, -counts))
        keep_i = np.sort(rank[: int(n_modes)])
        reps, counts = reps[keep_i], counts[keep_i]

    srt = np.argsort(np.abs(reps))
    lam = reps[srt]
    counts = counts[srt]
    wn = np.abs(lam)

    shapes = None
    if mode_shapes and lam.size:
        shapes = _nearest(lam, pole_sets, vector_sets, used_orders, columns=True)

    top = used_orders[-1]
    return ModalParameterResult(
        freq_hz=wn / (2.0 * math.pi),
        damping=np.where(wn > 0, -lam.real / wn, 0.0),
        poles=lam,
        mode_shapes=shapes,
        order=top,
        method="ERA",
        stabilization=diagram,
        extras={
            "dt": dt,
            "orders": used_orders,
            "singular_values": s,
            "cluster_counts": counts,
            "amplitude_coherence": _nearest(lam, pole_sets, coherence_sets, used_orders),
            "block_rows": r,
            "block_cols": c,
            "first": int(first),
            "window_ratio": rho,
            "A": models[top][0],
            "B": models[top][1],
            "C": models[top][2],
        },
    )

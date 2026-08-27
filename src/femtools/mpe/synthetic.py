"""Deterministic synthetic EMA/OMA data generators.

These helpers make the estimators in this package testable without hardware or a
full FE model: they build FRF matrices and operational (output-only) time
histories from a prescribed set of modal parameters, so identified values can be
compared against exactly known truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["SyntheticModal", "synthetic_frf", "synthetic_response", "modal_poles"]


def modal_poles(freq_hz: ArrayLike, damping: ArrayLike) -> np.ndarray:
    """Continuous-time poles from natural frequencies [Hz] and damping ratios."""
    f = np.asarray(freq_hz, dtype=float).ravel()
    z = np.broadcast_to(np.asarray(damping, dtype=float).ravel(), f.shape)
    wn = 2.0 * math.pi * f
    return -z * wn + 1j * wn * np.sqrt(np.maximum(1.0 - z**2, 0.0))


@dataclass
class SyntheticModal:
    """Truth model plus the data generated from it."""

    freq_hz: np.ndarray
    damping: np.ndarray
    poles: np.ndarray
    mode_shapes: np.ndarray
    frf: np.ndarray | None = None
    freq_axis: np.ndarray | None = None
    residues: np.ndarray | None = None
    data: np.ndarray | None = None
    fs: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def synthetic_frf(
    freq_hz: ArrayLike,
    modal_freq: ArrayLike,
    damping: ArrayLike | float = 0.02,
    *,
    mode_shapes: np.ndarray | None = None,
    n_out: int = 4,
    n_in: int = 1,
    input_dofs: Sequence[int] | None = None,
    kind: str = "receptance",
    modal_mass: ArrayLike | float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
    lower_residual: float = 0.0,
    upper_residual: float = 0.0,
) -> SyntheticModal:
    """Build a modal-superposition FRF matrix with known modal parameters.

    .. math::
        H_{oi}(\\omega) = \\sum_r \\frac{\\phi_{or}\\phi_{ir}/m_r}
                                       {\\omega_r^2 - \\omega^2 + 2j\\zeta_r\\omega_r\\omega}

    Parameters
    ----------
    freq_hz:
        Frequency axis of the generated FRF.
    modal_freq, damping:
        Truth natural frequencies [Hz] and damping ratios.
    mode_shapes:
        ``(n_out, n_modes)`` real/complex shapes; a deterministic sine-shaped
        set is generated when omitted.
    kind:
        ``"receptance"`` (displacement/force, default), ``"mobility"`` or
        ``"accelerance"``.
    noise:
        Relative complex Gaussian noise added to the FRF (std as a fraction of
        the RMS magnitude).

    Returns
    -------
    SyntheticModal
        With ``frf`` shaped ``(n_out, n_in, n_freq)``.
    """
    f = np.asarray(freq_hz, dtype=float).ravel()
    fr = np.asarray(modal_freq, dtype=float).ravel()
    n_modes = fr.size
    zt = np.broadcast_to(np.atleast_1d(np.asarray(damping, dtype=float)), (n_modes,)).astype(
        float
    )
    mr = np.broadcast_to(np.atleast_1d(np.asarray(modal_mass, dtype=float)), (n_modes,)).astype(
        float
    )

    if mode_shapes is None:
        x = np.arange(1, n_out + 1) / (n_out + 1.0)
        phi = np.array([np.sin((r + 1) * math.pi * x) for r in range(n_modes)]).T
    else:
        phi = np.asarray(mode_shapes)
        n_out = phi.shape[0]
    if input_dofs is None:
        input_dofs = list(range(min(n_in, n_out)))
    idx_in = np.asarray(input_dofs, dtype=int)[:n_in]
    phi_in = phi[idx_in, :]

    w = 2.0 * math.pi * f
    wr = 2.0 * math.pi * fr
    H = np.zeros((n_out, len(idx_in), f.size), dtype=complex)
    for r in range(n_modes):
        denom = wr[r] ** 2 - w**2 + 2j * zt[r] * wr[r] * w
        num = np.outer(phi[:, r], phi_in[:, r]) / mr[r]
        H += num[:, :, None] / denom[None, None, :]
    if lower_residual:
        with np.errstate(divide="ignore"):
            H += lower_residual * (-1.0 / np.maximum(w**2, 1e-300))[None, None, :]
    if upper_residual:
        H += upper_residual

    kind_l = kind.lower()
    if kind_l in ("mobility", "velocity"):
        H = H * (1j * w)[None, None, :]
    elif kind_l in ("accelerance", "inertance", "acceleration"):
        H = H * (-(w**2))[None, None, :]
    elif kind_l not in ("receptance", "compliance", "displacement"):
        raise ValueError(f"unknown FRF kind {kind!r}")

    if noise:
        rng = np.random.default_rng(seed)
        scale = noise * np.sqrt(np.mean(np.abs(H) ** 2))
        H = H + scale * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))

    residues = np.zeros((n_out, len(idx_in), n_modes), dtype=complex)
    lam = modal_poles(fr, zt)
    for r in range(n_modes):
        residues[:, :, r] = np.outer(phi[:, r], phi_in[:, r]) / (mr[r] * 2j * lam[r].imag)

    return SyntheticModal(
        freq_hz=fr,
        damping=zt,
        poles=lam,
        mode_shapes=phi,
        frf=H,
        freq_axis=f,
        residues=residues,
        extras={"kind": kind_l, "input_dofs": idx_in},
    )


def synthetic_response(
    modal_freq: ArrayLike,
    damping: ArrayLike | float = 0.02,
    *,
    mode_shapes: np.ndarray | None = None,
    n_out: int = 6,
    fs: float = 512.0,
    duration: float = 120.0,
    seed: int = 0,
    noise: float = 0.01,
    n_inputs: int | None = None,
) -> SyntheticModal:
    """Simulate ambient (white-noise driven) output-only responses.

    Each mode is integrated as an independent SDOF oscillator excited by white
    noise (exact zero-order-hold discrete-time step), then projected onto the
    sensors through the mode shapes.  This is the classic operational modal
    analysis test signal for :func:`femtools.mpe.fdd.fdd` / ``efdd``.

    ``n_inputs`` defaults to the number of modes so that the excitation has full
    rank, which is what output-only estimators assume.

    Returns
    -------
    SyntheticModal
        With ``data`` shaped ``(n_out, n_samples)`` and ``fs`` set.
    """
    fr = np.asarray(modal_freq, dtype=float).ravel()
    n_modes = fr.size
    zt = np.broadcast_to(np.atleast_1d(np.asarray(damping, dtype=float)), (n_modes,)).astype(
        float
    )
    if mode_shapes is None:
        x = np.arange(1, n_out + 1) / (n_out + 1.0)
        phi = np.array([np.sin((r + 1) * math.pi * x) for r in range(n_modes)]).T
    else:
        phi = np.asarray(mode_shapes, dtype=float)
        n_out = phi.shape[0]

    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    dt = 1.0 / fs
    q = np.zeros((n_modes, n))
    # FDD/EFDD assume broadband excitation of *every* mode: with fewer
    # independent forces than modes the spectral matrix is rank deficient and
    # the identified shapes degrade, so full rank is the default.
    n_in = n_modes if n_inputs is None else int(n_inputs)
    forcing = rng.standard_normal((n_in, n))
    part = rng.standard_normal((n_modes, n_in))
    f_modal = part @ forcing

    for r in range(n_modes):
        wn = 2.0 * math.pi * fr[r]
        z = zt[r]
        wd = wn * math.sqrt(max(1.0 - z**2, 1e-12))
        # exact discretisation of  q'' + 2 z wn q' + wn^2 q = u  (ZOH on u)
        e = math.exp(-z * wn * dt)
        c, s = math.cos(wd * dt), math.sin(wd * dt)
        A = np.array(
            [
                [e * (c + z * wn / wd * s), e * s / wd],
                [-e * wn**2 / wd * s, e * (c - z * wn / wd * s)],
            ]
        )
        B = np.array([(1.0 - e * (c + z * wn / wd * s)) / wn**2, e * s / wd])
        state = np.zeros(2)
        u = f_modal[r]
        out = np.empty(n)
        for k in range(n):
            out[k] = state[0]
            state = A @ state + B * u[k]
        q[r] = out

    data = phi @ q
    if noise:
        data = data + noise * np.std(data) * rng.standard_normal(data.shape)

    return SyntheticModal(
        freq_hz=fr,
        damping=zt,
        poles=modal_poles(fr, zt),
        mode_shapes=phi,
        data=data,
        fs=float(fs),
        extras={"modal_coordinates": q},
    )

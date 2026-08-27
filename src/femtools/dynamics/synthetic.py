"""Synthetic test-data generation (stand-in for the deferred hardware DAQ package).

Produces noisy FRFs and time histories from a known modal model so that parameter
estimation, correlation and updating code can be exercised against a ground truth.
All generators take an explicit ``seed`` and are therefore reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .damping import as_damping
from .frf import FRFResult, modal_frf
from .modal import as_modal
from .time_domain import TimeHistoryResult, time_history

__all__ = [
    "SyntheticTest",
    "band_limited_noise",
    "burst_random",
    "impulse",
    "synthetic_frf",
    "synthetic_time_response",
]


@dataclass
class SyntheticTest:
    """A synthetic measurement: the excitation, the response, and the truth behind them."""

    t: np.ndarray
    force: np.ndarray
    response: np.ndarray
    force_dofs: np.ndarray
    response_dofs: np.ndarray
    truth: dict[str, Any] = field(default_factory=dict)


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def synthetic_frf(
    modal: Any,
    inputs: Any,
    outputs: Any,
    freq_hz: Any,
    damping: Any = 0.01,
    *,
    noise: float = 0.0,
    seed: int | np.random.Generator | None = 0,
    response: str = "receptance",
) -> FRFResult:
    """Modal FRF with proportional complex Gaussian noise added.

    ``noise`` is the noise standard deviation as a fraction of the RMS magnitude of the
    clean FRF (so ``noise=0.02`` is roughly 2 % noise, i.e. ~34 dB SNR).
    """
    frf = modal_frf(modal, inputs, outputs, freq_hz, damping, response=response)
    if noise <= 0.0:
        return frf
    rng = _rng(seed)
    scale = float(np.sqrt(np.mean(np.abs(frf.H) ** 2))) * float(noise)
    perturbation = (rng.normal(size=frf.H.shape) + 1j * rng.normal(size=frf.H.shape)) / np.sqrt(2)
    return FRFResult(
        H=frf.H + scale * perturbation,
        freq_hz=frf.freq_hz,
        outputs=frf.outputs,
        inputs=frf.inputs,
        response=frf.response,
        method="synthetic",
        meta={**frf.meta, "noise": float(noise), "noise_sigma": scale},
    )


def impulse(n_steps: int, dt: float, amplitude: float = 1.0, index: int = 0) -> np.ndarray:
    """Unit impulse series of length ``n_steps`` (a single non-zero sample)."""
    x = np.zeros(int(n_steps))
    x[int(index)] = float(amplitude)
    return x


def band_limited_noise(
    n_steps: int,
    dt: float,
    band_hz: tuple[float, float] | None = None,
    rms: float = 1.0,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Gaussian noise band-limited to ``band_hz`` and scaled to the requested RMS."""
    rng = _rng(seed)
    x = rng.normal(size=int(n_steps))
    if band_hz is not None:
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(int(n_steps), d=float(dt))
        spectrum[(freqs < band_hz[0]) | (freqs > band_hz[1])] = 0.0
        x = np.fft.irfft(spectrum, n=int(n_steps))
    current = float(np.sqrt(np.mean(x**2)))
    return x * (float(rms) / current) if current > 0 else x


def burst_random(
    n_steps: int,
    dt: float,
    duty: float = 0.5,
    band_hz: tuple[float, float] | None = None,
    rms: float = 1.0,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Burst-random excitation: noise over the first ``duty`` fraction, then silence."""
    x = band_limited_noise(n_steps, dt, band_hz, rms, seed)
    cut = int(np.clip(duty, 0.0, 1.0) * n_steps)
    x[cut:] = 0.0
    return x


def synthetic_time_response(
    modal: Any,
    force_dofs: Any,
    response_dofs: Any,
    n_steps: int,
    dt: float,
    damping: Any = 0.01,
    *,
    excitation: str = "burst",
    band_hz: tuple[float, float] | None = None,
    noise: float = 0.0,
    seed: int | np.random.Generator | None = 0,
) -> SyntheticTest:
    """Simulate a modal-superposition test with a chosen excitation and output noise.

    ``excitation`` is ``"impulse"``, ``"random"`` or ``"burst"``. ``noise`` is the output
    noise standard deviation as a fraction of the response RMS.
    """
    mm = as_modal(modal).mass_normalized()
    rng = _rng(seed)
    f_idx = np.atleast_1d(np.asarray(force_dofs, dtype=int)).reshape(-1)
    r_idx = np.atleast_1d(np.asarray(response_dofs, dtype=int)).reshape(-1)

    rows = []
    for _ in f_idx:
        if excitation == "impulse":
            rows.append(impulse(n_steps, dt))
        elif excitation == "random":
            rows.append(band_limited_noise(n_steps, dt, band_hz, 1.0, rng))
        elif excitation == "burst":
            rows.append(burst_random(n_steps, dt, 0.5, band_hz, 1.0, rng))
        else:
            raise ValueError(f"unknown excitation {excitation!r}")
    F = np.vstack(rows)

    th: TimeHistoryResult = time_history(
        mm, F, dt, as_damping(damping), force_dofs=f_idx, outputs=r_idx
    )
    y = th.displacement
    if noise > 0.0:
        sigma = float(np.sqrt(np.mean(y**2))) * float(noise)
        y = y + rng.normal(scale=sigma, size=y.shape)

    return SyntheticTest(
        t=th.t,
        force=F,
        response=y,
        force_dofs=f_idx,
        response_dofs=r_idx,
        truth={
            "freq_hz": mm.freq_hz.copy(),
            "modes": mm.modes.copy(),
            "damping": damping,
            "excitation": excitation,
            "noise": float(noise),
        },
    )

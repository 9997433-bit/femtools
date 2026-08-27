"""Deterministic checks for H1/H2 FRF estimation and covariance SSI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.signal import freqz, lfilter


def _frequency_data(result: Any, value_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(result, tuple) and len(result) >= 2:
        return np.asarray(result[0], dtype=float), np.asarray(result[1])
    frequency = None
    for name in ("freq_hz", "frequency", "frequencies", "f"):
        if hasattr(result, name):
            frequency = np.asarray(getattr(result, name), dtype=float)
            break
    values = None
    for name in value_names:
        if hasattr(result, name):
            values = np.asarray(getattr(result, name))
            break
    if frequency is None or values is None:
        raise AssertionError("spectral estimate must expose its frequency axis and values")
    return frequency, values


def _curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    return values.reshape(-1, values.shape[-1])[0]


def test_h1_h2_recover_a_noise_free_linear_filter() -> None:
    estimation = pytest.importorskip("femtools.mpe.frf_estimation")
    estimate_h1 = getattr(estimation, "estimate_h1", None)
    estimate_h2 = getattr(estimation, "estimate_h2", None)
    estimate_coherence = getattr(estimation, "coherence", None)
    if estimate_h1 is None or estimate_h2 is None or estimate_coherence is None:
        pytest.skip("H1/H2/coherence APIs are not available")

    fs = 256.0
    nperseg = 1024
    rng = np.random.default_rng(4127)
    excitation = rng.standard_normal(2**16)
    numerator = np.array([0.5, 0.25])
    denominator = np.array([1.0, -0.3])
    response = lfilter(numerator, denominator, excitation)
    # Remove the filter start-up transient before spectral averaging.
    excitation = excitation[nperseg:]
    response = response[nperseg:]

    h1_result = estimate_h1(excitation, response, fs=fs, nperseg=nperseg)
    h2_result = estimate_h2(excitation, response, fs=fs, nperseg=nperseg)
    coherence_result = estimate_coherence(excitation, response, fs=fs, nperseg=nperseg)
    frequency, h1 = _frequency_data(h1_result, ("H", "h_complex", "frf", "values"))
    h2_frequency, h2 = _frequency_data(
        h2_result, ("H", "h_complex", "frf", "values")
    )
    coherence_frequency, gamma2 = _frequency_data(
        coherence_result, ("coherence", "gamma2", "values", "C")
    )
    h1 = _curve(h1)
    h2 = _curve(h2)
    gamma2 = _curve(gamma2).real

    np.testing.assert_allclose(h2_frequency, frequency)
    np.testing.assert_allclose(coherence_frequency, frequency)
    _, expected = freqz(numerator, denominator, worN=2.0 * np.pi * frequency / fs)
    use = (frequency >= 2.0) & (frequency <= 100.0)
    np.testing.assert_allclose(h1[use], expected[use], rtol=2.0e-2, atol=2.0e-3)
    np.testing.assert_allclose(h2[use], expected[use], rtol=2.0e-2, atol=2.0e-3)
    assert np.min(gamma2[use]) > 0.995


def test_ssi_cov_identifies_two_synthetic_ambient_modes() -> None:
    ssi = pytest.importorskip("femtools.mpe.ssi")
    ssi_cov = getattr(ssi, "ssi_cov", None)
    if ssi_cov is None:
        pytest.skip("covariance-driven SSI API is not available")

    from femtools.mpe.synthetic import synthetic_response

    expected = np.array([6.0, 15.0])
    synthetic = synthetic_response(
        expected,
        damping=np.array([0.01, 0.02]),
        n_out=4,
        fs=128.0,
        duration=40.0,
        noise=0.002,
        seed=912,
    )
    result = ssi_cov(synthetic.data, fs=synthetic.fs, order=12, n_modes=2)
    identified = np.asarray(result.freq_hz, dtype=float)
    identified = identified[np.isfinite(identified) & (identified > 0.0)]

    assert identified.size >= expected.size
    nearest = np.array([identified[np.argmin(abs(identified - value))] for value in expected])
    np.testing.assert_allclose(nearest, expected, rtol=0.06, atol=0.15)

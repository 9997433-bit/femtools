"""Stationary random response: PSD synthesis, RMS and the SDOF closed form."""

from __future__ import annotations

import numpy as np
import pytest

from femtools.dynamics.frf import modal_frf
from femtools.dynamics.modal import ModalModel
from femtools.dynamics.random import PSDResult, miles_rms, psd_response

FN_HZ = 25.0
ZETA = 0.02
S0 = 3.0e-3  # flat one-sided force PSD, N^2/Hz


def _sdof() -> ModalModel:
    """One mass-normalised mode on one DOF."""
    return ModalModel(freq_hz=[FN_HZ], modes=[[1.0]])


def _band(f_max_factor: float = 100.0, n_coarse: int = 4001, n_fine: int = 4001) -> np.ndarray:
    """Lines that resolve the half-power bandwidth and still cover the tail."""
    coarse = np.linspace(0.0, f_max_factor * FN_HZ, n_coarse)
    fine = np.linspace(0.9 * FN_HZ, 1.1 * FN_HZ, n_fine)
    return np.unique(np.concatenate([coarse, fine]))


def _two_dof() -> ModalModel:
    """A two-DOF chain in modal form; the shapes are mass-normalised by construction."""
    K = np.array([[2.0, -1.0], [-1.0, 1.0]]) * 6.0e4
    M = np.diag([2.0, 3.0])
    lam, phi = np.linalg.eigh(np.linalg.inv(np.linalg.cholesky(M)) @ K @ np.linalg.inv(
        np.linalg.cholesky(M).T
    ))
    modes = np.linalg.inv(np.linalg.cholesky(M).T) @ phi
    return ModalModel(freq_hz=np.sqrt(lam) / (2.0 * np.pi), modes=modes, eigenvalues=lam)


def test_sdof_white_noise_rms_matches_the_closed_form() -> None:
    """sigma^2 = S0 / (8 zeta omega_n^3) — the exact infinite-band SDOF result."""
    f = _band()
    result = psd_response(_sdof(), S0, f, ZETA)

    omega_n = 2.0 * np.pi * FN_HZ
    exact = np.sqrt(S0 / (8.0 * ZETA * omega_n**3))
    assert isinstance(result, PSDResult)
    assert result.rms.shape == (1,)
    assert float(result.rms[0]) == pytest.approx(exact, rel=2.0e-3)
    assert float(result.sigma[0]) == float(result.rms[0])
    assert float(result.three_sigma[0]) == pytest.approx(3.0 * exact, rel=2.0e-3)
    assert float(result.variance[0]) == pytest.approx(exact**2, rel=4.0e-3)
    assert miles_rms(FN_HZ, ZETA, S0) == pytest.approx(exact, rel=1e-12)


def test_psd_is_the_squared_frf_times_the_force_spectrum() -> None:
    f = np.linspace(1.0, 60.0, 401)
    modal = _sdof()
    spectrum = 1.0e-4 * (1.0 + f / 50.0)
    result = psd_response(modal, spectrum, f, ZETA)
    H = modal_frf(modal, [0], [0], f, ZETA).H[0, 0, :]
    assert np.allclose(result.psd[0], np.abs(H) ** 2 * spectrum, rtol=1e-12)


def test_uncorrelated_inputs_add_their_variances() -> None:
    f = np.linspace(1.0, 200.0, 4001)
    modal = _two_dof()
    both = psd_response(modal, S0, f, ZETA, inputs=[0, 1], outputs=[0, 1])
    only_0 = psd_response(modal, S0, f, ZETA, inputs=[0], outputs=[0, 1])
    only_1 = psd_response(modal, S0, f, ZETA, inputs=[1], outputs=[0, 1])
    assert np.allclose(both.psd, only_0.psd + only_1.psd, rtol=1e-12)
    assert np.allclose(both.variance, only_0.variance + only_1.variance, rtol=1e-12)


def test_cross_psd_is_hermitian_and_carries_the_auto_spectra() -> None:
    f = np.linspace(1.0, 200.0, 801)
    modal = _two_dof()
    S_ff = np.zeros((2, 2, f.size), dtype=complex)
    S_ff[0, 0] = 2.0e-3
    S_ff[1, 1] = 5.0e-3
    S_ff[0, 1] = 1.0e-3 + 4.0e-4j  # partially correlated inputs
    S_ff[1, 0] = np.conj(S_ff[0, 1])

    result = psd_response(modal, S_ff, f, ZETA, inputs=[0, 1], outputs=[0, 1], cross=True)
    assert result.cross_psd is not None
    assert result.cross_psd.shape == (2, 2, f.size)
    assert np.allclose(
        result.cross_psd, np.conj(np.swapaxes(result.cross_psd, 0, 1)), atol=1e-18
    )
    assert np.allclose(result.psd, np.real(np.einsum("oof->of", result.cross_psd)))

    k = 300
    H = modal_frf(modal, [0, 1], [0, 1], f, ZETA).H[:, :, k]
    assert np.allclose(result.cross_psd[:, :, k], H @ S_ff[:, :, k] @ H.conj().T)


def test_correlation_between_the_inputs_changes_the_answer() -> None:
    f = np.linspace(1.0, 200.0, 2001)
    modal = _two_dof()
    level = 2.0e-3
    uncorrelated = psd_response(modal, level, f, ZETA, inputs=[0, 1], outputs=[0])
    coherent = np.full((2, 2, f.size), level, dtype=complex)
    correlated = psd_response(modal, coherent, f, ZETA, inputs=[0, 1], outputs=[0])
    assert not np.allclose(correlated.rms, uncorrelated.rms)
    # Fully coherent inputs are equivalent to one input driving the summed FRF column.
    H = modal_frf(modal, [0, 1], [0], f, ZETA).H[0, :, :]
    assert np.allclose(correlated.psd[0], np.abs(H.sum(axis=0)) ** 2 * level, rtol=1e-10)


@pytest.mark.parametrize("form", ["scalar", "per_frequency", "per_input", "auto", "matrix"])
def test_force_psd_shapes_agree_where_they_describe_the_same_input(form: str) -> None:
    f = np.linspace(1.0, 200.0, 1001)
    modal = _two_dof()
    level = 4.0e-3
    spec = {
        "scalar": level,
        "per_frequency": np.full(f.size, level),
        "per_input": np.full(2, level),
        "auto": np.full((2, f.size), level),
        "matrix": level * np.eye(2),
    }[form]
    result = psd_response(modal, spec, f, ZETA, inputs=[0, 1], outputs=[0, 1])
    reference = psd_response(modal, level, f, ZETA, inputs=[0, 1], outputs=[0, 1])
    assert np.allclose(result.psd, reference.psd, rtol=1e-12)


def test_accelerance_psd_is_the_receptance_psd_times_omega_to_the_fourth() -> None:
    f = np.linspace(1.0, 200.0, 501)
    modal = _sdof()
    disp = psd_response(modal, S0, f, ZETA)
    acc = psd_response(modal, S0, f, ZETA, response="accelerance")
    assert acc.response == "accelerance"
    assert np.allclose(acc.psd, disp.psd * (2.0 * np.pi * f) ** 4, rtol=1e-10)


def test_statistics_of_a_narrow_band_response() -> None:
    result = psd_response(_sdof(), S0, _band(), ZETA)
    # For an SDOF driven by white noise the upcrossing rate is exactly f_n.
    assert float(result.zero_crossing_rate_hz()[0]) == pytest.approx(FN_HZ, rel=0.02)
    assert float(result.moment(0)[0]) == pytest.approx(float(result.variance[0]), rel=1e-12)

    short, long = result.peak_factor(1.0), result.peak_factor(600.0)
    assert 2.0 < float(short[0]) < float(long[0]) < 5.0
    assert np.allclose(result.peak(60.0), result.peak_factor(60.0) * result.rms)

    running = result.cumulative_rms()
    assert running.shape == result.psd.shape
    assert float(running[0, 0]) == 0.0
    assert np.all(np.diff(running[0]) >= -1e-15)
    assert float(running[0, -1]) == pytest.approx(float(result.rms[0]), rel=1e-12)
    # A narrow-band response collects nearly all of its variance at the resonance: below
    # 0.8 f_n only the quasi-static part has accumulated, a few percent of the total.
    below = running[0, result.index_at(0.8 * FN_HZ)]
    assert (below / result.rms[0]) ** 2 < 0.05


@pytest.mark.slow
def test_predicted_rms_matches_a_simulated_time_history() -> None:
    """Wiener-Khinchin end to end: simulate the process and measure its RMS.

    A 300 s record of a 5 %-damped SDOF driven by discrete white noise gives roughly
    2*(2 zeta f_n)*T independent cycles, so the sampled RMS carries a couple of percent
    of statistical scatter around the spectral prediction — nothing that hides a factor
    of two in the PSD, the frequency-axis convention or the integration.
    """
    from femtools.dynamics.time_domain import time_history

    zeta = 0.05
    fs, duration = 1000.0, 300.0
    n_steps = int(fs * duration)
    rng = np.random.default_rng(20240827)
    # A flat one-sided PSD S0 up to Nyquist means a sample variance of S0 * fs / 2.
    force = rng.normal(0.0, np.sqrt(S0 * fs / 2.0), n_steps)

    modal = _sdof()
    history = time_history(modal, force, 1.0 / fs, zeta, force_dofs=[0], outputs=[0])
    settled = history.displacement[0, int(5.0 * fs) :]
    sampled = float(np.sqrt(np.mean(settled**2)))

    resonance = np.linspace(0.8 * FN_HZ, 1.2 * FN_HZ, 8001)
    f = np.unique(np.concatenate([np.linspace(0.0, fs / 2.0, 20001), resonance]))
    predicted = float(psd_response(modal, S0, f, zeta).rms[0])
    assert sampled == pytest.approx(predicted, rel=0.05)


def test_an_existing_frf_can_be_reused() -> None:
    f = np.linspace(1.0, 200.0, 501)
    modal = _two_dof()
    frf = modal_frf(modal, [0, 1], [0], f, ZETA)
    direct = psd_response(modal, S0, f, ZETA, inputs=[0, 1], outputs=[0])
    reused = psd_response(frf, S0)
    assert np.allclose(reused.psd, direct.psd, rtol=1e-12)
    assert np.allclose(reused.rms, direct.rms, rtol=1e-12)
    assert reused.inputs is not None and reused.inputs.tolist() == [0, 1]

    with pytest.raises(ValueError, match="already fixes"):
        psd_response(frf, S0, f, ZETA)


def test_invalid_force_psd_is_rejected() -> None:
    f = np.linspace(1.0, 50.0, 51)
    modal = _two_dof()
    with pytest.raises(ValueError, match="negative"):
        psd_response(modal, -1.0, f, ZETA, inputs=[0, 1])
    with pytest.raises(ValueError, match="Hermitian"):
        bad = np.zeros((2, 2, f.size), dtype=complex)
        bad[0, 0] = bad[1, 1] = 1.0
        bad[0, 1] = 0.5
        psd_response(modal, bad, f, ZETA, inputs=[0, 1])
    with pytest.raises(ValueError, match="2-D force_psd"):
        psd_response(modal, np.ones((3, 7)), f, ZETA, inputs=[0, 1])
    with pytest.raises(ValueError, match="freq_hz is required"):
        psd_response(modal, 1.0)


def test_miles_equation_arguments_are_validated() -> None:
    assert miles_rms(FN_HZ, ZETA, S0, modal_mass=2.0) == pytest.approx(
        miles_rms(FN_HZ, ZETA, S0) / 2.0
    )
    for bad in ({"freq_hz": 0.0}, {"zeta": 0.0}, {"psd_level": -1.0}):
        kwargs = {"freq_hz": FN_HZ, "zeta": ZETA, "psd_level": S0, **bad}
        with pytest.raises(ValueError):
            miles_rms(**kwargs)

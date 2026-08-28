"""Round 10 / O4: the Juang--Pappa Eigensystem Realization Algorithm.

Every synthetic signal is generated from an explicitly seeded stream, so the
frequencies, spectral-line spacing and MAC values quoted in
``.agent_workspace/reports/R10-O4.md`` are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtools.mpe.common import ModalParameterResult, mac
from femtools.mpe.era import era, era_realization, markov_hankel
from femtools.mpe.lsce import irf_from_frf, lsce
from femtools.mpe.p_lscf import poly_lscf
from femtools.mpe.ssi import block_hankel, ssi_data
from femtools.mpe.synthetic import synthetic_frf, synthetic_response

#: Spectral-line spacing of the FRF the gate is measured on [Hz].
DF = 0.25
#: DC-aligned frequency axis; `irf_from_frf` then needs no resampling.
FREQ = np.arange(0.0, 400.0 + DF, DF)


def two_dof_truth() -> tuple[np.ndarray, np.ndarray]:
    """A genuine 2-DOF chain: ``M = diag(1, 2)``, ``k1 = 2e5``, ``k2 = 1e5``.

    Both natural frequencies and both mode shapes come from the undamped
    eigenproblem, so the identified values are compared against analysis rather
    than against numbers the generator was handed.
    """
    M = np.diag([1.0, 2.0])
    K = np.array([[3.0e5, -1.0e5], [-1.0e5, 1.0e5]])
    w2, phi = np.linalg.eigh(np.linalg.solve(M, K))
    phi = phi / np.max(np.abs(phi), axis=0)
    return np.sqrt(w2) / (2.0 * np.pi), phi


def two_dof_frf(noise: float = 0.0, seed: int = 7):
    freq_hz, phi = two_dof_truth()
    return synthetic_frf(
        FREQ,
        freq_hz,
        damping=[0.010, 0.015],
        mode_shapes=phi,
        n_in=1,
        noise=noise,
        seed=seed,
    )


# ----------------------------------------------------------------------
# 1. the realization itself
# ----------------------------------------------------------------------
def test_era_realizes_a_known_discrete_system_exactly() -> None:
    """A rank-4 Markov sequence is reproduced by its 4-state realisation."""
    rng = np.random.default_rng(11)
    A_true = np.array(
        [
            [0.90, 0.20, 0.00, 0.00],
            [-0.20, 0.90, 0.00, 0.00],
            [0.00, 0.00, 0.50, 0.60],
            [0.00, 0.00, -0.60, 0.50],
        ]
    )
    B_true = rng.standard_normal((4, 2))
    C_true = rng.standard_normal((3, 4))

    n_t = 60
    markov = np.empty((3, 2, n_t))
    power = np.eye(4)
    for k in range(n_t):
        markov[:, :, k] = C_true @ power @ B_true
        power = A_true @ power

    A, B, C = era_realization(markov, 4, block_rows=10, block_cols=20)
    assert A.shape == (4, 4) and B.shape == (4, 2) and C.shape == (3, 4)

    rebuilt = np.empty_like(markov)
    power = np.eye(4)
    for k in range(n_t):
        rebuilt[:, :, k] = C @ power @ B
        power = A @ power

    # The realisation is minimal, so it is only unique up to a similarity
    # transform: the Markov parameters and the eigenvalues must match, the
    # matrices themselves need not.
    np.testing.assert_allclose(rebuilt, markov, atol=1.0e-11)
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(A)),
        np.sort_complex(np.linalg.eigvals(A_true)),
        atol=1.0e-11,
    )


def test_markov_hankel_stacks_and_shifts_the_markov_parameters() -> None:
    markov = np.arange(2 * 3 * 12, dtype=float).reshape(2, 3, 12)
    H0 = markov_hankel(markov, 4, 5)
    H1 = markov_hankel(markov, 4, 5, offset=1)

    assert H0.shape == (4 * 2, 5 * 3) and H1.shape == H0.shape
    for a in range(4):
        for b in range(5):
            np.testing.assert_allclose(
                H0[a * 2 : (a + 1) * 2, b * 3 : (b + 1) * 3], markov[:, :, a + b]
            )
            np.testing.assert_allclose(
                H1[a * 2 : (a + 1) * 2, b * 3 : (b + 1) * 3], markov[:, :, a + b + 1]
            )

    # A single-input sequence is exactly the SSI block Hankel of the outputs.
    single = markov[:, 0, :]
    np.testing.assert_allclose(
        markov_hankel(single, 4, 5, offset=2), block_hankel(single, 4, n_columns=5, offset=2)
    )

    with pytest.raises(ValueError):
        markov_hankel(markov, 8, 8)
    with pytest.raises(ValueError):
        markov_hankel(markov, 0, 5)


# ----------------------------------------------------------------------
# 2. the gate: a 2-DOF system within one spectral line, MAC > 0.99
# ----------------------------------------------------------------------
def test_era_identifies_a_two_dof_system_within_one_spectral_line() -> None:
    truth_f, truth_phi = two_dof_truth()
    syn = two_dof_frf()
    result = era(syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0))

    assert isinstance(result, ModalParameterResult)
    assert result.method == "ERA"
    assert result.n_modes == 2
    assert np.all(np.abs(result.freq_hz - truth_f) < DF)
    np.testing.assert_allclose(result.damping, [0.010, 0.015], atol=1.0e-4)

    assert result.mode_shapes is not None
    assert result.mode_shapes.shape == (2, 2)
    for k in range(2):
        assert mac(result.mode_shapes[:, k], truth_phi[:, k]) > 0.99

    # The Hankel matrix of a 2-mode response has 4 significant singular values
    # and drops by more than an order of magnitude at the fifth, which is the
    # paper's order-selection criterion.  The floor is not zero: band-limiting
    # the FRF at 400 Hz makes the sampled response only nearly a 4-state
    # exponential sum.
    sv = result.extras["singular_values"]
    assert sv[3] / sv[0] > 0.5
    assert sv[4] / sv[3] < 0.05
    assert np.all(result.extras["amplitude_coherence"] > 0.99)
    # `f_range` narrows the band before the inverse FFT, so the sampling rate of
    # the realised pulse response follows the retained upper limit.
    assert result.extras["dt"] == pytest.approx(1.0 / (2.0 * 200.0))


def test_era_holds_the_gate_on_a_noisy_two_dof_frf() -> None:
    truth_f, truth_phi = two_dof_truth()
    syn = two_dof_frf(noise=0.02, seed=7)
    result = era(syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0))

    assert np.all(np.abs(result.freq_hz - truth_f) < DF)
    for k in range(2):
        assert mac(result.mode_shapes[:, k], truth_phi[:, k]) > 0.99
    np.testing.assert_allclose(result.damping, [0.010, 0.015], atol=2.0e-3)


def test_era_accepts_impulse_responses_and_agrees_with_the_frf_path() -> None:
    truth_f, _ = two_dof_truth()
    syn = two_dof_frf()
    irf, dt = irf_from_frf(syn.frf, FREQ, window=None)

    from_time = era(irf, dt=dt, n_modes=2, order=16, f_range=(1.0, 200.0))
    from_freq = era(syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0))
    single = era(irf[0, 0], fs=1.0 / dt, n_modes=2, order=16, f_range=(1.0, 200.0))

    np.testing.assert_allclose(from_time.freq_hz, truth_f, atol=DF)
    np.testing.assert_allclose(from_time.freq_hz, from_freq.freq_hz, rtol=1.0e-4)
    # One sensor still sees both modes, but can no longer resolve their shapes.
    np.testing.assert_allclose(single.freq_hz, truth_f, atol=DF)
    assert single.mode_shapes.shape == (1, 2)


def test_era_undoes_the_exponential_window_of_the_inverse_fft() -> None:
    """Windowing scales ``A`` by ``rho``; the correction must be exact."""
    truth_f, _ = two_dof_truth()
    syn = two_dof_frf()
    windowed = era(syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0))
    plain = era(
        syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0), window=None
    )

    assert windowed.extras["window_ratio"] < 1.0
    assert plain.extras["window_ratio"] == 1.0
    # Without the correction the damping would be inflated by
    # -ln(rho) / (2 pi f), i.e. by ~2 % of critical at 32 Hz.
    np.testing.assert_allclose(windowed.damping, [0.010, 0.015], atol=1.0e-4)
    np.testing.assert_allclose(plain.damping, [0.010, 0.015], atol=1.0e-4)
    np.testing.assert_allclose(windowed.freq_hz, truth_f, atol=DF)


def test_era_handles_several_references_and_ranks_by_amplitude_coherence() -> None:
    syn = synthetic_frf(
        FREQ, [12.0, 41.0, 88.0], damping=0.01, n_out=6, n_in=2, input_dofs=[1, 4],
        noise=0.01, seed=3,
    )
    result = era(syn.frf, freq_hz=FREQ, n_modes=3, order=20, f_range=(1.0, 150.0))

    np.testing.assert_allclose(result.freq_hz, [12.0, 41.0, 88.0], atol=DF)
    for k in range(3):
        assert mac(result.mode_shapes[:, k], syn.mode_shapes[:, k]) > 0.99
    assert np.all(result.extras["amplitude_coherence"] > 0.99)
    assert result.extras["block_rows"] * 6 >= 20
    assert result.extras["block_cols"] * 2 >= 20

    # The coherence filter keeps the physical modes and only them.
    filtered = era(
        syn.frf, freq_hz=FREQ, order=20, f_range=(1.0, 150.0), min_coherence=0.99,
        stabilization=False,
    )
    assert filtered.n_modes == 3
    np.testing.assert_allclose(filtered.freq_hz, [12.0, 41.0, 88.0], atol=DF)


def test_era_runs_a_single_order_without_a_stabilization_sweep() -> None:
    truth_f, truth_phi = two_dof_truth()
    syn = two_dof_frf()
    single = era(
        syn.frf, freq_hz=FREQ, order=4, f_range=(1.0, 200.0), stabilization=False
    )

    assert single.stabilization is None
    assert single.extras["orders"] == [4]
    np.testing.assert_allclose(single.freq_hz, truth_f, atol=DF)
    for k in range(2):
        assert mac(single.mode_shapes[:, k], truth_phi[:, k]) > 0.99

    swept = era(syn.frf, freq_hz=FREQ, n_modes=2, order=16, f_range=(1.0, 200.0))
    assert swept.stabilization is not None
    assert len(swept.extras["orders"]) > 1
    # The minimal order is enough for the gate but not quite as accurate as the
    # over-determined one, which absorbs the band-limiting error into the extra
    # computational states.
    np.testing.assert_allclose(swept.freq_hz, single.freq_hz, rtol=2.0e-3)


def test_era_validates_its_arguments() -> None:
    syn = two_dof_frf()
    irf, dt = irf_from_frf(syn.frf, FREQ, window=None)

    with pytest.raises(ValueError):
        era(syn.frf)  # complex input without a frequency axis
    with pytest.raises(ValueError):
        era(irf)  # time-domain input without dt or fs
    with pytest.raises(ValueError):
        era(irf, dt=dt, order=1)
    with pytest.raises(ValueError):
        era(irf, dt=-1.0)
    with pytest.raises(ValueError):
        era(irf, dt=dt, order=20, block_rows=2)  # 2 * 2 rows below the order
    with pytest.raises(ValueError):
        era(irf, dt=dt, n_samples=6, order=8)
    with pytest.raises(ValueError):
        era_realization(irf, 400, block_rows=4, block_cols=8)
    with pytest.raises(ValueError):
        era(np.zeros((2, 2, 2, 8)), dt=dt)
    with pytest.raises(RuntimeError):
        era(irf, dt=dt, order=8, f_range=(300.0, 390.0))


# ----------------------------------------------------------------------
# 3. the neighbours stay put
# ----------------------------------------------------------------------
def test_era_agrees_with_lsce_and_poly_lscf_on_the_same_frf() -> None:
    truth_f, truth_phi = two_dof_truth()
    syn = two_dof_frf(noise=0.005, seed=19)
    shared = dict(freq_hz=FREQ, n_modes=2, f_range=(1.0, 200.0))

    identified = {
        "ERA": era(syn.frf, order=16, **shared),
        "LSCE": lsce(syn.frf, order=16, **shared),
        "PolyMAX": poly_lscf(syn.frf, FREQ, 20, n_modes=2, f_range=(1.0, 200.0)),
    }
    for name, result in identified.items():
        np.testing.assert_allclose(result.freq_hz, truth_f, atol=DF, err_msg=name)
        for k in range(2):
            assert mac(result.mode_shapes[:, k], truth_phi[:, k]) > 0.99, name


def test_ssi_data_is_unaffected_by_the_new_estimator() -> None:
    signal = synthetic_response(
        [5.0, 13.0], damping=0.02, n_out=6, fs=256.0, duration=240.0, noise=0.05, seed=23
    )
    result = ssi_data(signal.data, fs=signal.fs, order=20, n_modes=2, f_range=(1.0, 60.0))

    np.testing.assert_allclose(result.freq_hz, signal.freq_hz, rtol=0.02)
    for k in range(2):
        assert mac(result.mode_shapes[:, k], signal.mode_shapes[:, k]) > 0.95

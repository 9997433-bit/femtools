"""Round 6 / O4: updating UQ, node-based shape optimization and SSI-DATA.

Every random stream is seeded explicitly, so the numbers quoted in
``.agent_workspace/reports/R6-O4.md`` are reproducible.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from femtools.core.model import FEModel
from femtools.fea.eigen import solve_modes
from femtools.fea.verification import shell_plate
from femtools.mpe.common import ModalParameterResult, mac
from femtools.mpe.ssi import block_hankel, ssi_cov, ssi_data
from femtools.mpe.synthetic import synthetic_response
from femtools.optimization.shape import ShapeResult, element_size_ratios, shape_optimize
from femtools.updating import update_model
from femtools.updating.reference import make_updating_testcase
from femtools.updating.uq import UQResult, monte_carlo_update, parameter_covariance

# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
#: A well-conditioned linear response ``r = A p``: the least-squares estimator is
#: exact, so the Monte Carlo covariance has a closed-form value to compare with.
LINEAR_A = np.array(
    [[1.0, 0.2], [0.1, 1.0], [1.0, 1.0], [0.5, -0.3], [2.0, 0.4]], dtype=float
)
LINEAR_P = np.array([1.0, 2.0])


def _linear_response(p: np.ndarray) -> np.ndarray:
    return LINEAR_A @ np.asarray(p, dtype=float)


def two_bar_arch(apex: float = 0.05) -> FEModel:
    """Two axial bars pinned at both ends with a movable apex node.

    Raising the apex turns a nearly straight (and therefore transversely very
    soft) bar chain into an arch, so the first frequency and the compliance
    depend strongly -- and genuinely, not through discretisation error -- on
    one nodal coordinate.  That is the textbook shape-optimization case.
    """
    model = FEModel(name="two-bar-arch")
    model.add_node(id=1, xyz=(0.0, 0.0, 0.0))
    model.add_node(id=2, xyz=(0.5, apex, 0.0))
    model.add_node(id=3, xyz=(1.0, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="bar", material_id=1, A=2.0e-4)
    model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
    model.add_element(id=2, type="BAR2", nodes=(2, 3), property_id=1)
    model.add_spc(node_id=1, mask=(True,) * 6)
    model.add_spc(node_id=3, mask=(True,) * 6)
    model.add_spc(node_id=2, mask=(False, False, True, True, True, True))
    return model


def mass_chain(x_middle: float = 0.35) -> FEModel:
    """Wall - bar - mass - bar - wall, axial only.

    ``f1 = sqrt(EA (1/L1 + 1/L2) / m)`` grows without bound as either bar
    collapses, so maximising it is exactly the case where an unguarded shape
    optimisation destroys the mesh.
    """
    model = FEModel(name="mass-chain")
    for index, x in enumerate((0.0, x_middle, 1.0)):
        model.add_node(id=index + 1, xyz=(x, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="bar", material_id=1, A=1.0e-4)
    model.add_property(id=2, type="lumped", m=5.0)
    model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
    model.add_element(id=2, type="BAR2", nodes=(2, 3), property_id=1)
    model.add_element(id=3, type="MASS", nodes=(2,), property_id=2)
    model.add_spc(node_id=1, mask=(True,) * 6)
    model.add_spc(node_id=3, mask=(True,) * 6)
    model.add_spc(node_id=2, mask=(False, True, True, True, True, True))
    return model


def first_frequency(model: Any) -> float:
    return float(np.asarray(solve_modes(model, n_modes=1).freq_hz, dtype=float)[0])


# ----------------------------------------------------------------------
# 1. first-order parameter covariance
# ----------------------------------------------------------------------
def test_parameter_covariance_reproduces_the_closed_form_least_squares_covariance() -> None:
    sigma = 0.02
    uq = parameter_covariance(LINEAR_A, residual_cov=sigma**2)
    expected = sigma**2 * np.linalg.inv(LINEAR_A.T @ LINEAR_A)

    assert isinstance(uq, UQResult)
    np.testing.assert_allclose(uq.covariance, expected, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(uq.std, np.sqrt(np.diag(expected)), rtol=1.0e-12)
    np.testing.assert_allclose(np.diag(uq.correlation), np.ones(2))
    assert uq.samples is None
    assert uq.extras["dof"] == LINEAR_A.shape[0] - LINEAR_A.shape[1]


def test_parameter_covariance_sandwich_collapses_for_minimum_variance_weighting() -> None:
    variances = np.array([1.0e-4, 4.0e-4, 1.0e-4, 9.0e-4, 2.5e-4])
    Cr = np.diag(variances)
    plain = parameter_covariance(LINEAR_A, residual_cov=Cr)
    sandwich = parameter_covariance(LINEAR_A, residual_cov=Cr, weights=np.linalg.inv(Cr))

    # (J^T W J)^-1 (J^T W Cr W J) (J^T W J)^-1 == (J^T Cr^-1 J)^-1 for W = Cr^-1.
    np.testing.assert_allclose(sandwich.covariance, plain.covariance, rtol=1.0e-10)
    assert "sandwich" in sandwich.method

    # Gauss-Markov: a suboptimal (unit) weighting can only inflate the variance.
    unit = parameter_covariance(LINEAR_A, residual_cov=Cr, weights=np.eye(variances.size))
    assert np.all(np.diag(unit.covariance) >= np.diag(plain.covariance) - 1.0e-18)


def test_parameter_covariance_scales_with_the_residual_when_no_covariance_is_given() -> None:
    rng = np.random.default_rng(2024)
    residual = 0.01 * rng.standard_normal(LINEAR_A.shape[0])
    uq = parameter_covariance(LINEAR_A, residual=residual)
    dof = LINEAR_A.shape[0] - LINEAR_A.shape[1]
    sigma2 = float(residual @ residual) / dof

    np.testing.assert_allclose(
        uq.covariance, sigma2 * np.linalg.inv(LINEAR_A.T @ LINEAR_A), rtol=1.0e-12
    )
    assert uq.extras["sigma2"] == pytest.approx(sigma2)


def test_parameter_covariance_prior_shrinks_the_posterior() -> None:
    Cr = 1.0e-4 * np.eye(LINEAR_A.shape[0])
    posterior = parameter_covariance(LINEAR_A, residual_cov=Cr, prior_cov=1.0e-3)
    likelihood = parameter_covariance(LINEAR_A, residual_cov=Cr)

    assert np.all(np.diag(posterior.covariance) < np.diag(likelihood.covariance))


def test_parameter_covariance_accepts_an_update_result() -> None:
    response, p_true, p0, targets, _ = make_updating_testcase("beam", error=0.10, n_modes=4)
    result = update_model(
        response, ["E1", "E2"], targets, p0=p0, bounds=(0.5, 1.5), max_iter=30, tol=1.0e-8
    )
    uq = parameter_covariance(result, residual_cov=(0.005 * targets) ** 2)

    assert uq.parameter_names == ["E1", "E2"]
    np.testing.assert_allclose(uq.mean, result.x)
    assert np.all(np.isfinite(uq.std)) and np.all(uq.std > 0.0)
    # A clamped-free beam sees its root region far more strongly than its tip,
    # so the tip stiffness is always the less identifiable of the two.
    assert uq.std[0] < uq.std[1]
    lower, upper = uq.interval(0.95)
    assert np.all(lower < p_true) and np.all(p_true < upper)


# ----------------------------------------------------------------------
# 2. Monte Carlo updating
# ----------------------------------------------------------------------
def test_monte_carlo_update_matches_the_first_order_covariance_on_a_linear_problem() -> None:
    sigma = 0.02
    targets = _linear_response(LINEAR_P)
    analytic = parameter_covariance(LINEAR_A, residual_cov=sigma**2)
    mc = monte_carlo_update(
        _linear_response,
        2,
        targets,
        400,
        seed=11,
        noise_std=sigma,
        weights="unit",
        p0=[0.5, 0.5],
    )

    assert isinstance(mc, UQResult)
    assert mc.n_samples == 400
    assert mc.samples is not None and mc.samples.shape == (400, 2)
    np.testing.assert_allclose(mc.mean, LINEAR_P, atol=0.01)
    # 400 samples estimate a variance to about sqrt(2/400) = 7 %; 20 % is a safe
    # band that still fails loudly if the two estimators disagree structurally.
    np.testing.assert_allclose(mc.std, analytic.std, rtol=0.20)
    np.testing.assert_allclose(mc.correlation, analytic.correlation, atol=0.10)


def test_monte_carlo_update_is_reproducible_and_demands_a_seed() -> None:
    targets = _linear_response(LINEAR_P)
    kwargs = dict(noise_std=0.02, weights="unit", p0=[0.5, 0.5])
    first = monte_carlo_update(_linear_response, 2, targets, 25, seed=5, **kwargs)
    again = monte_carlo_update(_linear_response, 2, targets, 25, seed=5, **kwargs)
    other = monte_carlo_update(_linear_response, 2, targets, 25, seed=6, **kwargs)

    np.testing.assert_array_equal(first.samples, again.samples)
    assert not np.allclose(first.samples, other.samples)
    with pytest.raises(TypeError):
        monte_carlo_update(_linear_response, 2, targets, 5)  # type: ignore[call-arg]


def test_monte_carlo_update_resamples_residuals_and_start_points_on_the_beam() -> None:
    response, p_true, p0, targets, _ = make_updating_testcase("beam", error=0.10, n_modes=4)
    shared = dict(p0=p0, bounds=(0.5, 1.5), max_iter=20, tol=1.0e-8)

    noisy = monte_carlo_update(
        response, ["E1", "E2"], targets, 40, seed=31, noise_std=0.002, relative=True, **shared
    )
    boot = monte_carlo_update(
        response, ["E1", "E2"], targets, 20, seed=31, perturb="residual", **shared
    )
    starts = monte_carlo_update(
        response, ["E1", "E2"], targets, 20, seed=31, perturb="parameters",
        start_scatter=0.15, **shared
    )

    assert noisy.method == "monte-carlo(targets)"
    assert boot.method == "monte-carlo(residual)"
    assert starts.method == "monte-carlo(parameters)"
    assert noisy.extras["n_failed"] == 0
    # A 0.2 % measurement error keeps both stiffness multipliers within 1 %.
    np.testing.assert_allclose(noisy.mean, p_true, rtol=0.01)
    assert np.all(noisy.std < 0.02)
    # The clean problem is well posed: every start point converges to the same
    # answer, so the scattered-start cloud collapses onto the truth.
    np.testing.assert_allclose(starts.mean, p_true, rtol=1.0e-3)
    assert np.max(starts.std) < 1.0e-3
    # The residual bootstrap of an essentially perfect fit is equally tight.
    np.testing.assert_allclose(boot.mean, p_true, rtol=1.0e-2)


def test_uq_result_reports_intervals_percentiles_and_names() -> None:
    targets = _linear_response(LINEAR_P)
    mc = monte_carlo_update(
        _linear_response, ["a", "b"], targets, 60, seed=3, noise_std=0.02,
        weights="unit", p0=[0.5, 0.5],
    )
    lower, upper = mc.interval(0.9)
    median = mc.percentile(50.0)

    assert mc.parameter_names == ["a", "b"]
    assert mc["a"] == pytest.approx(mc.mean[0])
    assert np.all(lower < mc.mean) and np.all(mc.mean < upper)
    np.testing.assert_allclose(median, mc.mean, atol=5.0 * np.max(mc.std))
    assert set(mc.to_dict()) == {"a", "b"}
    assert "monte-carlo" in mc.summary()
    with pytest.raises(ValueError):
        parameter_covariance(LINEAR_A).percentile(50.0)


# ----------------------------------------------------------------------
# 3. shape optimization
# ----------------------------------------------------------------------
def test_shape_optimize_raises_the_first_frequency_of_a_two_bar_arch() -> None:
    model = two_bar_arch()
    before = first_frequency(model)
    result = shape_optimize(
        model, nodes=[2], directions="y", objective="frequency", move_limit=0.2, max_iter=40
    )

    assert isinstance(result, ShapeResult)
    assert result.success and result.feasible
    assert result.initial_value == pytest.approx(before, rel=1.0e-12)
    assert result.value > 2.0 * result.initial_value
    assert result.improvement > 1.0
    assert result.x[0] > 0.15  # the apex is pushed up
    # No element inverted or collapsed: stretching the bars keeps every ratio > 1.
    assert result.min_size_ratio > 1.0
    assert np.all(result.extras["size_ratios"] > 0.0)
    # The caller's model is untouched; the optimum lives on the returned copy.
    np.testing.assert_allclose(model.nodes[2].xyz, (0.5, 0.05, 0.0))
    np.testing.assert_allclose(result.model.nodes[2].xyz[1], 0.05 + result.x[0])
    assert first_frequency(result.model) == pytest.approx(result.value, rel=1.0e-9)


def test_shape_optimize_reduces_the_compliance_of_the_same_arch() -> None:
    model = two_bar_arch()
    result = shape_optimize(
        model,
        nodes=[2],
        directions="y",
        objective="compliance",
        loads={(2, 1): -1000.0},
        move_limit=0.2,
        max_iter=40,
    )

    assert result.objective == "compliance"
    assert result.value < 0.2 * result.initial_value
    assert result.improvement > 0.8
    assert result.min_size_ratio > 0.0


def test_shape_optimize_stops_at_the_mesh_quality_barrier() -> None:
    model = mass_chain()
    limit = 0.4
    result = shape_optimize(
        model,
        nodes=[2],
        directions="x",
        objective="frequency",
        move_limit=0.3,
        min_quality=limit,
        max_iter=60,
    )

    # Maximising f1 wants to collapse the short bar; the barrier stops it exactly
    # at the bound instead of letting the element invert.
    assert result.value > 1.2 * result.initial_value
    assert result.min_size_ratio == pytest.approx(limit, abs=1.0e-3)
    assert result.constraint_violation < 1.0e-6
    assert np.all(result.extras["size_ratios"] > 0.0)
    assert result.max_movement <= 0.3 + 1.0e-9


def test_element_size_ratios_flag_a_folded_shell_element() -> None:
    plate = shell_plate(2, 2, etype="QUAD4", side=1.0, thickness=0.01, clamped_edge=True)
    np.testing.assert_allclose(element_size_ratios(plate, plate), np.ones(4))

    folded = copy.deepcopy(plate)
    folded["nodes"][5]["xyz"] = (0.5, -0.4, 0.0)  # drag the centre node past an edge
    ratios = element_size_ratios(folded, plate)
    assert np.min(ratios) < 0.0
    assert np.sum(ratios < 0.0) == 2


def test_shape_optimize_keeps_a_quad4_plate_valid_while_improving_it() -> None:
    plate = shell_plate(2, 2, etype="QUAD4", side=1.0, thickness=0.01, clamped_edge=True)
    result = shape_optimize(
        plate,
        nodes=[5],
        directions="xy",
        objective="frequency",
        move_limit=0.2,
        min_quality=0.25,
        max_iter=30,
    )

    assert result.value >= result.initial_value
    assert result.min_size_ratio >= 0.25 - 1.0e-6
    assert np.all(result.extras["size_ratios"] > 0.0)
    assert result.extras["n_elements_monitored"] == 4
    assert len(result.variables) == 2 and result.variables[0] == (5, "x")


def test_shape_optimize_accepts_a_callable_objective_and_a_laplacian_regulariser() -> None:
    model = two_bar_arch()
    calls: list[int] = []

    def negative_frequency(m: Any) -> float:
        calls.append(1)
        return -first_frequency(m)

    result = shape_optimize(
        model,
        nodes=[2],
        directions="y",
        objective=negative_frequency,
        move_limit=0.2,
        smoothing=0.05,
        max_iter=30,
    )

    assert result.objective == "custom"
    assert calls  # the callable really drove the search
    assert result.value < result.initial_value  # a minimised objective
    assert result.extras["smoothing"] == pytest.approx(0.05)
    assert result.extras["laplacian"] > 0.0
    assert "node 2 dy" in result.summary()


def test_shape_optimize_rejects_impossible_requests() -> None:
    model = two_bar_arch()
    with pytest.raises(ValueError):
        shape_optimize(model, nodes=[2], directions="w")
    with pytest.raises(ValueError):
        shape_optimize(model, nodes=[2], directions="y", move_limit=-0.1)
    with pytest.raises(ValueError):
        shape_optimize(model, nodes=[2], directions="y", objective="mass")


# ----------------------------------------------------------------------
# 4. data-driven SSI
# ----------------------------------------------------------------------
def test_ssi_data_recovers_a_noisy_sdof_record() -> None:
    truth = 7.5
    signal = synthetic_response(
        [truth], damping=0.02, n_out=4, fs=200.0, duration=200.0, noise=0.05, seed=17
    )
    result = ssi_data(signal.data, fs=signal.fs, order=12, n_modes=1, f_range=(1.0, 40.0))

    assert isinstance(result, ModalParameterResult)
    assert result.method == "SSI-DATA"
    assert result.freq_hz.size == 1
    assert abs(result.freq_hz[0] - truth) / truth < 0.02
    assert abs(result.damping[0] - 0.02) < 0.01
    assert result.mode_shapes is not None
    assert mac(result.mode_shapes[:, 0], signal.mode_shapes[:, 0]) > 0.95


def test_ssi_data_and_ssi_cov_agree_on_a_noisy_two_dof_record() -> None:
    signal = synthetic_response(
        [5.0, 13.0], damping=0.02, n_out=6, fs=256.0, duration=240.0, noise=0.05, seed=23
    )
    data = ssi_data(signal.data, fs=signal.fs, order=20, n_modes=2, f_range=(1.0, 60.0))
    cov = ssi_cov(signal.data, fs=signal.fs, order=20, n_modes=2, f_range=(1.0, 60.0))

    assert type(data) is type(cov)
    assert set(cov.extras) <= set(data.extras)
    for identified in (data, cov):
        np.testing.assert_allclose(identified.freq_hz, signal.freq_hz, rtol=0.02)
        for k in range(2):
            assert mac(identified.mode_shapes[:, k], signal.mode_shapes[:, k]) > 0.95
    np.testing.assert_allclose(data.freq_hz, cov.freq_hz, rtol=0.01)


def test_ssi_data_supports_cva_weighting_and_reference_channels() -> None:
    signal = synthetic_response(
        [5.0, 13.0], damping=0.02, n_out=6, fs=256.0, duration=240.0, noise=0.05, seed=23
    )
    cva = ssi_data(
        signal.data, fs=signal.fs, order=20, n_modes=2, f_range=(1.0, 60.0), weighting="cva"
    )
    refs = ssi_data(
        signal.data,
        fs=signal.fs,
        order=16,
        n_modes=2,
        f_range=(1.0, 60.0),
        ref_channels=[1, 3, 4],
        block_rows=14,
    )

    assert cva.extras["weighting"] == "cva"
    np.testing.assert_allclose(cva.freq_hz, signal.freq_hz, rtol=0.02)
    np.testing.assert_array_equal(refs.extras["ref_channels"], [1, 3, 4])
    np.testing.assert_allclose(refs.freq_hz, signal.freq_hz, rtol=0.03)
    for k in range(2):
        assert mac(refs.mode_shapes[:, k], signal.mode_shapes[:, k]) > 0.95


def test_block_hankel_stacks_shifted_copies_of_the_record() -> None:
    y = np.arange(24.0).reshape(2, 12)
    past = block_hankel(y, 3, n_columns=5, offset=0)
    future = block_hankel(y, 3, n_columns=5, offset=3)

    assert past.shape == (6, 5) and future.shape == (6, 5)
    np.testing.assert_allclose(past[0:2], y[:, 0:5])
    np.testing.assert_allclose(past[2:4], y[:, 1:6])
    np.testing.assert_allclose(future[0:2], y[:, 3:8])
    np.testing.assert_allclose(block_hankel(y, 2, n_columns=4, channels=[1])[0], y[1, 0:4])
    with pytest.raises(ValueError):
        block_hankel(y, 20)


def test_ssi_data_validates_its_arguments() -> None:
    signal = synthetic_response([6.0], damping=0.02, n_out=2, fs=64.0, duration=20.0, seed=1)
    with pytest.raises(ValueError):
        ssi_data(signal.data, fs=64.0, dt=1.0 / 64.0)
    with pytest.raises(ValueError):
        ssi_data(signal.data, fs=64.0, order=1)
    with pytest.raises(ValueError):
        ssi_data(signal.data, fs=64.0, order=10, block_rows=1)
    with pytest.raises(ValueError):
        ssi_data(signal.data, fs=64.0, order=10, block_rows=6, weighting="magic")
    with pytest.raises(ValueError):
        ssi_data(signal.data[:, :20], fs=64.0, order=10, block_rows=8)
    with pytest.raises(ValueError):
        ssi_data(signal.data, fs=64.0, order=40, block_rows=4)

"""Canonical two-parameter beam recovery case."""

from __future__ import annotations

import numpy as np
import pytest

from femtools.updating import reference as reference_module
from femtools.updating import static_stress_response
from femtools.updating import updater as updater_module


@pytest.mark.golden
def test_updating_recovers_ten_percent_beam_stiffness_error() -> None:
    response, true_parameters, initial, targets, _ = reference_module.make_updating_testcase(
        "beam",
        error=0.10,
        n_modes=4,
    )

    result = updater_module.update_model(
        response,
        ["E1", "E2"],
        targets,
        p0=initial,
        bounds=(0.5, 1.5),
        max_iter=30,
        tol=1.0e-8,
    )
    recovered = np.asarray(result.values, dtype=float)
    relative_error = np.abs(recovered - true_parameters) / true_parameters

    assert isinstance(result, updater_module.UpdateResult)
    assert np.all(np.isfinite(recovered))
    assert np.max(relative_error) < 0.02
    assert result.rms_error < result.initial_rms_error


def test_static_stress_response_tracks_prescribed_strain_modulus(
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, data = axial_bar
    strain = 2.0e-4
    response = static_stress_response(
        model,
        [{"type": "material", "id": 1, "name": "E"}],
        elements=(1,),
        component="xx",
        solver_kwargs={"enforced": {(2, 0): strain * data["L"]}},
    )

    baseline = response(np.array([1.0]))
    half_modulus = response(np.array([0.5]))

    np.testing.assert_allclose(baseline, [data["E"] * strain], rtol=1.0e-12)
    np.testing.assert_allclose(half_modulus, 0.5 * baseline, rtol=1.0e-12)

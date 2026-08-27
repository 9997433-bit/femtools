"""Canonical two-parameter beam recovery case."""

from __future__ import annotations

import numpy as np
import pytest

reference_module = pytest.importorskip("femtools.updating.reference")
updater_module = pytest.importorskip("femtools.updating.updater")


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

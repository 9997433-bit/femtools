"""Single-parameter FRF updating recovery case."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from femtools.dynamics.frf import modal_frf
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes
from femtools.updating.frf_updating import update_from_frf
from femtools.updating.parameters import Parameter


def test_update_from_frf_recovers_sdof_stiffness(
    axial_bar: tuple[Any, dict[str, float]],
) -> None:
    model, _ = axial_bar
    truth = copy.deepcopy(model)
    expected_scale = 1.21
    truth.materials[1].E *= expected_scale
    assembly = assemble_km(truth)
    modal = solve_modes(truth, n_modes=1, assembly=assembly)
    tip_dof = assembly.dof_map.index(2, 0)
    frequency = np.linspace(0.6 * modal.freq_hz[0], 1.4 * modal.freq_hz[0], 81)
    measured = modal_frf(
        modal,
        inputs=[tip_dof],
        outputs=[tip_dof],
        freq_hz=frequency,
        damping=0.02,
    )
    parameter = Parameter(
        "E scale",
        kind="E",
        target=1,
        value=1.0,
        relative=True,
        lower=0.6,
        upper=1.6,
    )

    result = update_from_frf(
        model,
        [parameter],
        measured,
        freq_hz=frequency,
        inputs=[tip_dof],
        outputs=[tip_dof],
        damping=0.02,
        n_modes=1,
        p0=[1.0],
        bounds=(0.6, 1.6),
        max_iter=20,
        tol=1.0e-8,
    )

    recovered = np.asarray(getattr(result, "values", result.x), dtype=float)
    assert recovered[0] == pytest.approx(expected_scale, rel=0.01)
    assert result.rms_error < result.initial_rms_error

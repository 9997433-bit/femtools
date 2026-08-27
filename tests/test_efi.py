"""Kammer Effective Independence sensor selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from femtools.pretest import efi as efi_module


def _ranking_and_values(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(result, tuple) and len(result) >= 2:
        return np.asarray(result[0]), np.asarray(result[1])

    rank = None
    values = None
    for name in ("ranked_dofs", "ranking", "dof_ids", "selected_dofs"):
        if hasattr(result, name):
            rank = np.asarray(getattr(result, name))
            break
    for name in ("efi_values", "values", "scores"):
        if hasattr(result, name):
            values = np.asarray(getattr(result, name))
            break
    if rank is None or values is None:
        raise AssertionError("EFI result must contain ranked DOF ids and EFI values")
    return rank, values


def test_efi_retains_an_independent_two_sensor_set() -> None:
    mode_shapes = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0e-3, 2.0e-3],
            [2.0e-3, 1.0e-3],
            [1.0e-3, -1.5e-3],
            [-1.0e-3, 1.0e-3],
            [0.5e-3, 1.0e-3],
            [1.0e-3, 0.5e-3],
            [-0.5e-3, 1.5e-3],
            [1.5e-3, -0.5e-3],
        ]
    )
    candidate_dofs = np.arange(100, 110)

    result = efi_module.effective_independence(
        mode_shapes,
        n_sensors=2,
        candidate_dofs=candidate_dofs,
    )
    ranking, values = _ranking_and_values(result)
    selected_ids = ranking[:2].astype(int)
    selected_rows = np.array([np.flatnonzero(candidate_dofs == item)[0] for item in selected_ids])
    reduced = mode_shapes[selected_rows]

    numerator = abs(np.vdot(reduced[:, 0], reduced[:, 1])) ** 2
    denominator = np.vdot(reduced[:, 0], reduced[:, 0]).real
    denominator *= np.vdot(reduced[:, 1], reduced[:, 1]).real
    off_diagonal_mac = numerator / denominator

    assert set(selected_ids) == {100, 101}
    assert off_diagonal_mac < 0.15
    assert values.ndim == 1
    assert values.size >= 2
    assert np.all(np.isfinite(values))
    assert np.all(values >= -1.0e-12)

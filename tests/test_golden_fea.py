"""Analytical golden cases for the finite-element eigensolver."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import positive_frequencies
from femtools.fea import eigen


@pytest.mark.golden
def test_two_node_axial_bar_frequency(axial_bar: tuple[object, dict[str, float]]) -> None:
    """The one-free-DOF consistent-mass bar has an exact discrete frequency."""
    model, data = axial_bar

    modal = eigen.solve_modes(model, n_modes=1)
    actual = positive_frequencies(modal, 1)
    expected = np.sqrt(3.0 * data["E"] / data["rho"]) / (2.0 * np.pi * data["L"])

    assert actual.size == 1
    assert actual[0] == pytest.approx(expected, rel=1.0e-8)


@pytest.mark.golden
def test_euler_bernoulli_cantilever_first_three_modes_per_bending_plane(
    cantilever: tuple[object, dict[str, float]],
) -> None:
    model, data = cantilever

    modal = eigen.solve_modes(model, n_modes=6)
    actual = positive_frequencies(modal, 6)

    # A rectangular section has distinct analytical families in its two bending planes.
    beta_l = np.array([1.875104068711961, 4.694091132974174, 7.854757438237612])
    expected = np.sort(
        np.concatenate(
            [
                beta_l**2
                * np.sqrt(data["E"] * inertia / (data["rho"] * data["A"]))
                / (2.0 * np.pi * data["L"] ** 2)
                for inertia in (data["Iy"], data["Iz"])
            ]
        )
    )

    assert data["n_elements"] >= 10
    assert actual.size == 6
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.0)

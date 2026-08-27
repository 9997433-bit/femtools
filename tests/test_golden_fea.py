"""Analytical golden cases for the finite-element eigensolver."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import positive_frequencies


@pytest.mark.golden
def test_two_node_axial_bar_frequency(axial_bar: tuple[object, dict[str, float]]) -> None:
    """The one-free-DOF consistent-mass bar has an exact discrete frequency."""
    eigen = pytest.importorskip("femtools.fea.eigen")
    model, data = axial_bar

    modal = eigen.solve_modes(model, n_modes=1)
    actual = positive_frequencies(modal, 1)
    expected = np.sqrt(3.0 * data["E"] / data["rho"]) / (2.0 * np.pi * data["L"])

    assert actual.size == 1
    assert actual[0] == pytest.approx(expected, rel=1.0e-8)


@pytest.mark.golden
def test_euler_bernoulli_cantilever_first_three_bending_frequencies(
    cantilever: tuple[object, dict[str, float]],
) -> None:
    eigen = pytest.importorskip("femtools.fea.eigen")
    model, data = cantilever

    modal = eigen.solve_modes(model, n_modes=6)
    actual = positive_frequencies(modal, 3)

    # The circular/symmetric section has each bending mode in two planes.
    beta_l = np.array([1.875104068711961, 1.875104068711961, 4.694091132974174])
    scale = np.sqrt(data["E"] * data["I"] / (data["rho"] * data["A"]))
    expected = beta_l**2 * scale / (2.0 * np.pi * data["L"] ** 2)

    assert actual.size == 3
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.0)

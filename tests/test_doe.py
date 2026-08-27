"""Design-of-experiments invariants."""

from __future__ import annotations

import numpy as np

from femtools.optimization import doe as doe_module


def test_latin_hypercube_is_bounded_stratified_and_seeded() -> None:
    bounds = [(-2.0, 3.0), (10.0, 14.0), (0.25, 0.75)]
    n_samples = 24

    first = np.asarray(
        doe_module.latin_hypercube(bounds, n_samples=n_samples, seed=917),
        dtype=float,
    )
    second = np.asarray(
        doe_module.latin_hypercube(bounds, n_samples=n_samples, seed=917),
        dtype=float,
    )

    assert first.shape == (n_samples, len(bounds))
    np.testing.assert_array_equal(first, second)
    for column, (lower, upper) in enumerate(bounds):
        assert np.all(first[:, column] >= lower)
        assert np.all(first[:, column] <= upper)
        unit = (first[:, column] - lower) / (upper - lower)
        strata = np.minimum((unit * n_samples).astype(int), n_samples - 1)
        np.testing.assert_array_equal(np.sort(strata), np.arange(n_samples))


def test_full_factorial_contains_the_cartesian_product() -> None:
    levels = [[-1.0, 1.0], [10.0, 20.0, 30.0], [5.0]]

    design = np.asarray(doe_module.full_factorial(levels), dtype=float)
    actual = {tuple(row) for row in design}
    expected = {
        (first, second, third)
        for first in levels[0]
        for second in levels[1]
        for third in levels[2]
    }

    assert design.shape == (6, 3)
    assert actual == expected

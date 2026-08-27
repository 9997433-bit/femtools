"""Regression tests for the reusable HEX8 verification cases."""

from __future__ import annotations

import numpy as np

from femtools.fea.verification import (
    hex8_bending_ratio,
    hex8_patch_test_error,
    hex8_rigid_body_frequencies,
)


def test_hex8_default_avoids_shear_locking() -> None:
    default_ratio = hex8_bending_ratio()
    full_ratio = hex8_bending_ratio("full")

    assert default_ratio > 0.95
    assert 0.60 < full_ratio < 0.70
    assert default_ratio > full_ratio + 0.25


def test_hex8_distorted_patch_test_is_exact() -> None:
    assert hex8_patch_test_error() < 1.0e-10


def test_hex8_free_block_has_exactly_six_rigid_body_modes() -> None:
    frequencies = hex8_rigid_body_frequencies()

    assert frequencies.shape == (10,)
    assert np.all(np.isfinite(frequencies))
    assert np.all(np.diff(frequencies) >= 0.0)
    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0

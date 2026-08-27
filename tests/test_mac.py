"""Modal Assurance Criterion identities."""

from __future__ import annotations

import numpy as np

from femtools.correlation import mac as mac_module


def test_mac_of_orthonormal_basis_is_identity() -> None:
    rng = np.random.default_rng(20260827)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 5)))

    actual = np.asarray(mac_module.mac_matrix(basis, basis))

    np.testing.assert_allclose(actual, np.eye(5), rtol=0.0, atol=1.0e-12)


def test_mac_is_invariant_to_mode_scaling_phase_and_permutation() -> None:
    rng = np.random.default_rng(41)
    basis, _ = np.linalg.qr(rng.standard_normal((9, 4)))
    permutation = np.array([2, 0, 3, 1])
    factors = np.array([2.0, -3.0, 0.5j, -1.5j])
    transformed = basis[:, permutation].astype(complex) * factors

    actual = np.asarray(mac_module.mac_matrix(basis, transformed))
    expected = np.zeros((4, 4))
    expected[permutation, np.arange(4)] = 1.0

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    assert np.all((actual >= 0.0) & (actual <= 1.0 + 1.0e-14))

"""Golden low-mode check for free-interface Rubin component reduction."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from femtools.dynamics.cms_free import FreeCMSResult, rubin


def test_rubin_basis_retains_requested_free_interface_modes() -> None:
    n_dof = 6
    stiffness = np.diag(np.full(n_dof, 2.0))
    stiffness += np.diag(np.full(n_dof - 1, -1.0), 1)
    stiffness += np.diag(np.full(n_dof - 1, -1.0), -1)
    mass = np.diag(np.linspace(1.0, 1.5, n_dof))

    result = rubin(stiffness, mass, [0, n_dof - 1], n_modes=2)

    assert isinstance(result, FreeCMSResult)
    transformation = np.asarray(result.T)
    reduced_stiffness = np.asarray(result.K)
    reduced_mass = np.asarray(result.M)
    assert transformation.shape[0] == n_dof
    assert transformation.shape[1] == reduced_stiffness.shape[0]
    assert reduced_stiffness.shape == reduced_mass.shape
    np.testing.assert_allclose(reduced_stiffness, reduced_stiffness.T, atol=1.0e-12)
    np.testing.assert_allclose(reduced_mass, reduced_mass.T, atol=1.0e-12)

    full = eigh(stiffness, mass, eigvals_only=True)[:2]
    reduced = eigh(reduced_stiffness, reduced_mass, eigvals_only=True)[:2]
    np.testing.assert_allclose(reduced, full, rtol=1.0e-8, atol=1.0e-10)

"""Normalization contract shared by assembly and modal analysis."""

from __future__ import annotations

import numpy as np
import pytest

assemble_module = pytest.importorskip("femtools.fea.assemble")
eigen_module = pytest.importorskip("femtools.fea.eigen")


def test_modes_are_mass_normalized(
    cantilever: tuple[object, dict[str, float]],
) -> None:
    model, _ = cantilever
    assembly = assemble_module.assemble_km(model)
    modal = eigen_module.solve_modes(model, n_modes=5)

    phi = np.asarray(modal.modes)
    mass = assembly.M
    assert phi.ndim == 2
    assert mass.shape == (phi.shape[0], phi.shape[0])

    gram = phi.conj().T @ (mass @ phi)
    np.testing.assert_allclose(gram, np.eye(phi.shape[1]), rtol=0.0, atol=1.0e-8)

    generalized_mass = np.asarray(modal.generalized_mass)
    np.testing.assert_allclose(
        generalized_mass,
        np.ones(phi.shape[1]),
        rtol=0.0,
        atol=1.0e-8,
    )

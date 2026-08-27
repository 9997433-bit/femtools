"""Analytical checks for the Round 4 reduction and damped-mode APIs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg import eigh


def _api(module_name: str, name: str) -> Any:
    """Import an in-flight API, skipping until its implementation is present."""
    module = pytest.importorskip(module_name)
    function = getattr(module, name, None)
    if function is None:
        pytest.skip(f"{module_name}.{name} is not available")
    return function


def _transformation(result: Any) -> np.ndarray:
    if hasattr(result, "T"):
        return np.asarray(result.T)
    if isinstance(result, tuple):
        return np.asarray(result[0])
    return np.asarray(result)


def _reduced_stiffness(result: Any) -> np.ndarray:
    for name in ("K", "Kr", "Krr", "K_reduced"):
        if hasattr(result, name):
            return np.asarray(getattr(result, name))
    if isinstance(result, tuple) and len(result) >= 2:
        return np.asarray(result[1])
    raise AssertionError("Guyan result must expose the reduced stiffness matrix")


def _chain_matrices(n: int) -> tuple[np.ndarray, np.ndarray]:
    stiffness = np.diag(np.full(n, 2.0))
    stiffness += np.diag(np.full(n - 1, -1.0), 1)
    stiffness += np.diag(np.full(n - 1, -1.0), -1)
    mass = np.diag(np.linspace(1.0, 2.0, n))
    return stiffness, mass


def test_guyan_recovers_static_slave_displacements() -> None:
    reduction = pytest.importorskip("femtools.fea.reduction")
    guyan = getattr(reduction, "guyan", None)
    if guyan is None:
        pytest.skip("femtools.fea.reduction.guyan is not available")

    stiffness = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )
    result = guyan(stiffness, [0, 2])
    transformation = _transformation(result)
    expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    np.testing.assert_allclose(transformation, expected, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _reduced_stiffness(result),
        expected.T @ stiffness @ expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    # Static equilibrium of every slave column: K_ss T_s + K_sm = 0.
    np.testing.assert_allclose(
        stiffness[1:2, 1:2] @ transformation[1:2] + stiffness[1:2][:, [0, 2]],
        0.0,
        atol=1.0e-13,
    )


def test_irs_improves_the_first_guyan_eigenvalue() -> None:
    reduction = pytest.importorskip("femtools.fea.reduction")
    guyan = getattr(reduction, "guyan", None)
    irs = getattr(reduction, "irs", None)
    if guyan is None or irs is None:
        pytest.skip("Guyan and IRS reduction APIs are not both available")

    stiffness, mass = _chain_matrices(5)
    master = [0, 4]
    guyan_t = _transformation(guyan(stiffness, master))
    irs_t = _transformation(irs(stiffness, mass, master))

    full_first = eigh(stiffness, mass, eigvals_only=True)[0]
    guyan_first = eigh(
        guyan_t.T @ stiffness @ guyan_t,
        guyan_t.T @ mass @ guyan_t,
        eigvals_only=True,
    )[0]
    irs_first = eigh(
        irs_t.T @ stiffness @ irs_t,
        irs_t.T @ mass @ irs_t,
        eigvals_only=True,
    )[0]

    np.testing.assert_allclose(irs_t[master], np.eye(2), rtol=0.0, atol=1.0e-13)
    assert abs(irs_first - full_first) < 0.1 * abs(guyan_first - full_first)


def test_serep_exactly_reconstructs_the_retained_modal_subspace() -> None:
    serep = _api("femtools.fea.reduction", "serep")
    modes = np.array(
        [
            [1.0, 0.0],
            [0.2, 1.0],
            [1.0, 0.5],
            [-0.3, 0.4],
        ]
    )
    master = [0, 1]

    transformation = _transformation(serep(modes, master))

    np.testing.assert_allclose(transformation[master], np.eye(2), atol=1.0e-13)
    np.testing.assert_allclose(
        transformation @ modes[master],
        modes,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_complex_modes_recover_decoupled_frequency_and_damping() -> None:
    eigen = pytest.importorskip("femtools.fea.eigen")
    solve_complex_modes = getattr(eigen, "solve_complex_modes", None)
    result_type = getattr(eigen, "ComplexModalResult", None)
    if solve_complex_modes is None or result_type is None:
        pytest.skip("complex modal API is not available")

    expected_frequency = np.array([3.0, 8.0])
    expected_damping = np.array([0.02, 0.05])
    omega = 2.0 * np.pi * expected_frequency
    mass = np.eye(2)
    stiffness = np.diag(omega**2)
    damping = np.diag(2.0 * expected_damping * omega)

    result = solve_complex_modes(stiffness, mass, damping)

    assert isinstance(result, result_type)
    np.testing.assert_allclose(result.freq_hz, expected_frequency, rtol=2.0e-6)
    np.testing.assert_allclose(result.zeta, expected_damping, rtol=2.0e-6)
    modes = np.asarray(result.modes_complex)
    assert modes.shape == (2, 2)
    assert np.iscomplexobj(modes)

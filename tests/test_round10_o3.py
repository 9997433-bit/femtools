"""Round 10 O3 — expanded MAC: SEREP expansion composed with the MAC.

The gate is the SEREP fixed point (O'Callahan, Avitabile & Riemer, Proc. 7th
IMAC, 1989): an analytical mode set restricted to a master DOF subset and
expanded with the same basis must come back unchanged, so its MAC (Allemang)
against the original full-DOF modes has a unit diagonal.  The *whole* table is
the identity only when the reference modes are mutually uncorrelated under the
criterion — an orthonormal basis, or mass-normalized FE modes weighted by the
mass matrix — which is checked separately here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from femtools.core.model import FEModel
from femtools.correlation import expanded_mac as expanded_mac_from_package
from femtools.correlation.expansion import (
    ExpandedMACResult,
    ExpansionResult,
    expand_serep,
    expanded_mac,
)
from femtools.correlation.mac import mac_matrix, modal_scale_factor
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes

#: Tolerance of the identity gate.
TOL = 1.0e-10


def orthonormal_basis(n_dof: int = 60, n_mode: int = 8, seed: int = 20261010) -> np.ndarray:
    """A deterministic orthonormal stand-in for a full-DOF FE mode set."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n_dof, n_mode)))
    return np.asarray(basis)


def beam_model(n_elements: int = 16) -> FEModel:
    """A slender cantilever with unequal bending stiffnesses in the two planes."""
    model = FEModel(name="round10-o3-cantilever")
    length = 2.0
    for index in range(n_elements + 1):
        model.add_node(id=index + 1, xyz=(length * index / n_elements, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.3, rho=2700.0)
    model.add_property(id=1, type="beam", material_id=1, A=8.0e-4, Iy=3.0e-8, Iz=6.0e-8, J=9.0e-8)
    for index in range(n_elements):
        model.add_element(id=index + 1, type="BEAM2", nodes=(index + 1, index + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model


def beam_modes(n_modes: int = 6) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """``(modal, phi, master, mass)`` for the cantilever.

    The masters are the two lateral translations of every other node: enough
    to resolve the bending modes of both planes, far from every DOF of the
    model (16 of 102).
    """
    model = beam_model()
    assembly = assemble_km(model)
    modal = solve_modes(model, n_modes=n_modes, assembly=assembly)
    rows = []
    for node in range(2, 18, 2):
        base = (node - 1) * 6
        rows += [base + 1, base + 2]
    return modal, np.asarray(modal.modes, dtype=float), np.asarray(rows), assembly.M


# --------------------------------------------------------------------------
# the gate: expanding a mode set onto itself
# --------------------------------------------------------------------------


def test_serep_self_expansion_has_unit_mac_diagonal() -> None:
    """FE modes at the masters, SEREP-expanded, correlate at 1 with themselves."""
    modal, phi, master, _ = beam_modes()

    result = expanded_mac(phi[master], modal, master)

    assert isinstance(result, ExpandedMACResult)
    assert result.mac.shape == (phi.shape[1], phi.shape[1])
    np.testing.assert_allclose(result.diagonal, np.ones(phi.shape[1]), rtol=0.0, atol=TOL)
    assert result.diagonal_error <= TOL
    # The fit is exact, so nothing of the "measurement" is left unrepresented.
    assert float(np.max(result.residual)) <= TOL


def test_expanded_mac_of_orthonormal_basis_is_the_identity() -> None:
    """With an orthonormal reference the whole table, not only its diagonal, is I."""
    basis = orthonormal_basis()
    rng = np.random.default_rng(7)
    master = np.sort(rng.choice(basis.shape[0], 20, replace=False))

    result = expanded_mac(basis[master], basis, master)

    np.testing.assert_allclose(np.asarray(result), np.eye(basis.shape[1]), rtol=0.0, atol=TOL)
    assert result.identity_error <= TOL
    assert result.max_off_diagonal <= TOL


def test_mass_weighted_expanded_mac_of_fe_modes_is_the_identity() -> None:
    """Mass-normalized FE modes are orthogonal under ``weights=M``, so the table is I."""
    modal, phi, master, mass = beam_modes()

    result = expanded_mac(phi[master], modal, master, weights=mass)

    np.testing.assert_allclose(np.asarray(result), np.eye(phi.shape[1]), rtol=0.0, atol=TOL)
    assert result.identity_error <= TOL


def test_unweighted_off_diagonal_is_the_auto_mac_of_the_fe_modes() -> None:
    """What the plain MAC leaves off the diagonal belongs to the modes, not to SEREP."""
    modal, phi, master, _ = beam_modes()

    result = expanded_mac(phi[master], modal, master)

    auto = mac_matrix(phi, phi)
    np.testing.assert_allclose(np.asarray(result), auto, rtol=0.0, atol=TOL)
    # The FE beam modes are mass- but not Euclidean-orthogonal: this is why
    # the gate is the diagonal and not the full identity.
    assert result.max_off_diagonal > 0.1
    assert result.diagonal_error <= TOL


def test_expanded_shapes_reproduce_the_fe_modes_up_to_scaling() -> None:
    """The unit diagonal is not a MAC artefact: the shapes themselves come back."""
    modal, phi, master, _ = beam_modes()

    result = expanded_mac(phi[master], modal, master)

    expanded = np.asarray(result.expansion)
    rescaled = expanded * modal_scale_factor(phi, expanded)
    assert float(np.abs(rescaled - phi).max() / np.abs(phi).max()) <= TOL


def test_full_size_test_matrix_is_restricted_to_the_masters() -> None:
    """``expanded_mac(modal, modal, master)`` is the self-check spelled short."""
    modal, phi, master, _ = beam_modes()

    convenience = expanded_mac(modal, modal, master)
    explicit = expanded_mac(phi[master], modal, master)

    np.testing.assert_allclose(convenience.mac, explicit.mac, rtol=0.0, atol=0.0)
    assert convenience.n_master == master.size
    # Frequencies travel with the modal result and reach the report.
    assert convenience.freq_test is not None
    np.testing.assert_allclose(convenience.freq_test, np.asarray(modal.freq_hz, dtype=float))
    assert "master DOF" in convenience.table()


def test_boolean_mask_and_dof_keys_select_the_same_masters() -> None:
    modal, phi, master, _ = beam_modes()
    mask = np.zeros(phi.shape[0], dtype=bool)
    mask[master] = True
    keys = [f"{node}{component}" for node in range(2, 18, 2) for component in ("Y", "Z")]

    by_mask = expanded_mac(modal, modal, mask)
    by_key = expanded_mac(modal, modal, keys)

    np.testing.assert_array_equal(by_mask.master, master)
    np.testing.assert_array_equal(by_key.master, master)
    np.testing.assert_allclose(by_key.mac, by_mask.mac, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------
# composition, container and argument handling
# --------------------------------------------------------------------------


def test_result_unpacks_as_mac_and_expansion() -> None:
    basis = orthonormal_basis(n_dof=40, n_mode=5, seed=3)
    master = np.arange(0, 40, 3)

    result = expanded_mac(basis[master], basis, master)
    mac, expansion = result

    assert isinstance(expansion, ExpansionResult)
    assert expansion.method == "serep"
    np.testing.assert_array_equal(mac, np.asarray(result))
    np.testing.assert_array_equal(result[0], mac[0])
    assert result.shape == mac.shape
    np.testing.assert_array_equal(result.modes, expansion.modes)
    np.testing.assert_array_equal(result.master, master)


def test_composition_matches_expand_serep_then_mac_matrix() -> None:
    """The function is the two published steps and nothing else."""
    basis = orthonormal_basis(n_dof=30, n_mode=4, seed=11)
    rng = np.random.default_rng(101)
    master = np.sort(rng.choice(30, 12, replace=False))
    measured = basis[master] @ np.array([[1.0, 0.4], [0.0, 1.0], [-0.3, 0.2], [0.1, -0.5]])

    result = expanded_mac(measured, basis, master)
    expansion = expand_serep(measured, basis, master)
    expected = mac_matrix(expansion.modes, basis)

    np.testing.assert_allclose(result.mac, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(result.expansion), np.asarray(expansion), atol=0.0)


def test_complex_modes_keep_the_unit_diagonal() -> None:
    """Damped (complex) shapes go through the Hermitian MAC unchanged."""
    real = orthonormal_basis(n_dof=32, n_mode=4, seed=13)
    imag = orthonormal_basis(n_dof=32, n_mode=4, seed=19)
    basis = real + 0.25j * imag
    master = np.arange(0, 32, 2)

    result = expanded_mac(basis[master], basis, master)

    assert np.iscomplexobj(result.expansion.modes)
    np.testing.assert_allclose(result.diagonal, np.ones(4), rtol=0.0, atol=TOL)


def test_reference_argument_correlates_against_another_mode_set() -> None:
    """A permuted, rescaled reference gives the permutation matrix back."""
    basis = orthonormal_basis(n_dof=36, n_mode=4, seed=5)
    master = np.arange(0, 36, 2)
    permutation = np.array([2, 0, 3, 1])
    reference = basis[:, permutation] * np.array([2.0, -3.0, 0.5, -1.5])

    result = expanded_mac(basis[master], basis, master, reference=reference)

    expected = np.zeros((4, 4))
    expected[permutation, np.arange(4)] = 1.0
    np.testing.assert_allclose(np.asarray(result), expected, rtol=0.0, atol=TOL)


def test_truncated_basis_keeps_the_retained_modes_and_reports_the_rest() -> None:
    """A shape outside the retained span is filtered out, and the residual says so."""
    basis = orthonormal_basis(n_dof=50, n_mode=6, seed=17)
    master = np.arange(0, 50, 2)
    # A fifth "measured" shape the retained basis cannot follow at the masters.
    rng = np.random.default_rng(23)
    outside = rng.standard_normal(master.size)
    measured = np.column_stack([basis[master, :4], outside])

    result = expanded_mac(measured, basis, master, n_modes=4)

    assert result.mac.shape == (5, 4)
    np.testing.assert_allclose(result.diagonal, np.ones(4), rtol=0.0, atol=TOL)
    assert float(np.max(result.residual[:4])) <= TOL
    assert float(result.residual[4]) > 0.5
    assert result.reference.shape == (50, 4)
    # The rectangular table still reports every measured mode, unpaired ones
    # included.
    assert len(result.table().splitlines()) == 5 + 3


def test_return_transform_exposes_the_serep_expansion_matrix() -> None:
    basis = orthonormal_basis(n_dof=24, n_mode=3, seed=29)
    master = np.arange(0, 24, 2)

    result = expanded_mac(basis[master], basis, master, return_transform=True)

    transform = result.expansion.transform
    assert transform is not None and transform.shape == (24, master.size)
    np.testing.assert_allclose(transform @ basis[master], np.asarray(result.expansion), atol=1e-12)


def test_master_set_blind_to_one_mode_loses_its_diagonal_entry() -> None:
    """A mode with no amplitude at any sensor cannot survive the fixed point."""
    basis = orthonormal_basis(n_dof=40, n_mode=6, seed=31)
    master = np.arange(0, 40, 2)
    # The last mode lives entirely on the unmeasured DOFs, the way an
    # antisymmetric mode does for sensors placed on the symmetry line.
    invisible = np.zeros(basis.shape[0])
    invisible[np.setdiff1d(np.arange(40), master)] = 1.0
    reference = np.column_stack([basis[:, :5], invisible / np.linalg.norm(invisible)])

    result = expanded_mac(reference[master], reference, master)

    np.testing.assert_allclose(result.diagonal[:5], np.ones(5), rtol=0.0, atol=TOL)
    assert result.diagonal[5] == 0.0
    assert result.min_diagonal == 0.0


def test_invalid_arguments_are_rejected() -> None:
    basis = orthonormal_basis(n_dof=20, n_mode=3, seed=41)
    master = np.arange(0, 20, 2)

    with pytest.raises(ValueError, match="reference has"):
        expanded_mac(basis[master], basis, master, reference=np.ones((7, 3)))
    with pytest.raises(ValueError, match="n_modes"):
        expanded_mac(basis[master], basis, master, n_modes=9)
    with pytest.raises(ValueError, match="no master DOF"):
        expanded_mac(basis[master], basis, np.zeros(20, dtype=bool))


def test_exported_from_the_correlation_package() -> None:
    import femtools.correlation as correlation

    assert expanded_mac_from_package is expanded_mac
    assert "expanded_mac" in correlation.__all__
    assert "ExpandedMACResult" in correlation.__all__
    from femtools.correlation import expansion

    assert "expanded_mac" in expansion.__all__
    assert "ExpandedMACResult" in expansion.__all__

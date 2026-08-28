"""Round 10 / O2: ``residual_flexibility`` as the upper residual of a truncated FRF.

A modal sum knows nothing about the modes it left out, and what it is missing in the
retained band is almost entirely their *static* compliance — they are excited far below
their own resonances, so they respond like springs. Subtracting the retained content
from ``K^-1 F`` leaves exactly that compliance (MacNeal residual flexibility; Ewins,
*Modal Testing*, §4.2; Craig & Kurdila, *Fundamentals of Structural Dynamics*, ch. 21),
and adding it back as ``modal_frf(..., upper_residual=...)`` is the cheapest correction
there is: one static solve, no extra eigenpairs.

The gate is the last test here — four modes of the sixteen-element cantilever, with and
without the residual, against ``direct_frf`` on the same physical matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.linalg as sla

from femtools.dynamics import residual_flexibility as residual_flexibility_export
from femtools.dynamics.frf import direct_frf, modal_frf, retained_band_lines
from femtools.dynamics.modal import ModalModel
from femtools.dynamics.residuals import (
    ResidualVectorResult,
    residual_flexibility,
    residual_vectors,
)

STIFF = 4.3e4
MASS = 1.7


def _chain(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Spring chain of ``n`` equal masses, grounded at DOF 0."""
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i] += STIFF
        K[i + 1, i + 1] += STIFF
        K[i, i + 1] -= STIFF
        K[i + 1, i] -= STIFF
    K[0, 0] += STIFF
    return K, MASS * np.eye(n)


def _basis(K: np.ndarray, M: np.ndarray, n_modes: int | None = None) -> ModalModel:
    """Mass-normalised modal model, optionally truncated to the lowest ``n_modes``."""
    lam, phi = sla.eigh(K, M)
    lam = np.clip(lam, 0.0, None)
    if n_modes is not None:
        lam, phi = lam[:n_modes], phi[:, :n_modes]
    return ModalModel(
        freq_hz=np.sqrt(lam) / (2.0 * np.pi),
        modes=phi,
        generalized_mass=np.ones(lam.size),
        eigenvalues=lam,
    )


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _rayleigh(basis: ModalModel, zeta: float) -> dict[str, float]:
    """Rayleigh coefficients hitting ``zeta`` at the first and last retained mode.

    A physical ``C`` keeps the modal and the direct side damped identically, so the two
    differ by modal truncation and nothing else.
    """
    w_first, w_last = 2.0 * np.pi * np.asarray(basis.freq_hz)[[0, -1]]
    return {
        "alpha": 2.0 * zeta * w_first * w_last / (w_first + w_last),
        "beta": 2.0 * zeta / (w_first + w_last),
    }


# ----------------------------------------------------------------------
# what the matrix is
# ----------------------------------------------------------------------
def test_residual_flexibility_is_the_static_compliance_the_basis_is_missing() -> None:
    """``K^-1 F`` minus the retained modal content, term by term."""
    K, M = _chain(12)
    kept = _basis(K, M, 4)
    drive, sense = 11, 7

    got = residual_flexibility(K, M, kept, inputs=[drive], outputs=[sense])

    full_compliance = np.linalg.inv(K)[sense, drive]
    retained = float(
        np.sum(kept.modes[sense, :] * kept.modes[drive, :] / np.asarray(kept.eigenvalues))
    )
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(full_compliance - retained, rel=1.0e-12)
    # The modes left out are a positive-definite pile of springs in series with the
    # retained ones, so a drive-point residual can only be positive.
    assert residual_flexibility(K, M, kept, inputs=[drive], outputs=[drive])[0, 0] > 0.0


def test_a_complete_basis_leaves_no_residual() -> None:
    K, M = _chain(10)
    full = _basis(K, M)
    UR = residual_flexibility(K, M, full, inputs=[9], outputs=None)
    assert UR.shape == (10, 1)
    assert np.abs(UR).max() < 1.0e-12 * np.abs(np.linalg.inv(K)).max()


def test_the_block_is_the_attribute_of_residual_vectors_selected() -> None:
    """The function is a view on ``ResidualVectorResult.residual_flexibility``."""
    K, M = _chain(12)
    kept = _basis(K, M, 5)
    loads = [3, 11]
    result = residual_vectors(K, M, kept, loads)

    assert isinstance(result, ResidualVectorResult)
    assert isinstance(result.residual_flexibility, np.ndarray)
    assert result.residual_flexibility.shape == (12, 2)
    np.testing.assert_allclose(
        residual_flexibility(K, M, kept, loads, outputs=[7]),
        result.residual_flexibility[[7], :],
        rtol=0.0,
        atol=0.0,
    )
    # An already-solved result can be re-sliced without a second static solve.
    np.testing.assert_array_equal(
        residual_flexibility(result, outputs=[7], inputs=[11]),
        result.upper_residual([7], [11]),
    )


def test_shape_follows_the_output_and_input_selection() -> None:
    K, M = _chain(12)
    kept = _basis(K, M, 3)
    assert residual_flexibility(K, M, kept, [2, 5, 9]).shape == (12, 3)
    assert residual_flexibility(K, M, kept, [2, 5, 9], outputs=[1, 4]).shape == (2, 3)
    assert residual_flexibility(K, M, kept, [2, 5, 9], inputs=[5]).shape == (12, 1)
    assert residual_flexibility(K, M, kept, inputs=[2, 9], outputs=[0]).shape == (1, 2)


def test_a_free_free_residual_carries_no_rigid_body_content() -> None:
    """Inertia relief, inherited from ``residual_vectors``: no drift in the residual."""
    n = 8
    K, M = _chain(n)
    K[0, 0] -= STIFF  # unground it
    kept = _basis(K, M, 4)
    UR = residual_flexibility(K, M, kept, inputs=[n - 1], outputs=None)
    rigid = kept.mass_normalized().modes[:, :1]
    assert np.abs(rigid.T @ M @ UR).max() < 1.0e-9 * np.abs(UR).max()


def test_bad_calls_are_refused() -> None:
    K, M = _chain(6)
    kept = _basis(K, M, 2)
    with pytest.raises(TypeError, match="needs both M and modal"):
        residual_flexibility(K, M)
    with pytest.raises(TypeError, match="only outputs/inputs"):
        residual_flexibility(residual_vectors(K, M, kept, [5]), M, kept)


def test_the_name_is_exported() -> None:
    from femtools.dynamics import residuals

    assert residual_flexibility_export is residual_flexibility
    assert "residual_flexibility" in residuals.__all__
    assert "residual_vectors" in residuals.__all__


# ----------------------------------------------------------------------
# the gate: the residual lowers the truncated-FRF error against direct_frf
# ----------------------------------------------------------------------
def test_upper_residual_lowers_the_truncated_frf_error_on_a_chain() -> None:
    K, M = _chain(24)
    kept = _basis(K, M, 4)
    drive, sense = 23, 12
    out = [drive, sense]
    f = retained_band_lines(kept, 300)
    damping = _rayleigh(kept, 0.02)

    reference = direct_frf(K, M, [drive], out, f, damping).H
    plain = modal_frf(kept, [drive], out, f, damping).H
    UR = residual_flexibility(K, M, kept, inputs=[drive], outputs=out)
    corrected = modal_frf(kept, [drive], out, f, damping, upper_residual=UR).H

    err_plain = _rel_l2(plain, reference)
    err_corrected = _rel_l2(corrected, reference)
    assert err_corrected < err_plain
    assert err_corrected < 0.25 * err_plain


def test_upper_residual_lowers_the_truncated_frf_error_on_the_cantilever(
    cantilever: tuple[Any, dict[str, float]],
) -> None:
    """The gate, on the FE model the 20-mode 5 % golden is measured on.

    Four modes instead of twenty, one static solve instead of sixteen extra eigenpairs.
    """
    from femtools.fea.assemble import assemble_km
    from femtools.fea.eigen import solve_modes

    model, data = cantilever
    assembly = assemble_km(model)
    n_retained = 4
    solved = solve_modes(model, n_modes=n_retained, assembly=assembly)
    kept = ModalModel(
        freq_hz=np.asarray(solved.freq_hz)[:n_retained],
        modes=np.asarray(solved.modes)[assembly.free_dof, :n_retained],
        generalized_mass=np.asarray(solved.generalized_mass)[:n_retained],
        eigenvalues=np.asarray(solved.eigenvalues)[:n_retained],
    )

    def free_index(node_id: int, component: int) -> int:
        return int(
            np.flatnonzero(assembly.free_dof == assembly.dof_map.index(node_id, component))[0]
        )

    tip = free_index(int(data["n_elements"]) + 1, 2)
    mid = free_index(int(data["n_elements"]) // 2 + 1, 2)
    f = retained_band_lines(kept, 300)
    damping = _rayleigh(kept, 0.02)

    reference = direct_frf(assembly.Kff, assembly.Mff, [tip], [tip, mid], f, damping).H
    plain = modal_frf(kept, [tip], [tip, mid], f, damping).H
    UR = residual_flexibility(assembly.Kff, assembly.Mff, kept, inputs=[tip], outputs=[tip, mid])
    corrected = modal_frf(kept, [tip], [tip, mid], f, damping, upper_residual=UR).H

    err_plain = _rel_l2(plain, reference)
    err_corrected = _rel_l2(corrected, reference)
    assert UR.shape == (2, 1)
    assert err_corrected < err_plain
    assert err_corrected < 0.25 * err_plain

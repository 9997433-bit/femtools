"""Free-interface CMS: a chain split in two must re-assemble into the full model.

The parent structure is a 40-mass spring chain grounded at its left end. It is cut at
mass 20, whose mass is shared equally by the two halves, so the *exact* assembly of the
two components reproduces the parent matrices — any error in the reduced result is
therefore modal truncation and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtools.dynamics.cms_free import (
    FreeCMSResult,
    free_interface_assembly,
    macneal,
    rubin,
)
from femtools.dynamics.mba import modal_based_assembly
from femtools.dynamics.modal import ModalModel

N_MASSES = 40
CUT = 20  # component A holds masses 1..CUT, component B holds masses CUT..N
MASS = 1.7
STIFF = 4.3e4


def _chain(n: int, mass: np.ndarray, grounded: bool) -> tuple[np.ndarray, np.ndarray]:
    """Spring chain of ``n`` masses; ``grounded`` adds a spring to ground at DOF 0."""
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i] += STIFF
        K[i + 1, i + 1] += STIFF
        K[i, i + 1] -= STIFF
        K[i + 1, i] -= STIFF
    if grounded:
        K[0, 0] += STIFF
    return K, np.diag(mass)


def _full_model() -> tuple[np.ndarray, np.ndarray]:
    return _chain(N_MASSES, np.full(N_MASSES, MASS), grounded=True)


def _components() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The two halves. A is grounded, B is free-free; the cut mass is split in half."""
    mass_a = np.full(CUT, MASS)
    mass_a[-1] = 0.5 * MASS
    mass_b = np.full(N_MASSES - CUT + 1, MASS)
    mass_b[0] = 0.5 * MASS
    return _chain(CUT, mass_a, grounded=True), _chain(N_MASSES - CUT + 1, mass_b, False)


def _frequencies(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    import scipy.linalg as sla

    lam = np.clip(sla.eigh(K, M, eigvals_only=True), 0.0, None)
    return np.sqrt(np.sort(lam)) / (2.0 * np.pi)


def _full_frequencies() -> np.ndarray:
    return _frequencies(*_full_model())


def _split_is_exact() -> None:
    """The two halves really do add up to the parent model (fixture self-check)."""
    (Ka, Ma), (Kb, Mb) = _components()
    K, M = _full_model()
    n = N_MASSES
    Kt = np.zeros((n, n))
    Mt = np.zeros((n, n))
    Kt[:CUT, :CUT] += Ka
    Mt[:CUT, :CUT] += Ma
    Kt[CUT - 1 :, CUT - 1 :] += Kb
    Mt[CUT - 1 :, CUT - 1 :] += Mb
    assert np.allclose(Kt, K)
    assert np.allclose(Mt, M)


def test_split_chain_fixture_reassembles_to_the_parent() -> None:
    _split_is_exact()


def _assembled(method: str, n_modes: int) -> np.ndarray:
    (Ka, Ma), (Kb, Mb) = _components()
    build = rubin if method == "rubin" else macneal
    a = build(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=n_modes)
    b = build(Kb, Mb, boundary_dofs=[0], n_modes=n_modes)
    assembled = free_interface_assembly(
        [("A", a), ("B", b)], [("A", CUT - 1, "B", 0)]
    )
    return assembled.freq_hz


@pytest.mark.parametrize("method", ["rubin", "macneal"])
def test_free_interface_cms_matches_the_full_model(method: str) -> None:
    """Six free modes per half plus residual flexibility: a few percent, worst case."""
    reference = _full_frequencies()
    got = _assembled(method, n_modes=6)

    n_check = 8
    error = np.abs(got[:n_check] - reference[:n_check]) / reference[:n_check]
    assert np.all(error < 0.03), (method, error)
    # The band the components actually resolve is reproduced far better than that.
    assert np.all(error[:4] < 2.0e-3), (method, error[:4])


def test_more_modes_converge_towards_the_full_model() -> None:
    reference = _full_frequencies()
    errors = []
    for n_modes in (4, 8, 12):
        got = _assembled("rubin", n_modes)
        errors.append(float(np.max(np.abs(got[:8] - reference[:8]) / reference[:8])))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-5


def test_residual_flexibility_is_what_buys_the_accuracy() -> None:
    """Coupling the same truncated modal bases without residuals is much worse."""
    reference = _full_frequencies()
    (Ka, Ma), (Kb, Mb) = _components()

    plain = []
    for K, M, in ((Ka, Ma), (Kb, Mb)):
        cms = rubin(K, M, boundary_dofs=[CUT - 1 if K is Ka else 0], n_modes=6)
        plain.append(
            ModalModel(
                freq_hz=cms.free_freq_hz,
                modes=cms.normal_modes,
                eigenvalues=(2.0 * np.pi * cms.free_freq_hz) ** 2,
            )
        )
    truncated = modal_based_assembly(
        [("A", plain[0]), ("B", plain[1])], [("A", CUT - 1, "B", 0)]
    ).freq_hz

    with_residuals = _assembled("rubin", n_modes=6)
    n_check = 8
    err_plain = np.abs(truncated[:n_check] - reference[:n_check]) / reference[:n_check]
    err_cms = np.abs(with_residuals[:n_check] - reference[:n_check]) / reference[:n_check]
    assert err_plain.max() > 10.0 * err_cms.max()


def test_rubin_component_is_a_modal_model_usable_by_mba() -> None:
    """A Rubin component has a regular mass matrix, so plain MBA can couple it."""
    reference = _full_frequencies()
    (Ka, Ma), (Kb, Mb) = _components()
    a = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=6).solve_modes()
    b = rubin(Kb, Mb, boundary_dofs=[0], n_modes=6).solve_modes()
    got = modal_based_assembly([("A", a), ("B", b)], [("A", CUT - 1, "B", 0)]).freq_hz
    error = np.abs(got[:8] - reference[:8]) / reference[:8]
    assert np.all(error < 0.03), error
    assert np.allclose(got[:8], _assembled("rubin", 6)[:8], rtol=1e-8)


def test_complete_basis_is_an_exact_change_of_basis() -> None:
    """With every free mode kept there is no residual left and CMS is exact."""
    (Ka, Ma), (Kb, Mb) = _components()
    a = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=CUT)
    b = rubin(Kb, Mb, boundary_dofs=[0], n_modes=N_MASSES - CUT + 1)
    assert a.n_residual == 0 and b.n_residual == 0
    got = free_interface_assembly([a, b], [(0, CUT - 1, 1, 0)]).freq_hz
    reference = _full_frequencies()
    assert np.allclose(got[: reference.size], reference, rtol=1e-8, atol=1e-8)


def test_reduced_model_has_the_textbook_block_structure() -> None:
    Ka, Ma = _components()[0]
    res = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=5)
    assert isinstance(res, FreeCMSResult)
    assert res.n_kept == 5 and res.n_residual == 1
    assert res.n_reduced == res.T.shape[1] == 6

    # M = I, K = diag(omega_r^2, residual): the residual modes are M- and K-orthogonal
    # to the retained normal modes by construction.
    assert np.allclose(res.M, np.eye(6), atol=1e-9)
    assert np.allclose(res.K, np.diag(np.diag(res.K)), atol=1e-6 * np.abs(res.K).max())
    assert np.allclose(
        np.sqrt(np.diag(res.K)[:5]) / (2.0 * np.pi), res.free_freq_hz, rtol=1e-8
    )
    # The residual pseudo-mode sits above the retained band, as it must.
    assert res.residual_freq_hz[0] > 1.5 * res.free_freq_hz[-1]
    assert res.meta["mass_coupling"] < 1e-8


def test_macneal_drops_only_the_residual_inertia() -> None:
    Ka, Ma = _components()[0]
    r = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=5)
    m = macneal(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=5)
    assert np.allclose(r.K, m.K)
    assert np.allclose(r.T, m.T)
    assert np.allclose(m.M[:5, :5], np.eye(5), atol=1e-9)
    assert np.allclose(m.M[5:, :], 0.0)
    # A massless coordinate cannot oscillate: the free component's own modes are the
    # retained normal modes again, and the residual set only shows up under coupling.
    assert np.allclose(m.solve_modes().freq_hz, m.free_freq_hz, rtol=1e-8)
    assert np.allclose(r.solve_modes().freq_hz[:5], r.free_freq_hz, rtol=1e-8)


def test_free_free_component_keeps_its_rigid_body_mode() -> None:
    Kb, Mb = _components()[1]
    res = rubin(Kb, Mb, boundary_dofs=[0], n_modes=6)
    assert res.meta["n_rigid"] == 1
    assert res.free_freq_hz[0] < 1e-6
    # Inertia relief must leave no rigid-body content in the residual flexibility.
    rigid = res.normal_modes[:, :1]
    assert np.abs(rigid.T @ Mb @ res.residual_flexibility).max() < 1e-8


def test_truncating_the_rigid_body_set_is_refused() -> None:
    """Inertia relief is silently wrong with a missing rigid-body mode, so refuse it."""
    Kb, Mb = _components()[1]
    n = Kb.shape[0]
    K = np.zeros((2 * n, 2 * n))
    M = np.zeros((2 * n, 2 * n))
    for offset in (0, n):
        K[offset : offset + n, offset : offset + n] = Kb
        M[offset : offset + n, offset : offset + n] = Mb

    assert rubin(K, M, boundary_dofs=[0, n], n_modes=4).meta["n_rigid"] == 2
    with pytest.raises(ValueError, match="rigid-body set"):
        rubin(K, M, boundary_dofs=[0, n], n_modes=1)


def test_interface_flexibility_matches_the_static_residual() -> None:
    """G_d[b, b] is the static compliance the truncated modal model is missing."""
    Ka, Ma = _components()[0]
    res = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=6)
    exact = np.linalg.solve(Ka, np.eye(CUT)[:, CUT - 1])[CUT - 1]
    modal_part = float(
        np.sum(res.normal_modes[CUT - 1, :] ** 2 / (2.0 * np.pi * res.free_freq_hz) ** 2)
    )
    assert float(res.interface_flexibility()[0, 0]) == pytest.approx(
        exact - modal_part, rel=1e-8
    )


def test_supplied_modal_basis_is_used_as_is() -> None:
    Ka, Ma = _components()[0]
    solved = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=6)
    basis = ModalModel(
        freq_hz=solved.free_freq_hz,
        modes=solved.normal_modes,
        eigenvalues=(2.0 * np.pi * solved.free_freq_hz) ** 2,
    )
    reused = rubin(Ka, Ma, boundary_dofs=[CUT - 1], modal=basis)
    assert np.allclose(reused.K, solved.K)
    assert np.allclose(np.abs(reused.T), np.abs(solved.T))

    with pytest.raises(ValueError, match="n_modes is required"):
        rubin(Ka, Ma, boundary_dofs=[CUT - 1])


def test_elastic_interface_tie_softens_the_assembly() -> None:
    (Ka, Ma), (Kb, Mb) = _components()
    a = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=6)
    b = rubin(Kb, Mb, boundary_dofs=[0], n_modes=6)
    rigid = free_interface_assembly([a, b], [(0, CUT - 1, 1, 0)]).freq_hz
    soft = free_interface_assembly(
        [a, b], [(0, CUT - 1, 1, 0, 0.05 * STIFF)]
    ).freq_hz
    assert np.all(soft[:6] < rigid[:6])
    stiff = free_interface_assembly(
        [a, b], [(0, CUT - 1, 1, 0, 1.0e5 * STIFF)]
    ).freq_hz
    assert np.allclose(stiff[:6], rigid[:6], rtol=1e-4)


def test_rubin_basis_fixes_the_truncated_frf_of_a_beam(
    cantilever: tuple[object, dict[str, float]],
) -> None:
    """On an FE model: the same six modes, plus residual flexibility, against direct.

    Truncation shows up in an FRF as a missing static compliance, which is exactly what
    the residual mode restores — so the modal-vs-direct error of ``docs/CONTRACT_API.md``
    collapses without retaining a single extra normal mode.
    """
    from femtools.dynamics.frf import direct_frf, modal_frf, retained_band_lines
    from femtools.fea.assemble import assemble_km

    model, data = cantilever
    assembly = assemble_km(model)
    tip_dof = int(
        np.flatnonzero(
            assembly.free_dof == assembly.dof_map.index(int(data["n_elements"]) + 1, 2)
        )[0]
    )

    cms = rubin(assembly.Kff, assembly.Mff, boundary_dofs=[tip_dof], n_modes=6)
    truncated = ModalModel(
        freq_hz=cms.free_freq_hz,
        modes=cms.normal_modes,
        eigenvalues=(2.0 * np.pi * cms.free_freq_hz) ** 2,
    )
    f = retained_band_lines(truncated, 300)
    damping = {"beta": 2.0 * 0.02 / (2.0 * np.pi * float(cms.free_freq_hz[-1]))}

    reference = direct_frf(assembly.Kff, assembly.Mff, [tip_dof], [tip_dof], f, damping).H

    def error(basis: ModalModel) -> float:
        H = modal_frf(basis, [tip_dof], [tip_dof], f, damping).H
        return float(np.linalg.norm(H - reference) / np.linalg.norm(reference))

    assert cms.n_residual == 1
    assert error(cms.solve_modes()) < 0.1 * error(truncated)
    assert error(cms.solve_modes()) < 1.0e-3


def test_sparse_matrices_give_the_same_reduction() -> None:
    import scipy.sparse as sp

    reference = _full_frequencies()
    (Ka, Ma), (Kb, Mb) = _components()
    a = rubin(sp.csr_matrix(Ka), sp.csr_matrix(Ma), boundary_dofs=[CUT - 1], n_modes=6)
    b = rubin(sp.csr_matrix(Kb), sp.csr_matrix(Mb), boundary_dofs=[0], n_modes=6)
    assert a.meta["sparse"] and b.meta["sparse"]
    assert b.meta["n_rigid"] == 1  # shift-invert below zero found the free-free mode
    got = free_interface_assembly([a, b], [(0, CUT - 1, 1, 0)]).freq_hz
    assert np.allclose(got[:8], _assembled("rubin", 6)[:8], rtol=1e-8)
    assert np.max(np.abs(got[:8] - reference[:8]) / reference[:8]) < 1e-3


def test_assembly_reports_the_same_motion_on_both_sides_of_the_tie() -> None:
    (Ka, Ma), (Kb, Mb) = _components()
    a = rubin(Ka, Ma, boundary_dofs=[CUT - 1], n_modes=6)
    b = rubin(Kb, Mb, boundary_dofs=[0], n_modes=6)
    assembled = free_interface_assembly([("A", a), ("B", b)], [("A", CUT - 1, "B", 0)])
    left = assembled.component_modes("A")[CUT - 1, :]
    right = assembled.component_modes("B")[0, :]
    assert np.allclose(left, right, atol=1e-10)
    assert assembled.component_names() == ("A", "B")
    assert assembled.meta["n_rigid_links"] == 1

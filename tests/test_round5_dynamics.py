"""Round-5 regression tests for the bugs reproduced in ``.agent_workspace/reports/R5-O2.md``.

Every test here failed before the corresponding fix. The references are closed forms
written so that they are themselves well conditioned in the limit under test — the whole
point of the main bug is that the *obvious* way to write these expressions is not
computable in floating point near ``omega = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from femtools.dynamics.cms_free import free_interface_assembly, macneal, rubin
from femtools.dynamics.craig_bampton import craig_bampton
from femtools.dynamics.frf import modal_frf
from femtools.dynamics.modal import ModalModel
from femtools.dynamics.time_domain import _ramp_coefficients, time_history

TWO_PI = 2.0 * np.pi


def _sdof(freq_hz: float) -> ModalModel:
    return ModalModel(freq_hz=np.array([float(freq_hz)]), modes=np.array([[1.0]]))


def _undamped_step(w: float, t: np.ndarray) -> np.ndarray:
    """``(1 - cos(w t)) / w^2`` in a form that stays accurate (and finite) as ``w -> 0``."""
    return 0.5 * t**2 * np.sinc(0.5 * w * t / np.pi) ** 2


# ---------------------------------------------------------------------------
# 1. ramp-invariant coefficients near omega = 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "freq_hz", [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
)
def test_undamped_step_response_across_ten_decades_of_frequency(freq_hz):
    """A near-rigid mode must integrate as accurately as an exactly rigid one.

    Before the fix the load coefficients were evaluated as a bracket cancelling to
    ``O((w dt)^2)`` divided by ``w^2``: exactly ``0 Hz`` took a separate branch and was
    fine, but ``1e-6 Hz`` — what ``eigh`` actually returns for a rigid-body mode — came
    out 100 % wrong.
    """
    dt, n = 1e-3, 500
    t = dt * np.arange(n)
    got = time_history(_sdof(freq_hz), np.ones((1, n)), dt, 0.0).modal_displacement[0]
    ref = _undamped_step(TWO_PI * freq_hz, t)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_near_rigid_mode_matches_the_exactly_rigid_one_under_alpha_damping():
    """Rayleigh ``alpha`` makes a near-rigid mode wildly over-damped (zeta ~ 1e6).

    That combination used to lose every digit, and further out it overflowed to ``nan``.
    """
    dt, n, alpha = 1e-3, 400, 3.0
    t = dt * np.arange(n)
    ref = t / alpha - (1.0 - np.exp(-alpha * t)) / alpha**2  # exact for q'' + c q' = 1
    for freq_hz in (0.0, 1e-9, 1e-7):
        got = time_history(
            _sdof(freq_hz), np.ones((1, n)), dt, {"alpha": alpha}
        ).modal_displacement[0]
        assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))
        assert np.isfinite(got).all()


@pytest.mark.parametrize("zeta", [0.0, 0.02, 0.7, 1.0, 1.5, 10.0, 1e3, 1e5])
def test_ramp_coefficients_are_finite_and_stable_for_any_damping(zeta):
    """Critical and heavily over-damped modes produced ``nan`` through the old fudge
    ``z -> z (1 - 1e-7) - 1e-9`` and the ``sqrt(1 - z^2)`` branch behind it."""
    w = np.array([2.0 * np.pi * 5.0])
    coeffs = _ramp_coefficients(w, 2.0 * zeta * w, 1e-3)
    assert all(np.isfinite(c).all() for c in coeffs)


def test_critically_damped_step_response_is_exact():
    """``zeta = 1`` has a removable singularity that used to be dodged by perturbing
    ``zeta``, costing ~7 digits."""
    fn, dt, n = 4.0, 1e-3, 600
    w = TWO_PI * fn
    t = dt * np.arange(n)
    got = time_history(_sdof(fn), np.ones((1, n)), dt, 1.0).modal_displacement[0]
    ref = (1.0 - np.exp(-w * t) * (1.0 + w * t)) / w**2
    assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))


def test_recurrence_is_exact_for_a_piecewise_linear_force():
    """The defining property: refining ``dt`` must not move the coarse-grid samples."""
    rng = np.random.default_rng(7)
    coarse_dt, n = 2e-3, 200
    nodes = rng.normal(size=n)
    mm = _sdof(6.0)
    coarse = time_history(mm, nodes.reshape(1, -1), coarse_dt, 0.03).modal_displacement[0]
    fine_nodes = np.interp(
        np.arange(4 * (n - 1) + 1) * (coarse_dt / 4),
        coarse_dt * np.arange(n),
        nodes,
    )
    fine = time_history(
        mm, fine_nodes.reshape(1, -1), coarse_dt / 4, 0.03
    ).modal_displacement[0]
    assert np.allclose(coarse, fine[::4], rtol=0, atol=1e-13 * np.max(np.abs(coarse)))


def test_free_free_chain_drifts_at_the_rigid_body_acceleration():
    """End-to-end: ``eigh`` reports this chain's rigid eigenvalue as ``-1.8e-11``, i.e.
    ``6.8e-07 Hz``, which is squarely inside the old failure region."""
    n_masses, mass, stiff = 12, 1.3, 2.2e4
    K = np.zeros((n_masses, n_masses))
    for i in range(n_masses - 1):
        K[i, i] += stiff
        K[i + 1, i + 1] += stiff
        K[i, i + 1] -= stiff
        K[i + 1, i] -= stiff
    M = np.eye(n_masses) * mass
    lam, phi = sla.eigh(K, M)
    assert 0.0 < np.sqrt(abs(lam[0])) / TWO_PI < 1e-5  # not exactly zero, as expected
    mm = ModalModel(
        freq_hz=np.sqrt(np.abs(lam)) / TWO_PI, modes=phi, eigenvalues=np.abs(lam)
    )

    dt, n_steps = 5e-4, 3000
    t = dt * np.arange(n_steps)
    force = np.zeros((n_masses, n_steps))
    force[0, :] = 1.0
    th = time_history(mm, force, dt, 0.0)
    total_mass = n_masses * mass
    centre = (M @ th.displacement).sum(axis=0) / total_mass
    ref = 0.5 * t**2 / total_mass
    assert np.max(np.abs(centre - ref)) <= 1e-9 * np.max(ref)


def test_newmark_still_tracks_the_exact_recurrence():
    """The Newmark branch was not touched; keep it pinned to the exact one."""
    rng = np.random.default_rng(11)
    n, dt = 2000, 2e-4
    force = np.vstack([rng.normal(size=n), np.zeros(n)])
    mm = ModalModel(freq_hz=np.array([5.0, 17.0]), modes=np.eye(2))
    exact = time_history(mm, force, dt, 0.02, method="exact").displacement
    newmark = time_history(mm, force, dt, 0.02, method="newmark").displacement
    assert np.max(np.abs(exact - newmark)) < 1e-3 * np.max(np.abs(exact))


# ---------------------------------------------------------------------------
# 2. initial-condition validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["q0", "qd0"])
def test_initial_modal_state_must_match_the_number_of_modes(key):
    """A length-1 ``q0`` used to broadcast: "displace mode 0 by one" started *every*
    mode at one, silently."""
    mm = ModalModel(freq_hz=np.array([1.0, 2.0, 3.0, 4.0]), modes=np.eye(4))
    with pytest.raises(ValueError, match="one entry per retained mode"):
        time_history(mm, np.zeros((4, 5)), 1e-3, 0.01, **{key: [1.0]})
    ok = time_history(mm, np.zeros((4, 5)), 1e-3, 0.01, **{key: np.ones(4)})
    assert ok.modal_displacement.shape == (4, 5)


# ---------------------------------------------------------------------------
# 3. Craig-Bampton DOF partition
# ---------------------------------------------------------------------------
def test_craig_bampton_rejects_a_partition_that_drops_dofs():
    """A DOF in neither set got a zero row in ``T``, which reduces the structure with
    that DOF held at zero while still reporting the parent's ``ndof``."""
    K = np.diag([2.0, 3.0, 4.0, 5.0])
    M = np.eye(4)
    with pytest.raises(ValueError, match="cover all 4 DOFs"):
        craig_bampton(K, M, [0], 1, interior_dofs=[1])
    full = craig_bampton(K, M, [0], 1, interior_dofs=[1, 2, 3])
    assert np.allclose(full.K, craig_bampton(K, M, [0], 1).K)


# ---------------------------------------------------------------------------
# 4. lower_residual at DC
# ---------------------------------------------------------------------------
def test_zero_lower_residual_is_a_no_op_even_on_a_dc_line():
    """``0 * inf`` used to turn the whole DC line into ``nan``."""
    mm = ModalModel(freq_hz=np.array([1.0, 2.0]), modes=np.eye(2))
    freq = np.array([0.0, 1.5])
    with_residual = modal_frf(mm, [0], [0], freq, 0.01, lower_residual=0.0).H
    without = modal_frf(mm, [0], [0], freq, 0.01).H
    assert np.isfinite(with_residual).all()
    assert np.array_equal(with_residual, without)


def test_nonzero_lower_residual_on_a_dc_line_is_refused():
    mm = ModalModel(freq_hz=np.array([1.0, 2.0]), modes=np.eye(2))
    with pytest.raises(ValueError, match="singular at f = 0"):
        modal_frf(mm, [0], [0], np.array([0.0, 1.5]), 0.01, lower_residual=1e-6)


def test_lower_residual_subtracts_the_inertia_term_above_dc():
    mm = ModalModel(freq_hz=np.array([1.0, 2.0]), modes=np.eye(2))
    freq = np.array([0.5, 1.5])
    delta = (
        modal_frf(mm, [0], [0], freq, 0.01, lower_residual=1e-6).H
        - modal_frf(mm, [0], [0], freq, 0.01).H
    )
    assert np.allclose(delta[0, 0], -1e-6 / (TWO_PI * freq) ** 2)


# ---------------------------------------------------------------------------
# 5. the Rubin accuracy the round-4 landing was accepted on must not move
# ---------------------------------------------------------------------------
def _split_chain():
    """40-mass grounded chain cut at mass 20, the cut mass shared by the two halves."""
    n_masses, cut, mass, stiff = 40, 20, 1.7, 4.3e4

    def chain(n, m, grounded):
        K = np.zeros((n, n))
        for i in range(n - 1):
            K[i, i] += stiff
            K[i + 1, i + 1] += stiff
            K[i, i + 1] -= stiff
            K[i + 1, i] -= stiff
        if grounded:
            K[0, 0] += stiff
        return K, np.diag(m)

    mass_a = np.full(cut, mass)
    mass_a[-1] = 0.5 * mass
    mass_b = np.full(n_masses - cut + 1, mass)
    mass_b[0] = 0.5 * mass
    parent = chain(n_masses, np.full(n_masses, mass), True)
    return (
        parent,
        chain(cut, mass_a, True),
        chain(n_masses - cut + 1, mass_b, False),
        cut,
    )


@pytest.mark.parametrize(
    ("method", "n_modes", "tolerance"),
    [(rubin, 6, 2.8e-4), (rubin, 12, 1.3e-6), (macneal, 6, 3.8e-3)],
)
def test_free_interface_cms_accuracy_is_unchanged(method, n_modes, tolerance):
    """Guard rail for the round-4 headline: Rubin with 6 free modes per half reproduces
    the first 8 frequencies of the unsplit chain to 0.028 %."""
    (K, M), (Ka, Ma), (Kb, Mb), cut = _split_chain()
    a = method(Ka, Ma, [cut - 1], n_modes)
    b = method(Kb, Mb, [0], n_modes)
    assembled = free_interface_assembly(
        [("A", a), ("B", b)], [("A", cut - 1, "B", 0)]
    )
    ref = np.sqrt(np.clip(sla.eigh(K, M, eigvals_only=True), 0.0, None))[:8] / TWO_PI
    got = np.asarray(assembled.freq_hz)[:8]
    assert np.max(np.abs(got - ref) / ref) < tolerance

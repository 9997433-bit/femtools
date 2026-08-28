"""Numerical goldens for the ACCEPTANCE cases that had no test yet.

Covers cases 3b (stiffness orthogonality), 5 (cantilever effective mass),
11 (static tip closed forms), 13 (harmonic force identification), 15 (RBPE on a
synthetic rigid body) and 16 (FDD on a synthetic 2-DOF record).  Every
construction is the one written down in `docs/ACCEPTANCE.md`; the stochastic
ones are seeded.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from femtools.core.model import FEModel
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes
from femtools.fea.static import solve_static
from femtools.mpe.common import mac
from femtools.mpe.fdd import fdd
from femtools.mpe.synthetic import synthetic_response
from femtools.pretest.candidates import node_coordinates
from femtools.pretest.target_modes import effective_mass
from femtools.rbpe import (
    mount_stiffness_matrix,
    rigid_body_mass_matrix,
    rigid_body_properties,
    rigid_body_transform,
)
from femtools.updating.force_id import identify_harmonic_forces

E, NU, RHO = 210.0e9, 0.3, 7850.0
LENGTH = 2.0
AREA, IY, IZ, JT = 8.0e-4, 3.0e-8, 6.0e-8, 9.0e-8


def beam_cantilever(n_elements: int) -> FEModel:
    """Uniform BEAM2 cantilever along x, clamped at node 1."""
    model = FEModel(name=f"cantilever-{n_elements}")
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=AREA, Iy=IY, Iz=IZ, J=JT)
    for i in range(n_elements + 1):
        model.add_node(id=i + 1, xyz=(LENGTH * i / n_elements, 0.0, 0.0))
    for i in range(n_elements):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True,) * 6)
    return model


def rod_chain(n_elements: int, *, clamped: bool = True) -> FEModel:
    """Pin-jointed BAR2 chain along x; only the axial motion is left free.

    With ``clamped=False`` the root is free too, so the axial DOFs are the
    complete free set of the model — a free-free rod.
    """
    model = FEModel(name=f"rod-{n_elements}")
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="bar", material_id=1, A=AREA)
    for i in range(n_elements + 1):
        model.add_node(id=i + 1, xyz=(LENGTH * i / n_elements, 0.0, 0.0))
    for i in range(n_elements):
        model.add_element(id=i + 1, type="BAR2", nodes=(i + 1, i + 2), property_id=1)
    # A chain of rods carries no transverse stiffness, so every node is held
    # everywhere except along the axis.
    for i in range(n_elements + 1):
        model.add_spc(node_id=i + 1, mask=(clamped and i == 0,) + (True,) * 5)
    return model


# ----------------------------------------------------------------------
# case 3b -- stiffness orthogonality, Phi^T K Phi = Lambda
# ----------------------------------------------------------------------
def test_modes_are_stiffness_orthogonal_to_the_eigenvalue_diagonal() -> None:
    """`docs/ACCEPTANCE.md` case 3b: rel dev of `Phi^T K Phi` from `Lambda` < 1e-6."""
    model = beam_cantilever(16)
    assembly = assemble_km(model)
    modal = solve_modes(model, n_modes=6)

    phi = np.asarray(modal.modes, dtype=float)
    stiffness = assembly.K
    gram = phi.T @ (stiffness @ phi)
    omega2 = (2.0 * math.pi * np.asarray(modal.freq_hz, dtype=float)) ** 2

    np.testing.assert_allclose(np.diag(gram), omega2, rtol=1.0e-6)
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) / np.max(omega2) < 1.0e-6


# ----------------------------------------------------------------------
# case 5 -- cantilever effective mass
# ----------------------------------------------------------------------
def test_cantilever_transverse_effective_mass_matches_the_classical_fractions() -> None:
    """Case 5: the first three transverse fractions are 0.6131 / 0.1883 / 0.0647."""
    model = beam_cantilever(20)
    modal = solve_modes(model, n_modes=24)
    result = effective_mass(
        modal, modal.M, dof_map=modal, coords=node_coordinates(model, modal)
    )

    tz = np.sort(result.fraction[:, result.directions.index("TZ")])[::-1]
    np.testing.assert_allclose(tz[:3], [0.6131, 0.1883, 0.0647], atol=1.0e-2)


def test_effective_masses_sum_to_the_rigid_body_mass() -> None:
    """Completeness: with every mode of the model, `sum_r L_r L_r^T = R^T M R`.

    The model has to be unconstrained for the identity to hold, because the
    right-hand side counts the mass sitting on the DOFs the modes cannot span.
    A free-free rod is used rather than a beam because a BEAM2 carries no
    rotary inertia about its own axis, so a beam has massless DOFs and no
    complete finite-frequency mode set.
    """
    n_elements = 5
    model = rod_chain(n_elements, clamped=False)
    modal = solve_modes(model, n_modes=n_elements + 1)
    assert np.asarray(modal.modes).shape[1] == n_elements + 1

    result = effective_mass(
        modal, modal.M, dof_map=modal, coords=node_coordinates(model, modal)
    )
    tx = result.directions.index("TX")
    np.testing.assert_allclose(
        result.effective_mass[:, tx].sum(), result.total_mass[tx], rtol=1.0e-6
    )


# ----------------------------------------------------------------------
# case 11 -- static tip closed forms
# ----------------------------------------------------------------------
def test_axial_bar_tip_extension_matches_fl_over_ea() -> None:
    """Case 11, first half: a linear rod is nodally exact for an end load."""
    n_elements = 5
    model = rod_chain(n_elements)
    force = 1.5e3
    tip = n_elements + 1
    model.add_load(node_id=tip, force=(force, 0.0, 0.0))

    u = solve_static(model, None)
    measured = float(u[model.dof_map()[(tip, 0)]])
    exact = force * LENGTH / (E * AREA)

    assert abs(measured - exact) / abs(exact) < 1.0e-12


def test_cantilever_tip_deflection_matches_fl3_over_3ei() -> None:
    """Case 11, second half: the Hermite BEAM2 is nodally exact for an end load.

    This is the assertion behind the `examples/update_static.py` measurement
    quoted in the acceptance status block.
    """
    n_elements = 8
    model = beam_cantilever(n_elements)
    force = -1.0e3
    tip = n_elements + 1
    model.add_load(node_id=tip, force=(0.0, 0.0, force))

    u = solve_static(model, None)
    measured = float(u[model.dof_map()[(tip, 2)]])
    exact = force * LENGTH**3 / (3.0 * E * IY)

    assert abs(measured - exact) / abs(exact) < 1.0e-12


# ----------------------------------------------------------------------
# case 13 -- harmonic force identification
# ----------------------------------------------------------------------
def two_dof_frf(freq_hz: np.ndarray) -> np.ndarray:
    """Receptance of a 2-DOF spring-mass-damper chain, `(2, 2, n_freq)`."""
    mass = np.diag([2.0, 1.0])
    stiffness = np.array([[3.0e5, -1.0e5], [-1.0e5, 1.0e5]])
    damping = 1.0e-3 * stiffness
    omega = 2.0 * math.pi * freq_hz
    return np.stack(
        [
            np.linalg.inv(stiffness - w**2 * mass + 1j * w * damping)
            for w in omega
        ],
        axis=2,
    )


def test_force_identification_is_exact_on_noiseless_two_dof_data() -> None:
    """Case 13: with `X = H F` and no noise, `F_hat = H^+ X` to 1e-8 relative."""
    freq = np.linspace(10.0, 90.0, 33)
    frf = two_dof_frf(freq)
    rng = np.random.default_rng(11)
    truth = rng.standard_normal((2, freq.size)) + 1j * rng.standard_normal((2, freq.size))
    response = np.einsum("oif,if->of", frf, truth)

    result = identify_harmonic_forces(frf, response, freq, method="pinv")

    forces = np.asarray(result.forces)
    error = np.linalg.norm(forces - truth) / np.linalg.norm(truth)
    assert error < 1.0e-8


# ----------------------------------------------------------------------
# case 15 -- RBPE on a synthetic rigid body
# ----------------------------------------------------------------------
RB_MASS = 10.0
RB_COG = np.array([0.1, 0.05, 0.2])
RB_INERTIA = np.diag([0.5, 0.8, 1.0])


def rbpe_dataset() -> dict[str, Any]:
    """A rigid block on six soft springs, measured over a mass-line band."""
    mass_matrix = rigid_body_mass_matrix(RB_MASS, RB_COG, RB_INERTIA)
    positions = np.repeat(
        [
            [sx, sy, sz]
            for sx in (-0.5, 0.5)
            for sy in (-0.4, 0.4)
            for sz in (-0.3, 0.3)
        ],
        3,
        axis=0,
    )
    directions = np.tile(np.eye(3), (8, 1))
    drive = [0, 4, 8, 13, 17, 22]
    mounts = [
        ((x, y, -0.3), axis, 900.0)
        for x in (-0.5, 0.5)
        for y in (-0.4, 0.4)
        for axis in ((0, 0, 1),)
    ] + [((0.0, y, -0.3), (1, 0, 0), 400.0) for y in (-0.4, 0.4)]
    stiffness = mount_stiffness_matrix(mounts)

    freq = np.linspace(6.0, 15.0, 21)
    t_out = rigid_body_transform(positions, directions, (0.0, 0.0, 0.0))
    t_in = t_out[drive]
    accelerance = np.stack(
        [
            -(w**2) * (t_out @ np.linalg.solve(stiffness - w**2 * mass_matrix, t_in.T))
            for w in 2.0 * math.pi * freq
        ],
        axis=2,
    )
    return {
        "frf": accelerance,
        "freq_hz": freq,
        "sensors": (positions, directions),
        "inputs": (positions[drive], directions[drive]),
        "mounts": mounts,
    }


def test_rbpe_recovers_mass_cog_and_inertia_of_a_synthetic_rigid_body() -> None:
    """Case 15: mass, CoG and inertia tensor back to 1e-8 relative."""
    data = rbpe_dataset()
    identified = rigid_body_properties(
        data["frf"],
        data["freq_hz"],
        sensors=data["sensors"],
        inputs=data["inputs"],
        band=(6.0, 15.0),
        mount_k=data["mounts"],
    )

    assert abs(identified.mass - RB_MASS) / RB_MASS < 1.0e-8
    np.testing.assert_allclose(identified.cog, RB_COG, rtol=1.0e-8, atol=1.0e-9)
    np.testing.assert_allclose(identified.inertia, RB_INERTIA, rtol=1.0e-8, atol=1.0e-9)
    assert identified.is_physical()


def test_rbpe_that_ignores_the_suspension_reads_the_mass_line_high() -> None:
    """The apparent negative mass `K / omega^2` is what `mount_k` removes."""
    data = rbpe_dataset()
    uncorrected = rigid_body_properties(
        data["frf"],
        data["freq_hz"],
        sensors=data["sensors"],
        inputs=data["inputs"],
        band=(6.0, 15.0),
    )

    assert abs(uncorrected.mass - RB_MASS) / RB_MASS > 1.0e-3


# ----------------------------------------------------------------------
# case 16 -- FDD on a synthetic 2-DOF record
# ----------------------------------------------------------------------
def test_fdd_recovers_a_synthetic_two_dof_record() -> None:
    """Case 16: peaks within one spectral line of truth, shape MAC > 0.99."""
    fs, nperseg = 256.0, 2048
    signal = synthetic_response(
        [5.0, 13.0], damping=0.01, n_out=6, fs=fs, duration=600.0, noise=0.0, seed=19
    )
    identified = fdd(signal.data, fs=fs, n_modes=2, nperseg=nperseg, f_range=(1.0, 60.0))

    df = fs / nperseg
    order = np.argsort(identified.freq_hz)
    np.testing.assert_array_less(
        np.abs(identified.freq_hz[order] - signal.freq_hz), df + 1.0e-12
    )
    for k, j in enumerate(order):
        assert mac(identified.mode_shapes[:, j], signal.mode_shapes[:, k]) > 0.99

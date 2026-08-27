"""Round 6 (R6-O1): per-node rotational frames for shell drilling.

A flat plate has no stiffness about its own normal, so the shell elements carry
a rank deficient drilling penalty and the assembler removes the drilling
rotations that receive nothing else.  Until this round that removal only worked
while the plate normal *was* a global axis; a tilted plate kept a zero-energy
drilling mechanism and reported a seventh rigid body mode.

``femtools.fea.nodal_frames`` gives each shell node a local triad whose third
axis is its averaged shell normal, so the drilling rotation is one degree of
freedom again at any orientation.  The tests below pin

* six rigid body modes for the tilted plate as well as the in-plane one,
* the whole free-free and clamped spectrum being orientation invariant,
* the frames being the *identity* wherever the normal already is a global axis,
  which is what keeps every existing golden case bit-for-bit unchanged, and
* those golden cases themselves (HEX8 tip ratio and rigid body count, patch
  test, MITC4 thin plate, BEAM2, ``solve_static(enforced=)``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from femtools.fea import assemble_km, solve_modes, solve_static
from femtools.fea.nodal_frames import (
    averaged_shell_normals,
    rotation_triad,
    shell_nodal_frames,
)
from femtools.fea.verification import (
    beam_cantilever,
    hex8_bending_ratio,
    hex8_patch_test_error,
    hex8_rigid_body_frequencies,
    shell_drilling_orientation_gap,
    shell_plate,
)

SHELLS = ["QUAD4", "TRIA3"]

#: A plate normal deliberately parallel to no global axis; the same rotation
#: :func:`shell_drilling_orientation_gap` uses.
OBLIQUE = np.linalg.qr(
    np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
)[0]

#: The plate rotated into the global x-z plane: oblique to *this* mesh but still
#: axis-aligned, so it must not need a frame at all.
XZ_PLANE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def assemble_quietly(model, **kwargs):
    """Assemble and return ``(assembly, warnings)`` without failing the run."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asm = assemble_km(model, **kwargs)
    return asm, [str(c.message) for c in caught]


# ----------------------------------------------------------------------
# the contract: six rigid body modes at any orientation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("etype", SHELLS)
def test_tilted_free_free_plate_has_six_rigid_body_modes(etype: str) -> None:
    """The acceptance case: six near-zero frequencies on the tilted plate, not seven."""
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_zero_modes"] == 6.0
    assert gap["oblique_warned"] == 0.0
    assert gap["oblique_frame_nodes"] > 0.0


@pytest.mark.parametrize("etype", SHELLS)
def test_in_plane_plate_still_has_six_rigid_body_modes(etype: str) -> None:
    """The path that already worked must keep working, and keep working the same way.

    The axis-aligned plate needs no local triad at all, so it goes down exactly
    the pre-existing elimination path.
    """
    gap = shell_drilling_orientation_gap(etype)

    assert gap["aligned_zero_modes"] == 6.0
    assert gap["aligned_warned"] == 0.0
    assert gap["aligned_frame_nodes"] == 0.0


@pytest.mark.parametrize("etype", SHELLS)
def test_tilted_plate_solves_the_same_system_as_the_aligned_one(etype: str) -> None:
    """Same plate, same equations: DOF counts and spectrum are orientation invariant."""
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_free_dof"] == gap["aligned_free_dof"]
    assert gap["oblique_drilling_dof"] == gap["aligned_drilling_dof"]
    assert gap["oblique_first_elastic_hz"] == pytest.approx(
        gap["aligned_first_elastic_hz"], rel=1.0e-9
    )


@pytest.mark.parametrize("etype", SHELLS)
def test_seventh_frequency_is_elastic_and_well_separated(etype: str) -> None:
    """Not merely six zeros: the seventh mode is the real first elastic one."""
    aligned = solve_modes(shell_plate(4, 3, etype=etype), n_modes=12).freq_hz
    oblique = solve_modes(
        shell_plate(4, 3, etype=etype, rotation=OBLIQUE), n_modes=12
    ).freq_hz

    assert np.count_nonzero(oblique < 1.0e-6) == 6
    assert oblique[6] > 1.0
    np.testing.assert_allclose(oblique[6:], aligned[6:], rtol=1.0e-9)


@pytest.mark.parametrize("etype", SHELLS)
def test_rigid_body_motion_of_the_tilted_plate_is_energy_free(etype: str) -> None:
    """The six zeros are the six rigid body modes, not five of them and a mechanism.

    Each analytic rigid body field is pushed through the *solved* partition of
    the tilted plate; a genuine rigid body mode stores no strain energy there.
    """
    model = shell_plate(4, 3, etype=etype, rotation=OBLIQUE)
    asm = assemble_km(model)
    coords = np.array([model["nodes"][nid]["xyz"] for nid in asm.dof_map.node_ids])
    centroid = coords.mean(axis=0)

    fields = []
    for axis in np.eye(3):
        translation = np.zeros((coords.shape[0], 6))
        translation[:, :3] = axis
        rotation = np.zeros((coords.shape[0], 6))
        rotation[:, :3] = np.cross(axis, coords - centroid)
        rotation[:, 3:] = axis
        fields += [translation, rotation]

    basis = asm.restrict(asm.from_basic(np.column_stack([f.ravel() for f in fields])))
    strain = np.einsum("ij,ij->j", basis, asm.Kff @ basis)
    kinetic = np.einsum("ij,ij->j", basis, asm.Mff @ basis)

    assert np.all(kinetic > 0.0)
    assert np.max(strain / kinetic) < 1.0e-6
    assert np.linalg.matrix_rank(basis, tol=1.0e-10 * np.abs(basis).max()) == 6


@pytest.mark.parametrize("etype", SHELLS)
def test_clamped_tilted_plate_matches_the_clamped_aligned_one(etype: str) -> None:
    """Constrained models are orientation invariant too, drilling DOFs and all."""
    aligned = solve_modes(
        shell_plate(3, 3, etype=etype, clamped_edge=True), n_modes=6
    ).freq_hz
    oblique = solve_modes(
        shell_plate(3, 3, etype=etype, rotation=OBLIQUE, clamped_edge=True), n_modes=6
    ).freq_hz

    np.testing.assert_allclose(oblique, aligned, rtol=1.0e-9)


def test_tilted_plate_static_response_is_orientation_invariant() -> None:
    """A tip-loaded cantilever plate deflects by the same amount however it is hung."""

    def tip_deflection(rotation):
        rot = np.eye(3) if rotation is None else rotation
        model = shell_plate(6, 6, clamped_edge=True, rotation=rotation)
        asm = assemble_km(model)
        normal = rot @ np.array([0.0, 0.0, 1.0])
        tip = [
            nid
            for nid, node in model["nodes"].items()
            if abs((rot.T @ np.asarray(node["xyz"]))[0] - 1.0) < 1.0e-12
        ]
        loads = {(nid, comp): normal[comp] / len(tip) for nid in tip for comp in range(3)}
        u = asm.to_basic(solve_static(model, loads, assembly=asm))
        return float(
            np.mean([u[asm.dof_map.node_dofs(nid)][:3] @ normal for nid in tip])
        )

    assert tip_deflection(OBLIQUE) == pytest.approx(tip_deflection(None), rel=1.0e-9)


# ----------------------------------------------------------------------
# the frames themselves
# ----------------------------------------------------------------------


def test_frames_are_the_identity_for_an_axis_aligned_normal() -> None:
    """No frame where none is needed -- this is what protects the goldens."""
    for rotation in (None, XZ_PLANE):
        model = shell_plate(3, 3, rotation=rotation)
        asm = assemble_km(model)
        assert asm.framed_nodes == []
        assert asm.frames.is_identity


def test_axis_aligned_assembly_is_untouched_by_the_frames() -> None:
    """Bit-for-bit: turning the frames off changes nothing for an aligned model."""
    model = shell_plate(3, 3, clamped_edge=True)
    with_frames = assemble_km(model, nodal_frames=True)
    without = assemble_km(model, nodal_frames=False)

    assert (with_frames.K - without.K).nnz == 0
    assert (with_frames.M - without.M).nnz == 0
    np.testing.assert_array_equal(with_frames.free_dof, without.free_dof)
    np.testing.assert_array_equal(with_frames.drilling_dof, without.drilling_dof)


def test_every_shell_node_of_a_tilted_plate_is_framed_on_its_normal() -> None:
    model = shell_plate(3, 3, rotation=OBLIQUE)
    asm = assemble_km(model)
    normal = OBLIQUE @ np.array([0.0, 0.0, 1.0])
    normals = averaged_shell_normals(model)

    assert len(asm.framed_nodes) == len(model["nodes"])
    for nid in asm.framed_nodes:
        triad = asm.frames.frame(nid)
        np.testing.assert_allclose(triad.T @ triad, np.eye(3), atol=1.0e-14)
        assert np.linalg.det(triad) == pytest.approx(1.0)
        assert abs(abs(float(triad[:, 2] @ normal)) - 1.0) < 1.0e-12
        np.testing.assert_allclose(np.abs(normals[nid]), np.abs(normal), atol=1.0e-12)


def test_rotation_triad_is_right_handed_and_carries_the_normal() -> None:
    for normal in (np.array([0.3, -0.5, 0.81]), np.array([1.0, 1.0, 1.0])):
        triad = rotation_triad(normal)
        np.testing.assert_allclose(triad.T @ triad, np.eye(3), atol=1.0e-14)
        np.testing.assert_allclose(
            triad[:, 2], normal / np.linalg.norm(normal), atol=1.0e-14
        )
        np.testing.assert_allclose(
            np.cross(triad[:, 0], triad[:, 1]), triad[:, 2], atol=1.0e-14
        )


def test_frame_transformation_is_orthogonal_and_leaves_translations_alone() -> None:
    model = shell_plate(3, 3, rotation=OBLIQUE)
    asm = assemble_km(model)
    lam = asm.frames.matrix().toarray()

    np.testing.assert_allclose(lam.T @ lam, np.eye(lam.shape[0]), atol=1.0e-14)
    translations = np.concatenate(
        [asm.dof_map.node_dofs(nid)[:3] for nid in asm.dof_map.node_ids]
    )
    np.testing.assert_array_equal(
        lam[np.ix_(translations, translations)], np.eye(translations.size)
    )
    # A round trip through the two frames is the identity for any field.
    field = np.random.default_rng(6).standard_normal(asm.n_dof)
    np.testing.assert_allclose(asm.from_basic(asm.to_basic(field)), field, atol=1.0e-13)


def test_nodes_with_a_rotational_spc_keep_the_basic_frame() -> None:
    """An SPC is written in the basic frame, so it is only one DOF there."""
    model = shell_plate(3, 3, rotation=OBLIQUE, clamped_edge=True)
    asm = assemble_km(model)
    clamped = {spc["node_id"] for spc in model["spcs"]}

    assert clamped
    assert clamped.isdisjoint(asm.framed_nodes)
    assert set(asm.framed_nodes) == set(model["nodes"]) - clamped


def test_enforced_rotation_on_a_framed_node_is_refused() -> None:
    """Silently reinterpreting the value would be worse than not accepting it."""
    model = shell_plate(3, 3, rotation=OBLIQUE)
    asm = assemble_km(model)
    node = asm.framed_nodes[0]

    with pytest.raises(ValueError, match="local shell triad"):
        solve_static(model, {}, assembly=asm, enforced={(node, 3): 0.01})
    # Translations are never rotated, so driving one is still fine.
    solve_static(model, {}, assembly=asm, enforced={(node, 0): 0.01})


def test_no_frame_is_built_for_a_model_without_shells() -> None:
    model = beam_cantilever(8)
    asm = assemble_km(model)

    assert asm.frames.is_identity
    assert averaged_shell_normals(model) == {}
    assert shell_nodal_frames(model, asm.dof_map).is_identity


def test_folded_shell_stays_quiet_and_keeps_its_ridge_rotations() -> None:
    """No false positive, and no stiffness thrown away where the patch is not flat.

    The ridge nodes have no common normal, so their drilling direction is not a
    single DOF in any frame; those rotations must survive the elimination.
    """
    model = shell_plate(4, 2, side=2.0)
    theta = np.radians(35.0)
    ridge = set()
    for nid, node in model["nodes"].items():
        x, y, _z = node["xyz"]
        if x > 1.0:
            node["xyz"] = (1.0 + (x - 1.0) * np.cos(theta), y, (x - 1.0) * np.sin(theta))
        elif abs(x - 1.0) < 1.0e-12:
            ridge.add(nid)

    asm, caught = assemble_quietly(model)
    dropped = set(asm.drilling_dof)

    assert not caught
    assert ridge
    for nid in ridge:
        assert dropped.isdisjoint(asm.dof_map.node_dofs(nid))
    assert solve_modes(model, n_modes=8, assembly=asm).freq_hz[6] > 1.0


# ----------------------------------------------------------------------
# goldens that must not move
# ----------------------------------------------------------------------


def test_hex8_cantilever_tip_ratio_golden() -> None:
    """Wilson-Taylor incompatible modes still reach 98.5% of Timoshenko."""
    assert hex8_bending_ratio() == pytest.approx(0.9854730473, rel=1.0e-6)


def test_hex8_goldens_are_unchanged() -> None:
    frequencies = hex8_rigid_body_frequencies()

    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0
    assert hex8_patch_test_error() < 1.0e-10


def test_beam2_euler_bernoulli_cantilever_golden() -> None:
    """First bending frequency of the BEAM2 cantilever against ``beta L = 1.8751``."""
    data = {"E": 70.0e9, "rho": 2700.0, "L": 2.0, "A": 8.0e-4, "Iy": 3.0e-8}
    first = solve_modes(beam_cantilever(16), n_modes=2).freq_hz[0]
    expected = (
        1.875104068711961**2
        * np.sqrt(data["E"] * data["Iy"] / (data["rho"] * data["A"]))
        / (2.0 * np.pi * data["L"] ** 2)
    )

    assert first == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("thickness", [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5])
def test_mitc4_thin_plate_does_not_shear_lock(thickness: float) -> None:
    """Four decades of slenderness at a constant answer is the no-locking statement.

    A displacement-based Mindlin quad integrated 2x2 loses its tip deflection
    roughly as ``t**2`` here; the MITC4 assumed shear holds the ratio to the
    thin-plate reference flat.
    """
    E, nu, side = 70.0e9, 0.3, 1.0
    model = shell_plate(8, 8, side=side, thickness=thickness, E=E, nu=nu, clamped_edge=True)
    asm = assemble_km(model)
    tip = [
        nid
        for nid, node in model["nodes"].items()
        if abs(node["xyz"][0] - side) < 1.0e-12
    ]
    loads = {(nid, 2): 1.0 / len(tip) for nid in tip}
    u = solve_static(model, loads, assembly=asm)
    deflection = float(np.mean([u[asm.dof_map.index(nid, 2)] for nid in tip]))

    rigidity = E * thickness**3 / (12.0 * (1.0 - nu**2))
    assert deflection / (side**3 / (3.0 * rigidity * side)) == pytest.approx(1.025, abs=0.01)


def test_enforced_displacement_on_a_free_dof_still_holds() -> None:
    """The R5-O1 fix: driving a DOF the assembler left free keeps equilibrium."""
    model = {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3, 4)},
        "elements": {
            i: {"type": "SPRING", "nodes": (i, i + 1), "k": 1000.0, "c1": 0}
            for i in (1, 2, 3)
        },
        "materials": {},
        "properties": {},
        "spcs": [{"node_id": 1, "dofs": (0,)}],
    }
    asm = assemble_km(model)
    u = solve_static(model, {}, assembly=asm, enforced={(3, 0): 0.03})

    np.testing.assert_allclose(
        [u[asm.dof_map.index(nid, 0)] for nid in (1, 2, 3, 4)],
        [0.0, 0.015, 0.03, 0.03],
        atol=1.0e-12,
    )

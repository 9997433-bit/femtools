"""Round 7 (R7-O1): element stress recovery and ``RBE2`` rigid bodies.

Two things the kernel could not do before this round.

``femtools.fea.recover`` turns a solved displacement field into the centroid
stress and strain of every ``BAR2``, ``BEAM2``, ``TRIA3``, ``QUAD4``, ``TET4``
and ``HEX8`` element, using each element's own strain-displacement operator.
The contract is the constant-strain patch test: a mesh carrying an exactly
linear displacement field must return the exact constant stress state, and it
does so to round-off for all six types on deliberately irregular meshes.

``femtools.fea.mpc`` applies the ``RBE2`` records the core model has been
carrying (``FEModel.add_rbe2``) as what they are -- a kinematic statement, not
a stiff spring.  The dependent DOFs are eliminated exactly, so welding two
nodes leaves a free-free structure with exactly six rigid body modes and the
analytic rigid body mass matrix, and a load on a rigid offset arrives at the
independent node as a force *and* the moment of the offset.

The last section re-measures the goldens that must not move.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtools.core.model import FEModel
from femtools.fea import (
    ConstraintTransform,
    apply_rbe2,
    assemble_km,
    recover_strain,
    recover_stress,
    solve_modes,
    solve_static,
)
from femtools.fea.mpc import MAX_CHAIN_DEPTH
from femtools.fea.recover import COMPONENTS, von_mises
from femtools.fea.verification import (
    PATCH_TYPES,
    beam_cantilever,
    hex8_bending_ratio,
    hex8_patch_test_error,
    hex8_rigid_body_frequencies,
    hex_cantilever,
    rbe2_offset_moment,
    rbe2_rigid_pair,
    shell_drilling_orientation_gap,
    shell_plate,
    stress_patch_error,
)

#: The same oblique rotation the round-6 shell work uses.
OBLIQUE = np.linalg.qr(
    np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
)[0]


def rod_chain(force: float = 250.0, area: float = 4.0e-4) -> tuple[dict, dict]:
    """Three ``BAR2`` elements in a row, clamped at one end, pulled at the other."""
    model = {
        "nodes": {i + 1: {"xyz": (0.5 * i, 0.0, 0.0)} for i in range(4)},
        "elements": {
            i + 1: {"type": "BAR2", "property_id": 1, "nodes": (i + 1, i + 2)} for i in range(3)
        },
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": area}},
        "spcs": [
            {"node_id": 1, "dofs": (0, 1, 2)},
            {"node_id": 2, "dofs": (1, 2)},
            {"node_id": 3, "dofs": (1, 2)},
            {"node_id": 4, "dofs": (1, 2)},
        ],
    }
    return model, {(4, 0): force}


def membrane_strip(
    nx: int = 3, ny: int = 2, *, length: float = 3.0, width: float = 1.0,
    thickness: float = 0.02, force: float = 5.0e4, etype: str = "QUAD4",
) -> tuple[dict, dict, float]:
    """Plane-stress strip pulled along ``x``; the exact answer is ``F / (w t)``."""
    ids: dict[tuple[int, int], int] = {}
    nodes: dict[int, dict] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            ids[(i, j)] = counter
            nodes[counter] = {"xyz": (length * i / nx, width * j / ny, 0.0)}
            counter += 1
    elements: dict[int, dict] = {}
    eid = 1
    for i in range(nx):
        for j in range(ny):
            n1, n2, n3, n4 = ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]
            if etype == "TRIA3":
                for conn in ((n1, n2, n3), (n1, n3, n4)):
                    elements[eid] = {"type": "TRIA3", "property_id": 1, "nodes": conn}
                    eid += 1
            else:
                elements[eid] = {"type": "QUAD4", "property_id": 1, "nodes": (n1, n2, n3, n4)}
                eid += 1

    tip = [ids[(nx, j)] for j in range(ny + 1)]
    weights = [0.5 if j in (0, ny) else 1.0 for j in range(ny + 1)]
    total = sum(weights)
    model = {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": 70.0e9, "nu": 0.3, "rho": 2700.0}},
        # A membrane property leaves the out-of-plane DOFs stiffness free, so
        # the assembler removes them and the strip is a pure 2D problem.
        "properties": {1: {"type": "membrane", "material_id": 1, "t": thickness}},
        "spcs": [{"node_id": ids[(0, j)], "dofs": (0,)} for j in range(ny + 1)]
        + [{"node_id": ids[(0, 0)], "dofs": (1,)}],
    }
    loads = {(nid, 0): force * w / total for nid, w in zip(tip, weights, strict=True)}
    return model, loads, force / (width * thickness)


def welded_pair(components=(1, 2, 3, 4, 5, 6), offset=(0.4, 0.0, 0.3)) -> FEModel:
    """A ``FEModel`` cantilever beam with a rigid arm on its tip."""
    model = FEModel(name="rigid-arm")
    model.add_node(1, (0.0, 0.0, 0.0))
    model.add_node(2, (1.0, 0.0, 0.0))
    model.add_node(3, tuple(np.array([1.0, 0.0, 0.0]) + np.asarray(offset, dtype=float)))
    model.add_material(1, E=2.1e11, nu=0.3, rho=7800.0)
    model.add_property(1, "beam", material_id=1, A=6.0e-4, Iy=5.0e-8, Iz=5.0e-8, J=8.0e-8)
    model.add_element(1, "BEAM2", (1, 2), property_id=1)
    model.add_spc(1, (True,) * 6)
    model.add_rbe2(1, independent=2, dependents=(3,), components=components)
    return model


# ----------------------------------------------------------------------
# stress recovery: the contract
# ----------------------------------------------------------------------


@pytest.mark.parametrize("etype", PATCH_TYPES)
def test_constant_strain_patch_test_is_exact(etype: str) -> None:
    """The acceptance case: exact constant stress on an irregular mesh, all six types.

    The boundary of a small distorted patch is driven with an exact linear
    displacement field and the enclosed node is left free; every element must
    then report the analytic constant state.  Measured in the *basic* frame, so
    the element frames the recovery reports are part of what is checked.
    """
    error = stress_patch_error(etype)

    assert error["elements"] >= 3.0
    assert error["displacement"] < 1.0e-12
    assert error["stress"] < 1.0e-12
    assert error["strain"] < 1.0e-12


def test_patch_test_holds_at_the_top_and_bottom_shell_fibres() -> None:
    """A membrane state is constant through the thickness, layer or no layer."""
    for etype in ("QUAD4", "TRIA3"):
        for layer in ("top", "bottom", 0.25):
            model, loads, exact = membrane_strip(etype=etype)
            asm = assemble_km(model)
            u = solve_static(model, loads, assembly=asm)
            stress = recover_stress(model, u, assembly=asm, layer=layer).stress_basic
            assert np.max(np.abs(stress[:, 0] - exact)) / exact < 1.0e-12


def test_uniaxial_membrane_stress_is_the_applied_traction() -> None:
    """``sxx = F / (w t)`` exactly, no transverse stress, Poisson contraction free."""
    model, loads, exact = membrane_strip()
    asm = assemble_km(model)
    result = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    stress = result.stress_basic
    strain = result.strain_basic

    np.testing.assert_allclose(stress[:, 0], exact, rtol=1.0e-12)
    assert np.max(np.abs(stress[:, [1, 2, 3, 4, 5]])) < 1.0e-9 * exact
    # Free lateral edges: the strip contracts by exactly -nu * exx.
    np.testing.assert_allclose(strain[:, 1], -0.3 * strain[:, 0], rtol=1.0e-9)
    np.testing.assert_allclose(strain[:, 0], exact / 70.0e9, rtol=1.0e-12)
    np.testing.assert_allclose(result.von_mises, exact, rtol=1.0e-12)


def test_rod_axial_stress_is_the_force_over_the_area() -> None:
    """A statically determinate rod chain: every element carries the tip load."""
    force, area = 250.0, 4.0e-4
    model, loads = rod_chain(force, area)
    asm = assemble_km(model)
    result = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    np.testing.assert_allclose(result.stress[:, 0], force / area, rtol=1.0e-12)
    np.testing.assert_allclose(result.strain[:, 0], force / (area * 2.1e11), rtol=1.0e-12)
    # The transverse strain of a rod is its Poisson contraction, and it carries
    # no transverse stress at all.
    np.testing.assert_allclose(result.strain[:, 1], -0.3 * result.strain[:, 0], rtol=1.0e-12)
    assert np.max(np.abs(result.stress[:, 1:])) == 0.0
    for eid in result.element_ids:
        assert result.extras[eid]["axial_force"] == pytest.approx(force, rel=1.0e-12)


def test_hex8_cantilever_centroid_carries_the_transverse_shear() -> None:
    """The neutral axis of a bent beam is in pure shear, ``V / A``."""
    model, _tip, loads = hex_cantilever(6, 1, 1, tip_force=1.0, width=1.0, height=1.0)
    asm = assemble_km(model)
    result = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    # sigma_zx = V / A with V = 1 and A = 1; the direct stresses vanish on the
    # neutral axis, which is where the centroid of a single-element-deep mesh is.
    np.testing.assert_allclose(result.stress[:, 5], 1.0, rtol=2.0e-2)
    assert np.max(np.abs(result.stress[:, :3])) < 1.0e-9


def test_beam_end_forces_are_the_analytic_cantilever_moments() -> None:
    """``BEAM2`` reports what a beam is checked with: end forces and moments."""
    model = beam_cantilever(8)
    force = 300.0
    asm = assemble_km(model)
    u = solve_static(model, {(9, 2): force}, assembly=asm)
    result = recover_stress(model, u, assembly=asm)

    root = result.extras[1]
    # Local frame of a beam along +x is the global one, so My at node 1 of the
    # root element is the fixed-end moment -P L.
    assert root["moments"][0] == pytest.approx(force * 2.0, rel=1.0e-9)
    assert root["end_forces"][2] == pytest.approx(-force, rel=1.0e-9)
    # The centroid of a beam section sits on its neutral axis: no axial stress
    # from bending, and the transverse load produces none either.
    assert abs(result.stress[0, 0]) < 1.0e-9
    # Every element's end forces are self-equilibrated: the two force vectors
    # cancel and the two moments balance the couple of the shear over the span.
    for eid in result.element_ids:
        forces = result.extras[eid]["end_forces"]
        span = np.array([result.extras[eid]["length"], 0.0, 0.0])
        np.testing.assert_allclose(forces[:3] + forces[6:9], 0.0, atol=1.0e-7 * force)
        np.testing.assert_allclose(
            forces[3:6] + forces[9:12] + np.cross(span, forces[6:9]), 0.0, atol=1.0e-7 * force
        )


def test_shell_bending_stress_is_linear_through_the_thickness() -> None:
    """Mid-surface zero, equal and opposite fibres, ``6 M / t^2`` at the surface."""
    thickness = 0.02
    model = shell_plate(4, 4, side=1.0, thickness=thickness, clamped_edge=True)
    asm = assemble_km(model)
    tip = [nid for nid, node in model["nodes"].items() if abs(node["xyz"][0] - 1.0) < 1.0e-12]
    u = solve_static(model, {(nid, 2): 1.0 / len(tip) for nid in tip}, assembly=asm)

    mid = recover_stress(model, u, assembly=asm, layer="mid")
    top = recover_stress(model, u, assembly=asm, layer="top")
    bottom = recover_stress(model, u, assembly=asm, layer="bottom")
    quarter = recover_stress(model, u, assembly=asm, layer=0.25)

    in_plane = [0, 1, 3]
    peak = float(np.max(np.abs(top.stress[:, in_plane])))
    assert peak > 0.0
    assert np.max(np.abs(mid.stress[:, in_plane])) < 1.0e-10 * peak
    np.testing.assert_allclose(
        bottom.stress[:, in_plane], -top.stress[:, in_plane], atol=1.0e-10 * peak
    )
    np.testing.assert_allclose(quarter.stress[:, in_plane], 0.5 * top.stress[:, in_plane],
                               rtol=1.0e-12)
    # The transverse shear is a section average: it does not vary with z.
    np.testing.assert_array_equal(bottom.stress[:, 4:], top.stress[:, 4:])
    # The surface stress of a plate is 6 M / t^2.
    for eid in top.element_ids:
        moment = top.extras[eid]["moment"]
        np.testing.assert_allclose(
            top.stress[top.index_of(eid), [0, 1, 3]],
            6.0 * moment / thickness**2,
            rtol=1.0e-12,
        )


def test_mitc4_transverse_shear_resultant_balances_the_tip_load() -> None:
    """The assumed shear strain is a real quantity, not just a locking cure."""
    width = 0.5
    model = shell_plate(8, 2, side=1.0, thickness=0.02, clamped_edge=True)
    for node in model["nodes"].values():
        x, y, z = node["xyz"]
        node["xyz"] = (x, y * width, z)
    asm = assemble_km(model)
    tip = [nid for nid, node in model["nodes"].items() if abs(node["xyz"][0] - 1.0) < 1.0e-12]
    load = 40.0
    u = solve_static(model, {(nid, 2): load / len(tip) for nid in tip}, assembly=asm)
    result = recover_stress(model, u, assembly=asm)

    interior = [
        eid
        for eid in result.element_ids
        if 0.3 < result.centroid[result.index_of(eid), 0] < 0.7
    ]
    per_length = np.array([result.extras[eid]["transverse_shear"][0] for eid in interior])
    assert np.mean(per_length) * width == pytest.approx(load, rel=0.05)


# ----------------------------------------------------------------------
# stress recovery: frames, options and plumbing
# ----------------------------------------------------------------------


def test_stress_is_orientation_invariant_and_rotates_with_the_model() -> None:
    """The same plate hung obliquely returns the same stress, rotated.

    This is also the check that the recovery reads the *analysis* frame of a
    tilted shell correctly rather than mistaking its nodal triads for basic
    rotations.
    """

    def solve(rotation):
        rot = np.eye(3) if rotation is None else rotation
        model = shell_plate(4, 4, clamped_edge=True, rotation=rotation)
        asm = assemble_km(model)
        normal = rot @ np.array([0.0, 0.0, 1.0])
        tip = [
            nid
            for nid, node in model["nodes"].items()
            if abs((rot.T @ np.asarray(node["xyz"]))[0] - 1.0) < 1.0e-12
        ]
        loads = {(nid, comp): normal[comp] / len(tip) for nid in tip for comp in range(3)}
        u = solve_static(model, loads, assembly=asm)
        return recover_stress(model, u, assembly=asm, layer="top")

    aligned = solve(None)
    tilted = solve(OBLIQUE)

    np.testing.assert_allclose(tilted.von_mises, aligned.von_mises, rtol=1.0e-9)
    # Each element frame is the aligned one carried along by the rotation.
    for i in range(len(aligned)):
        np.testing.assert_allclose(
            tilted.frame[i], aligned.frame[i] @ OBLIQUE.T, atol=1.0e-9
        )
    np.testing.assert_allclose(tilted.stress, aligned.stress, rtol=1.0e-8, atol=1.0e-6)
    np.testing.assert_allclose(
        tilted.centroid, aligned.centroid @ OBLIQUE.T, atol=1.0e-12
    )


def test_frames_are_orthonormal_and_solids_report_in_the_basic_frame() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    solid = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    np.testing.assert_array_equal(solid.frame, np.tile(np.eye(3), (len(solid), 1, 1)))
    np.testing.assert_array_equal(solid.stress, solid.stress_basic)

    plate = shell_plate(3, 3, rotation=OBLIQUE, clamped_edge=True)
    asm = assemble_km(plate)
    tip = [nid for nid in plate["nodes"] if nid not in {s["node_id"] for s in plate["spcs"]}]
    u = solve_static(plate, {(tip[0], 2): 1.0}, assembly=asm)
    shell = recover_stress(plate, u, assembly=asm)
    for R in shell.frame:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1.0e-14)
        assert float(np.linalg.det(R)) == pytest.approx(1.0)


def test_von_mises_and_principal_stresses_agree() -> None:
    model, _tip, loads = hex_cantilever(4, 1, 1)
    asm = assemble_km(model)
    result = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    p = result.principal

    assert np.all(np.diff(p, axis=1) <= 1.0e-9 * np.abs(p).max())
    from_principal = np.sqrt(
        0.5 * ((p[:, 0] - p[:, 1]) ** 2 + (p[:, 1] - p[:, 2]) ** 2 + (p[:, 2] - p[:, 0]) ** 2)
    )
    np.testing.assert_allclose(result.von_mises, from_principal, rtol=1.0e-10)
    np.testing.assert_allclose(result.max_shear, 0.5 * (p[:, 0] - p[:, 2]), rtol=1.0e-12)
    # Frame independent, so the rotated tensors give the same equivalent stress.
    np.testing.assert_allclose(von_mises(result.stress_basic), result.von_mises, rtol=1.0e-10)


def test_result_bookkeeping_is_usable() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    result = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    assert len(result) == result.n_elements == 3
    assert result.components == COMPONENTS == ("xx", "yy", "zz", "xy", "yz", "zx")
    assert result.etypes == ["HEX8"] * 3
    assert result.location == "centroid"
    record = result.element(2)
    assert record["element_id"] == 2
    np.testing.assert_array_equal(record["stress"], result.stress[result.index_of(2)])
    np.testing.assert_allclose(result.tensor(2), np.asarray(result.tensor(2)).T)
    with pytest.raises(KeyError, match="not in this result"):
        result.index_of(4321)


def test_recovery_accepts_a_static_result_a_vector_or_no_assembly() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    static = solve_static(model, loads, assembly=asm, full_result=True)

    from_result = recover_stress(model, static)
    from_vector = recover_stress(model, static.u, assembly=asm)
    bare = recover_stress(model, static.u)

    np.testing.assert_array_equal(from_result.stress, from_vector.stress)
    np.testing.assert_array_equal(from_result.stress, bare.stress)
    np.testing.assert_array_equal(
        recover_strain(model, static).strain, from_result.strain
    )


def test_recovery_reports_what_it_cannot_do() -> None:
    model = {
        "nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in (1, 2)},
        "elements": {
            1: {"type": "BAR2", "property_id": 1, "nodes": (1, 2)},
            2: {"type": "MASS", "nodes": (2,), "m": 3.0},
            3: {"type": "SPRING", "nodes": (1, 2), "k": 10.0, "c1": 0},
        },
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": 1.0e-4}},
        "spcs": [],
    }
    result = recover_stress(model, np.zeros(12))

    assert result.element_ids == [1]
    assert set(result.skipped) == {2, 3}
    assert "no stress state" in result.skipped[2]

    unknown = {**model, "elements": {9: {"type": "CQUAD8", "nodes": (1, 2)}}}
    with pytest.raises(KeyError, match="unknown element type"):
        recover_stress(unknown, np.zeros(12))
    assert recover_stress(unknown, np.zeros(12), on_unknown="skip").skipped[9]


def test_recovery_arguments_are_checked() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    u = solve_static(model, loads, assembly=asm)

    assert len(recover_stress(model, u, assembly=asm, elements=[1, 3])) == 2
    assert len(recover_stress(model, u, assembly=asm, elements=lambda eid, _e: eid == 2)) == 1
    with pytest.raises(ValueError, match="unknown layer"):
        recover_stress(model, u, assembly=asm, layer="middle-ish")
    with pytest.raises(ValueError, match="outside the section"):
        recover_stress(model, u, assembly=asm, layer=1.5)
    with pytest.raises(ValueError, match="entries but the model has"):
        recover_stress(model, np.zeros(3), assembly=asm)
    with pytest.raises(ValueError, match="one displacement field at a time"):
        recover_stress(model, np.column_stack([u, u]), assembly=asm)


# ----------------------------------------------------------------------
# RBE2: the contract
# ----------------------------------------------------------------------


def test_welded_pair_is_free_free_with_exactly_six_rigid_body_modes() -> None:
    """The acceptance case: a rigid body element adds no stiffness at all."""
    result = rbe2_rigid_pair()

    assert result["zero_modes"] == 6.0
    assert result["free_dof"] == 6.0
    assert result["dependent_dof"] == 6.0
    assert result["stiffness_norm"] == 0.0


def test_welded_pair_reduces_to_the_analytic_rigid_body_mass_matrix() -> None:
    """Not merely six zeros: the six equations are the right ones.

    The reduced mass has to be the textbook rigid body mass matrix of the pair
    about the independent node, offset coupling and parallel-axis term included.
    """
    for offset in ((1.0, 0.5, -0.3), (0.0, 0.0, 0.0), (2.5, 0.0, 0.0)):
        result = rbe2_rigid_pair(offset)
        assert result["mass_error"] < 1.0e-12
        assert result["zero_modes"] == 6.0


def test_rigid_offset_delivers_force_and_moment() -> None:
    """The second acceptance case: a rigid arm carries a moment.

    An axial force on a node held out on a rigid arm has to reach the beam tip
    as that force plus ``arm * force``.  Measured against a model where the two
    are applied directly *and* against the analytic Euler-Bernoulli cantilever.
    """
    result = rbe2_offset_moment()

    assert result["direct_gap"] < 1.0e-12
    assert result["rigid_kinematics"] < 1.0e-12
    assert result["tip_axial"] == pytest.approx(result["analytic_axial"], rel=1.0e-9)
    assert result["tip_deflection"] == pytest.approx(result["analytic_deflection"], rel=1.0e-9)
    assert result["tip_rotation"] == pytest.approx(result["analytic_rotation"], rel=1.0e-9)
    # Without the moment the tip would not rotate at all: this is the whole point.
    assert abs(result["tip_rotation"]) > 1.0e-3


def test_rigid_arm_kinematics_hold_after_the_solve() -> None:
    """``u_dependent == u_independent + theta_independent x r``, exactly."""
    model = welded_pair()
    asm = assemble_km(model)
    u = solve_static(model, {(3, 2): -500.0, (3, 1): 200.0}, assembly=asm)

    tip = u[asm.dof_map.node_dofs(2)]
    arm = u[asm.dof_map.node_dofs(3)]
    lever = model.nodes[3].xyz - model.nodes[2].xyz

    np.testing.assert_allclose(arm[3:], tip[3:], rtol=1.0e-14)
    np.testing.assert_allclose(
        arm[:3], tip[:3] + np.cross(tip[3:], lever), rtol=1.0e-12
    )
    assert np.max(np.abs(arm)) > 0.0


def test_a_rigid_link_inside_a_structure_keeps_six_rigid_body_modes() -> None:
    """A free-free beam with a rigid arm is still a free-free structure.

    Adding a rigid body must not add energy anywhere: the six zero frequencies
    stay six, and the first elastic mode is untouched by the arm.
    """
    base = beam_cantilever(6)
    tip = len(base["nodes"])
    free = {**base, "spcs": []}
    with_arm = {
        **free,
        "nodes": {**free["nodes"], 99: {"xyz": (2.0, 0.3, 0.4)}},
        "rbe2": [{"id": 1, "independent": tip, "dependents": (99,)}],
    }

    frequencies = solve_modes(with_arm, n_modes=9).freq_hz
    np.testing.assert_allclose(frequencies, solve_modes(free, n_modes=9).freq_hz, rtol=1.0e-9)
    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0


def test_mode_shapes_carry_the_dependent_motion_and_stay_orthonormal() -> None:
    """Filling the eliminated DOFs is free: the transform is idempotent."""
    model = welded_pair()
    result = solve_modes(model, n_modes=5)
    asm = result.assembly
    arm = asm.dof_map.node_dofs(3)

    assert np.max(np.abs(result.modes[arm, :])) > 0.0
    gram = result.modes.T @ (asm.M @ result.modes)
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1.0e-8)
    # ... and the filled shape is exactly the rigid image of the tip motion.
    lever = np.asarray(model.nodes[3].xyz) - np.asarray(model.nodes[2].xyz)
    tip = result.modes[asm.dof_map.node_dofs(2), :]
    np.testing.assert_allclose(
        result.modes[arm[:3], :], tip[:3] + np.cross(tip[3:].T, lever).T, atol=1.0e-12
    )


def test_stress_recovery_sees_the_motion_of_a_rigidly_driven_node() -> None:
    """An element hanging off a rigid body must be recovered from its real motion."""
    model = welded_pair()
    asm = assemble_km(model)
    u = solve_static(model, {(3, 0): 900.0}, assembly=asm)
    result = recover_stress(model, u, assembly=asm)

    direct = {
        "nodes": {nid: {"xyz": tuple(node.xyz)} for nid, node in model.nodes.items()},
        "elements": {1: {"type": "BEAM2", "property_id": 1, "nodes": (1, 2)}},
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {
            1: {"type": "beam", "material_id": 1, "A": 6.0e-4, "Iy": 5.0e-8, "Iz": 5.0e-8,
                "J": 8.0e-8}
        },
        "spcs": [{"node_id": 1, "dofs": (0, 1, 2, 3, 4, 5)}],
    }
    lever = np.asarray(model.nodes[3].xyz) - np.asarray(model.nodes[2].xyz)
    moment = np.cross(lever, np.array([900.0, 0.0, 0.0]))
    reference = recover_stress(
        direct,
        solve_static(
            direct,
            {(2, 0): 900.0, (2, 3): moment[0], (2, 4): moment[1], (2, 5): moment[2]},
        ),
    )

    np.testing.assert_allclose(result.stress, reference.stress, rtol=1.0e-12)
    assert abs(result.extras[1]["axial_force"] - 900.0) < 1.0e-9


# ----------------------------------------------------------------------
# RBE2: the transform itself
# ----------------------------------------------------------------------


def test_transform_is_an_exact_idempotent_projector() -> None:
    model = welded_pair()
    transform = apply_rbe2(model)

    G = transform.G.toarray()
    np.testing.assert_allclose(G @ G, G, atol=1.0e-15)
    np.testing.assert_array_equal(transform.dependent, np.arange(12, 18))
    np.testing.assert_array_equal(transform.independent, np.arange(12))
    assert transform.T.shape == (18, 12)
    assert transform.dependent_nodes() == [3]
    assert transform.independent_nodes() == [2]
    assert set(transform.nodes()) == {2, 3}
    assert not transform.is_identity
    assert ConstraintTransform.identity(transform.dof_map).is_identity


def test_assembly_honours_the_model_table_and_can_be_told_not_to() -> None:
    model = welded_pair()
    default = assemble_km(model)
    explicit = assemble_km(model, mpc=apply_rbe2(model))
    records = assemble_km(model, mpc=model.rbe2)
    without = assemble_km(model, mpc=False)

    assert default.mpc is not None
    assert (default.K - explicit.K).nnz == 0
    assert (default.K - records.K).nnz == 0
    np.testing.assert_array_equal(default.mpc_dof, np.arange(12, 18))
    assert without.mpc is None
    assert without.mpc_dof.size == 0
    # Without the rigid body the arm node is unattached, so its DOFs are empty
    # rather than dependent -- a different statement with the same DOF count.
    assert set(without.null_dof) == set(range(12, 18))


def test_transform_matches_the_analytic_rigid_body_rows() -> None:
    """The coefficients are ``u_m = u_n - skew(r) theta_n``, nothing else."""
    model = welded_pair(offset=(0.4, -0.2, 0.3))
    transform = apply_rbe2(model)
    r = np.asarray(model.nodes[3].xyz) - np.asarray(model.nodes[2].xyz)
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])

    block = transform.G.toarray()[12:18, 6:12]
    expected = np.block([[np.eye(3), -skew], [np.zeros((3, 3)), np.eye(3)]])
    np.testing.assert_allclose(block, expected, atol=1.0e-15)


def test_only_the_listed_components_are_eliminated() -> None:
    """A translation-only ``RBE2`` is a ball joint, not a weld."""
    model = welded_pair(components=(1, 2, 3))
    asm = assemble_km(model)

    np.testing.assert_array_equal(asm.mpc_dof, np.arange(12, 15))
    # The dependent node keeps its own rotations; nothing drives them, so the
    # assembler removes them as empty rather than as dependent.
    assert set(asm.null_dof) == {15, 16, 17}
    # A moment on the arm node is now lost instead of being carried across.
    u = solve_static(model, {(3, 4): 100.0}, assembly=asm)
    assert abs(u[asm.dof_map.index(2, 4)]) < 1.0e-18


def test_chained_rigid_bodies_resolve_to_their_root() -> None:
    """A dependent node acting as an independent one is substituted through."""
    coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 0.8, 0.0)}
    model = {"nodes": {nid: {"xyz": xyz} for nid, xyz in coords.items()}, "elements": {}}
    chained = apply_rbe2(
        model,
        [
            {"id": 1, "independent": 1, "dependents": (2,)},
            {"id": 2, "independent": 2, "dependents": (3,)},
        ],
    )
    direct = apply_rbe2(model, [{"id": 1, "independent": 1, "dependents": (2, 3)}])

    np.testing.assert_allclose(chained.G.toarray(), direct.G.toarray(), atol=1.0e-15)
    # Every dependent row references the root node only -- node 2 has been
    # substituted away -- which is what makes G idempotent.
    G = chained.G.toarray()
    assert np.count_nonzero(G[6:18, 6:]) == 0
    np.testing.assert_allclose(chained.G @ chained.G.toarray(), G, atol=1.0e-15)


def test_pathological_rigid_bodies_are_rejected() -> None:
    coords = {i: {"xyz": (float(i), 0.0, 0.0)} for i in range(1, 5)}
    model = {"nodes": coords, "elements": {}}

    with pytest.raises(ValueError, match="circular"):
        apply_rbe2(
            model,
            [
                {"id": 1, "independent": 1, "dependents": (2,)},
                {"id": 2, "independent": 2, "dependents": (1,)},
            ],
        )
    with pytest.raises(ValueError, match="already dependent"):
        apply_rbe2(
            model,
            [
                {"id": 1, "independent": 1, "dependents": (3,)},
                {"id": 2, "independent": 2, "dependents": (3,)},
            ],
        )
    with pytest.raises(ValueError, match="cannot also be dependent"):
        apply_rbe2(model, [{"id": 1, "independent": 1, "dependents": (1,)}])
    with pytest.raises(KeyError, match="not in the model"):
        apply_rbe2(model, [{"id": 1, "independent": 1, "dependents": (99,)}])
    with pytest.raises(ValueError, match="out of range"):
        apply_rbe2(model, [{"id": 1, "independent": 1, "dependents": (2,), "components": (7,)}])
    assert MAX_CHAIN_DEPTH >= 8


def test_a_dependent_dof_cannot_also_be_constrained_or_enforced() -> None:
    model = welded_pair()
    model.add_spc(3, (True, False, False, False, False, False))

    with pytest.raises(ValueError, match="both single point constrained and dependent"):
        assemble_km(model)

    free = welded_pair()
    asm = assemble_km(free)
    with pytest.raises(ValueError, match="no equation to enforce"):
        solve_static(free, {}, assembly=asm, enforced={(3, 0): 0.001})
    # The independent node is still drivable.
    solve_static(free, {}, assembly=asm, enforced={(2, 0): 0.001})


def test_component_lists_accept_the_nastran_spellings() -> None:
    model = welded_pair()
    reference = apply_rbe2(model).G.toarray()
    for spelling in ("123456", 123456, (1, 2, 3, 4, 5, 6), [1, 2, 3, 4, 5, 6], None):
        transform = apply_rbe2(
            model, [{"id": 1, "independent": 2, "dependents": (3,), "components": spelling}]
        )
        np.testing.assert_array_equal(transform.G.toarray(), reference)


def test_the_core_femodel_rbe2_table_is_what_gets_applied() -> None:
    """``FEModel.add_rbe2`` is consumed as-is; the kernel defines no second table."""
    model = welded_pair()
    assert [rbe.id for rbe in model.rbe2] == [1]
    assert model.rbe2[0].components == (1, 2, 3, 4, 5, 6)

    plain = {
        "nodes": {nid: {"xyz": tuple(node.xyz)} for nid, node in model.nodes.items()},
        "elements": {1: {"type": "BEAM2", "property_id": 1, "nodes": (1, 2)}},
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {
            1: {"type": "beam", "material_id": 1, "A": 6.0e-4, "Iy": 5.0e-8, "Iz": 5.0e-8,
                "J": 8.0e-8}
        },
        "spcs": [{"node_id": 1, "dofs": (0, 1, 2, 3, 4, 5)}],
        "rbe2": [{"id": 1, "independent": 2, "dependents": (3,), "components": (1, 2, 3, 4, 5, 6)}],
    }
    loads = {(3, 2): -750.0}
    np.testing.assert_allclose(
        solve_static(model, loads), solve_static(plain, loads), atol=1.0e-18
    )


def test_a_model_without_rigid_bodies_is_untouched() -> None:
    """Bit-for-bit: the new code path must not exist for an ordinary model."""
    model = beam_cantilever(8)
    with_default = assemble_km(model)
    disabled = assemble_km(model, mpc=False)

    assert with_default.mpc is None
    assert (with_default.K - disabled.K).nnz == 0
    assert (with_default.M - disabled.M).nnz == 0
    np.testing.assert_array_equal(with_default.free_dof, disabled.free_dof)
    assert apply_rbe2(model).is_identity


# ----------------------------------------------------------------------
# goldens that must not move
# ----------------------------------------------------------------------


def test_hex8_cantilever_tip_ratio_golden() -> None:
    assert hex8_bending_ratio() == pytest.approx(0.9854730473, rel=1.0e-6)


def test_hex8_patch_and_rigid_body_goldens() -> None:
    frequencies = hex8_rigid_body_frequencies()

    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0
    assert hex8_patch_test_error() < 1.0e-10


@pytest.mark.parametrize("etype", ["QUAD4", "TRIA3"])
def test_tilted_shell_still_has_six_rigid_body_modes(etype: str) -> None:
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_zero_modes"] == 6.0
    assert gap["aligned_zero_modes"] == 6.0
    assert gap["oblique_warned"] == 0.0


@pytest.mark.parametrize("thickness", [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5])
def test_mitc4_thin_plate_does_not_shear_lock(thickness: float) -> None:
    E, nu, side = 70.0e9, 0.3, 1.0
    model = shell_plate(8, 8, side=side, thickness=thickness, E=E, nu=nu, clamped_edge=True)
    asm = assemble_km(model)
    tip = [nid for nid, node in model["nodes"].items() if abs(node["xyz"][0] - side) < 1.0e-12]
    u = solve_static(model, {(nid, 2): 1.0 / len(tip) for nid in tip}, assembly=asm)
    deflection = float(np.mean([u[asm.dof_map.index(nid, 2)] for nid in tip]))

    rigidity = E * thickness**3 / (12.0 * (1.0 - nu**2))
    assert deflection / (side**3 / (3.0 * rigidity * side)) == pytest.approx(1.025, abs=0.01)


def test_enforced_displacement_on_a_free_dof_still_holds() -> None:
    model = {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3, 4)},
        "elements": {
            i: {"type": "SPRING", "nodes": (i, i + 1), "k": 1000.0, "c1": 0} for i in (1, 2, 3)
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

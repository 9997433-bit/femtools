"""Round 8 (R8-O1): ``RBE3`` interpolation constraints and nodal stress averaging.

Two additions to the kernel, both of which are defined by what they are *not*.

``femtools.fea.mpc.apply_rbe3`` applies the ``RBE3`` records the core model has
been carrying (``FEModel.add_rbe3``) as the interpolation multipoint constraint
of the literature (Cook §13.5; Zienkiewicz & Taylor, master-slave elimination):
one dependent grid point whose listed components are the **weighted average**
of the same components of its independents.  It is not the rigid-body
kinematics of ``RBE2`` -- the independents are not welded to each other -- and
it is not a penalty spring: a mass hung on the reference grid of a free-free
spider leaves exactly six rigid body modes, and a force on it is shared out in
proportion to the weights, equally for equal weights whatever the geometry.
``assemble_km`` composes it with ``model.rbe2`` into a single transform.

``femtools.fea.recover.average_nodal`` smooths the centroid stresses onto the
nodes with the classical ``1 / n_adjacent`` share.  It is not Zienkiewicz-Zhu
superconvergent patch recovery: no polynomial, no patch, no error estimate.
Its contract is that a constant stress state comes through exactly at every
node, which it does to round-off for all six element types.

The last section re-measures the goldens that must not move.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtools.core.model import FEModel
from femtools.fea import (
    NodalStressResult,
    apply_mpc,
    apply_rbe2,
    apply_rbe3,
    assemble_km,
    average_nodal,
    recover_strain,
    recover_stress,
    solve_modes,
    solve_static,
)
from femtools.fea.mpc import ConstraintTransform, is_rbe3, rbe2_records, rbe3_records
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
    rbe3_load_path,
    rbe3_spider,
    shell_drilling_orientation_gap,
    shell_plate,
    stress_patch_error,
)


def spider(
    weights=None, components=(1, 2, 3), dependent=(0.0, 0.0, 0.0)
) -> FEModel:
    """A ``FEModel`` triangle of ``BAR2`` rods with a mass on an ``RBE3`` reference grid."""
    model = FEModel(name="rbe3-spider")
    for i, angle in enumerate(np.deg2rad([90.0, 210.0, 330.0])):
        model.add_node(i + 1, (float(0.6 * np.cos(angle)), float(0.6 * np.sin(angle)), 0.0))
    model.add_node(4, tuple(float(v) for v in dependent))
    model.add_material(1, E=2.1e11, nu=0.3, rho=7800.0)
    model.add_property(1, "bar", material_id=1, A=4.0e-4)
    model.add_property(2, "lumped", m=2.5)
    for eid, conn in enumerate(((1, 2), (2, 3), (3, 1)), start=1):
        model.add_element(eid, "BAR2", conn, property_id=1)
    model.add_element(4, "MASS", (4,), property_id=2)
    model.add_rbe3(
        1, dependent=4, independents=(1, 2, 3), components=components, weights=weights
    )
    return model


def plate_patch(nx: int = 3, ny: int = 3) -> tuple[dict, dict, float]:
    """Plane-stress ``QUAD4`` strip pulled along ``x``: constant ``sxx = F / (w t)``."""
    thickness, width, length, force = 0.02, 1.0, 3.0, 5.0e4
    ids = {}
    nodes: dict[int, dict] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            ids[(i, j)] = counter
            nodes[counter] = {"xyz": (length * i / nx, width * j / ny, 0.0)}
            counter += 1
    elements = {}
    eid = 1
    for i in range(nx):
        for j in range(ny):
            elements[eid] = {
                "type": "QUAD4",
                "property_id": 1,
                "nodes": (ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]),
            }
            eid += 1
    tip = [ids[(nx, j)] for j in range(ny + 1)]
    share = [0.5 if j in (0, ny) else 1.0 for j in range(ny + 1)]
    model = {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": 70.0e9, "nu": 0.3, "rho": 2700.0}},
        "properties": {1: {"type": "membrane", "material_id": 1, "t": thickness}},
        "spcs": [{"node_id": ids[(0, j)], "dofs": (0,)} for j in range(ny + 1)]
        + [{"node_id": ids[(0, 0)], "dofs": (1,)}],
    }
    loads = {(nid, 0): force * w / sum(share) for nid, w in zip(tip, share, strict=True)}
    return model, loads, force / (width * thickness)


# ----------------------------------------------------------------------
# RBE3: the contract
# ----------------------------------------------------------------------


def test_free_free_spider_keeps_exactly_six_rigid_body_modes() -> None:
    """The acceptance case: an interpolation constraint adds no stiffness at all.

    A pin-jointed triangle is an exactly determinate rigid body in space; hang
    a mass on a fourth node tied to its vertices by one ``RBE3`` and it must
    still be a free-free structure with six zero frequencies and not a trace of
    stiffness from the constraint.
    """
    result = rbe3_spider()

    assert result["zero_modes"] == 6.0
    assert result["first_elastic_hz"] > 1.0
    assert result["free_dof"] == 9.0
    assert result["dependent_dof"] == 3.0
    assert result["constraint_stiffness"] == 0.0


def test_the_dependent_mass_arrives_in_full_at_the_weighted_centroid() -> None:
    """Not merely six zeros: the reduced rigid body mass is the right one.

    Because the dependent motion is a weighted average, a rigid body motion of
    the independents carries the reference grid to their *weighted centroid* --
    so the reduced rigid body mass matrix has to be the bare triangle's plus a
    point mass sitting exactly there, for any weights.  Moving the reference
    grid off the centroid does not change that: the mass is still delivered in
    full, and still to the centroid rather than to where the node was drawn.
    """
    for case in (
        {},
        {"weights": (3.0, 1.0, 1.0)},
        {"weights": (1.0, 2.0, 5.0)},
        {"dependent_xyz": (0.2, -0.1, 0.35)},
    ):
        result = rbe3_spider(**case)
        assert result["rigid_mass_error"] < 1.0e-12
        assert result["zero_modes"] == 6.0


def test_a_force_on_the_reference_grid_is_shared_out_by_weight() -> None:
    """The second acceptance case: ``G^T f``, equal weights giving equal shares.

    Three parallel rods are the whole load path, so what each carries is the
    share the constraint handed it.  Equal weights give exactly one third each
    whatever the geometry -- an ``RBE2`` cap would have distributed by
    stiffness instead -- and unequal weights give exactly their normalised
    fractions.
    """
    equal = rbe3_load_path()

    assert equal["share_error"] < 1.0e-12
    assert equal["min_share"] == pytest.approx(1.0 / 3.0, rel=1.0e-12)
    assert equal["max_share"] == pytest.approx(1.0 / 3.0, rel=1.0e-12)
    assert equal["extension_error"] < 1.0e-12

    uneven = rbe3_load_path(weights=(4.0, 1.0, 1.0))

    assert uneven["share_error"] < 1.0e-12
    assert uneven["max_share"] == pytest.approx(4.0 / 6.0, rel=1.0e-12)
    assert uneven["min_share"] == pytest.approx(1.0 / 6.0, rel=1.0e-12)
    # The legs no longer move together, and the reference grid follows the
    # weighted average of three different extensions: this is not a rigid cap.
    assert uneven["dependent"] == pytest.approx(uneven["analytic_dependent"], rel=1.0e-12)
    assert uneven["average_gap"] < 1.0e-12
    assert uneven["dependent"] > equal["dependent"]


def test_the_load_shares_are_the_transpose_of_the_averaging_rows() -> None:
    """Virtual work, spelled out: ``f_independent = G^T f_dependent``."""
    model = spider(weights=(2.0, 1.0, 1.0))
    transform = apply_rbe3(model)
    dofs = transform.dof_map

    force = np.zeros(dofs.n_dof)
    force[dofs.index(4, 2)] = 600.0
    shared = transform.to_independent(force)

    np.testing.assert_allclose(
        [shared[dofs.index(nid, 2)] for nid in (1, 2, 3)],
        [300.0, 150.0, 150.0],
        rtol=1.0e-14,
    )
    assert shared[dofs.index(4, 2)] == 0.0
    # Nothing is lost or created on the way: the shares sum to the force.
    assert shared.sum() == pytest.approx(600.0, rel=1.0e-14)


def test_interpolation_kinematics_hold_after_a_solve() -> None:
    """``u_dependent == sum_i w_i u_i / sum_j w_j``, exactly, all six components.

    A ``BEAM2`` cantilever carries real rotations as well as translations, so
    the identity can be read component by component on a solved field rather
    than only on the transform.  Node 99 is driven by the mid-span and the tip
    with a three-to-one weighting, and lands three quarters of the way from the
    tip to mid-span in every component at once.
    """
    base = beam_cantilever(4)
    model = {**base, "nodes": {**base["nodes"], 99: {"xyz": (2.0, 0.4, -0.3)}}}
    model["rbe3"] = [
        {
            "id": 1,
            "dependent": 99,
            "independents": (3, 5),
            "components": (1, 2, 3, 4, 5, 6),
            "weights": (3.0, 1.0),
        }
    ]
    asm = assemble_km(model)
    u = solve_static(model, {(5, 2): -400.0, (5, 4): 60.0, (4, 1): 250.0}, assembly=asm)

    for comp in range(6):
        expected = (
            0.75 * u[asm.dof_map.index(3, comp)] + 0.25 * u[asm.dof_map.index(5, comp)]
        )
        assert u[asm.dof_map.index(99, comp)] == pytest.approx(expected, rel=1.0e-12)
    assert np.max(np.abs(u[asm.dof_map.node_dofs(99)])) > 0.0
    # The reference grid is *not* on the rigid image of either driver: it sits
    # between them, which is exactly what an RBE2 could not have produced.
    assert not np.allclose(
        u[asm.dof_map.node_dofs(99)], u[asm.dof_map.node_dofs(3)], atol=1.0e-12
    )


def test_an_rbe3_is_not_an_rbe2() -> None:
    """The distinction the round is about, measured on one model.

    Welding the same four nodes with an ``RBE2`` makes the triangle a rigid
    body: nine degrees of freedom collapse to the six of the reference grid and
    the three elastic modes of the triangle disappear.  Tying them with an
    ``RBE3`` instead eliminates only the reference grid's own three components
    and leaves every elastic mode of the triangle where it was.
    """
    interpolated = spider()
    welded = FEModel(name="rbe2-cap")
    for nid, node in interpolated.nodes.items():
        welded.add_node(nid, tuple(node.xyz))
    welded.add_material(1, E=2.1e11, nu=0.3, rho=7800.0)
    welded.add_property(1, "bar", material_id=1, A=4.0e-4)
    welded.add_property(2, "lumped", m=2.5)
    for eid, el in interpolated.elements.items():
        welded.add_element(eid, el.type, tuple(el.nodes), property_id=el.property_id)
    welded.add_rbe2(1, independent=4, dependents=(1, 2, 3))

    soft = assemble_km(interpolated)
    rigid = assemble_km(welded)

    assert soft.n_free == 9
    assert soft.mpc_dof.size == 3
    assert rigid.n_free == 6
    assert rigid.mpc_dof.size == 18
    soft_hz = solve_modes(interpolated, n_modes=9, assembly=soft).freq_hz
    assert np.count_nonzero(soft_hz < 1.0e-6) == 6
    assert np.count_nonzero(soft_hz > 1.0e-6) == 3
    # The weld leaves a body that cannot deform at all: six equations, no
    # stiffness in any of them (bar round-off), so no elastic mode is left.
    scale = abs(assemble_km(welded, mpc=False).K).max()
    assert abs(rigid.Kff).max() < 1.0e-12 * scale
    assert np.max(solve_modes(welded, n_modes=6, assembly=rigid).freq_hz) < 1.0e-3


def test_only_the_listed_components_are_eliminated() -> None:
    """``REFC`` means what it says: an unlisted component is not driven."""
    model = spider(components=(1, 2))
    asm = assemble_km(model)
    dofs = asm.dof_map

    np.testing.assert_array_equal(asm.mpc_dof, [dofs.index(4, 0), dofs.index(4, 1)])
    # uz of the reference grid is now nobody's business: it carries the mass
    # but nothing drives it, so it is a free DOF of its own.
    assert dofs.index(4, 2) in set(asm.free_dof)
    # ... which means the structure has a seventh zero frequency, the mass
    # bobbing on nothing.  Stating it is the point: the constraint did not
    # quietly supply the component the card left out.
    assert np.count_nonzero(solve_modes(model, n_modes=8).freq_hz < 1.0e-6) == 7


def test_a_component_the_independents_do_not_carry_is_refused() -> None:
    """No silent rigid-body term: an average cannot manufacture a rotation."""
    model = spider(components=(1, 2, 3))
    with pytest.raises(ValueError, match="not among the independent components"):
        apply_rbe3(
            model,
            [
                {
                    "id": 1,
                    "dependent": 4,
                    "independents": (1, 2, 3),
                    "components": (1, 2, 3, 4, 5, 6),
                    "independent_components": (1, 2, 3),
                }
            ],
        )


# ----------------------------------------------------------------------
# RBE3: the transform, and living with RBE2
# ----------------------------------------------------------------------


def test_transform_rows_are_the_normalised_weights() -> None:
    model = spider(weights=(2.0, 1.0, 1.0))
    transform = apply_rbe3(model)
    G = transform.G.toarray()
    dofs = transform.dof_map

    for comp in range(3):
        row = G[dofs.index(4, comp)]
        np.testing.assert_allclose(
            [row[dofs.index(nid, comp)] for nid in (1, 2, 3)], [0.5, 0.25, 0.25], atol=1.0e-15
        )
        assert row.sum() == pytest.approx(1.0, rel=1.0e-14)
    np.testing.assert_allclose(G @ G, G, atol=1.0e-15)
    np.testing.assert_array_equal(transform.dependent, np.arange(18, 21))
    assert transform.dependent_nodes() == [4]
    assert sorted(transform.independent_nodes()) == [1, 2, 3]
    assert not transform.is_identity


def test_equal_weights_are_the_default_and_can_be_spelled_out() -> None:
    reference = apply_rbe3(spider()).G.toarray()
    model = spider()
    for weights in (None, (1.0, 1.0, 1.0), (7.5, 7.5, 7.5), [2, 2, 2]):
        transform = apply_rbe3(
            model,
            [{"id": 1, "dependent": 4, "independents": (1, 2, 3), "weights": weights}],
        )
        np.testing.assert_allclose(transform.G.toarray(), reference, atol=1.0e-15)


def chained_arm() -> dict:
    """A cantilever with an interpolated node 98 and a rigid arm 99 hanging off it."""
    base = beam_cantilever(4)
    return {
        **base,
        "nodes": {
            **base["nodes"],
            98: {"xyz": (1.5, 0.0, 0.0)},
            99: {"xyz": (1.5, 0.35, 0.2)},
        },
        "rbe3": [
            {
                "id": 1,
                "dependent": 98,
                "independents": (3, 5),
                "components": (1, 2, 3, 4, 5, 6),
            }
        ],
        "rbe2": [{"id": 1, "independent": 98, "dependents": (99,)}],
    }


def test_assembly_composes_rbe2_and_rbe3_into_one_transform() -> None:
    """A rigid arm on a node that is itself interpolated: one ``G``, resolved through."""
    model = chained_arm()
    asm = assemble_km(model)
    transform = apply_mpc(model)

    assert asm.mpc is not None
    np.testing.assert_array_equal(asm.mpc_dof, transform.dependent)
    # Node 99 hangs rigidly off node 98, which is itself the average of nodes 3
    # and 5; the chain is substituted away, so no dependent row references
    # another dependent DOF and G stays idempotent.
    G = transform.G.toarray()
    assert set(transform.dependent_nodes()) == {98, 99}
    assert sorted(transform.independent_nodes()) == [3, 5]
    assert np.count_nonzero(G[np.ix_(transform.dependent, transform.dependent)]) == 0
    np.testing.assert_allclose(G @ G, G, atol=1.0e-14)

    # ... and the two kinematics hold at once on a solved field.
    u = solve_static(model, {(99, 0): 700.0, (5, 2): -120.0}, assembly=asm)
    dofs = asm.dof_map
    average = 0.5 * (u[dofs.node_dofs(3)] + u[dofs.node_dofs(5)])
    arm = u[dofs.node_dofs(99)]
    lever = np.array([0.0, 0.35, 0.2])

    np.testing.assert_allclose(u[dofs.node_dofs(98)], average, rtol=1.0e-11)
    np.testing.assert_allclose(arm[3:], average[3:], rtol=1.0e-11)
    np.testing.assert_allclose(arm[:3], average[:3] + np.cross(average[3:], lever),
                               rtol=1.0e-11)
    assert np.max(np.abs(arm)) > 0.0


def test_mpc_false_disables_both_tables() -> None:
    model = chained_arm()
    default = assemble_km(model)
    without = assemble_km(model, mpc=False)

    assert default.mpc is not None
    assert default.mpc_dof.size == 12
    assert without.mpc is None
    assert without.mpc_dof.size == 0
    assert apply_mpc(model, rbe2=(), rbe3=()).is_identity
    # Untied, the two extra nodes are simply unattached rather than dependent.
    assert set(without.null_dof) >= set(default.mpc_dof)


def test_explicit_records_replace_the_model_tables_either_way() -> None:
    model = spider()
    default = assemble_km(model)

    for explicit in (model.rbe3, list(model.rbe3), model.rbe3[0]):
        assert (assemble_km(model, mpc=explicit).K - default.K).nnz == 0
    assert (assemble_km(model, mpc=apply_rbe3(model)).K - default.K).nnz == 0
    # A plain-dictionary spelling of the same card, read duck-typed.
    plain = [{"id": 1, "refgrid": 4, "gi": (1, 2, 3), "refc": 123}]
    np.testing.assert_allclose(
        apply_rbe3(model, plain).G.toarray(), apply_rbe3(model).G.toarray(), atol=1.0e-15
    )
    with pytest.raises(TypeError, match="not a bare matrix"):
        assemble_km(model, mpc=np.eye(24))


def test_the_two_card_types_are_told_apart_in_a_mixed_container() -> None:
    """One ``mpc`` list may hold both; each reader picks out its own."""
    rigid = {"id": 1, "independent": 1, "dependents": (2,)}
    interpolation = {"id": 2, "dependent": 4, "independents": (1, 2, 3)}
    model = {"nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in (1, 2, 3, 4)}, "elements": {}}
    mixed = [rigid, interpolation]

    assert not is_rbe3(rigid)
    assert is_rbe3(interpolation)
    assert [rid for rid, _ in rbe2_records(model, mixed)] == [1]
    assert [rid for rid, _ in rbe3_records(model, mixed)] == [2]
    assert rbe2_records(model, interpolation) == []
    assert rbe3_records(model, rigid) == []

    both = apply_mpc(model, rbe2=mixed, rbe3=mixed)
    assert set(both.dependent_nodes()) == {2, 4}


def test_the_core_femodel_rbe3_table_is_what_gets_applied() -> None:
    """``FEModel.add_rbe3`` is consumed as-is; the kernel defines no second table."""
    model = spider(weights=(2.0, 1.0, 1.0))
    assert [rbe.id for rbe in model.rbe3] == [1]
    assert model.rbe3[0].dependent == 4
    assert model.rbe3[0].independents == (1, 2, 3)
    assert model.rbe3[0].weights == (2.0, 1.0, 1.0)

    duck = {
        "nodes": {nid: {"xyz": tuple(node.xyz)} for nid, node in model.nodes.items()},
        "elements": {
            eid: {
                "type": el.type,
                "property_id": el.property_id,
                "nodes": tuple(el.nodes),
                **({"m": 2.5} if el.type == "MASS" else {}),
            }
            for eid, el in model.elements.items()
        },
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": 4.0e-4}},
        "spcs": [],
        "rbe3": [
            {
                "id": 1,
                "dependent": 4,
                "independents": (1, 2, 3),
                "components": (1, 2, 3),
                "weights": (2.0, 1.0, 1.0),
            }
        ],
    }
    np.testing.assert_allclose(
        solve_modes(model, n_modes=8).freq_hz, solve_modes(duck, n_modes=8).freq_hz, rtol=1.0e-12
    )


def test_pathological_interpolation_constraints_are_rejected() -> None:
    model = {"nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in range(1, 6)}, "elements": {}}

    def build(**record):
        return apply_rbe3(model, [{"id": 1, **record}])

    with pytest.raises(ValueError, match="cannot also be independent"):
        build(dependent=1, independents=(1, 2))
    with pytest.raises(ValueError, match="at least one independent node"):
        build(dependent=1, independents=())
    with pytest.raises(ValueError, match="no dependent"):
        build(independents=(1, 2))
    with pytest.raises(ValueError, match="2 weights for 3 independent nodes"):
        build(dependent=1, independents=(2, 3, 4), weights=(1.0, 2.0))
    with pytest.raises(ValueError, match="weights must be positive"):
        build(dependent=1, independents=(2, 3), weights=(1.0, -1.0))
    with pytest.raises(ValueError, match="weights must be finite"):
        build(dependent=1, independents=(2, 3), weights=(1.0, np.inf))
    with pytest.raises(ValueError, match="out of range"):
        build(dependent=1, independents=(2, 3), components=(7,))
    with pytest.raises(KeyError, match="not in the model"):
        build(dependent=1, independents=(2, 99))
    with pytest.raises(KeyError, match="not in the model"):
        build(dependent=99, independents=(1, 2))
    with pytest.raises(ValueError, match="already dependent"):
        apply_rbe3(
            model,
            [
                {"id": 1, "dependent": 1, "independents": (2, 3)},
                {"id": 2, "dependent": 1, "independents": (4, 5)},
            ],
        )
    # ... and the same DOF claimed by a rigid body and an interpolation card.
    with pytest.raises(ValueError, match="already dependent"):
        apply_mpc(
            model,
            rbe2=[{"id": 1, "independent": 2, "dependents": (1,), "components": (1, 2, 3)}],
            rbe3=[{"id": 2, "dependent": 1, "independents": (3, 4)}],
        )


def test_a_circular_interpolation_is_caught() -> None:
    model = {"nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in (1, 2, 3)}, "elements": {}}
    with pytest.raises(ValueError, match="circular"):
        apply_rbe3(
            model,
            [
                {"id": 1, "dependent": 1, "independents": (2, 3)},
                {"id": 2, "dependent": 2, "independents": (1, 3)},
            ],
        )


def test_an_empty_rbe3_table_leaves_the_rbe2_transform_bit_identical() -> None:
    """The new code path must not exist for a model that does not use it."""
    coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.4, 0.5, -0.3)}
    base = {"nodes": {nid: {"xyz": xyz} for nid, xyz in coords.items()}, "elements": {}}
    records = [{"id": 1, "independent": 2, "dependents": (3,)}]

    reference = apply_rbe2(base, records)
    with_empty = apply_mpc({**base, "rbe3": []}, rbe2=records)

    assert (reference.G != with_empty.G).nnz == 0
    np.testing.assert_array_equal(reference.G.data, with_empty.G.data)
    np.testing.assert_array_equal(reference.G.indices, with_empty.G.indices)
    np.testing.assert_array_equal(reference.dependent, with_empty.dependent)

    beam = beam_cantilever(8)
    assert apply_rbe3(beam).is_identity
    assert apply_mpc(beam).is_identity
    assert assemble_km(beam).mpc is None


# ----------------------------------------------------------------------
# nodal averaging: the contract
# ----------------------------------------------------------------------


@pytest.mark.parametrize("etype", PATCH_TYPES)
def test_constant_stress_survives_the_average_at_every_node(etype: str) -> None:
    """The acceptance case: averaging cannot damage a constant field.

    The same constant-strain patch that pins the element recovery, re-measured
    after the centroid values have been smoothed onto the nodes.  Interior and
    boundary nodes alike, whatever the number of elements meeting there.
    """
    error = stress_patch_error(etype)

    assert error["nodes"] >= 4.0
    assert error["nodal"] < 1.0e-12
    assert error["stress"] < 1.0e-12


def test_the_average_is_one_over_the_number_of_adjacent_elements() -> None:
    """No weighting by area, volume or distance: an equal share, and that is all."""
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    nodal = average_nodal(stress, model)

    incident: dict[int, list[int]] = {}
    for i, eid in enumerate(stress.element_ids):
        for nid in model["elements"][eid]["nodes"]:
            incident.setdefault(nid, []).append(i)

    assert set(nodal.node_ids) == set(incident)
    for nid, rows in incident.items():
        assert nodal.count[nodal.index_of(nid)] == len(rows)
        np.testing.assert_allclose(
            nodal.stress[nodal.index_of(nid)],
            stress.stress_basic[rows].mean(axis=0),
            rtol=1.0e-14,
        )
        np.testing.assert_allclose(
            nodal.strain[nodal.index_of(nid)],
            stress.strain_basic[rows].mean(axis=0),
            rtol=1.0e-14,
        )
    # The end nodes of a single-file mesh touch one element, the interior ones two.
    assert sorted(set(nodal.count.tolist())) == [1, 2]


def test_the_average_is_taken_in_the_basic_frame() -> None:
    """Tensors written in different element frames cannot simply be added.

    The same plate hung obliquely must give the same nodal field, rotated --
    which only holds if the contributions were rotated out of their element
    frames first.
    """
    oblique = np.linalg.qr(
        np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
    )[0]

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
        return average_nodal(recover_stress(model, u, assembly=asm, layer="top"), model)

    aligned = solve(None)
    tilted = solve(oblique)

    assert tilted.node_ids == aligned.node_ids
    np.testing.assert_array_equal(tilted.count, aligned.count)
    np.testing.assert_allclose(tilted.von_mises, aligned.von_mises, rtol=1.0e-8, atol=1.0e-6)
    np.testing.assert_allclose(tilted.xyz, aligned.xyz @ oblique.T, atol=1.0e-12)


def test_the_average_smooths_a_real_gradient_without_moving_the_mean() -> None:
    """What averaging is for: a continuous field whose element mean is unchanged."""
    model, loads, exact = plate_patch()
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    nodal = average_nodal(stress, model)

    # This membrane state is uniform, so the nodal field is the element one.
    np.testing.assert_allclose(nodal.stress[:, 0], exact, rtol=1.0e-12)
    assert np.max(np.abs(nodal.stress[:, 1:])) < 1.0e-9 * exact
    np.testing.assert_allclose(nodal.von_mises, exact, rtol=1.0e-12)

    # A bending gradient is what actually gets smoothed: the nodal peak of a
    # cantilever plate is bounded by the element peak but not equal to it.
    plate = shell_plate(6, 6, thickness=0.02, clamped_edge=True)
    asm = assemble_km(plate)
    tip = [nid for nid, node in plate["nodes"].items() if abs(node["xyz"][0] - 1.0) < 1.0e-12]
    u = solve_static(plate, {(nid, 2): 1.0 / len(tip) for nid in tip}, assembly=asm)
    bending = recover_stress(plate, u, assembly=asm, layer="top")
    smoothed = average_nodal(bending, plate)

    assert 0.0 < smoothed.von_mises.max() <= bending.von_mises.max() * (1.0 + 1.0e-12)
    assert smoothed.von_mises.max() > 0.5 * bending.von_mises.max()
    assert len(smoothed) == len(plate["nodes"])


def test_von_mises_is_taken_of_the_averaged_tensor() -> None:
    """A nonlinear function of the tensor: averaging it first is the exact route."""
    model, _tip, loads = hex_cantilever(4, 2, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    nodal = average_nodal(stress, model)

    np.testing.assert_allclose(von_mises(nodal.stress), nodal.von_mises, rtol=1.0e-12)
    p = nodal.principal
    assert np.all(np.diff(p, axis=1) <= 1.0e-9 * np.abs(p).max())
    np.testing.assert_allclose(
        nodal.tensor(nodal.node_ids[0]), nodal.tensor(nodal.node_ids[0]).T, rtol=1.0e-14
    )


def test_nodal_result_bookkeeping_is_usable() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    nodal = average_nodal(stress, model)

    assert isinstance(nodal, NodalStressResult)
    assert len(nodal) == nodal.n_nodes == len(model["nodes"])
    assert nodal.components == COMPONENTS
    assert nodal.location == "node"
    assert nodal.element_ids == stress.element_ids
    assert nodal.etypes == ["HEX8"] * 3
    record = nodal.node(nodal.node_ids[0])
    np.testing.assert_array_equal(record["stress"], nodal.stress[0])
    assert record["count"] == int(nodal.count[0])
    np.testing.assert_allclose(record["xyz"], model["nodes"][nodal.node_ids[0]]["xyz"])
    with pytest.raises(KeyError, match="carries no averaged stress"):
        nodal.index_of(4321)
    with pytest.raises(TypeError, match="StressResult"):
        average_nodal(np.zeros((3, 6)), model)

    # ``nodes=`` restricts the report without changing any value.
    wanted = nodal.node_ids[:2]
    subset = average_nodal(stress, model, nodes=wanted)
    assert subset.node_ids == wanted
    np.testing.assert_array_equal(subset.stress, nodal.stress[:2])
    assert average_nodal(stress, model, nodes=lambda nid: nid == wanted[0]).n_nodes == 1
    assert average_nodal(stress, model, nodes=()).n_nodes == 0


def test_elements_without_a_stress_state_are_not_averaged() -> None:
    """A concentrated mass contributes nothing, and neither does its node."""
    model = {
        "nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in (1, 2, 3)},
        "elements": {
            1: {"type": "BAR2", "property_id": 1, "nodes": (1, 2)},
            2: {"type": "MASS", "nodes": (3,), "m": 3.0},
        },
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": 1.0e-4}},
        "spcs": [
            {"node_id": 1, "dofs": (0, 1, 2)},
            {"node_id": 2, "dofs": (1, 2)},
            {"node_id": 3, "dofs": (0, 1, 2)},
        ],
    }
    asm = assemble_km(model)
    u = solve_static(model, {(2, 0): 500.0}, assembly=asm)
    stress = recover_stress(model, u, assembly=asm)
    nodal = average_nodal(stress, model)

    assert stress.skipped[2]
    assert nodal.node_ids == [1, 2]
    np.testing.assert_allclose(nodal.stress[:, 0], 500.0 / 1.0e-4, rtol=1.0e-12)
    np.testing.assert_array_equal(nodal.count, [1, 1])
    # Strains average the same way, and ``recover_strain`` feeds the same object.
    np.testing.assert_array_equal(
        average_nodal(recover_strain(model, u, assembly=asm), model).strain, nodal.strain
    )


def test_averaging_reaches_a_node_driven_by_an_interpolation_constraint() -> None:
    """The whole chain: solve with an ``RBE3``, recover, average.

    A uniform bar stretched through a two-node interpolation constraint still
    reports its exact uniform stress at every node, dependent one included.
    """
    model = {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3, 4)},
        "elements": {
            1: {"type": "BAR2", "property_id": 1, "nodes": (1, 2)},
            2: {"type": "BAR2", "property_id": 1, "nodes": (3, 4)},
        },
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": 2.0e-4}},
        "spcs": [{"node_id": nid, "dofs": (1, 2)} for nid in (1, 2, 3, 4)]
        + [{"node_id": 1, "dofs": (0,)}],
        # Node 3 simply follows node 2: a one-node "average" is a plain link.
        "rbe3": [{"id": 1, "dependent": 3, "independents": (2,), "components": (1,)}],
    }
    asm = assemble_km(model)
    u = solve_static(model, {(4, 0): 400.0}, assembly=asm)
    stress = recover_stress(model, u, assembly=asm)
    nodal = average_nodal(stress, model)

    assert u[asm.dof_map.index(3, 0)] == pytest.approx(u[asm.dof_map.index(2, 0)], rel=1.0e-14)
    np.testing.assert_allclose(stress.stress[:, 0], 400.0 / 2.0e-4, rtol=1.0e-12)
    np.testing.assert_allclose(nodal.stress[:, 0], 400.0 / 2.0e-4, rtol=1.0e-12)
    assert sorted(nodal.node_ids) == [1, 2, 3, 4]


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


def test_the_rbe2_goldens_are_untouched() -> None:
    pair = rbe2_rigid_pair()
    offset = rbe2_offset_moment()

    assert pair["zero_modes"] == 6.0
    assert pair["mass_error"] < 1.0e-12
    assert pair["stiffness_norm"] == 0.0
    assert offset["direct_gap"] < 1.0e-12
    assert offset["rigid_kinematics"] < 1.0e-12
    assert offset["tip_axial"] == pytest.approx(offset["analytic_axial"], rel=1.0e-9)


def test_enforced_displacement_still_holds_with_an_interpolated_node() -> None:
    """``solve_static(enforced=)`` and ``RBE3`` in the same model."""
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

    # The same chain with node 4 interpolated from nodes 2 and 3 instead of
    # sprung to node 3: the enforced DOF still lands exactly, and node 4 lands
    # on the average of the two nodes that drive it.
    interpolated = {
        **model,
        "elements": {i: model["elements"][i] for i in (1, 2)},
        "rbe3": [{"id": 1, "dependent": 4, "independents": (2, 3), "components": (1,)}],
    }
    asm = assemble_km(interpolated)
    u = solve_static(interpolated, {}, assembly=asm, enforced={(3, 0): 0.03})
    values = [u[asm.dof_map.index(nid, 0)] for nid in (1, 2, 3, 4)]

    np.testing.assert_allclose(values[:3], [0.0, 0.015, 0.03], atol=1.0e-12)
    assert values[3] == pytest.approx(0.5 * (values[1] + values[2]), rel=1.0e-12)


def test_a_dependent_rbe3_dof_cannot_also_be_constrained() -> None:
    model = spider()
    model.add_spc(4, (True, False, False, False, False, False))

    with pytest.raises(ValueError, match="both single point constrained and dependent"):
        assemble_km(model)


def test_the_frozen_import_paths_resolve() -> None:
    """``femtools.fea.mpc.apply_rbe3`` and ``femtools.fea.recover.average_nodal``."""
    from femtools.fea.mpc import apply_rbe3 as from_mpc
    from femtools.fea.recover import average_nodal as from_recover

    assert from_mpc is apply_rbe3
    assert from_recover is average_nodal


def test_the_identity_transform_is_still_free() -> None:
    beam = beam_cantilever(4)
    identity = ConstraintTransform.identity(assemble_km(beam).dof_map)

    assert identity.is_identity
    assert identity.n_dependent == 0
    assert (assemble_km(beam, mpc=identity).K - assemble_km(beam).K).nnz == 0

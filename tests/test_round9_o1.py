"""Round 9 (R9-O1): ``apply_mpc`` frozen as the public composer of the two MPC tables.

``femtools.fea.mpc`` grew in two rounds -- ``RBE2`` rigid bodies first, ``RBE3``
interpolation constraints after -- and this module pins the shape the pair
settled into, which is one composer and two wrappers.

``apply_mpc(model, rbe2=..., rbe3=...)`` reads ``model.rbe2`` *and*
``model.rbe3``, puts both card types into a single set of dependent rows and
resolves them together, so a chain crossing from one type to the other is
substituted away like any other and ``G`` stays idempotent whichever card sits
on top.  That is what ``assemble_km`` applies.  ``apply_rbe2`` and
``apply_rbe3`` are the same call with the other table set to ``()``, and the
tests below check that literally: the CSR ``data``, ``indices`` and ``indptr``
of the transform they return are the composer's, entry for entry.

The content of neither card moves in this round.  An ``RBE2`` is still the
rigid weld ``u_m = u_n + theta_n x r`` and an ``RBE3`` still the weighted
average ``u_d[c] = sum_i w_i u_i[c] / sum_j w_j`` -- one component at a time,
reading no coordinates at all (Cook, *Concepts and Applications of Finite
Element Analysis*, §13.5; Zienkiewicz & Taylor, master-slave elimination).  The
last section re-measures the goldens that must not move.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp

import femtools.fea as fea
from femtools.core.model import FEModel
from femtools.fea import (
    ConstraintTransform,
    apply_mpc,
    apply_rbe2,
    apply_rbe3,
    assemble_km,
    solve_modes,
    solve_static,
)
from femtools.fea.mpc import resolve_mpc
from femtools.fea.verification import (
    MPC_CHAIN_DIRECTIONS,
    beam_cantilever,
    hex8_bending_ratio,
    hex8_patch_test_error,
    hex8_rigid_body_frequencies,
    mpc_mixed_chain,
    rbe2_offset_moment,
    rbe2_rigid_pair,
    rbe3_load_path,
    rbe3_spider,
    shell_drilling_orientation_gap,
    shell_plate,
)

ARM = (0.0, 0.3, -0.2)


def chained(rbe2: bool = True, rbe3: bool = True) -> dict:
    """``BEAM2`` cantilever, a rigid arm on node 3 and an ``RBE3`` on the arm.

    Node 98 is welded to node 3 by an ``RBE2`` and node 99 is the reference
    grid of an ``RBE3`` averaging node 98 with the tip -- an interpolation
    constraint hanging off a rigid body.  The two tables can be dropped
    independently, which is what the bit-identity checks compare against.
    """
    base = beam_cantilever(4)
    offset = np.asarray(base["nodes"][3]["xyz"], dtype=float) + np.asarray(ARM)
    model = {
        **base,
        "nodes": {
            **base["nodes"],
            98: {"xyz": tuple(float(v) for v in offset)},
            99: {"xyz": (1.7, 0.1, 0.05)},
        },
    }
    if rbe2:
        model["rbe2"] = [{"id": 1, "independent": 3, "dependents": (98,)}]
    if rbe3:
        model["rbe3"] = [
            {
                "id": 2,
                "dependent": 99,
                "independents": (98, 5),
                "components": (1, 2, 3, 4, 5, 6),
            }
        ]
    return model


def bit_identical(left: ConstraintTransform, right: ConstraintTransform) -> None:
    """The same transform down to the CSR arrays, not merely to a tolerance."""
    np.testing.assert_array_equal(left.G.data, right.G.data)
    np.testing.assert_array_equal(left.G.indices, right.G.indices)
    np.testing.assert_array_equal(left.G.indptr, right.G.indptr)
    np.testing.assert_array_equal(left.dependent, right.dependent)
    assert left.G.shape == right.G.shape
    assert left.sources == right.sources


# ----------------------------------------------------------------------
# the frozen entry point
# ----------------------------------------------------------------------


def test_the_frozen_import_path_resolves() -> None:
    """``from femtools.fea.mpc import apply_mpc``, and the package re-export."""
    from femtools.fea.mpc import apply_mpc as from_module

    assert from_module is apply_mpc
    assert fea.apply_mpc is apply_mpc
    assert "apply_mpc" in fea.__all__
    assert {"apply_rbe2", "apply_rbe3", "ConstraintTransform"} <= set(fea.__all__)


def test_the_composer_signature_is_frozen() -> None:
    """One positional model; the tables and the numbering are keyword-only."""
    parameters = inspect.signature(apply_mpc).parameters

    assert list(parameters) == ["model", "rbe2", "rbe3", "dof_map", "dofs_per_node", "index"]
    assert parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name, default in (("rbe2", None), ("rbe3", None), ("dof_map", None), ("index", None)):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is default
    assert parameters["dofs_per_node"].default == 6

    # The wrappers keep their own table in the second positional slot, which is
    # what ``apply_rbe2(model, records)`` has always meant.
    assert list(inspect.signature(apply_rbe2).parameters)[:2] == ["model", "rbe2"]
    assert list(inspect.signature(apply_rbe3).parameters)[:2] == ["model", "rbe3"]


def test_an_empty_pair_of_tables_is_the_identity() -> None:
    """No records, no transform: the composer does not manufacture work."""
    plain = beam_cantilever(4)
    transform = apply_mpc(plain)

    assert transform.is_identity
    assert transform.n_dependent == 0
    assert transform.dependent.size == 0
    assert transform.dependent_nodes() == []
    assert transform.independent_nodes() == []
    assert (transform.G - sp.identity(transform.n_dof, format="csr")).nnz == 0
    np.testing.assert_array_equal(transform.independent, np.arange(transform.n_dof))

    # ... and every use of it is a no-op, not a multiplication by one.
    asm = assemble_km(plain)
    vector = np.arange(float(transform.n_dof))
    assert transform.to_full(vector) is vector
    assert transform.to_independent(vector) is vector
    assert (transform.congruence(asm.K) - asm.K).nnz == 0
    assert asm.mpc is None
    assert asm.mpc_dof.size == 0

    # An explicit pair of empty tables says the same thing on a model that
    # does carry records, and assembling with it is assembling without them.
    model = chained()
    assert apply_mpc(model, rbe2=(), rbe3=()).is_identity
    assert (assemble_km(model, mpc=apply_mpc(model, rbe2=(), rbe3=())).K
            - assemble_km(model, mpc=False).K).nnz == 0


def test_an_empty_rbe3_table_leaves_apply_rbe2_bit_identical() -> None:
    """The rigid-body transform of a model that uses no interpolation card."""
    both = chained()
    rigid_only = chained(rbe3=False)

    reference = apply_mpc(rigid_only)
    bit_identical(reference, apply_rbe2(rigid_only))
    bit_identical(reference, apply_mpc(rigid_only, rbe3=()))
    bit_identical(reference, apply_mpc(both, rbe3=()))
    # ``apply_rbe2`` on the mixed model reads its own table and nothing else.
    bit_identical(reference, apply_rbe2(both))
    assert reference.dependent_nodes() == [98]

    # The same statement for an explicit record list against an empty table.
    records = [{"id": 1, "independent": 3, "dependents": (98,)}]
    bit_identical(apply_rbe2(rigid_only, records), apply_mpc(both, rbe2=records, rbe3=()))


def test_an_empty_rbe2_table_leaves_apply_rbe3_bit_identical() -> None:
    """... and the mirror image, for a model that uses no rigid body."""
    both = chained()
    interpolation_only = chained(rbe2=False)

    reference = apply_mpc(interpolation_only)
    bit_identical(reference, apply_rbe3(interpolation_only))
    bit_identical(reference, apply_mpc(interpolation_only, rbe2=()))
    bit_identical(reference, apply_mpc(both, rbe2=()))
    bit_identical(reference, apply_rbe3(both))
    assert reference.dependent_nodes() == [99]

    records = interpolation_only["rbe3"]
    bit_identical(apply_rbe3(interpolation_only, records), apply_mpc(both, rbe2=(), rbe3=records))


def test_the_composition_is_the_product_of_the_two_maps() -> None:
    """One ``G``, and it is exactly the two transforms applied in chain order.

    Node 99 is the average of node 98 and the tip; node 98 is welded to node 3.
    Composing the tables must therefore give the same map as substituting the
    rigid transform into the interpolation one -- ``G_rbe3 @ G_rbe2``, the
    outer card first -- which it does to the last bit.  The result references
    no dependent column, which is what makes it idempotent and safe to apply to
    an already-filled vector.
    """
    model = chained()
    composed = apply_mpc(model)
    rigid = apply_rbe2(model)
    interpolation = apply_rbe3(model)

    np.testing.assert_array_equal(
        (interpolation.G @ rigid.G).toarray(), composed.G.toarray()
    )
    assert composed.n_dependent == rigid.n_dependent + interpolation.n_dependent
    np.testing.assert_array_equal(
        composed.dependent, np.union1d(rigid.dependent, interpolation.dependent)
    )

    G = composed.G.toarray()
    assert np.count_nonzero(G[np.ix_(composed.dependent, composed.dependent)]) == 0
    np.testing.assert_array_equal(G @ G, G)
    assert sorted(composed.dependent_nodes()) == [98, 99]
    assert sorted(composed.independent_nodes()) == [3, 5]

    # Reversing which card hangs off which reverses the product, and nothing
    # else: the composer does not care which table was written first.
    base = beam_cantilever(4)
    other = {
        **base,
        "nodes": {**base["nodes"], 98: {"xyz": (1.5, 0.0, 0.0)}, 99: {"xyz": (1.5, 0.35, 0.2)}},
        "rbe3": [
            {
                "id": 1,
                "dependent": 98,
                "independents": (3, 5),
                "components": (1, 2, 3, 4, 5, 6),
            }
        ],
        "rbe2": [{"id": 2, "independent": 98, "dependents": (99,)}],
    }
    np.testing.assert_array_equal(
        (apply_rbe2(other).G @ apply_rbe3(other).G).toarray(), apply_mpc(other).G.toarray()
    )


def test_explicit_records_replace_the_model_tables_on_either_side() -> None:
    """``rbe2=`` / ``rbe3=`` override; ``None`` reads the model's own."""
    model = chained()
    reference = apply_mpc(model)

    bit_identical(reference, apply_mpc(model, rbe2=model["rbe2"], rbe3=model["rbe3"]))
    bit_identical(reference, apply_mpc(model, rbe2=model["rbe2"][0], rbe3=model["rbe3"][0]))
    # One mixed container handed to both readers: each picks out its own cards.
    mixed = [*model["rbe2"], *model["rbe3"]]
    bit_identical(reference, apply_mpc(model, rbe2=mixed, rbe3=mixed))

    # A plain-dictionary spelling on the public Nastran field names.
    nastran = apply_mpc(
        model,
        rbe2=[{"id": 1, "gn": 3, "gm": (98,), "cm": 123456}],
        rbe3=[{"id": 2, "refgrid": 99, "gi": (98, 5), "refc": 123456}],
    )
    np.testing.assert_allclose(nastran.G.toarray(), reference.G.toarray(), atol=1.0e-15)


def test_the_composer_takes_cards_and_not_a_ready_made_operator() -> None:
    model = chained()

    for bad in (np.eye(6 * len(model["nodes"])), sp.identity(6, format="csr")):
        with pytest.raises(TypeError, match="not a bare matrix"):
            apply_mpc(model, rbe2=bad)
        with pytest.raises(TypeError, match="not a bare matrix"):
            apply_mpc(model, rbe3=bad)
    with pytest.raises(TypeError, match="not a ConstraintTransform"):
        apply_mpc(model, rbe2=apply_mpc(model))
    # ``assemble_km(mpc=...)`` is where a built transform goes, and it says so.
    with pytest.raises(TypeError, match="not a bare matrix"):
        assemble_km(model, mpc=np.eye(6 * len(model["nodes"])))


# ----------------------------------------------------------------------
# the two card types chained together
# ----------------------------------------------------------------------


@pytest.mark.parametrize("direction", MPC_CHAIN_DIRECTIONS)
@pytest.mark.parametrize("weights", [None, (3.0, 1.0, 1.0), (1.0, 2.0, 5.0)])
def test_a_mixed_chain_keeps_exactly_six_rigid_body_modes(direction, weights) -> None:
    """The acceptance case: composing the two cards is still exact.

    A free-free ``BEAM2`` triangle with a mass hung two constraints away --
    a rigid arm on an interpolated reference grid, or an interpolated grid on
    a rigid arm.  Neither order may add stiffness, invent a seventh zero mode
    or lose a gram of the mass on the way.
    """
    case = mpc_mixed_chain(direction, weights=weights)

    assert case["zero_modes"] == 6.0
    assert case["first_elastic_hz"] > 1.0
    assert case["free_dof"] == 18.0
    assert case["dependent_dof"] == 12.0
    assert case["constraint_stiffness"] == 0.0
    assert case["idempotency"] == 0.0
    assert case["rigid_mass_error"] < 1.0e-12


def test_the_hung_mass_is_delivered_where_the_two_kinematics_carry_it() -> None:
    """Not merely six zeros: the reduced rigid body mass is the right one.

    An ``RBE2`` moves the mass by its own lever and an ``RBE3`` carries it to
    the weighted centroid of its independents, so the delivery point of the
    chain is the composition of the two -- and it moves with the weights, not
    with where the dependent node happens to be drawn.
    """
    equal = mpc_mixed_chain("rbe2_on_rbe3")
    skewed = mpc_mixed_chain("rbe2_on_rbe3", weights=(3.0, 1.0, 1.0))

    # Equal weights put the reference grid at the triangle's centroid, so the
    # arm delivers the mass at its own drawn position.
    assert (equal["delivered_x"], equal["delivered_y"], equal["delivered_z"]) == (
        pytest.approx(0.15),
        pytest.approx(-0.1),
        pytest.approx(0.4),
    )
    assert skewed["delivered_y"] != pytest.approx(equal["delivered_y"])
    assert equal["rigid_mass_error"] < 1.0e-12
    assert skewed["rigid_mass_error"] < 1.0e-12

    # Placing the reference grid off the centroid changes neither statement.
    moved = mpc_mixed_chain("rbe3_on_rbe2", arm=(0.4, 0.25, -0.6))
    assert moved["zero_modes"] == 6.0
    assert moved["rigid_mass_error"] < 1.0e-12


def test_both_kinematics_hold_at_once_on_a_solved_field() -> None:
    """Rigid weld and weighted average, read component by component after a solve."""
    model = chained()
    asm = assemble_km(model)
    u = np.asarray(solve_static(model, {(99, 0): 800.0, (5, 2): -250.0}, assembly=asm))
    dofs = asm.dof_map
    node3, arm, tip, reference = (u[dofs.node_dofs(nid)] for nid in (3, 98, 5, 99))

    np.testing.assert_allclose(arm[3:], node3[3:], rtol=1.0e-12)
    np.testing.assert_allclose(
        arm[:3], node3[:3] + np.cross(node3[3:], np.asarray(ARM)), rtol=1.0e-12
    )
    np.testing.assert_allclose(reference, 0.5 * (arm + tip), rtol=1.0e-12)
    assert np.max(np.abs(reference)) > 0.0


def test_a_load_at_the_far_end_of_the_chain_arrives_as_force_and_moment() -> None:
    """``f -> G^T f`` through both cards: half to the tip, half onto the arm.

    Virtual work is the whole content of the load path.  The interpolation card
    hands half the force to node 98 and half to the tip; the rigid card carries
    node 98's half back to node 3 as that force *plus* the moment of the arm.
    Applying those three things directly must give the identical beam.
    """
    model = chained()
    force = 800.0
    asm = assemble_km(model)
    u = np.asarray(solve_static(model, {(99, 0): force, (5, 2): -250.0}, assembly=asm))

    moment = np.cross(np.asarray(ARM), np.array([0.5 * force, 0.0, 0.0]))
    direct_model = beam_cantilever(4)
    loads = {(5, 2): -250.0, (5, 0): 0.5 * force, (3, 0): 0.5 * force}
    loads.update({(3, 3 + k): moment[k] for k in range(3)})
    direct = np.asarray(solve_static(direct_model, loads))

    beam_dofs = np.concatenate([asm.dof_map.node_dofs(nid) for nid in range(1, 6)])
    gap = np.max(np.abs(u[beam_dofs] - direct[: beam_dofs.size]))
    assert gap / np.max(np.abs(direct)) < 1.0e-12

    # The same, read off the transform: nothing is lost or created.
    transform = apply_mpc(model)
    f = np.zeros(transform.n_dof)
    f[transform.dof_map.index(99, 0)] = force
    shared = transform.to_independent(f)
    assert shared[transform.dof_map.index(5, 0)] == pytest.approx(0.5 * force, rel=1.0e-14)
    assert shared[transform.dof_map.index(3, 0)] == pytest.approx(0.5 * force, rel=1.0e-14)
    np.testing.assert_allclose(
        shared[transform.dof_map.node_dofs(3)][3:], moment, rtol=1.0e-12
    )
    assert shared[transform.dof_map.index(99, 0)] == 0.0


def test_the_rbe3_rows_are_still_the_normalised_weights_and_read_no_coordinates() -> None:
    """The content of the interpolation card does not move in this round.

    ``u_d[c] = sum_i w_i u_i[c] / sum_j w_j``, one component at a time, with no
    lever anywhere: moving every node of the model leaves ``G`` bit-identical,
    which is the difference from the rigid card that no tolerance can blur.
    """
    coords = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0), 4: (0.3, 0.3, 0.0)}
    records = [
        {
            "id": 1,
            "dependent": 4,
            "independents": (1, 2, 3),
            "components": (1, 2, 3, 4, 5, 6),
            "weights": (2.0, 1.0, 1.0),
        }
    ]

    def build(scale):
        model = {
            "nodes": {nid: {"xyz": tuple(scale * v for v in xyz)} for nid, xyz in coords.items()},
            "elements": {},
        }
        return apply_mpc(model, rbe3=records)

    transform = build(1.0)
    G = transform.G.toarray()
    dofs = transform.dof_map
    for comp in range(6):
        row = G[dofs.index(4, comp)]
        np.testing.assert_array_equal(
            [row[dofs.index(nid, comp)] for nid in (1, 2, 3)], [0.5, 0.25, 0.25]
        )
        # ... and zero everywhere else: components do not mix.
        assert row.sum() == pytest.approx(1.0, rel=1.0e-15)
        assert np.count_nonzero(row) == 3

    bit_identical(transform, build(7.0))
    bit_identical(transform, build(-3.5))

    # The rigid card, by contrast, is all coordinates.
    rigid = [{"id": 1, "independent": 1, "dependents": (4,)}]
    moved = {
        "nodes": {nid: {"xyz": tuple(2.0 * v for v in xyz)} for nid, xyz in coords.items()},
        "elements": {},
    }
    plain = {"nodes": {nid: {"xyz": xyz} for nid, xyz in coords.items()}, "elements": {}}
    assert not np.array_equal(
        apply_mpc(plain, rbe2=rigid).G.data, apply_mpc(moved, rbe2=rigid).G.data
    )


def test_an_interpolation_card_still_works_with_fewer_dofs_per_node() -> None:
    """A weighted average names components, so three DOFs per node are enough."""
    model = {
        "nodes": {nid: {"xyz": (float(nid), 0.0, 0.0)} for nid in (1, 2, 3)},
        "elements": {},
    }
    transform = apply_mpc(
        model,
        rbe3=[{"id": 1, "dependent": 3, "independents": (1, 2), "components": (1, 2, 3)}],
        dofs_per_node=3,
    )

    assert transform.n_dof == 9
    np.testing.assert_array_equal(transform.dependent, [6, 7, 8])
    # A rigid body with an offset cannot be written without the rotations.
    with pytest.raises(ValueError, match="rigid rotation term"):
        apply_mpc(
            model,
            rbe2=[{"id": 1, "independent": 1, "dependents": (3,), "components": (1, 2, 3)}],
            dofs_per_node=3,
        )


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------


def test_a_dof_eliminated_twice_is_refused_whichever_cards_claim_it() -> None:
    """One DOF, one elimination: the master-slave transform admits no ambiguity."""
    model = {"nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in range(1, 7)}, "elements": {}}
    rigid = {"id": 1, "independent": 2, "dependents": (1,), "components": (1, 2, 3)}
    interpolation = {"id": 2, "dependent": 1, "independents": (3, 4)}

    with pytest.raises(ValueError, match="already dependent"):
        apply_mpc(model, rbe2=[rigid], rbe3=[interpolation])
    with pytest.raises(ValueError, match="already dependent"):
        apply_mpc(model, rbe2=[rigid, {**rigid, "id": 3, "independent": 5}])
    with pytest.raises(ValueError, match="already dependent"):
        apply_mpc(model, rbe3=[interpolation, {**interpolation, "id": 4, "independents": (5, 6)}])
    # A mixed container claiming the same DOF from both sides is the same clash.
    with pytest.raises(ValueError, match="already dependent"):
        apply_mpc(model, rbe2=[rigid, interpolation], rbe3=[rigid, interpolation])
    # Only the overlapping component: the rigid card leaves 4..6 alone.
    assert apply_mpc(
        model,
        rbe2=[rigid],
        rbe3=[{"id": 2, "dependent": 1, "independents": (3, 4), "components": (4, 5, 6)}],
    ).n_dependent == 6


def test_a_chain_that_loops_across_the_card_types_is_caught() -> None:
    model = {"nodes": {i: {"xyz": (float(i), 0.0, 0.0)} for i in (1, 2, 3)}, "elements": {}}

    with pytest.raises(ValueError, match="circular"):
        apply_mpc(
            model,
            rbe2=[{"id": 1, "independent": 2, "dependents": (1,)}],
            rbe3=[{"id": 2, "dependent": 2, "independents": (1, 3)}],
        )


def test_a_dependent_dof_cannot_also_be_single_point_constrained() -> None:
    model = FEModel(name="mpc-spc-clash")
    for nid in (1, 2, 3):
        model.add_node(nid, (float(nid), 0.0, 0.0))
    model.add_rbe3(1, dependent=3, independents=(1, 2), components=(1, 2, 3))
    model.add_spc(3, (True, False, False, False, False, False))

    with pytest.raises(ValueError, match="both single point constrained and dependent"):
        assemble_km(model)


# ----------------------------------------------------------------------
# what the assembler does with it
# ----------------------------------------------------------------------


def test_assemble_km_applies_the_composed_transform_by_default() -> None:
    model = chained()
    asm = assemble_km(model)
    transform = apply_mpc(model)

    assert asm.mpc is not None
    np.testing.assert_array_equal(asm.mpc_dof, transform.dependent)
    assert asm.mpc_dof.size == 12
    # ``resolve_mpc`` is the assembler's door to the composer, and ``None``
    # means "read both tables".
    bit_identical(transform, resolve_mpc(model, None, dof_map=asm.dof_map))
    assert resolve_mpc(model, transform, dof_map=asm.dof_map) is transform


def test_mpc_false_disables_both_tables() -> None:
    model = chained()
    default = assemble_km(model)
    without = assemble_km(model, mpc=False)

    assert default.mpc is not None
    assert without.mpc is None
    assert without.mpc_dof.size == 0
    assert resolve_mpc(model, False, dof_map=default.dof_map).is_identity
    # Untied, the two hung nodes are simply unattached rather than dependent.
    assert set(without.null_dof) >= set(default.mpc_dof)
    assert without.n_free == default.n_free


def test_the_core_femodel_tables_are_what_gets_composed() -> None:
    """``FEModel.add_rbe2`` / ``add_rbe3`` are consumed as they stand."""
    model = FEModel(name="mpc-compose")
    for nid, xyz in ((1, (0.0, 0.0, 0.0)), (2, (1.0, 0.0, 0.0)), (3, (1.0, 0.4, 0.0))):
        model.add_node(nid, xyz)
    model.add_node(4, (0.5, 0.2, 0.0))
    model.add_rbe2(1, independent=1, dependents=(2,), components=(1, 2, 3, 4, 5, 6))
    model.add_rbe3(2, dependent=4, independents=(2, 3), components=(1, 2, 3))

    transform = apply_mpc(model)

    assert [record.id for record in model.rbe2] == [1]
    assert [record.id for record in model.rbe3] == [2]
    assert sorted(transform.dependent_nodes()) == [2, 4]
    assert sorted(transform.independent_nodes()) == [1, 3]
    assert transform.sources == dict.fromkeys(range(6, 12), 1) | dict.fromkeys(range(18, 21), 2)
    # Node 4 averages a rigid node and a free one, so its row reaches through
    # the weld to node 1 rather than stopping at node 2: half of node 3's uy,
    # half of node 1's uy and half of the lever term that carries node 1's
    # rotation out to where node 2 sits.
    G = transform.G.toarray()
    dofs = transform.dof_map
    row = G[dofs.index(4, 1)]
    assert np.count_nonzero(row) == 3
    assert row[dofs.index(1, 1)] == 0.5
    assert row[dofs.index(3, 1)] == 0.5
    assert row[dofs.index(1, 5)] == 0.5 * 1.0
    assert row[dofs.index(2, 1)] == 0.0
    np.testing.assert_array_equal(G @ G, G)


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


@pytest.mark.parametrize("etype", ["QUAD4", "TRIA3"])
def test_tilted_shell_still_has_six_rigid_body_modes(etype: str) -> None:
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_zero_modes"] == 6.0
    assert gap["aligned_zero_modes"] == 6.0
    assert gap["oblique_warned"] == 0.0


def test_the_rbe2_and_rbe3_goldens_are_untouched() -> None:
    pair = rbe2_rigid_pair()
    offset = rbe2_offset_moment()
    spider = rbe3_spider()
    shares = rbe3_load_path()

    assert pair["zero_modes"] == 6.0
    assert pair["mass_error"] < 1.0e-12
    assert pair["stiffness_norm"] == 0.0
    assert offset["direct_gap"] < 1.0e-12
    assert offset["rigid_kinematics"] < 1.0e-12
    assert offset["tip_axial"] == pytest.approx(offset["analytic_axial"], rel=1.0e-9)
    assert spider["zero_modes"] == 6.0
    assert spider["rigid_mass_error"] < 1.0e-12
    assert spider["constraint_stiffness"] == 0.0
    assert shares["share_error"] < 1.0e-12
    assert shares["min_share"] == pytest.approx(1.0 / 3.0, rel=1.0e-12)


def test_enforced_displacement_still_holds_through_a_mixed_chain() -> None:
    """``solve_static(enforced=)`` with both card types in the same model."""
    springs = {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3, 4, 5)},
        "elements": {
            i: {"type": "SPRING", "nodes": (i, i + 1), "k": 1000.0, "c1": 0} for i in (1, 2)
        },
        "materials": {},
        "properties": {},
        "spcs": [{"node_id": 1, "dofs": (0,)}],
        # Node 4 averages nodes 2 and 3; node 5 is welded to node 4.
        "rbe3": [{"id": 1, "dependent": 4, "independents": (2, 3), "components": (1,)}],
        "rbe2": [{"id": 2, "independent": 4, "dependents": (5,), "components": (1,)}],
    }
    asm = assemble_km(springs)
    u = solve_static(springs, {}, assembly=asm, enforced={(3, 0): 0.03})
    values = [u[asm.dof_map.index(nid, 0)] for nid in range(1, 6)]

    np.testing.assert_allclose(values[:3], [0.0, 0.015, 0.03], atol=1.0e-12)
    assert values[3] == pytest.approx(0.5 * (values[1] + values[2]), rel=1.0e-12)
    assert values[4] == pytest.approx(values[3], rel=1.0e-12)
    assert set(asm.mpc_dof) == {asm.dof_map.index(4, 0), asm.dof_map.index(5, 0)}


def test_a_free_free_model_with_both_cards_still_has_six_zero_modes() -> None:
    """The gate once more, straight off ``solve_modes`` rather than a case builder."""
    base = beam_cantilever(4)
    free = {
        **base,
        "spcs": [],
        "nodes": {
            **base["nodes"],
            98: {"xyz": (2.0, 0.3, -0.2)},
            99: {"xyz": (1.5, 0.1, 0.0)},
        },
        "elements": {**base["elements"], 90: {"type": "MASS", "nodes": (99,), "m": 1.5}},
        "rbe2": [{"id": 1, "independent": 5, "dependents": (98,)}],
        "rbe3": [
            {
                "id": 2,
                "dependent": 99,
                "independents": (98, 3),
                "components": (1, 2, 3, 4, 5, 6),
            }
        ],
    }
    frequencies = solve_modes(free, n_modes=10).freq_hz

    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0

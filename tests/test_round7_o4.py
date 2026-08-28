"""Round 7 / O4: topometry optimization and the static-displacement response.

Every case is deterministic: the optimizers here have no random component, and
the numbers quoted in ``.agent_workspace/reports/R7-O4.md`` come straight from
this file.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from femtools.core.model import DOFSet, FEModel
from femtools.fea.elements import element_matrices
from femtools.fea.static import solve_static
from femtools.fea.verification import shell_plate
from femtools.optimization import TopometryResult, topometry_optimize
from femtools.optimization.shape import element_size_ratios
from femtools.optimization.topometry import _Problem
from femtools.updating import update_model
from femtools.updating.reference import make_updating_testcase
from femtools.updating.responses import static_displacement_response

TIP_LOAD = -1.0e3


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def cantilever_plate(
    nx: int = 6,
    ny: int = 3,
    *,
    lx: float = 0.9,
    ly: float = 0.45,
    t: float = 5.0e-3,
) -> tuple[FEModel, int]:
    """Clamped-free flat plate with a transverse point load at the free edge.

    The classic topometry demonstrator: the mesh is fixed, and the only freedom
    is how much thickness each element carries.  Bending dominates, so material
    is worth much more at the root than at the tip and a uniform start is far
    from optimal.
    """
    model = FEModel(name="cantilever-plate")
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.33, rho=2700.0)
    model.add_property(id=1, type="shell", material_id=1, t=t)
    ids: dict[tuple[int, int], int] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            ids[(i, j)] = counter
            model.add_node(id=counter, xyz=(lx * i / nx, ly * j / ny, 0.0))
            counter += 1
    eid = 1
    for i in range(nx):
        for j in range(ny):
            model.add_element(
                id=eid,
                type="QUAD4",
                nodes=(ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]),
                property_id=1,
            )
            eid += 1
    for j in range(ny + 1):
        model.add_spc(node_id=ids[(0, j)], mask=(True,) * 6)
    tip = ids[(nx, ny // 2)]
    model.add_load(node_id=tip, force=(0.0, 0.0, TIP_LOAD))
    return model, tip


def plate_node(i: int, j: int, ny: int) -> int:
    """Node id of grid position ``(i, j)`` in :func:`cantilever_plate`."""
    return i * (ny + 1) + j + 1


def solid_cantilever(nx: int = 4, ny: int = 2, nz: int = 2, a: float = 0.05) -> tuple[FEModel, int]:
    """HEX8 block clamped on one face, loaded at a free corner."""
    model = FEModel(name="solid-cantilever")
    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="solid", material_id=1)
    ids: dict[tuple[int, int, int], int] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                ids[(i, j, k)] = counter
                model.add_node(id=counter, xyz=(a * i, a * j, a * k))
                counter += 1
    eid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                model.add_element(
                    id=eid,
                    type="HEX8",
                    nodes=(
                        ids[(i, j, k)],
                        ids[(i + 1, j, k)],
                        ids[(i + 1, j + 1, k)],
                        ids[(i, j + 1, k)],
                        ids[(i, j, k + 1)],
                        ids[(i + 1, j, k + 1)],
                        ids[(i + 1, j + 1, k + 1)],
                        ids[(i, j + 1, k + 1)],
                    ),
                    property_id=1,
                )
                eid += 1
    for j in range(ny + 1):
        for k in range(nz + 1):
            model.add_spc(node_id=ids[(0, j, k)], mask=(True,) * 6)
    tip = ids[(nx, ny, nz)]
    model.add_load(node_id=tip, force=(0.0, 0.0, -5.0e3))
    return model, tip


def central_difference_gradient(problem: _Problem, x: np.ndarray, rel: float = 1.0e-4):
    """Central-difference compliance gradient, used to audit the analytic one."""
    out = np.empty_like(x)
    for i in range(x.size):
        h = rel * x[i]
        plus, minus = x.copy(), x.copy()
        plus[i] += h
        minus[i] -= h
        out[i] = (problem.compliance(plus) - problem.compliance(minus)) / (2.0 * h)
    return out


# ----------------------------------------------------------------------
# topometry -- the cantilever plate
# ----------------------------------------------------------------------
def test_topometry_reduces_cantilever_plate_compliance() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=60)

    assert isinstance(result, TopometryResult)
    assert result.design == "thickness"
    assert len(result.element_ids) == len(model.elements)
    # Redistributing the same material must stiffen the plate substantially.
    assert result.compliance < 0.5 * result.initial_compliance
    assert result.improvement > 0.5
    assert result.compliance == pytest.approx(min(result.compliance_history))
    assert result.iterations >= 1


def test_topometry_holds_the_volume_constraint() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=40)

    assert result.extras["constraint"] == "initial_volume"
    assert result.volume == pytest.approx(result.initial_volume, rel=1.0e-9)
    assert result.volume <= result.volume_limit * (1.0 + 1.0e-9)
    assert result.feasible
    # Sum of element area times thickness, recomputed independently.
    areas = np.array([0.15 * 0.15] * len(result.element_ids))
    assert float(areas @ result.x) == pytest.approx(result.volume, rel=1.0e-12)


def test_topometry_never_distorts_or_inverts_an_element() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=30)

    lo, hi = result.bounds
    assert np.all(result.x > 0.0)
    assert np.all(result.x >= lo - 1.0e-15)
    assert np.all(result.x <= hi + 1.0e-15)
    # No node moved, so every element keeps its exact size measure.
    ratios = element_size_ratios(result.model, model)
    assert np.allclose(ratios, 1.0, atol=0.0, rtol=0.0)
    assert result.min_size_ratio == 1.0


def test_topometry_leaves_the_input_model_untouched() -> None:
    model, _tip = cantilever_plate()
    before = copy.deepcopy(model)
    result = topometry_optimize(model, max_iter=10)

    assert model.properties[1].t == before.properties[1].t
    assert len(model.properties) == len(before.properties) == 1
    assert [e.property_id for e in model.elements.values()] == [
        e.property_id for e in before.elements.values()
    ]
    # The returned model carries one private property per designed element.
    assert len(result.model.properties) == 1 + len(result.element_ids)
    assert len({e.property_id for e in result.model.elements.values()}) == len(
        result.element_ids
    )


def test_topometry_returned_model_reproduces_the_reported_compliance() -> None:
    model, tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=25)

    u = solve_static(result.model, {(tip, "uz"): TIP_LOAD})
    dof = result.model.dof_map()[(tip, 2)]
    assert float(u[dof] * TIP_LOAD) == pytest.approx(result.compliance, rel=1.0e-10)


def test_topometry_thickens_the_root_and_thins_the_tip() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=60)

    thickness = result.to_dict()
    # Elements are numbered column by column: ids 1..3 are the root column,
    # the last three the tip column.
    root = np.mean([thickness[e] for e in (1, 2, 3)])
    tip_column = np.mean([thickness[e] for e in (16, 17, 18)])
    assert root > tip_column


# ----------------------------------------------------------------------
# topometry -- gradients
# ----------------------------------------------------------------------
def test_thickness_sensitivities_match_finite_differences() -> None:
    model, _tip = cantilever_plate(nx=3, ny=2)
    problem = _Problem(model, "thickness", None, None, 3.0, 1.0e-9, {})
    x = np.linspace(3.0e-3, 7.0e-3, len(problem.design))

    analytic = problem.compliance_gradient(x)
    numeric = central_difference_gradient(problem, x)
    assert np.all(analytic < 0.0)  # more thickness is always stiffer
    assert np.max(np.abs(analytic - numeric)) < 1.0e-5 * np.max(np.abs(numeric))


def test_density_sensitivities_match_finite_differences() -> None:
    model, _tip = solid_cantilever(nx=2, ny=1, nz=1)
    problem = _Problem(model, "density", None, None, 3.0, 1.0e-9, {})
    x = np.linspace(0.4, 0.95, len(problem.design))

    analytic = problem.compliance_gradient(x)
    numeric = central_difference_gradient(problem, x)
    assert np.all(analytic < 0.0)
    assert np.max(np.abs(analytic - numeric)) < 1.0e-5 * np.max(np.abs(numeric))


def test_shell_stiffness_is_exactly_cubic_in_the_thickness() -> None:
    """``K_e(t) = t A + t^3 B`` -- the identity the exact gradients rest on."""
    model, _tip = cantilever_plate(nx=2, ny=1)
    problem = _Problem(model, "thickness", None, None, 3.0, 1.0e-9, {})
    design = problem.design[0]
    assert design.exact

    for t in (1.0e-3, 4.5e-3, 1.2e-2):
        design.prop.t = t
        built = np.asarray(
            element_matrices(problem.work, design.eid, design.element).k, dtype=float
        )
        model_form = t * design.k_lin + t**3 * design.k_cub
        assert np.max(np.abs(built - model_form)) < 1.0e-9 * np.max(np.abs(built))


def test_topometry_reports_exact_sensitivities() -> None:
    model, _tip = cantilever_plate(nx=2, ny=2)
    result = topometry_optimize(model, max_iter=3)
    assert result.extras["exact_sensitivities"] is True


# ----------------------------------------------------------------------
# topometry -- constraints, bounds, subsets, methods
# ----------------------------------------------------------------------
def test_mean_thickness_constraint_is_met() -> None:
    model, _tip = cantilever_plate()
    target = 4.0e-3
    result = topometry_optimize(model, mean_thickness=target, max_iter=40)

    assert result.extras["constraint"] == "mean_thickness"
    assert result.mean_thickness == pytest.approx(target, rel=1.0e-9)
    assert result.volume < result.initial_volume  # 4 mm mean out of a 5 mm start


def test_volume_fraction_constraint_on_a_solid_block() -> None:
    model, _tip = solid_cantilever()
    result = topometry_optimize(model, design="density", volume_fraction=0.5, max_iter=40)

    total = float(np.sum(result.measures))
    assert result.design == "density"
    assert result.volume == pytest.approx(0.5 * total, rel=1.0e-9)
    assert result.initial_volume == pytest.approx(0.5 * total, rel=1.0e-9)
    # Same amount of material as the uniform start, better placed.
    assert result.compliance < result.initial_compliance
    assert np.all(result.x >= 1.0e-3) and np.all(result.x <= 1.0)


def test_explicit_bounds_are_respected() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, bounds=(3.0e-3, 8.0e-3), max_iter=30)

    assert np.all(result.x >= 3.0e-3 - 1.0e-15)
    assert np.all(result.x <= 8.0e-3 + 1.0e-15)
    assert np.min(result.x) == pytest.approx(3.0e-3, rel=1.0e-9)


def test_undesigned_elements_stay_frozen() -> None:
    model, _tip = cantilever_plate()
    designed = [1, 2, 3, 4, 5, 6]
    result = topometry_optimize(model, elements=designed, max_iter=20)

    assert result.element_ids == designed
    frozen = [
        result.model.properties[result.model.elements[eid].property_id].t
        for eid in model.elements
        if eid not in designed
    ]
    assert np.allclose(frozen, 5.0e-3, rtol=0.0, atol=0.0)
    assert result.compliance < result.initial_compliance


def test_slsqp_and_oc_agree_on_the_design_trend() -> None:
    model, _tip = cantilever_plate(nx=4, ny=2)
    oc = topometry_optimize(model, method="oc", max_iter=60)
    slsqp = topometry_optimize(model, method="slsqp", max_iter=60)

    assert slsqp.method == "SLSQP" and oc.method == "OC"
    for result in (oc, slsqp):
        assert result.compliance < result.initial_compliance
        assert result.volume <= result.volume_limit * (1.0 + 1.0e-6)
    # Both put material at the root rather than at the tip (elements are
    # numbered column by column, two per column here).
    for result in (oc, slsqp):
        assert float(np.mean(result.x[:2])) > float(np.mean(result.x[-2:]))


def test_filtering_smooths_the_design() -> None:
    model, _tip = cantilever_plate()
    plain = topometry_optimize(model, max_iter=40)
    filtered = topometry_optimize(
        model, max_iter=40, filter_radius=0.35, filter="density"
    )

    assert filtered.extras["filter"] == "density"
    assert plain.extras["filter"] == "none"
    assert float(np.ptp(filtered.x)) < float(np.ptp(plain.x))
    assert filtered.volume == pytest.approx(filtered.initial_volume, rel=1.0e-6)


def test_sensitivity_filter_runs_and_keeps_the_constraint() -> None:
    model, _tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=25, filter_radius=0.3)

    assert result.extras["filter"] == "sensitivity"
    assert result.volume == pytest.approx(result.initial_volume, rel=1.0e-9)
    assert result.compliance < result.initial_compliance


# ----------------------------------------------------------------------
# topometry -- duck typed models and frames
# ----------------------------------------------------------------------
def _oblique_rotation() -> np.ndarray:
    return np.linalg.qr(
        np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
    )[0]


def test_topometry_is_invariant_to_the_orientation_of_the_plate() -> None:
    """The same plate, rotated: same optimum, only the frames differ.

    The oblique mesh is the one whose shell nodes are solved in a local triad,
    so this exercises the analysis-frame to basic-frame conversion behind the
    element sensitivities as well as the duck-typed (dict) model path.
    """
    aligned: dict[str, Any] = shell_plate(nx=3, ny=3, side=0.6, thickness=4.0e-3,
                                          clamped_edge=True)
    R = _oblique_rotation()
    oblique: dict[str, Any] = shell_plate(nx=3, ny=3, side=0.6, thickness=4.0e-3,
                                          rotation=R, clamped_edge=True)
    tip = 4 * 4  # node (3, 1) of the 4x4 grid: on the free edge
    force = np.array([0.0, 0.0, -2.0e3])
    aligned_loads = {(tip, i): float(force[i]) for i in range(3)}
    oblique_loads = {(tip, i): float(v) for i, v in enumerate(R @ force)}

    a = topometry_optimize(aligned, loads=aligned_loads, max_iter=15)
    b = topometry_optimize(oblique, loads=oblique_loads, max_iter=15)

    assert a.compliance == pytest.approx(b.compliance, rel=1.0e-8)
    assert a.initial_compliance == pytest.approx(b.initial_compliance, rel=1.0e-8)
    assert np.allclose(a.x, b.x, rtol=1.0e-7, atol=0.0)


# ----------------------------------------------------------------------
# topometry -- result object and error handling
# ----------------------------------------------------------------------
def test_topometry_result_helpers() -> None:
    model, _tip = cantilever_plate(nx=2, ny=2)
    result = topometry_optimize(model, max_iter=5)

    assert np.asarray(result).shape == (len(result),)
    assert result[0] == result.x[0]
    assert result.to_dict()[result.element_ids[1]] == pytest.approx(result.x[1])
    assert result.fun == result.compliance
    assert set(result.to_dict()) == set(result.element_ids)
    assert result.mean_thickness == pytest.approx(
        result.volume / float(np.sum(result.measures))
    )
    assert np.allclose(result.thickness, result.x)
    with pytest.raises(AttributeError):
        _ = result.density
    assert "topometry_optimize(thickness" in result.summary()
    assert result.strain_energy is not None
    assert float(np.sum(result.strain_energy)) == pytest.approx(
        0.5 * result.compliance, rel=1.0e-9
    )


def test_topometry_rejects_contradictory_or_unknown_input() -> None:
    model, _tip = cantilever_plate(nx=2, ny=2)
    with pytest.raises(ValueError, match="volume constraint"):
        topometry_optimize(model, volume_fraction=0.5, max_volume=1.0e-3)
    with pytest.raises(ValueError, match="unknown design"):
        topometry_optimize(model, design="wobble")
    with pytest.raises(ValueError, match="unknown method"):
        topometry_optimize(model, method="genetic")
    with pytest.raises(ValueError, match="objective"):
        topometry_optimize(model, objective="mass")
    with pytest.raises(ValueError, match="mean_thickness"):
        topometry_optimize(model, design="density", mean_thickness=1.0e-3)
    with pytest.raises(KeyError):
        topometry_optimize(model, elements=[999])
    with pytest.raises(ValueError, match="bounds"):
        topometry_optimize(model, bounds=(8.0e-3, 2.0e-3))


def test_topometry_needs_a_loaded_model() -> None:
    model, _tip = cantilever_plate(nx=2, ny=2)
    model.loads.clear()
    with pytest.raises(ValueError, match="no load"):
        topometry_optimize(model)


# ----------------------------------------------------------------------
# static displacement response
# ----------------------------------------------------------------------
def test_static_displacement_response_matches_a_direct_solve() -> None:
    model, tip = cantilever_plate(nx=3, ny=2)
    dofs = [(tip, "uz"), (tip, "ry")]
    response = static_displacement_response(model, {"E": {"kind": "E", "relative": True}}, dofs)

    u = solve_static(model, {(tip, "uz"): TIP_LOAD})
    table = model.dof_map()
    expected = np.array([u[table[(tip, 2)]], u[table[(tip, 4)]]])
    assert np.allclose(response(np.array([1.0])), expected, rtol=1.0e-12, atol=0.0)
    # Halving E roughly doubles the deflection: only "roughly", because a
    # `kind="E"` parameter writes the modulus and leaves the shear modulus the
    # material record stores alongside it, and the MITC4 shear term uses G.
    halved = response(np.array([0.5]))
    assert np.allclose(halved, 2.0 * expected, rtol=1.0e-3)
    assert np.all(np.abs(halved) > np.abs(expected))


def test_static_displacement_response_accepts_every_dof_spelling() -> None:
    model, tip = cantilever_plate(nx=3, ny=2)
    table = model.dof_map()
    parameters = {"E": {"kind": "E", "relative": True}}
    p = np.array([1.0])

    by_label = static_displacement_response(model, parameters, [(tip, "uz")])(p)
    by_index = static_displacement_response(model, parameters, [(tip, 2)])(p)
    by_mapping = static_displacement_response(model, parameters, {tip: "z"})(p)
    by_global = static_displacement_response(model, parameters, [table[(tip, 2)]])(p)
    by_dofset = static_displacement_response(model, parameters, DOFSet.from_nodes(
        "sensor", [tip], dofs=(2,)
    ))(p)
    defaulted = static_displacement_response(model, parameters)(p)

    for other in (by_index, by_mapping, by_global, by_dofset, defaulted):
        assert np.allclose(other, by_label, rtol=0.0, atol=0.0)


def test_static_displacement_response_recovers_a_wrong_modulus() -> None:
    """Static updating: one measured deflection pins one modulus."""
    model, _tip = cantilever_plate(nx=4, ny=2)
    # Three deflections along the free edge (a dial gauge sweep), all non-zero.
    dofs = [(plate_node(4, j, 2), "uz") for j in range(3)]
    parameters = [{"type": "material", "id": 1, "name": "E", "lower": 0.5, "upper": 2.0}]

    truth = static_displacement_response(model, parameters, dofs)
    targets = truth(np.array([1.0]))

    # The analyst's model is 20 % too stiff.
    wrong = copy.deepcopy(model)
    wrong.materials[1].E *= 1.2
    response = static_displacement_response(wrong, parameters, dofs)
    result = update_model(
        wrong, parameters, targets, response=response, p0=[1.0], bounds=(0.5, 2.0)
    )

    assert result.converged
    assert abs(result.x[0] - 1.0 / 1.2) < 1.0e-9
    assert result.rms_error < 1.0e-9


def test_static_displacement_response_scaling_and_custom_solver() -> None:
    model, tip = cantilever_plate(nx=3, ny=2)
    parameters = {"E": {"kind": "E", "relative": True}}
    plain = static_displacement_response(model, parameters, [(tip, "uz")])
    scaled = static_displacement_response(model, parameters, [(tip, "uz")], scale=1.0e3)
    assert scaled(np.array([1.0])) == pytest.approx(1.0e3 * plain(np.array([1.0])))

    calls: list[Any] = []

    def bare_vector_solver(m: Any, loads: Any) -> np.ndarray:
        calls.append(loads)
        return np.asarray(solve_static(m, loads))

    custom = static_displacement_response(
        model, parameters, [(tip, "uz")], solver=bare_vector_solver
    )
    assert custom(np.array([1.0])) == pytest.approx(plain(np.array([1.0])))
    assert calls and calls[0] == {(tip, 2): TIP_LOAD}


def test_static_displacement_response_rejects_bad_requests() -> None:
    model, tip = cantilever_plate(nx=2, ny=2)
    parameters = {"E": {"kind": "E", "relative": True}}
    with pytest.raises(ValueError, match="component"):
        static_displacement_response(model, parameters, [(tip, "wiggle")])
    with pytest.raises(TypeError, match="measurement dof"):
        static_displacement_response(model, parameters, [(tip, 2, 3)])
    bare = copy.deepcopy(model)
    bare.loads.clear()
    with pytest.raises(ValueError, match="no measurement DOFs"):
        static_displacement_response(bare, parameters)


def test_static_and_modal_targets_can_share_one_residual() -> None:
    """A displacement response composes with any other response callable."""
    model, tip = cantilever_plate(nx=3, ny=2)
    parameters = {"E": {"kind": "E", "relative": True}}
    static = static_displacement_response(model, parameters, [(tip, "uz")])

    from femtools.updating.responses import modal_response_function

    modal = modal_response_function(model, parameters, n_modes=2)

    def combined(p: np.ndarray) -> np.ndarray:
        return np.concatenate([1.0e3 * static(p), modal(p)])

    targets = combined(np.array([1.0]))
    result = update_model(
        model, parameters, targets, response=combined, p0=[1.25], bounds=(0.5, 2.0)
    )
    assert abs(result.x[0] - 1.0) < 1.0e-6


@pytest.mark.golden
def test_ten_percent_beam_modulus_recovery_invariant() -> None:
    """The Round-4 updating golden, re-measured on this tree."""
    response, true_parameters, initial, targets, _model = make_updating_testcase(
        "beam", error=0.10, n_modes=4
    )
    result = update_model(
        response, ["E1", "E2"], targets, p0=initial, bounds=(0.5, 2.0), tol=1.0e-12
    )
    error = np.max(np.abs(result.x - true_parameters) / np.abs(true_parameters))
    assert error < 1.0e-7

"""Round 8 / O4: static updating and the stress residual.

Every case here is deterministic — there is no random component anywhere in
the static path — so the numbers quoted in ``.agent_workspace/reports/R8-O4.md``
come straight out of this file.

Two responses are exercised:

* :func:`femtools.updating.updater.update_from_static`, which wraps
  :func:`~femtools.updating.responses.static_displacement_response` and
  :func:`~femtools.updating.updater.update_model` into the static counterpart of
  a modal update (Friswell & Mottershead, *Finite Element Model Updating in
  Structural Dynamics*, ch. 3);
* :func:`~femtools.updating.responses.static_stress_response`, which feeds
  :func:`femtools.fea.recover.recover_stress` into the same loop, so a
  constant-stress patch pins the modulus down from measured stress alone.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from femtools.core.model import FEModel
from femtools.fea.eigen import solve_modes
from femtools.fea.materials import MaterialData, solid_D
from femtools.fea.recover import recover_stress
from femtools.fea.static import solve_static
from femtools.optimization import topometry_optimize
from femtools.updating import update_model
from femtools.updating.reference import make_updating_testcase
from femtools.updating.responses import (
    static_displacement_response,
    static_stress_response,
)
from femtools.updating.updater import update_from_static
from femtools.updating.uq import parameter_covariance

#: The classic updating error: the analyst's modulus is 10 % too high.
E_ERROR = 1.10

#: A relative Young's-modulus multiplier over one material.
E_PARAMETER = [{"type": "material", "id": 1, "name": "E", "lower": 0.2, "upper": 5.0}]


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def axial_bar(
    *,
    stations: tuple[float, ...] = (0.0, 0.31, 0.77, 1.05, 1.5),
    area: float = 3.0e-4,
    modulus: float = 2.1e11,
    force: float = 1.0e4,
    material_split: int | None = None,
) -> tuple[FEModel, int]:
    """Clamped-free ``BAR2`` chain pulled along its own axis.

    The cleanest updating specimen there is: the tip extension is exactly
    ``F L / (A E)``, so the response is exactly inversely proportional to the
    modulus multiplier and the recovered parameter is limited only by round-off.
    The stations are deliberately unequal so a solver that assumed a uniform
    mesh would show up.

    ``material_split`` gives the elements before that index a second material,
    which turns the bar into a two-parameter problem.
    """
    model = FEModel(name="axial-bar")
    model.add_material(id=1, type="isotropic", E=modulus, nu=0.3, rho=7800.0)
    model.add_property(id=1, type="bar", material_id=1, A=area)
    if material_split is not None:
        model.add_material(id=2, type="isotropic", E=modulus, nu=0.3, rho=7800.0)
        model.add_property(id=2, type="bar", material_id=2, A=area)
    for i, x in enumerate(stations):
        model.add_node(id=i + 1, xyz=(float(x), 0.0, 0.0))
    for i in range(len(stations) - 1):
        pid = 2 if material_split is not None and i >= material_split else 1
        model.add_element(id=i + 1, type="BAR2", property_id=pid, nodes=(i + 1, i + 2))
    model.add_spc(node_id=1, mask=(True,) * 6)
    # A chain of pin-jointed rods is a mechanism transverse to its own axis, so
    # every interior node is held there and only the axial motion is solved for.
    for i in range(1, len(stations)):
        model.add_spc(node_id=i + 1, mask=(False, True, True, True, True, True))
    tip = len(stations)
    if force:
        model.add_load(node_id=tip, force=(force, 0.0, 0.0))
    return model, tip


def beam_cantilever(
    n_elements: int = 8, *, length: float = 2.0, modulus: float = 2.1e11
) -> tuple[FEModel, int]:
    """``BEAM2`` cantilever with a transverse tip load."""
    model = FEModel(name="beam-cantilever")
    model.add_material(id=1, type="isotropic", E=modulus, nu=0.3, rho=7800.0)
    model.add_property(
        id=1, type="beam", material_id=1, A=8.0e-4, Iy=3.0e-8, Iz=6.0e-8, J=9.0e-8
    )
    for i in range(n_elements + 1):
        model.add_node(id=i + 1, xyz=(length * i / n_elements, 0.0, 0.0))
    for i in range(n_elements):
        model.add_element(id=i + 1, type="BEAM2", property_id=1, nodes=(i + 1, i + 2))
    model.add_spc(node_id=1, mask=(True,) * 6)
    tip = n_elements + 1
    model.add_load(node_id=tip, force=(0.0, 0.0, -1.0e3))
    return model, tip


#: Unsymmetric displacement gradient of the solid patch test: a recovery that
#: dropped the rotational part of the gradient would not cancel out.
PATCH_GRADIENT = np.array(
    [[1.0e-4, 2.0e-5, 3.0e-5], [5.0e-6, -2.0e-4, 1.0e-5], [1.0e-5, 3.0e-5, 1.5e-4]]
)
PATCH_E = 1.0e7
PATCH_NU = 0.3
#: Unequal grid lines, one per axis, of the 3x3x3 patch mesh.
PATCH_STATIONS = ((0.0, 0.55, 1.2), (0.0, 0.4, 0.9), (0.0, 0.62, 1.1))


def hex_patch() -> tuple[FEModel, dict[tuple[int, int], float], np.ndarray, int]:
    """Eight ``HEX8`` bricks around one free, off-centre node.

    The classical constant-strain patch test: the 26 outer nodes are driven with
    the exact linear field ``u = PATCH_GRADIENT @ x`` and the enclosed node is
    left free.  Every element must then report the *same* stress, and that
    stress is exactly ``D(E) eps`` — which is what makes the modulus recoverable
    from a stress measurement alone.

    Returns the model, the enforced boundary displacements, the analytic Voigt
    stress and the id of the free node.
    """
    coords: dict[int, np.ndarray] = {}
    ids: dict[tuple[int, int, int], int] = {}
    counter = 1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                point = np.array(
                    [PATCH_STATIONS[0][i], PATCH_STATIONS[1][j], PATCH_STATIONS[2][k]]
                )
                if (i, j, k) == (1, 1, 1):
                    point = point + np.array([0.09, -0.07, 0.08])
                ids[(i, j, k)] = counter
                coords[counter] = point
                counter += 1

    model = FEModel(name="hex-patch")
    model.add_material(id=1, type="isotropic", E=PATCH_E, nu=PATCH_NU, rho=1.0)
    model.add_property(id=1, type="solid", material_id=1)
    for nid, point in coords.items():
        model.add_node(id=nid, xyz=tuple(point))
    eid = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                model.add_element(
                    id=eid,
                    type="HEX8",
                    property_id=1,
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
                )
                eid += 1

    centre = ids[(1, 1, 1)]
    boundary = [nid for nid in coords if nid != centre]
    for nid in boundary:
        model.add_spc(node_id=nid, mask=(True, True, True, False, False, False))
    enforced = {
        (nid, comp): float((PATCH_GRADIENT @ coords[nid])[comp])
        for nid in boundary
        for comp in range(3)
    }

    eps = 0.5 * (PATCH_GRADIENT + PATCH_GRADIENT.T)
    voigt = np.array(
        [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]]
    )
    exact = solid_D(MaterialData(E=PATCH_E, nu=PATCH_NU)) @ voigt
    return model, enforced, exact, centre


def cantilever_plate(nx: int = 4, ny: int = 2) -> tuple[FEModel, int]:
    """Clamped-free ``QUAD4`` plate with a transverse tip load — the topometry case."""
    lx, ly = 0.9, 0.45
    model = FEModel(name="cantilever-plate")
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.33, rho=2700.0)
    model.add_property(id=1, type="shell", material_id=1, t=5.0e-3)
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
    model.add_load(node_id=tip, force=(0.0, 0.0, -1.0e3))
    return model, tip


def measure(model: FEModel, dofs: list[tuple[int, Any]]) -> np.ndarray:
    """The truth model's deflections at ``dofs`` — the "test data"."""
    u = solve_static(model, None)
    table = model.dof_map()
    return np.array([float(u[table[(node, comp)]]) for node, comp in dofs])


def with_wrong_modulus(model: FEModel, factor: float = E_ERROR) -> FEModel:
    """A copy of ``model`` whose every material is ``factor`` times too stiff."""
    wrong = copy.deepcopy(model)
    for material in wrong.materials.values():
        material.E *= factor
    return wrong


# ----------------------------------------------------------------------
# update_from_static -- the headline case
# ----------------------------------------------------------------------
def test_update_from_static_recovers_a_ten_percent_modulus_error_on_an_axial_bar() -> None:
    """One measured tip extension pins the modulus down to round-off."""
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])[0]
    wrong = with_wrong_modulus(truth)

    result = update_from_static(wrong, {(tip, "ux"): target})

    assert result.converged
    # The parameter is a multiplier on the wrong (10 % too stiff) modulus, so
    # the answer it is chasing is 1/1.1.
    assert abs(result["E"] * E_ERROR - 1.0) < 1.0e-11
    assert result.rms_error < 1.0e-12
    assert result.improvement > 1.0 - 1.0e-11


def test_update_from_static_returns_the_modulus_of_the_truth_model() -> None:
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])[0]
    wrong = with_wrong_modulus(truth)

    result = update_from_static(wrong, {(tip, "ux"): target})

    recovered = result.model.materials[1].E
    assert abs(recovered - truth.materials[1].E) / truth.materials[1].E < 1.0e-11
    # The input model is never touched.
    assert wrong.materials[1].E == pytest.approx(E_ERROR * truth.materials[1].E)


def test_update_from_static_reaches_the_same_order_as_the_modal_path() -> None:
    """The static residual is no worse than the modal one on the same bar."""
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])[0]
    frequencies = np.asarray(solve_modes(truth, n_modes=3).freq_hz, dtype=float)

    static = update_from_static(with_wrong_modulus(truth), {(tip, "ux"): target})
    modal = update_model(
        with_wrong_modulus(truth),
        E_PARAMETER,
        frequencies,
        p0=[1.0],
        bounds=(0.2, 5.0),
        tol=1.0e-12,
    )

    static_error = abs(static.x[0] * E_ERROR - 1.0)
    modal_error = abs(modal.x[0] * E_ERROR - 1.0)
    assert static_error < 1.0e-9 and modal_error < 1.0e-9
    assert static_error <= 1.0e3 * max(modal_error, 1.0e-16)


def test_update_from_static_recovers_the_modulus_of_a_beam_cantilever() -> None:
    """A dial-gauge sweep along a cantilever: three deflections, one modulus."""
    truth, tip = beam_cantilever()
    dofs = [(tip, 2), (tip, 4), (5, 2)]
    targets = measure(truth, dofs)
    wrong = with_wrong_modulus(truth)

    result = update_from_static(
        wrong, targets, E_PARAMETER, [(tip, "uz"), (tip, "ry"), (5, "uz")]
    )

    assert abs(result.x[0] * E_ERROR - 1.0) < 1.0e-9
    assert result.rms_error < 1.0e-10


def test_update_from_static_separates_two_materials() -> None:
    """Two moduli, four measured stations: both parameters come back."""
    truth, tip = axial_bar(material_split=2)
    truth.materials[2].E *= 0.7  # the outboard half really is softer
    dofs = [(nid, 0) for nid in (2, 3, 4, 5)]
    targets = measure(truth, dofs)

    wrong = copy.deepcopy(truth)
    wrong.materials[1].E *= E_ERROR
    wrong.materials[2].E *= 0.85
    parameters = [
        {"type": "material", "id": 1, "name": "E1", "kind": "E", "lower": 0.2, "upper": 5.0},
        {"type": "material", "id": 2, "name": "E2", "kind": "E", "lower": 0.2, "upper": 5.0},
    ]

    result = update_from_static(wrong, targets, parameters, [(n, "ux") for n, _ in dofs])

    assert abs(result["E1"] * E_ERROR - 1.0) < 1.0e-9
    assert abs(result["E2"] / (1.0 / 0.85) - 1.0) < 1.0e-9
    assert result.model.materials[1].E == pytest.approx(truth.materials[1].E, rel=1.0e-9)
    assert result.model.materials[2].E == pytest.approx(truth.materials[2].E, rel=1.0e-9)


# ----------------------------------------------------------------------
# update_from_static -- the `measured` argument
# ----------------------------------------------------------------------
def test_update_from_static_accepts_every_measured_spelling() -> None:
    truth, tip = beam_cantilever(n_elements=4)
    dofs = [(tip, "uz"), (3, "uz")]
    targets = measure(truth, [(tip, 2), (3, 2)])
    wrong = with_wrong_modulus(truth)

    by_pairs = update_from_static(wrong, dict(zip(dofs, targets, strict=True)))
    by_nested = update_from_static(
        wrong, {tip: {"uz": targets[0]}, 3: {"uz": targets[1]}}
    )
    by_record = update_from_static(wrong, {"u": targets, "dofs": dofs})
    by_vector = update_from_static(wrong, targets, dofs=dofs)

    for other in (by_nested, by_record, by_vector):
        assert other.x[0] == pytest.approx(by_pairs.x[0], rel=0.0, abs=0.0)
    assert abs(by_pairs.x[0] * E_ERROR - 1.0) < 1.0e-9


def test_update_from_static_defaults_to_the_loaded_dofs() -> None:
    """No ``dofs``: the drive point of the test is the measurement point."""
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])
    wrong = with_wrong_modulus(truth)

    defaulted = update_from_static(wrong, target)
    explicit = update_from_static(wrong, target, dofs=[(tip, "ux")])

    assert defaulted.x[0] == pytest.approx(explicit.x[0], rel=0.0, abs=0.0)


def test_update_from_static_defaults_to_one_relative_modulus() -> None:
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])[0]

    result = update_from_static(with_wrong_modulus(truth), {(tip, "ux"): target})

    assert result.parameter_names == ["E"]
    assert result.initial == pytest.approx(np.array([1.0]))
    assert result.sensitivity is not None
    assert result.sensitivity.shape == (1, 1)


def test_update_from_static_rejects_contradictory_input() -> None:
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])
    wrong = with_wrong_modulus(truth)

    with pytest.raises(ValueError, match="empty"):
        update_from_static(wrong, [])
    with pytest.raises(ValueError, match="measurement DOFs"):
        update_from_static(wrong, target, dofs=[(tip, "ux"), (2, "ux")])
    with pytest.raises(ValueError, match="contradictory"):
        update_from_static(wrong, {(tip, "ux"): target[0]}, dofs=[(tip, "ux")])
    with pytest.raises(ValueError, match="either in `measured`"):
        update_from_static(wrong, {"u": target, "dofs": [(tip, "ux")]}, dofs=[(tip, "ux")])
    with pytest.raises(ValueError, match="unexpected keys"):
        update_from_static(wrong, {"u": target, "wobble": 1.0})
    with pytest.raises(ValueError, match="cannot interpret"):
        update_from_static(wrong, {tip: 1.0e-3})


def test_update_from_static_forwards_solver_options() -> None:
    """``update_model``'s own knobs still reach the loop through ``**kwargs``."""
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])
    wrong = with_wrong_modulus(truth)
    seen: list[int] = []

    result = update_from_static(
        wrong,
        target,
        max_iter=3,
        weights="unit",
        callback=lambda it, x, cost: seen.append(it),
    )

    assert seen == list(range(1, result.n_iter + 1))
    assert result.n_iter <= 3
    assert result.history[0]["weights"] == "unit"


def test_update_from_static_accepts_a_scale_and_a_custom_solver() -> None:
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)]) * 1.0e3  # millimetres, as a gauge reads
    wrong = with_wrong_modulus(truth)
    calls: list[Any] = []

    def bare_vector_solver(model: Any, loads: Any) -> np.ndarray:
        calls.append(loads)
        return np.asarray(solve_static(model, loads))

    scaled = update_from_static(wrong, target, scale=1.0e3)
    custom = update_from_static(wrong, target, scale=1.0e3, solver=bare_vector_solver)

    assert abs(scaled.x[0] * E_ERROR - 1.0) < 1.0e-9
    assert custom.x[0] == pytest.approx(scaled.x[0], rel=1.0e-12)
    assert calls and calls[0] == {(tip, 0): 1.0e4}


def test_update_from_static_result_feeds_parameter_covariance() -> None:
    truth, tip = beam_cantilever()
    dofs = [(tip, 2), (tip, 4), (5, 2)]
    targets = measure(truth, dofs)
    wrong = with_wrong_modulus(truth)

    result = update_from_static(
        wrong, targets, E_PARAMETER, [(tip, "uz"), (tip, "ry"), (5, "uz")]
    )
    uq = parameter_covariance(result, residual_cov=(0.01 * targets) ** 2)

    assert uq.parameter_names == ["E"]
    assert np.all(np.isfinite(uq.std)) and uq.std[0] > 0.0
    lower, upper = uq.interval(0.95)
    assert lower[0] < result.x[0] < upper[0]


# ----------------------------------------------------------------------
# static_stress_response -- the constant-stress patch
# ----------------------------------------------------------------------
def test_stress_response_is_constant_over_the_hex8_patch() -> None:
    """The patch test the stress residual rests on."""
    model, enforced, exact, centre = hex_patch()
    result = solve_static(model, None, enforced=enforced, full_result=True)

    table = model.dof_map()
    got = np.array([result.u[table[(centre, comp)]] for comp in range(3)])
    assert np.allclose(got, PATCH_GRADIENT @ np.asarray(model.nodes[centre].xyz), atol=1.0e-18)

    stress = recover_stress(model, result)
    assert len(stress) == 8
    assert np.max(np.abs(stress.stress - exact)) < 1.0e-12 * np.max(np.abs(exact))

    response = static_stress_response(
        model, E_PARAMETER, solver_kwargs={"enforced": enforced}
    )
    von_mises = response(np.array([1.0]))
    assert von_mises.shape == (8,)
    assert float(np.ptp(von_mises)) < 1.0e-12 * float(np.mean(von_mises))


def test_stress_residual_recovers_the_modulus_on_the_hex8_patch() -> None:
    """A constant-stress patch measured, not deflected: the modulus still comes back."""
    truth, enforced, _exact, _centre = hex_patch()
    measured = static_stress_response(
        truth, E_PARAMETER, solver_kwargs={"enforced": enforced}
    )(np.array([1.0]))

    wrong = with_wrong_modulus(truth)
    response = static_stress_response(
        wrong, E_PARAMETER, solver_kwargs={"enforced": enforced}
    )
    result = update_from_static(wrong, measured, E_PARAMETER, response=response)

    assert abs(result.x[0] * E_ERROR - 1.0) < 1.0e-11
    assert result.model.materials[1].E == pytest.approx(truth.materials[1].E, rel=1.0e-11)


def test_stress_residual_recovers_the_modulus_on_a_stretched_bar_chain() -> None:
    """A chain of rods stretched by a prescribed end displacement.

    The axial force is constant through the chain, so the stress is constant
    too — a one-dimensional constant-stress patch, and ``sigma = E delta / L``
    is exactly linear in the modulus.
    """
    truth, tip = axial_bar(force=0.0)
    enforced = {(tip, 0): 1.0e-4}
    response_kwargs = {"solver_kwargs": {"enforced": enforced}, "component": "xx"}
    measured = static_stress_response(truth, E_PARAMETER, **response_kwargs)(np.array([1.0]))

    assert float(np.ptp(measured)) < 1.0e-12 * float(np.mean(measured))
    assert measured[0] == pytest.approx(2.1e11 * 1.0e-4 / 1.5, rel=1.0e-12)

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **response_kwargs),
    )
    assert abs(result.x[0] * E_ERROR - 1.0) < 1.0e-12


def test_stress_response_selects_elements_components_and_frames() -> None:
    model, enforced, exact, _centre = hex_patch()
    kwargs = {"solver_kwargs": {"enforced": enforced}}
    p = np.array([1.0])

    subset = static_stress_response(model, E_PARAMETER, [3, 1], **kwargs)(p)
    everything = static_stress_response(model, E_PARAMETER, **kwargs)(p)
    assert subset.shape == (2,)
    assert np.allclose(subset, everything[[2, 0]], rtol=0.0, atol=0.0)

    for name, column in (("xx", 0), ("yy", 1), ("xy", 3)):
        component = static_stress_response(model, E_PARAMETER, [1], component=name, **kwargs)
        assert component(p)[0] == pytest.approx(exact[column], rel=1.0e-12)

    # Solids are recovered in the basic frame already, so the two agree.
    basic = static_stress_response(
        model, E_PARAMETER, [1], component="xx", frame="basic", **kwargs
    )
    assert basic(p)[0] == pytest.approx(exact[0], rel=1.0e-12)

    strain = static_stress_response(
        model, E_PARAMETER, [1], component="xx", quantity="strain", **kwargs
    )
    assert strain(p)[0] == pytest.approx(PATCH_GRADIENT[0, 0], rel=1.0e-12)

    scaled = static_stress_response(
        model, E_PARAMETER, [1], component="xx", scale=1.0e-6, **kwargs
    )
    assert scaled(p)[0] == pytest.approx(1.0e-6 * exact[0], rel=1.0e-12)


def test_stress_response_rejects_bad_requests() -> None:
    model, enforced, _exact, _centre = hex_patch()
    kwargs: dict[str, Any] = {"solver_kwargs": {"enforced": enforced}}

    with pytest.raises(ValueError, match="unknown stress component"):
        static_stress_response(model, E_PARAMETER, component="wobble", **kwargs)
    with pytest.raises(ValueError, match="quantity"):
        static_stress_response(model, E_PARAMETER, quantity="temperature", **kwargs)
    with pytest.raises(ValueError, match="frame"):
        static_stress_response(model, E_PARAMETER, component="xx", frame="modal", **kwargs)
    with pytest.raises(ValueError, match="von Mises is a stress measure"):
        static_stress_response(model, E_PARAMETER, quantity="strain", **kwargs)
    with pytest.raises(KeyError):
        static_stress_response(model, E_PARAMETER, [999], **kwargs)(np.array([1.0]))


# ----------------------------------------------------------------------
# invariants Round 8 must not break
# ----------------------------------------------------------------------
@pytest.mark.golden
def test_ten_percent_modal_modulus_recovery_invariant() -> None:
    """The Round-4 modal updating golden, re-measured on this tree."""
    response, true_parameters, initial, targets, _model = make_updating_testcase(
        "beam", error=0.10, n_modes=4
    )
    result = update_model(
        response, ["E1", "E2"], targets, p0=initial, bounds=(0.5, 2.0), tol=1.0e-12
    )
    error = np.max(np.abs(result.x - true_parameters) / np.abs(true_parameters))
    assert error < 1.0e-7


def test_static_displacement_response_still_matches_a_direct_solve() -> None:
    model, tip = beam_cantilever(n_elements=4)
    response = static_displacement_response(
        model, {"E": {"kind": "E", "relative": True}}, [(tip, "uz")]
    )
    assert np.allclose(
        response(np.array([1.0])), measure(model, [(tip, 2)]), rtol=1.0e-12, atol=0.0
    )


def test_topometry_optimize_still_reduces_the_compliance() -> None:
    model, _tip = cantilever_plate(nx=4, ny=2)
    result = topometry_optimize(model, max_iter=30)
    assert result.compliance < result.initial_compliance
    assert result.volume == pytest.approx(result.initial_volume, rel=1.0e-9)

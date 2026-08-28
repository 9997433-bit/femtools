"""Round 9 / O4: the frozen ``static_stress_response`` contract.

Round 8 landed the kernel; Round 9 pins it.  Nothing here is a new estimator —
:func:`femtools.updating.responses.static_stress_response` solves the linear
static problem, pushes the answer through :func:`femtools.fea.recover.recover_stress`
and hands one number per element to the same damped Gauss--Newton loop as every
other residual (Friswell & Mottershead, *Finite Element Model Updating in
Structural Dynamics*, ch. 3).

The one thing a stress residual demands of its load case is that stress actually
depend on the parameter.  Under **dead load** a statically determinate structure
carries ``sigma = F / A`` whatever its modulus, so the residual is blind to
``E``; driven by **prescribed displacement** the same structure carries
``sigma = E delta / L``, exactly linear in it.  Measured *strain* is the mirror
image, informative under dead load and blind under enforced displacement.  Both
halves of that statement are asserted below, because they are the reason every
recovery case in this file is displacement-driven.

Every case is deterministic — there is no random component anywhere in the
static path — so the numbers quoted in ``.agent_workspace/reports/R9-O4.md``
come straight out of this file.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import pytest

import femtools.updating as updating_pkg
from femtools.core.model import FEModel
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

#: The prescribed tip extension that drives the bar chain.
TIP_STRETCH = 1.0e-4


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def axial_bar(
    *,
    stations: tuple[float, ...] = (0.0, 0.31, 0.77, 1.05, 1.5),
    area: float = 3.0e-4,
    modulus: float = 2.1e11,
    force: float = 1.0e4,
) -> tuple[FEModel, int]:
    """Clamped-free ``BAR2`` chain along its own axis, on unequal stations.

    With ``force`` it is the Round-8 dead-load specimen (tip extension
    ``F L / (A E)``); with ``force=0`` and a prescribed tip displacement it is a
    one-dimensional constant-stress patch (``sigma = E delta / L``).
    """
    model = FEModel(name="axial-bar")
    model.add_material(id=1, type="isotropic", E=modulus, nu=0.3, rho=7800.0)
    model.add_property(id=1, type="bar", material_id=1, A=area)
    for i, x in enumerate(stations):
        model.add_node(id=i + 1, xyz=(float(x), 0.0, 0.0))
    for i in range(len(stations) - 1):
        model.add_element(id=i + 1, type="BAR2", property_id=1, nodes=(i + 1, i + 2))
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


def parameter_error(result: Any) -> float:
    """``|p * 1.1 - 1|``: how far the recovered multiplier is from ``1/1.1``."""
    return abs(float(result.x[0]) * E_ERROR - 1.0)


# ----------------------------------------------------------------------
# the frozen import and signature
# ----------------------------------------------------------------------
def test_static_stress_response_is_importable_under_its_frozen_name() -> None:
    """``from femtools.updating.responses import static_stress_response``."""
    from femtools.updating.responses import static_stress_response as frozen

    assert frozen is static_stress_response
    assert updating_pkg.static_stress_response is static_stress_response
    assert "static_stress_response" in updating_pkg.__all__
    from femtools.updating import responses

    assert "static_stress_response" in responses.__all__
    assert callable(static_stress_response)


def test_static_stress_response_signature_is_frozen() -> None:
    """The argument list Round 9 promises, in the order it promises it."""
    signature = inspect.signature(static_stress_response)
    assert list(signature.parameters) == [
        "model",
        "parameters",
        "elements",
        "component",
        "quantity",
        "frame",
        "layer",
        "loads",
        "enforced",
        "solver",
        "solver_kwargs",
        "scale",
    ]
    defaults = {
        name: p.default
        for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "elements": None,
        "component": "von_mises",
        "quantity": "stress",
        "frame": "element",
        "layer": "mid",
        "loads": None,
        "enforced": None,
        "solver": None,
        "solver_kwargs": None,
        "scale": None,
    }
    # Everything past `elements` is keyword-only, so the order above is the
    # documented one and not a positional contract callers can trip over.
    kinds = {name: p.kind for name, p in signature.parameters.items()}
    assert kinds["elements"] is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        kinds[name] is inspect.Parameter.KEYWORD_ONLY
        for name in ("component", "quantity", "frame", "layer", "loads", "enforced",
                     "solver", "solver_kwargs", "scale")
    )


# ----------------------------------------------------------------------
# the headline gate: displacement-driven 10 % E recovery from stress
# ----------------------------------------------------------------------
def test_stress_residual_recovers_the_modulus_on_the_stretched_bar_chain() -> None:
    """BAR2 constant-stress patch: one prescribed extension pins the modulus."""
    truth, tip = axial_bar(force=0.0)
    kwargs: dict[str, Any] = {"enforced": {(tip, 0): TIP_STRETCH}, "component": "xx"}
    measured = static_stress_response(truth, E_PARAMETER, **kwargs)(np.array([1.0]))

    # A constant-stress state: sigma = E delta / L through the whole chain.
    assert measured.shape == (4,)
    assert float(np.ptp(measured)) < 1.0e-12 * float(np.mean(measured))
    assert measured[0] == pytest.approx(2.1e11 * TIP_STRETCH / 1.5, rel=1.0e-12)

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **kwargs),
    )

    assert result.converged
    assert parameter_error(result) < 1.0e-12
    assert result.model.materials[1].E == pytest.approx(truth.materials[1].E, rel=1.0e-12)
    # The input model is never touched.
    assert wrong.materials[1].E == pytest.approx(E_ERROR * truth.materials[1].E)


def test_stress_residual_recovers_the_modulus_on_the_hex8_patch() -> None:
    """HEX8 constant-stress patch: 8 measured von Mises values, one modulus."""
    truth, enforced, exact, centre = hex_patch()
    kwargs: dict[str, Any] = {"enforced": enforced}
    measured = static_stress_response(truth, E_PARAMETER, **kwargs)(np.array([1.0]))

    # The patch test itself: free node on the linear field, one stress everywhere.
    result = solve_static(truth, None, enforced=enforced, full_result=True)
    table = truth.dof_map()
    got = np.array([result.u[table[(centre, comp)]] for comp in range(3)])
    assert np.allclose(
        got, PATCH_GRADIENT @ np.asarray(truth.nodes[centre].xyz), atol=1.0e-18
    )
    recovered = recover_stress(truth, result)
    assert len(recovered) == 8
    assert np.max(np.abs(recovered.stress - exact)) < 1.0e-12 * np.max(np.abs(exact))
    assert measured.shape == (8,)
    assert float(np.ptp(measured)) < 1.0e-12 * float(np.mean(measured))

    wrong = with_wrong_modulus(truth)
    update = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **kwargs),
    )

    assert update.converged
    assert parameter_error(update) < 1.0e-11
    assert update.model.materials[1].E == pytest.approx(truth.materials[1].E, rel=1.0e-11)


def test_stress_residual_recovers_the_modulus_from_one_voigt_component() -> None:
    """The same patch read as a single strain-gauge direction rather than von Mises."""
    truth, enforced, exact, _centre = hex_patch()
    kwargs: dict[str, Any] = {"enforced": enforced, "component": "xx"}
    measured = static_stress_response(truth, E_PARAMETER, [1, 4, 8], **kwargs)(
        np.array([1.0])
    )
    assert np.allclose(measured, exact[0], rtol=1.0e-12, atol=0.0)

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, [1, 4, 8], **kwargs),
    )
    assert parameter_error(result) < 1.0e-11


def test_displacement_driven_stress_is_exactly_linear_in_the_modulus() -> None:
    """``sigma(p) = p sigma(1)`` — why the recovery above is round-off limited."""
    model, enforced, _exact, _centre = hex_patch()
    response = static_stress_response(model, E_PARAMETER, enforced=enforced)
    base = response(np.array([1.0]))

    # Scaling by a power of two is exact in binary floating point, so there the
    # linearity has to show up bit for bit; elsewhere it is round-off limited.
    for factor in (0.5, 2.0, 4.0):
        assert np.array_equal(response(np.array([factor])), factor * base)
    for factor in (0.3, 3.7):
        assert np.allclose(
            response(np.array([factor])), factor * base, rtol=1.0e-15, atol=0.0
        )


# ----------------------------------------------------------------------
# why the drive has to be a displacement
# ----------------------------------------------------------------------
def test_dead_load_stress_is_blind_to_the_modulus() -> None:
    """``sigma = F / A``: the residual carries no information about ``E``."""
    model, _tip = axial_bar()
    response = static_stress_response(model, E_PARAMETER, component="xx")
    base = response(np.array([1.0]))

    for factor in (0.5, 2.0, 3.7):
        assert np.allclose(response(np.array([factor])), base, rtol=1.0e-14, atol=0.0)
    assert base[0] == pytest.approx(1.0e4 / 3.0e-4, rel=1.0e-12)


def test_a_dead_load_stress_residual_does_not_recover_the_modulus() -> None:
    """The negative control the displacement drive exists to avoid.

    Nothing is wrong with the loop — the Jacobian really is zero — so it stops
    at the first iteration with the parameter still 10 % out.  This is asserted
    rather than merely noted, because a future change that made it *look* like
    it worked would mean the response had stopped being a stress.
    """
    truth, _tip = axial_bar()
    kwargs: dict[str, Any] = {"component": "xx"}
    measured = static_stress_response(truth, E_PARAMETER, **kwargs)(np.array([1.0]))

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **kwargs),
    )

    assert parameter_error(result) > 0.05  # still ~10 % out: nothing was learned
    assert np.max(np.abs(np.asarray(result.sensitivity))) < 1.0e-6 * float(measured[0])


def test_a_dead_load_strain_residual_does_recover_the_modulus() -> None:
    """The mirror image: ``eps = F / (A E)`` is informative under dead load."""
    truth, _tip = axial_bar()
    kwargs: dict[str, Any] = {"component": "xx", "quantity": "strain"}
    measured = static_stress_response(truth, E_PARAMETER, **kwargs)(np.array([1.0]))
    assert measured[0] == pytest.approx(1.0e4 / (3.0e-4 * 2.1e11), rel=1.0e-12)

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **kwargs),
    )
    assert parameter_error(result) < 1.0e-9


def test_enforced_displacement_strain_is_blind_to_the_modulus() -> None:
    """And the other half of the mirror: ``eps = delta / L`` carries no ``E``."""
    model, tip = axial_bar(force=0.0)
    response = static_stress_response(
        model,
        E_PARAMETER,
        component="xx",
        quantity="strain",
        enforced={(tip, 0): TIP_STRETCH},
    )
    base = response(np.array([1.0]))

    assert base[0] == pytest.approx(TIP_STRETCH / 1.5, rel=1.0e-12)
    for factor in (0.5, 2.0, 3.7):
        assert np.allclose(response(np.array([factor])), base, rtol=1.0e-14, atol=0.0)


# ----------------------------------------------------------------------
# the `enforced=` spelling
# ----------------------------------------------------------------------
def test_enforced_matches_the_solver_kwargs_spelling_bit_for_bit() -> None:
    model, enforced, _exact, _centre = hex_patch()
    p = np.array([1.0])

    spelled_out = static_stress_response(model, E_PARAMETER, enforced=enforced)(p)
    through_kwargs = static_stress_response(
        model, E_PARAMETER, solver_kwargs={"enforced": enforced}
    )(p)

    assert np.array_equal(spelled_out, through_kwargs)


def test_enforced_is_copied_and_the_caller_mapping_is_not_read_again() -> None:
    """A response built once keeps the load case it was built with."""
    model, tip = axial_bar(force=0.0)
    enforced = {(tip, 0): TIP_STRETCH}
    response = static_stress_response(model, E_PARAMETER, component="xx", enforced=enforced)
    first = response(np.array([1.0]))

    enforced[(tip, 0)] = 5.0 * TIP_STRETCH
    assert np.array_equal(response(np.array([1.0])), first)


def test_enforced_rejects_a_contradictory_or_bypassed_request() -> None:
    model, tip = axial_bar(force=0.0)
    enforced = {(tip, 0): TIP_STRETCH}

    with pytest.raises(ValueError, match="not both"):
        static_stress_response(
            model, E_PARAMETER, enforced=enforced, solver_kwargs={"enforced": enforced}
        )
    with pytest.raises(ValueError, match="custom `solver`"):
        static_stress_response(
            model, E_PARAMETER, enforced=enforced, solver=lambda m, loads: None
        )


# ----------------------------------------------------------------------
# selection, framing and determinism
# ----------------------------------------------------------------------
def test_stress_response_selects_elements_components_and_frames() -> None:
    model, enforced, exact, _centre = hex_patch()
    kwargs: dict[str, Any] = {"enforced": enforced}
    p = np.array([1.0])

    subset = static_stress_response(model, E_PARAMETER, [3, 1], **kwargs)(p)
    everything = static_stress_response(model, E_PARAMETER, **kwargs)(p)
    assert subset.shape == (2,)
    assert np.array_equal(subset, everything[[2, 0]])

    for name, column in (("xx", 0), ("yy", 1), ("zz", 2), ("xy", 3), ("yz", 4), ("zx", 5)):
        component = static_stress_response(model, E_PARAMETER, [1], component=name, **kwargs)
        assert component(p)[0] == pytest.approx(exact[column], rel=1.0e-11)

    # Solids are recovered in the basic frame already, so the two agree exactly.
    basic = static_stress_response(
        model, E_PARAMETER, [1], component="xx", frame="basic", **kwargs
    )
    element = static_stress_response(model, E_PARAMETER, [1], component="xx", **kwargs)
    assert np.array_equal(basic(p), element(p))

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
    kwargs: dict[str, Any] = {"enforced": enforced}

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


def test_stress_response_is_deterministic_and_leaves_the_model_alone() -> None:
    model, enforced, _exact, _centre = hex_patch()
    before = copy.deepcopy(model)
    response = static_stress_response(model, E_PARAMETER, enforced=enforced)

    first = response(np.array([1.0]))
    response(np.array([0.3]))  # a different point in between
    assert np.array_equal(response(np.array([1.0])), first)
    assert model.materials[1].E == before.materials[1].E
    assert len(model.elements) == len(before.elements)


# ----------------------------------------------------------------------
# Round-8 invariants Round 9 must not move
# ----------------------------------------------------------------------
def test_update_from_static_displacement_path_is_unchanged() -> None:
    """The Round-8 headline: one tip extension, parameter error 4.4e-16."""
    truth, tip = axial_bar()
    target = measure(truth, [(tip, 0)])[0]
    wrong = with_wrong_modulus(truth)

    result = update_from_static(wrong, {(tip, "ux"): target})

    assert result.converged
    assert parameter_error(result) < 1.0e-14
    assert result.rms_error < 1.0e-12
    assert result.improvement > 1.0 - 1.0e-11
    assert result.model.materials[1].E == pytest.approx(truth.materials[1].E, rel=1.0e-11)
    assert wrong.materials[1].E == pytest.approx(E_ERROR * truth.materials[1].E)


def test_update_from_static_still_recovers_a_beam_cantilever_sweep() -> None:
    truth, tip = beam_cantilever()
    dofs = [(tip, 2), (tip, 4), (5, 2)]
    targets = measure(truth, dofs)

    result = update_from_static(
        with_wrong_modulus(truth), targets, E_PARAMETER, [(tip, "uz"), (tip, "ry"), (5, "uz")]
    )

    assert parameter_error(result) < 1.0e-9
    assert result.rms_error < 1.0e-10


def test_static_displacement_response_still_matches_a_direct_solve() -> None:
    model, tip = beam_cantilever(n_elements=4)
    response = static_displacement_response(
        model, {"E": {"kind": "E", "relative": True}}, [(tip, "uz")]
    )
    assert np.allclose(
        response(np.array([1.0])), measure(model, [(tip, 2)]), rtol=1.0e-12, atol=0.0
    )


def test_parameter_covariance_still_reads_a_static_update() -> None:
    truth, tip = beam_cantilever()
    dofs = [(tip, 2), (tip, 4), (5, 2)]
    targets = measure(truth, dofs)

    result = update_from_static(
        with_wrong_modulus(truth),
        targets,
        E_PARAMETER,
        [(tip, "uz"), (tip, "ry"), (5, "uz")],
    )
    uq = parameter_covariance(result, residual_cov=(0.01 * targets) ** 2)

    assert uq.parameter_names == ["E"]
    assert np.all(np.isfinite(uq.std)) and uq.std[0] > 0.0
    lower, upper = uq.interval(0.95)
    assert lower[0] < result.x[0] < upper[0]


def test_parameter_covariance_still_reads_a_stress_update() -> None:
    """The uncertainty path over the stress residual, not just the displacement one."""
    truth, enforced, _exact, _centre = hex_patch()
    kwargs: dict[str, Any] = {"enforced": enforced}
    measured = static_stress_response(truth, E_PARAMETER, **kwargs)(np.array([1.0]))

    wrong = with_wrong_modulus(truth)
    result = update_from_static(
        wrong,
        measured,
        E_PARAMETER,
        response=static_stress_response(wrong, E_PARAMETER, **kwargs),
    )
    uq = parameter_covariance(result, residual_cov=(0.01 * measured) ** 2)

    assert uq.parameter_names == ["E"]
    assert np.all(np.isfinite(uq.std)) and uq.std[0] > 0.0


def test_topometry_optimize_still_reduces_the_compliance() -> None:
    model, _tip = cantilever_plate(nx=4, ny=2)
    result = topometry_optimize(model, max_iter=30)
    assert result.compliance < result.initial_compliance
    assert result.volume == pytest.approx(result.initial_volume, rel=1.0e-9)


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

#!/usr/bin/env python3
"""Probe frozen API boundaries without requiring every package module to exist.

Missing femtools modules are reported as skips and never produce a failing exit
status. Implemented modules are checked against deterministic contract cases.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
RNG = np.random.default_rng(0)


class ProbeUnavailable(RuntimeError):
    """An optional boundary has not landed on this tree yet."""


def probe_core_model() -> dict[str, Any]:
    from femtools.core.model import FEModel

    model = FEModel(name="boundary-probe")
    model.add_node(id=1, xyz=(0.0, 0.0, 0.0))
    model.add_node(id=2, xyz=(1.0, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="bar", material_id=1, A=1.0e-4)
    model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))

    if sorted(model.nodes) != [1, 2] or sorted(model.elements) != [1]:
        raise AssertionError("FEModel did not retain contract entities by integer id")
    if np.asarray(model.nodes[2].xyz).shape != (3,):
        raise AssertionError("Node.xyz is not a three-component array")
    return {"nodes": len(model.nodes), "elements": len(model.elements)}


def probe_mac() -> dict[str, Any]:
    from femtools.correlation.mac import mac_matrix

    modes, _ = np.linalg.qr(RNG.standard_normal((64, 6)), mode="reduced")
    scales = np.array([-2.0, 0.5, 4.0, -1.0, 3.0, 0.25])
    mac = np.asarray(mac_matrix(modes, modes * scales))
    expected = np.eye(6)
    error = float(np.max(np.abs(mac - expected)))
    if mac.shape != expected.shape or error > 1.0e-10:
        raise AssertionError(f"MAC scale/sign invariance error is {error:.3e}")
    return {"shape": list(mac.shape), "max_identity_error": error}


def probe_eigen() -> dict[str, Any]:
    from benchmarks.bench_eigen import build_axial_chain

    from femtools.fea.assemble import assemble_km
    from femtools.fea.eigen import solve_modes

    model = build_axial_chain(12)
    modal = solve_modes(model, n_modes=4, shift=0.0)
    frequencies = np.asarray(modal.freq_hz, dtype=float)
    modes = np.asarray(modal.modes)
    if frequencies.shape != (4,) or np.any(np.diff(frequencies) < 0.0):
        raise AssertionError("eigen frequencies are not a four-value ascending vector")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise AssertionError("eigen frequencies are non-finite or negative")

    assembly = assemble_km(model)
    mass_gram = modes.conj().T @ (assembly.M @ modes)
    normalization_error = float(np.max(np.abs(mass_gram - np.eye(4))))
    if normalization_error > 1.0e-8:
        raise AssertionError(
            f"modal mass-normalization error is {normalization_error:.3e}"
        )
    return {
        "frequencies_hz": frequencies.tolist(),
        "mass_normalization_error": normalization_error,
    }


def probe_modal_frf() -> dict[str, Any]:
    from benchmarks.bench_frf import make_modal_data, response_array

    from femtools.dynamics.frf import modal_frf

    modal = make_modal_data(n_dof=48, n_modes=8, seed=0)
    channels = np.array([0, 7], dtype=int)
    frequency = np.array([0.0, 1.0, 17.5, 2_500.0])
    result = modal_frf(
        modal,
        channels,
        channels,
        frequency,
        np.full(8, 0.02),
    )
    response = response_array(result)
    expected_shape = (2, 2, 4)
    if response.shape != expected_shape:
        raise AssertionError(f"FRF shape {response.shape} does not match {expected_shape}")
    if not np.iscomplexobj(response) or not np.all(np.isfinite(response)):
        raise AssertionError("FRF response is non-complex or non-finite")
    return {"shape": list(response.shape), "norm": float(np.linalg.norm(response))}


def probe_mock_unv() -> dict[str, Any]:
    from femtools.io.unv import read_unv

    fixture = ROOT / "tests" / "perf" / "fixtures" / "mock.unv"
    loaded = read_unv(fixture)
    model = getattr(loaded, "model", loaded)
    if sorted(model.nodes) != [1, 2]:
        raise AssertionError("UNV dataset 2411 did not produce the two fixture nodes")
    xyz = np.asarray(model.nodes[2].xyz, dtype=float)
    if not np.allclose(xyz, (1.0, 0.0, 0.0), rtol=0.0, atol=1.0e-15):
        raise AssertionError(f"UNV node coordinates were parsed as {xyz}")
    return {"fixture": str(fixture.relative_to(ROOT)), "nodes": len(model.nodes)}


def probe_guyan() -> dict[str, Any]:
    from femtools.fea.reduction import guyan

    stiffness = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )
    result = guyan(stiffness, [0, 2])
    if hasattr(result, "T"):
        transform = np.asarray(result.T)
        reduced_stiffness = np.asarray(result.K)
    else:
        transform, reduced_stiffness = (np.asarray(value) for value in result[:2])

    expected_transform = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    expected_stiffness = expected_transform.T @ stiffness @ expected_transform
    transform_error = float(np.max(np.abs(transform - expected_transform)))
    stiffness_error = float(np.max(np.abs(reduced_stiffness - expected_stiffness)))
    if transform.shape != (3, 2) or transform_error > 1.0e-12:
        raise AssertionError(f"Guyan transformation error is {transform_error:.3e}")
    if reduced_stiffness.shape != (2, 2) or stiffness_error > 1.0e-12:
        raise AssertionError(f"Guyan reduced-stiffness error is {stiffness_error:.3e}")
    return {
        "shape": list(transform.shape),
        "transform_error": transform_error,
        "stiffness_error": stiffness_error,
    }


def probe_h1() -> dict[str, Any]:
    from femtools.mpe.frf_estimation import estimate_h1

    fs = 1_024.0
    excitation = RNG.standard_normal(8_192)
    gain = 2.5
    result = estimate_h1(excitation, gain * excitation, fs=fs, nperseg=1_024)
    if hasattr(result, "H"):
        estimate = np.asarray(result.H)
    elif hasattr(result, "frf"):
        estimate = np.asarray(result.frf)
    elif isinstance(result, tuple):
        estimate = np.asarray(result[1])
    else:
        estimate = np.asarray(result)

    estimate = np.squeeze(estimate)
    if estimate.ndim != 1:
        raise AssertionError(f"H1 estimate has unexpected shape {estimate.shape}")
    finite = np.isfinite(estimate)
    if np.count_nonzero(finite) < 0.95 * estimate.size:
        raise AssertionError("H1 estimate has too many non-finite frequency lines")
    gain_error = float(np.max(np.abs(estimate[finite] - gain)))
    if gain_error > 1.0e-10:
        raise AssertionError(f"H1 constant-gain error is {gain_error:.3e}")
    return {
        "frequency_lines": int(estimate.size),
        "finite_lines": int(np.count_nonzero(finite)),
        "gain_error": gain_error,
    }


def probe_read_inp() -> dict[str, Any]:
    from femtools.io.inp import read_inp

    deck = """\
*HEADING
femtools boundary probe
*NODE, NSET=ALLNODES
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE=C3D8, ELSET=SOLID
1, 1, 2, 3, 4, 5, 6, 7, 8
*MATERIAL, NAME=STEEL
*ELASTIC
210000000000, 0.3
*DENSITY
7850
*SOLID SECTION, ELSET=SOLID, MATERIAL=STEEL
*BOUNDARY
1, 1, 3
"""
    with TemporaryDirectory(prefix="femtools-inp-probe-") as tmp:
        path = Path(tmp) / "cube.inp"
        path.write_text(deck, encoding="utf-8")
        loaded = read_inp(path)

    model = getattr(loaded, "model", loaded)
    if sorted(model.nodes) != list(range(1, 9)) or sorted(model.elements) != [1]:
        raise AssertionError("INP C3D8 deck did not produce eight nodes and one element")
    xyz = np.asarray(model.nodes[8].xyz, dtype=float)
    if not np.array_equal(xyz, np.array([0.0, 1.0, 1.0])):
        raise AssertionError(f"INP node 8 coordinates were parsed as {xyz}")
    element = model.elements[1]
    if tuple(element.nodes) != tuple(range(1, 9)):
        raise AssertionError(f"INP C3D8 connectivity was parsed as {element.nodes}")
    if str(element.type).upper() != "HEX8":
        raise AssertionError(f"INP C3D8 was mapped to {element.type!r}, not HEX8")
    return {
        "nodes": len(model.nodes),
        "elements": len(model.elements),
        "materials": len(model.materials),
        "properties": len(model.properties),
    }


def probe_read_k() -> dict[str, Any]:
    from femtools.io.kfile import read_k

    deck = """\
*KEYWORD
*PART
solid cube
1, 1, 1, 0, 0, 0, 0, 0
*SECTION_SOLID
1, 1
*MAT_ELASTIC
1, 7850, 210000000000, 0.3
*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT_SOLID
1, 1, 1, 2, 3, 4, 5, 6, 7, 8
*BOUNDARY_SPC_NODE
1, 0, 1, 1, 1, 0, 0, 0
*END
"""
    with TemporaryDirectory(prefix="femtools-k-probe-") as tmp:
        path = Path(tmp) / "cube.k"
        path.write_text(deck, encoding="utf-8")
        loaded = read_k(path)

    model = getattr(loaded, "model", loaded)
    if sorted(model.nodes) != list(range(1, 9)) or sorted(model.elements) != [1]:
        raise AssertionError("K ELEMENT_SOLID deck did not produce eight nodes and one element")
    xyz = np.asarray(model.nodes[7].xyz, dtype=float)
    if not np.array_equal(xyz, np.array([1.0, 1.0, 1.0])):
        raise AssertionError(f"K node 7 coordinates were parsed as {xyz}")
    element = model.elements[1]
    if tuple(element.nodes) != tuple(range(1, 9)):
        raise AssertionError(f"K ELEMENT_SOLID connectivity was parsed as {element.nodes}")
    if str(element.type).upper() != "HEX8":
        raise AssertionError(f"K ELEMENT_SOLID was mapped to {element.type!r}, not HEX8")
    return {
        "nodes": len(model.nodes),
        "elements": len(model.elements),
        "materials": len(model.materials),
        "properties": len(model.properties),
    }


def probe_nmd() -> dict[str, Any]:
    from femtools.correlation.mac import nmd

    reference = np.array([1.0, 0.0])
    same = float(np.asarray(nmd(reference, reference)).squeeze())
    orthogonal = float(np.asarray(nmd(reference, np.array([0.0, 1.0]))).squeeze())
    diagonal = float(np.asarray(nmd(reference, np.array([1.0, 1.0]))).squeeze())
    expected = np.array([0.0, 1.0, np.sqrt(0.5)])
    actual = np.array([same, orthogonal, diagonal])
    error = float(np.max(np.abs(actual - expected)))
    if error > 1.0e-12:
        raise AssertionError(f"NMD sqrt(1-MAC) error is {error:.3e}")
    return {"values": actual.tolist(), "max_formula_error": error}


def probe_ssi_data() -> dict[str, Any]:
    from femtools.mpe.ssi import ssi_data

    fs = 64.0
    target_hz = 5.0
    damping = 0.03
    time = np.arange(256, dtype=float) / fs
    omega = 2.0 * np.pi * target_hz
    omega_d = omega * np.sqrt(1.0 - damping**2)
    decay = np.exp(-damping * omega * time)
    data = np.vstack((decay * np.sin(omega_d * time), decay * np.cos(omega_d * time)))
    result = ssi_data(
        data,
        fs=fs,
        order=2,
        block_rows=8,
        stabilization=False,
        n_modes=1,
        f_range=(1.0, 12.0),
        max_damping=0.2,
    )
    frequencies = np.asarray(result.freq_hz, dtype=float).reshape(-1)
    dampings = np.asarray(result.damping, dtype=float).reshape(-1)
    if frequencies.size != 1 or dampings.size != 1:
        raise AssertionError(
            f"SSI-DATA returned {frequencies.size} frequencies and {dampings.size} damping values"
        )
    frequency_error = float(abs(frequencies[0] - target_hz) / target_hz)
    damping_error = float(abs(dampings[0] - damping))
    if not np.all(np.isfinite(frequencies)) or frequency_error > 0.02:
        raise AssertionError(f"SSI-DATA frequency error is {frequency_error:.3%}")
    if not np.all(np.isfinite(dampings)) or damping_error > 0.02:
        raise AssertionError(f"SSI-DATA damping error is {damping_error:.3e}")
    return {
        "frequency_hz": float(frequencies[0]),
        "damping": float(dampings[0]),
        "relative_frequency_error": frequency_error,
    }


def probe_parameter_covariance() -> dict[str, Any]:
    from femtools.updating.uq import parameter_covariance

    jacobian = np.diag([2.0, 4.0])
    residual_covariance = np.diag([4.0, 9.0])
    result = parameter_covariance(jacobian, residual_covariance)
    covariance = np.asarray(getattr(result, "covariance", result), dtype=float)
    expected = np.diag([1.0, 9.0 / 16.0])
    error = float(np.max(np.abs(covariance - expected)))
    if covariance.shape != (2, 2) or error > 1.0e-12:
        raise AssertionError(f"first-order parameter covariance error is {error:.3e}")
    return {"diagonal": np.diag(covariance).tolist(), "max_formula_error": error}


def probe_modal_strain_energy() -> dict[str, Any]:
    from femtools.dynamics.energy import modal_strain_energy

    modes = np.eye(2)
    stiffness = np.diag([2.0, 8.0])
    result = np.asarray(modal_strain_energy(modes, stiffness), dtype=float).squeeze()
    half_quadratic = np.array([1.0, 4.0])
    full_quadratic = 2.0 * half_quadratic
    half_error = float(np.max(np.abs(result - half_quadratic)))
    full_error = float(np.max(np.abs(result - full_quadratic)))
    error = min(half_error, full_error)
    if result.shape != (2,) or not np.all(np.isfinite(result)) or error > 1.0e-12:
        raise AssertionError(f"modal strain-energy quadratic-form error is {error:.3e}")
    convention = "half-quadratic" if half_error <= full_error else "full-quadratic"
    return {"energy": result.tolist(), "convention": convention}


def _bar_model(name: str) -> Any:
    from femtools.core.model import FEModel

    model = FEModel(name=name)
    model.add_node(id=1, xyz=(0.0, 0.0, 0.0))
    model.add_node(id=2, xyz=(1.0, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="bar", material_id=1, A=1.0e-4)
    model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model


def _solid_cube_model(name: str) -> Any:
    from femtools.core.model import FEModel

    model = FEModel(name=name)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    for node_id, xyz in enumerate(coordinates, start=1):
        model.add_node(id=node_id, xyz=xyz)
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.3, rho=2700.0)
    model.add_property(id=1, type="solid", material_id=1)
    model.add_element(id=1, type="HEX8", nodes=tuple(range(1, 9)), property_id=1)
    return model


def _tet10_model(name: str) -> Any:
    from femtools.core.model import FEModel

    model = FEModel(name=name)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    )
    for node_id, xyz in enumerate(coordinates, start=1):
        model.add_node(id=node_id, xyz=xyz)
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.25, rho=2700.0)
    model.add_property(id=1, type="solid", material_id=1)
    model.add_element(id=1, type="TET10", nodes=tuple(range(1, 11)), property_id=1)
    return model


def _result_field(
    result: Any,
    names: tuple[str, ...],
    element_id: int | None = 1,
) -> np.ndarray:
    value = result
    for name in names:
        if hasattr(result, name):
            value = getattr(result, name)
            break
        if isinstance(result, Mapping) and name in result:
            value = result[name]
            break
    if isinstance(value, Mapping) and element_id is not None:
        if element_id not in value:
            raise AssertionError(f"result field does not contain element {element_id}")
        value = value[element_id]
    elif isinstance(value, Mapping):
        value = [value[key] for key in sorted(value)]
    array = np.asarray(value, dtype=float).squeeze()
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise AssertionError("result field is empty or non-finite")
    return array


def probe_recover_stress() -> dict[str, Any]:
    from femtools.fea.recover import recover_stress

    model = _bar_model("stress-boundary-probe")
    displacement = np.zeros(model.ndof)
    displacement[model.dof_map()[(2, 0)]] = 1.0e-3
    result = recover_stress(model, displacement)
    stress = _result_field(
        result,
        ("stress", "stresses", "element_stress", "values", "data"),
    )
    expected = 210.0e6
    recovered = float(np.max(np.abs(stress)))
    relative_error = abs(recovered - expected) / expected
    if relative_error > 1.0e-10:
        raise AssertionError(
            f"BAR2 constant axial-stress error is {relative_error:.3e}"
        )
    return {
        "shape": list(stress.shape),
        "max_abs_stress": recovered,
        "relative_error": relative_error,
    }


def probe_apply_rbe3() -> dict[str, Any]:
    from femtools.core.model import FEModel
    from femtools.fea.mpc import apply_rbe3

    model = FEModel(name="rbe3-boundary-probe")
    for node_id, xyz in enumerate(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.5, 0.5, 0.0)),
        start=1,
    ):
        model.add_node(node_id, xyz)
    weights = np.array([1.0, 2.0, 3.0])
    model.add_rbe3(
        id=1,
        dependent=4,
        independents=(1, 2, 3),
        components=(1,),
        independent_components=(1,),
        weights=weights,
    )

    transform = apply_rbe3(model)
    dofs = model.dof_map()
    independent_values = np.array([1.0, 4.0, -2.0])
    displacement = np.zeros(model.ndof)
    for node_id, value in zip((1, 2, 3), independent_values, strict=True):
        displacement[dofs[(node_id, 0)]] = value
    displacement[dofs[(4, 0)]] = 99.0
    constrained = np.asarray(transform.to_full(displacement), dtype=float)

    normalized = weights / weights.sum()
    expected_motion = float(normalized @ independent_values)
    motion_error = abs(float(constrained[dofs[(4, 0)]]) - expected_motion)
    if motion_error > 1.0e-14:
        raise AssertionError(f"RBE3 weighted-average motion error is {motion_error:.3e}")

    dependent_force = np.zeros(model.ndof)
    dependent_force[dofs[(4, 0)]] = 1.0
    distributed = np.asarray(transform.to_independent(dependent_force), dtype=float)
    shares = np.array([distributed[dofs[(node_id, 0)]] for node_id in (1, 2, 3)])
    load_error = float(np.max(np.abs(shares - normalized)))
    if load_error > 1.0e-14 or distributed[dofs[(4, 0)]] != 0.0:
        raise AssertionError(f"RBE3 virtual-work load distribution error is {load_error:.3e}")
    return {
        "weighted_motion": float(constrained[dofs[(4, 0)]]),
        "force_shares": shares.tolist(),
        "motion_error": motion_error,
        "load_error": load_error,
    }


def _nodal_stress_rows(result: Any) -> tuple[list[Any], np.ndarray]:
    values = result
    for name in ("stress", "stresses", "values", "data"):
        if hasattr(result, name):
            values = getattr(result, name)
            break
    if isinstance(values, Mapping):
        node_ids = sorted(values)
        rows = np.asarray([values[node_id] for node_id in node_ids], dtype=float)
        return node_ids, np.atleast_2d(rows)

    node_ids = None
    for name in ("node_ids", "point_ids", "ids", "element_ids"):
        if hasattr(result, name):
            node_ids = list(getattr(result, name))
            break
    if node_ids is None:
        raise AssertionError("nodal stress result has no node identifiers")
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != len(node_ids):
        raise AssertionError(
            f"nodal stress values have shape {rows.shape} for {len(node_ids)} node ids"
        )
    return node_ids, rows


def probe_average_nodal() -> dict[str, Any]:
    from femtools.fea.recover import average_nodal, recover_stress

    model = _bar_model("nodal-average-boundary-probe")
    model.add_node(id=3, xyz=(2.0, 0.0, 0.0))
    model.add_element(id=2, type="BAR2", nodes=(2, 3), property_id=1)
    strain = 1.0e-3
    displacement = np.zeros(model.ndof)
    dofs = model.dof_map()
    displacement[dofs[(2, 0)]] = strain
    displacement[dofs[(3, 0)]] = 2.0 * strain

    centroid = recover_stress(model, displacement)
    nodal = average_nodal(centroid, model)
    node_ids, stresses = _nodal_stress_rows(nodal)
    expected_ids = [1, 2, 3]
    if node_ids != expected_ids:
        raise AssertionError(
            f"nodal averaging returned node ids {node_ids}, expected {expected_ids}"
        )
    expected = np.zeros((3, 6))
    expected[:, 0] = 210.0e6
    error = float(np.max(np.abs(stresses - expected)))
    if stresses.shape != expected.shape or error > 1.0e-6:
        raise AssertionError(f"constant-stress nodal averaging error is {error:.3e} Pa")
    return {
        "node_ids": node_ids,
        "shape": list(stresses.shape),
        "max_patch_error_pa": error,
    }


def _probe_hex8_writer(
    writer: Callable[[Any, Path], Any],
    reader: Callable[[Path], Any],
    suffix: str,
    markers: tuple[str, ...],
) -> dict[str, Any]:
    from femtools.fea.assemble import assemble_km

    model = _solid_cube_model(f"{suffix[1:]}-writer-boundary-probe")
    reference = assemble_km(model).K.toarray()
    with TemporaryDirectory(prefix=f"femtools-{suffix[1:]}-writer-probe-") as tmp:
        path = Path(tmp) / f"cube{suffix}"
        writer(model, path)
        text = path.read_text(encoding="utf-8")
        loaded_result = reader(path)
        loaded = getattr(loaded_result, "model", loaded_result)
        size = path.stat().st_size

    upper = text.upper()
    missing = [marker for marker in markers if marker not in upper]
    if missing:
        raise AssertionError(f"{suffix} deck is missing records {missing}")
    if sorted(loaded.nodes) != list(range(1, 9)) or sorted(loaded.elements) != [1]:
        raise AssertionError(f"{suffix} HEX8 round-trip changed model entities")
    element = loaded.elements[1]
    if str(element.type).upper() != "HEX8" or tuple(element.nodes) != tuple(range(1, 9)):
        raise AssertionError(f"{suffix} HEX8 round-trip changed connectivity")

    roundtrip = assemble_km(loaded).K.toarray()
    scale = max(float(np.max(np.abs(reference))), 1.0)
    stiffness_error = float(np.max(np.abs(roundtrip - reference)) / scale)
    if roundtrip.shape != reference.shape or stiffness_error > 1.0e-10:
        raise AssertionError(
            f"{suffix} round-trip stiffness error is {stiffness_error:.3e}"
        )
    return {
        "bytes": size,
        "nodes": len(loaded.nodes),
        "elements": len(loaded.elements),
        "relative_stiffness_error": stiffness_error,
    }


def probe_write_cdb() -> dict[str, Any]:
    from femtools.io.cdb import read_cdb, write_cdb

    return _probe_hex8_writer(write_cdb, read_cdb, ".cdb", ("NBLOCK", "EBLOCK"))


def probe_write_k() -> dict[str, Any]:
    from femtools.io.kfile import read_k, write_k

    return _probe_hex8_writer(
        write_k,
        read_k,
        ".k",
        ("*NODE", "*ELEMENT_SOLID"),
    )


def probe_map_nearest_nodes() -> dict[str, Any]:
    from femtools.correlation.dofmap import map_nearest_nodes

    model = _solid_cube_model("nearest-node-boundary-probe")
    xyz_fe = np.vstack([model.nodes[node_id].xyz for node_id in model.node_ids()])
    translation = np.array([0.125, -0.25, 0.0625])
    fe_ids, distances = map_nearest_nodes(xyz_fe + translation, model)
    fe_ids = np.asarray(fe_ids, dtype=int).reshape(-1)
    distances = np.asarray(distances, dtype=float).reshape(-1)
    expected_ids = np.arange(1, 9)
    expected_distance = float(np.linalg.norm(translation))
    distance_error = float(np.max(np.abs(distances - expected_distance)))
    if not np.array_equal(fe_ids, expected_ids):
        raise AssertionError(f"translated cube mapped to node ids {fe_ids.tolist()}")
    if distances.shape != (8,) or distance_error > 1.0e-12:
        raise AssertionError(
            f"translated-cube nearest-node distance error is {distance_error:.3e}"
        )
    return {
        "node_ids": fe_ids.tolist(),
        "distance": expected_distance,
        "max_distance_error": distance_error,
    }


def _cantilever_plate_model() -> Any:
    from femtools.core.model import FEModel

    model = FEModel(name="topometry-boundary-probe")
    nx, ny = 4, 2

    def node_id(ix: int, iy: int) -> int:
        return ix * (ny + 1) + iy + 1

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            model.add_node(node_id(ix, iy), (float(ix), float(iy), 0.0))
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.3, rho=2700.0)
    element_id = 0
    for ix in range(nx):
        for iy in range(ny):
            element_id += 1
            model.add_property(
                id=element_id,
                type="shell",
                material_id=1,
                t=0.05,
            )
            model.add_element(
                id=element_id,
                type="QUAD4",
                nodes=(
                    node_id(ix, iy),
                    node_id(ix + 1, iy),
                    node_id(ix + 1, iy + 1),
                    node_id(ix, iy + 1),
                ),
                property_id=element_id,
            )
    for iy in range(ny + 1):
        model.add_spc(
            node_id=node_id(0, iy),
            mask=(True, True, True, True, True, True),
        )
    return model, node_id(nx, ny // 2)


def probe_topometry_optimize() -> dict[str, Any]:
    from femtools.optimization.topometry import topometry_optimize

    model, tip_node = _cantilever_plate_model()
    result = topometry_optimize(
        model,
        loads={(tip_node, 2): -1.0e3},
        volume_fraction=0.7,
        max_iter=6,
    )
    design = _result_field(
        result,
        (
            "thickness",
            "thicknesses",
            "density",
            "densities",
            "design",
            "element_values",
            "x",
            "rho",
        ),
        element_id=None,
    ).reshape(-1)
    if design.shape != (len(model.elements),) or np.any(design <= 0.0):
        raise AssertionError(
            f"topometry design has shape {design.shape} or non-positive values"
        )

    compliance_value = None
    for name in ("compliance", "final_compliance", "value", "objective_value"):
        if hasattr(result, name):
            compliance_value = float(getattr(result, name))
            break
    if compliance_value is None or not np.isfinite(compliance_value) or compliance_value <= 0.0:
        raise AssertionError("topometry returned no finite positive compliance")

    initial = getattr(result, "initial_compliance", None)
    if initial is not None and compliance_value > float(initial) * (1.0 + 1.0e-8):
        raise AssertionError("topometry increased compliance from its uniform start")
    return {
        "elements": len(model.elements),
        "compliance": compliance_value,
        "design_min": float(np.min(design)),
        "design_max": float(np.max(design)),
    }


def probe_nastran_punch_driver() -> dict[str, Any]:
    from femtools.core.errors import SolverError
    from femtools.drivers.base import SolverDriver
    from femtools.drivers.nastran import NastranPunchDriver

    executable = "__femtools_boundary_probe_missing_nastran__"
    driver = NastranPunchDriver(executable=executable)
    if not isinstance(driver, SolverDriver):
        raise AssertionError("NastranPunchDriver does not satisfy SolverDriver")
    if driver.is_available():
        raise AssertionError(f"unexpected executable found for {executable!r}")

    with TemporaryDirectory(prefix="femtools-nastran-driver-probe-") as tmp:
        deck = Path(driver.write_input(_bar_model("nastran-driver-probe"), tmp))
        text = deck.read_text(encoding="utf-8").upper()
        if "SOL 103" not in text or "PUNCH" not in text:
            raise AssertionError("Nastran modal deck lacks SOL 103 punch requests")
        try:
            driver.run(deck, timeout=1.0)
        except SolverError:
            pass
        else:
            raise AssertionError("missing Nastran executable did not raise SolverError")
    return {
        "name": driver.name,
        "available": False,
        "missing_executable_raises": "SolverError",
    }


def _probe_text_solver_driver(
    driver: Any,
    *,
    deck_suffix: str,
    deck_markers: tuple[str, ...],
    binary_suffix: str,
) -> dict[str, Any]:
    from femtools.core.errors import SolverError
    from femtools.core.results import ModalResult
    from femtools.drivers.base import SolverDriver
    from femtools.io.pch import write_pch

    if not isinstance(driver, SolverDriver):
        raise AssertionError(f"{type(driver).__name__} does not satisfy SolverDriver")
    if driver.is_available():
        raise AssertionError(f"unexpected executable found for {driver.executable!r}")

    with TemporaryDirectory(prefix=f"femtools-{driver.name}-probe-") as tmp:
        directory = Path(tmp)
        deck = Path(driver.write_input(_bar_model(f"{driver.name}-probe"), directory))
        if deck.suffix.lower() != deck_suffix:
            raise AssertionError(f"{type(driver).__name__} wrote {deck.suffix}, not {deck_suffix}")
        text = deck.read_text(encoding="utf-8").upper()
        missing = [marker for marker in deck_markers if marker not in text]
        if missing:
            raise AssertionError(f"{deck_suffix} driver deck is missing records {missing}")
        try:
            driver.run(deck, timeout=1.0)
        except SolverError:
            pass
        else:
            raise AssertionError(f"missing {driver.name} executable did not raise SolverError")

        frequency_hz = 3.0
        eigenvalue = (2.0 * np.pi * frequency_hz) ** 2
        modal = ModalResult(
            freq_hz=np.array([frequency_hz]),
            eigenvalues=np.array([eigenvalue]),
            modes=np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
            generalized_mass=np.ones(1),
            dof_index=tuple((1, dof) for dof in range(6)),
        )
        punch = directory / "mode.pch"
        write_pch(punch, modal)
        loaded = driver.read_modal(punch)
        loaded_frequency = np.asarray(loaded.freq_hz, dtype=float).reshape(-1)
        frequency_error = float(np.max(np.abs(loaded_frequency - frequency_hz)))
        if loaded_frequency.shape != (1,) or frequency_error > 1.0e-8:
            raise AssertionError(
                f"{type(driver).__name__} text-modal frequency error is {frequency_error:.3e}"
            )

        binary = directory / f"mode{binary_suffix}"
        binary.write_bytes(b"femtools boundary probe")
        try:
            driver.read_modal(binary)
        except SolverError as exc:
            if binary_suffix[1:].upper() not in str(exc).upper():
                raise AssertionError(
                    f"{type(driver).__name__} binary rejection did not name "
                    f"{binary_suffix.upper()}"
                ) from exc
        else:
            raise AssertionError(
                f"{type(driver).__name__} accepted unsupported {binary_suffix.upper()} results"
            )

    return {
        "name": driver.name,
        "deck_suffix": deck_suffix,
        "text_frequency_error": frequency_error,
        "binary_rejected": binary_suffix[1:].upper(),
        "missing_executable_raises": "SolverError",
    }


def probe_ansys_cdb_driver() -> dict[str, Any]:
    from femtools.drivers.ansys import AnsysCdbDriver

    return _probe_text_solver_driver(
        AnsysCdbDriver(executable="__femtools_boundary_probe_missing_ansys__"),
        deck_suffix=".cdb",
        deck_markers=("NBLOCK", "EBLOCK"),
        binary_suffix=".rst",
    )


def probe_abaqus_inp_driver() -> dict[str, Any]:
    from femtools.drivers.abaqus import AbaqusInpDriver

    return _probe_text_solver_driver(
        AbaqusInpDriver(executable="__femtools_boundary_probe_missing_abaqus__"),
        deck_suffix=".inp",
        deck_markers=("*NODE", "*ELEMENT"),
        binary_suffix=".odb",
    )


def probe_dump_frf() -> dict[str, Any]:
    from femtools.dynamics.frf import FRFResult

    try:
        from femtools.dynamics.frf import dump_frf, load_frf
    except ImportError:
        from femtools.dynamics.superelement import dump_frf, load_frf

    frequency = np.array([0.0, 1.25, 9.5, 80.0])
    real = np.arange(24, dtype=float).reshape(2, 3, 4)
    response = FRFResult(
        H=real + 1j * (real[::-1] + 0.5),
        freq_hz=frequency,
        outputs=np.array([2, 7]),
        inputs=np.array([1, 4, 9]),
        response="mobility",
        method="boundary-probe",
    )
    with TemporaryDirectory(prefix="femtools-frf-dump-probe-") as tmp:
        requested = Path(tmp) / "response"
        returned = dump_frf(response, requested)
        archive = Path(returned) if returned is not None else requested.with_suffix(".npz")
        loaded = load_frf(archive)
        size = archive.stat().st_size

    loaded_response = getattr(loaded, "H", getattr(loaded, "h_complex", None))
    if loaded_response is None:
        raise AssertionError("loaded FRF has neither H nor h_complex data")
    h_equal = bool(np.array_equal(np.asarray(loaded_response), response.H))
    frequency_equal = bool(np.array_equal(np.asarray(loaded.freq_hz), frequency))
    if not h_equal or not frequency_equal:
        raise AssertionError(
            f"FRF npz is not bit-identical (H={h_equal}, freq_hz={frequency_equal})"
        )
    return {
        "shape": list(np.asarray(loaded_response).shape),
        "bytes": size,
        "H_bit_identical": h_equal,
        "frequency_bit_identical": frequency_equal,
    }


def probe_plot_stress() -> dict[str, Any]:
    from femtools.fea.recover import recover_stress
    from femtools.viz.plots import plot_stress

    model = _solid_cube_model("plot-stress-boundary-probe")
    displacement = np.zeros(model.ndof)
    dofs = model.dof_map()
    for node_id, node in model.nodes.items():
        displacement[dofs[(node_id, 0)]] = 1.0e-4 * float(node.xyz[0])
    stress = recover_stress(model, displacement)

    with TemporaryDirectory(prefix="femtools-plot-stress-probe-") as tmp:
        image = Path(tmp) / "stress.png"
        figure = plot_stress(model, stress, outfile=image)
        size = image.stat().st_size
        axes = list(getattr(figure, "axes", ()))
        color_values = [
            np.asarray(collection.get_array())
            for axis in axes
            for collection in axis.collections
            if collection.get_array() is not None
        ]
        figure.clear()

    if size <= 0 or not axes:
        raise AssertionError("plot_stress did not return a non-empty matplotlib figure")
    if not color_values or not all(np.all(np.isfinite(values)) for values in color_values):
        raise AssertionError("plot_stress produced no finite stress color data")
    return {
        "image_bytes": size,
        "axes": len(axes),
        "colored_artists": len(color_values),
        "max_von_mises": float(np.max(stress.von_mises)),
    }


def probe_update_from_static() -> dict[str, Any]:
    from femtools.updating.updater import update_from_static

    true_modulus = 210.0e9
    model = _bar_model("static-update-boundary-probe")
    model.materials[1].E = 1.1 * true_modulus
    model.add_spc(node_id=2, mask=(False, True, True, True, True, True))
    model.add_load(node_id=2, force=(1.0, 0.0, 0.0))
    measured = np.array([1.0 / (true_modulus * model.properties[1].A)])
    parameters = [
        {
            "type": "material",
            "id": 1,
            "name": "E",
            "lower": 0.5,
            "upper": 1.5,
        }
    ]
    result = update_from_static(
        model,
        measured,
        parameters=parameters,
        dofs=[(2, "ux")],
        bounds=(0.5, 1.5),
        tol=1.0e-12,
    )
    values = np.asarray(getattr(result, "x", getattr(result, "values", result)), dtype=float)
    expected = 1.0 / 1.1
    relative_error = abs(float(values[0]) - expected) / expected
    if values.shape != (1,) or relative_error > 1.0e-8:
        raise AssertionError(f"static Young's-modulus recovery error is {relative_error:.3e}")
    return {
        "recovered_multiplier": float(values[0]),
        "expected_multiplier": expected,
        "relative_error": relative_error,
    }


def probe_mapped_mode_matrix() -> dict[str, Any]:
    from femtools.correlation.dofmap import map_nearest_nodes, mapped_mode_matrix
    from femtools.correlation.mac import mac_matrix

    model = _solid_cube_model("mapped-mode-boundary-probe")
    xyz = np.vstack([model.nodes[node_id].xyz for node_id in model.node_ids()])
    translated = xyz + np.array([0.125, -0.25, 0.0625])
    fe_ids, _distances = map_nearest_nodes(translated, model)
    fe_ids = np.asarray(fe_ids, dtype=int)
    nodal_modes = np.column_stack(
        (
            np.linspace(1.0, 2.0, len(fe_ids)),
            np.array([1.0, -1.0] * (len(fe_ids) // 2)),
        )
    )
    modes = np.zeros((model.ndof, nodal_modes.shape[1]))
    dof_map = model.dof_map()
    for row, node_id in enumerate(model.node_ids()):
        modes[dof_map[(node_id, 0)], :] = nodal_modes[row]
    mapped = np.asarray(
        mapped_mode_matrix(modes, fe_ids, dof_map=dof_map, dofs=("x",))
    )
    expected = nodal_modes[fe_ids - 1]
    if mapped.shape != expected.shape or not np.array_equal(mapped, expected):
        raise AssertionError(
            f"mapped mode matrix has shape {mapped.shape}; expected translated-cube "
            f"selection {expected.shape}"
        )
    mac = np.asarray(mac_matrix(mapped, expected), dtype=float)
    diagonal_error = float(np.max(np.abs(np.diag(mac) - 1.0)))
    if diagonal_error > 1.0e-12:
        raise AssertionError(f"mapped-mode MAC diagonal error is {diagonal_error:.3e}")
    return {
        "shape": list(mapped.shape),
        "node_ids": fe_ids.tolist(),
        "max_mac_diagonal_error": diagonal_error,
    }


def probe_apply_mpc() -> dict[str, Any]:
    from femtools.core.model import FEModel
    from femtools.fea.mpc import apply_mpc

    model = FEModel(name="composed-mpc-boundary-probe")
    for node_id, xyz in enumerate(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, xyz)
    model.add_rbe2(id=1, independent=1, dependents=(2,), components=(1,))
    model.add_rbe3(
        id=2,
        dependent=4,
        independents=(2, 3),
        components=(1,),
        independent_components=(1,),
        weights=(1.0, 3.0),
    )

    transform = apply_mpc(model)
    dofs = model.dof_map()
    displacement = np.zeros(model.ndof)
    displacement[dofs[(1, 0)]] = 2.0
    displacement[dofs[(3, 0)]] = 10.0
    constrained = np.asarray(transform.to_full(displacement), dtype=float)
    expected = np.array([2.0, 8.0])
    actual = constrained[[dofs[(2, 0)], dofs[(4, 0)]]]
    error = float(np.max(np.abs(actual - expected)))
    if transform.n_dependent != 2 or error > 1.0e-14:
        raise AssertionError(
            f"composed RBE2/RBE3 motion is {actual.tolist()}, expected {expected.tolist()}"
        )
    if not apply_mpc(model, rbe2=(), rbe3=()).is_identity:
        raise AssertionError("apply_mpc with both MPC tables disabled is not an identity")
    return {
        "dependent_dofs": transform.n_dependent,
        "dependent_motion": actual.tolist(),
        "max_motion_error": error,
    }


def probe_static_stress_response() -> dict[str, Any]:
    from femtools.updating.responses import static_stress_response
    from femtools.updating.updater import update_from_static

    true_modulus = 210.0e9
    parameters = [
        {
            "type": "material",
            "id": 1,
            "name": "E",
            "lower": 0.5,
            "upper": 1.5,
        }
    ]
    enforced = {(2, 0): 1.0e-3}

    truth = _bar_model("static-stress-response-boundary-probe")
    truth.add_spc(node_id=2, mask=(False, True, True, True, True, True))
    measured = np.asarray(
        static_stress_response(
            truth,
            parameters,
            component="xx",
            solver_kwargs={"enforced": enforced},
        )(np.array([1.0])),
        dtype=float,
    ).reshape(-1)
    expected_stress = true_modulus * enforced[(2, 0)]
    stress_error = abs(float(measured[0]) - expected_stress) / expected_stress
    if measured.shape != (1,) or stress_error > 1.0e-10:
        raise AssertionError(
            f"displacement-driven BAR2 stress error is {stress_error:.3e}"
        )

    wrong = _bar_model("static-stress-update-boundary-probe")
    wrong.add_spc(node_id=2, mask=(False, True, True, True, True, True))
    wrong.materials[1].E = 1.1 * true_modulus
    response = static_stress_response(
        wrong,
        parameters,
        component="xx",
        solver_kwargs={"enforced": enforced},
    )
    result = update_from_static(
        wrong,
        measured,
        parameters,
        response=response,
        tol=1.0e-12,
    )
    recovered = float(np.asarray(result.x, dtype=float).reshape(-1)[0])
    expected_multiplier = 1.0 / 1.1
    recovery_error = abs(recovered - expected_multiplier) / expected_multiplier
    if recovery_error > 1.0e-9:
        raise AssertionError(
            f"stress-residual Young's-modulus recovery error is {recovery_error:.3e}"
        )
    return {
        "stress_pa": float(measured[0]),
        "stress_relative_error": stress_error,
        "recovered_multiplier": recovered,
        "recovery_relative_error": recovery_error,
    }


def probe_dump_psd() -> dict[str, Any]:
    from femtools.dynamics.random import PSDResult, dump_psd, load_psd

    frequency = np.array([0.0, 2.5, 17.0, 64.0])
    spectra = np.arange(1.0, 9.0).reshape(2, 4) * 1.0e-6
    rms = np.sqrt(
        np.asarray(
            (getattr(np, "trapezoid", None) or np.trapz)(
                spectra,
                x=frequency,
                axis=-1,
            )
        )
    )
    source = PSDResult(
        psd=spectra,
        freq_hz=frequency,
        rms=rms,
        outputs=np.array([2, 7]),
        inputs=np.array([1]),
        response="accelerance",
        meta={"probe": "boundary"},
    )
    with TemporaryDirectory(prefix="femtools-psd-dump-probe-") as tmp:
        requested = Path(tmp) / "response"
        returned = dump_psd(source, requested)
        archive = Path(returned) if returned is not None else requested.with_suffix(".npz")
        loaded = load_psd(archive)
        size = archive.stat().st_size

    spectra_equal = bool(np.array_equal(np.asarray(loaded.psd), spectra))
    frequency_equal = bool(np.array_equal(np.asarray(loaded.freq_hz), frequency))
    if not spectra_equal or not frequency_equal:
        raise AssertionError(
            "PSD npz is not bit-identical "
            f"(psd={spectra_equal}, freq_hz={frequency_equal})"
        )
    return {
        "shape": list(np.asarray(loaded.psd).shape),
        "bytes": size,
        "psd_bit_identical": spectra_equal,
        "frequency_bit_identical": frequency_equal,
    }


def probe_mapped_mac() -> dict[str, Any]:
    from femtools.correlation.dofmap import mapped_mac

    model = _solid_cube_model("mapped-mac-boundary-probe")
    xyz = np.vstack([model.nodes[node_id].xyz for node_id in model.node_ids()])
    translated = xyz + np.array([0.125, -0.25, 0.0625])
    test_modes = np.column_stack(
        (
            np.linspace(1.0, 2.0, len(xyz)),
            np.array([1.0, -1.0] * (len(xyz) // 2)),
        )
    )
    fe_modes = np.zeros((model.ndof, test_modes.shape[1]))
    dof_map = model.dof_map()
    for row, node_id in enumerate(model.node_ids()):
        fe_modes[dof_map[(node_id, 0)], :] = test_modes[row]

    result = mapped_mac(
        test_modes,
        translated,
        fe_modes,
        model,
        dof_map=dof_map,
        dofs=("x",),
    )
    matrix = np.asarray(getattr(result, "mac", result), dtype=float)
    expected_shape = (test_modes.shape[1], test_modes.shape[1])
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if matrix.shape != expected_shape or diagonal_error > 1.0e-12:
        raise AssertionError(
            f"translated-block mapped MAC has shape {matrix.shape} and "
            f"diagonal error {diagonal_error:.3e}"
        )
    return {
        "shape": list(matrix.shape),
        "max_diagonal_error": diagonal_error,
    }


def probe_nastran_static() -> dict[str, Any]:
    from inspect import signature

    from femtools.drivers.nastran import NastranPunchDriver

    driver = NastranPunchDriver(executable="__femtools_boundary_probe_missing_nastran__")
    read_static = getattr(driver, "read_static", None)
    if not callable(read_static):
        raise ImportError("NastranPunchDriver.read_static is not available")

    write_static = getattr(driver, "write_static_input", None)
    if not callable(write_static):
        write_static = getattr(driver, "write_static", None)
    supports_sol = "sol" in signature(driver.write_input).parameters
    if not callable(write_static) and not supports_sol:
        raise ImportError(
            "NastranPunchDriver has neither a static writer nor write_input(..., sol=101)"
        )

    punch_text = """\
$TITLE = FEMTOOLS STATIC BOUNDARY PROBE
$SUBTITLE = SOL 101
$LABEL =
$DISPLACEMENTS
$REAL OUTPUT
$SUBCASE ID = 1
       1       G      1.000000E-04      2.000000E-04      3.000000E-04
-CONT-               4.000000E-04      5.000000E-04      6.000000E-04
       2       G     -1.500000E-03      2.500000E-03     -3.500000E-03
-CONT-               4.500000E-03     -5.500000E-03      6.500000E-03
"""
    with TemporaryDirectory(prefix="femtools-nastran-static-probe-") as tmp:
        directory = Path(tmp)
        if callable(write_static):
            deck = Path(write_static(_bar_model("nastran-static-probe"), directory))
        else:
            deck = Path(
                driver.write_input(_bar_model("nastran-static-probe"), directory, sol=101)
            )
        text = deck.read_text(encoding="utf-8").upper()
        if "SOL 101" not in text or "DISPLACEMENT(PUNCH)" not in text:
            raise AssertionError("Nastran static deck lacks SOL 101 punch requests")

        punch = directory / "static.pch"
        punch.write_text(punch_text, encoding="utf-8")
        loaded = read_static(punch)

    values = getattr(loaded, "u", getattr(loaded, "displacements", None))
    if values is None:
        raise AssertionError("Nastran static result has neither u nor displacements")
    displacement = np.asarray(values, dtype=float).reshape(-1)
    expected = np.array(
        [
            1.0e-4,
            2.0e-4,
            3.0e-4,
            4.0e-4,
            5.0e-4,
            6.0e-4,
            -1.5e-3,
            2.5e-3,
            -3.5e-3,
            4.5e-3,
            -5.5e-3,
            6.5e-3,
        ]
    )
    error = (
        float(np.max(np.abs(displacement - expected)))
        if displacement.shape == expected.shape
        else float("inf")
    )
    if error > 1.0e-15:
        raise AssertionError(
            f"Nastran static punch displacement shape/error is "
            f"{displacement.shape}/{error:.3e}"
        )
    return {
        "deck_suffix": deck.suffix,
        "displacement_dofs": int(displacement.size),
        "max_displacement_error": error,
    }


def probe_tet10_patch() -> dict[str, Any]:
    # Importing the public symbol is an intentional capability check.  Merely
    # accepting "TET10" in FEModel does not mean the assembly kernel has landed.
    from femtools.fea.assemble import assemble_km
    from femtools.fea.elements import tet10  # noqa: F401
    from femtools.fea.materials import solid_D
    from femtools.fea.recover import recover_stress

    model = _tet10_model("tet10-patch-boundary-probe")
    assembly = assemble_km(model)
    if assembly.free_dof.size != 30:
        raise AssertionError(
            f"TET10 assembly retained {assembly.free_dof.size} active DOFs, expected 30"
        )

    # An affine displacement is represented exactly by a quadratic tetrahedron.
    gradient = np.array(
        [
            [1.0e-3, 2.0e-4, -1.0e-4],
            [3.0e-4, -5.0e-4, 4.0e-4],
            [2.0e-4, -3.0e-4, 7.0e-4],
        ]
    )
    expected_strain = np.array(
        [
            gradient[0, 0],
            gradient[1, 1],
            gradient[2, 2],
            gradient[0, 1] + gradient[1, 0],
            gradient[1, 2] + gradient[2, 1],
            gradient[2, 0] + gradient[0, 2],
        ]
    )
    expected_stress = solid_D(model.materials[1]) @ expected_strain
    displacement = np.zeros(model.ndof)
    for node_id, node in model.nodes.items():
        motion = gradient @ np.asarray(node.xyz, dtype=float)
        for component in range(3):
            displacement[assembly.dof_map.index(node_id, component)] = motion[component]

    recovered = recover_stress(model, displacement, assembly=assembly)
    if recovered.element_ids != [1] or recovered.etypes != ["TET10"]:
        raise AssertionError("TET10 stress recovery did not return the source element")
    strain_scale = max(float(np.max(np.abs(expected_strain))), 1.0e-30)
    stress_scale = max(float(np.max(np.abs(expected_stress))), 1.0)
    strain_error = float(np.max(np.abs(recovered.strain[0] - expected_strain)) / strain_scale)
    stress_error = float(np.max(np.abs(recovered.stress[0] - expected_stress)) / stress_scale)

    volume = 1.0 / 6.0
    expected_energy = 0.5 * volume * float(expected_strain @ expected_stress)
    assembled_energy = 0.5 * float(displacement @ (assembly.K @ displacement))
    energy_error = abs(assembled_energy - expected_energy) / expected_energy
    patch_error = max(strain_error, stress_error, energy_error)
    if patch_error > 1.0e-12:
        raise AssertionError(f"TET10 constant-strain patch error is {patch_error:.3e}")
    return {
        "active_dofs": int(assembly.free_dof.size),
        "strain_relative_error": strain_error,
        "stress_relative_error": stress_error,
        "energy_relative_error": energy_error,
    }


def probe_recover_spr() -> dict[str, Any]:
    from femtools.fea.recover import recover_spr, recover_stress

    model = _bar_model("spr-boundary-probe")
    model.add_node(id=3, xyz=(2.0, 0.0, 0.0))
    model.add_element(id=2, type="BAR2", nodes=(2, 3), property_id=1)
    displacement = np.zeros(model.ndof)
    dofs = model.dof_map()
    displacement[dofs[(2, 0)]] = 1.0e-3
    displacement[dofs[(3, 0)]] = 2.0e-3
    centroid = recover_stress(model, displacement)

    parameters = list(inspect.signature(recover_spr).parameters)
    if parameters and parameters[0].lower() in {"model", "mesh"}:
        recovered = recover_spr(model, displacement)
    else:
        recovered = recover_spr(centroid, model)
    node_ids, stresses = _nodal_stress_rows(recovered)
    expected = np.zeros((3, 6))
    expected[:, 0] = 210.0e6
    error = (
        float(np.max(np.abs(stresses - expected)))
        if stresses.shape == expected.shape
        else float("inf")
    )
    if node_ids != [1, 2, 3] or error > 1.0e-6:
        raise AssertionError(
            f"SPR constant-stress patch returned nodes {node_ids} with error {error:.3e} Pa"
        )
    return {
        "node_ids": node_ids,
        "shape": list(stresses.shape),
        "max_patch_error_pa": error,
    }


def probe_era() -> dict[str, Any]:
    from femtools.correlation.mac import mac_matrix
    from femtools.mpe.era import era

    fs = 256.0
    n_samples = 1_024
    truth_frequency = np.array([8.0, 23.0])
    truth_damping = np.array([0.01, 0.02])
    truth_shapes = np.array([[1.0, 0.2], [0.3, 1.0]])
    participation = np.array([1.0, 0.7])
    time = np.arange(n_samples, dtype=float) / fs
    impulse = np.zeros((2, 1, n_samples))
    for mode in range(2):
        omega = 2.0 * np.pi * truth_frequency[mode]
        omega_d = omega * np.sqrt(1.0 - truth_damping[mode] ** 2)
        coordinate = (
            participation[mode]
            * np.exp(-truth_damping[mode] * omega * time)
            * np.sin(omega_d * time)
            / omega_d
        )
        impulse[:, 0, :] += truth_shapes[:, [mode]] * coordinate

    parameters = inspect.signature(era).parameters
    kwargs: dict[str, Any] = {}
    if "fs" in parameters:
        kwargs["fs"] = fs
    elif "dt" in parameters:
        kwargs["dt"] = 1.0 / fs
    else:
        raise AssertionError("ERA exposes neither an fs nor dt sampling argument")
    if "order" in parameters:
        kwargs["order"] = 4
    if "block_rows" in parameters:
        kwargs["block_rows"] = 80
    elif "n_block_rows" in parameters:
        kwargs["n_block_rows"] = 80
    if "n_modes" in parameters:
        kwargs["n_modes"] = 2
    if "f_range" in parameters:
        kwargs["f_range"] = (1.0, 60.0)
    if "max_damping" in parameters:
        kwargs["max_damping"] = 0.2
    result = era(impulse, **kwargs)

    frequencies = np.asarray(result.freq_hz, dtype=float).reshape(-1)
    shapes_raw = getattr(result, "mode_shapes", getattr(result, "modes", None))
    if shapes_raw is None:
        raise AssertionError("ERA result has no mode shapes")
    shapes = np.asarray(shapes_raw)
    if frequencies.size < 2 or shapes.ndim != 2 or shapes.shape[1] != frequencies.size:
        raise AssertionError(
            f"ERA returned frequencies {frequencies.shape} and modes {shapes.shape}"
        )

    choices = [
        (i, j)
        for i in range(frequencies.size)
        for j in range(frequencies.size)
        if i != j
    ]
    first, second = min(
        choices,
        key=lambda pair: abs(frequencies[pair[0]] - truth_frequency[0])
        + abs(frequencies[pair[1]] - truth_frequency[1]),
    )
    selected = np.array([first, second])
    frequency_error = np.abs(frequencies[selected] - truth_frequency)
    matrix = np.asarray(mac_matrix(truth_shapes, shapes[:, selected]), dtype=float)
    paired_mac = np.diag(matrix)
    spectral_line = fs / n_samples
    if np.any(frequency_error > spectral_line * (1.0 + 1.0e-10)):
        raise AssertionError(
            f"ERA frequency errors {frequency_error.tolist()} exceed df={spectral_line:g} Hz"
        )
    if np.any(paired_mac < 0.99):
        raise AssertionError(f"ERA paired MAC values are {paired_mac.tolist()}")
    return {
        "frequencies_hz": frequencies[selected].tolist(),
        "frequency_error_hz": frequency_error.tolist(),
        "spectral_line_hz": spectral_line,
        "paired_mac": paired_mac.tolist(),
    }


def probe_expanded_mac() -> dict[str, Any]:
    from femtools.correlation.expansion import expanded_mac

    modes, _ = np.linalg.qr(RNG.standard_normal((12, 3)), mode="reduced")
    master = np.array([0, 2, 4, 6, 8, 10])
    measured = modes[master]
    result = expanded_mac(measured, modes, master)
    if hasattr(result, "mac"):
        matrix = np.asarray(result.mac, dtype=float)
    elif isinstance(result, tuple):
        matrix = np.asarray(result[0], dtype=float)
    else:
        matrix = np.asarray(result, dtype=float)
    expected = np.eye(3)
    error = (
        float(np.max(np.abs(matrix - expected)))
        if matrix.shape == expected.shape
        else float("inf")
    )
    if error > 1.0e-10:
        raise AssertionError(
            f"expanded MAC identity has shape {matrix.shape} and error {error:.3e}"
        )
    return {"shape": list(matrix.shape), "max_identity_error": error}


def probe_residual_flexibility() -> dict[str, Any]:
    import scipy.linalg

    from femtools.dynamics.frf import direct_frf, modal_frf
    from femtools.dynamics.modal import ModalModel
    from femtools.dynamics.residuals import residual_flexibility

    n_dof = 8
    difference = np.eye(n_dof)
    difference[1:, :-1] -= np.eye(n_dof - 1)
    stiffness = 4.0e5 * (difference.T @ difference)
    mass = np.diag(np.linspace(1.0, 1.7, n_dof))
    eigenvalues, modes = scipy.linalg.eigh(stiffness, mass)
    frequencies = np.sqrt(eigenvalues) / (2.0 * np.pi)
    full = ModalModel(
        freq_hz=frequencies,
        eigenvalues=eigenvalues,
        modes=modes,
        generalized_mass=np.ones(n_dof),
    )
    truncated = full.truncate(n_modes=2)
    drive = n_dof - 1

    parameters = inspect.signature(residual_flexibility).parameters
    positional = [stiffness]
    names = [name.lower() for name in parameters]
    if len(names) > 1 and names[1] in {"m", "mass"}:
        positional.append(mass)
    positional.append(truncated)
    kwargs: dict[str, Any] = {}
    if "force_dofs" in parameters:
        kwargs["force_dofs"] = [drive]
    elif "inputs" in parameters:
        kwargs["inputs"] = [drive]
    if "outputs" in parameters:
        kwargs["outputs"] = [drive]
    residual_result = residual_flexibility(*positional, **kwargs)
    residual = np.asarray(
        getattr(residual_result, "residual_flexibility", residual_result),
        dtype=float,
    )
    if residual.ndim == 0:
        residual = residual.reshape(1, 1)
    elif residual.shape == (n_dof,):
        residual = residual[[drive], None]
    elif residual.shape == (n_dof, 1):
        residual = residual[[drive], :]
    if residual.shape != (1, 1) or not np.all(np.isfinite(residual)):
        raise AssertionError(f"residual-flexibility block has shape {residual.shape}")

    frequency = np.linspace(
        0.2 * truncated.freq_hz[-1],
        0.8 * truncated.freq_hz[-1],
        240,
    )
    damping = {"eta": 0.02}
    exact = direct_frf(
        stiffness,
        mass,
        inputs=[drive],
        outputs=[drive],
        freq_hz=frequency,
        damping=damping,
    ).H
    plain = modal_frf(
        truncated,
        inputs=[drive],
        outputs=[drive],
        freq_hz=frequency,
        damping=damping,
    ).H
    corrected = modal_frf(
        truncated,
        inputs=[drive],
        outputs=[drive],
        freq_hz=frequency,
        damping=damping,
        upper_residual=residual,
    ).H
    plain_error = float(np.linalg.norm(plain - exact) / np.linalg.norm(exact))
    corrected_error = float(np.linalg.norm(corrected - exact) / np.linalg.norm(exact))
    if not corrected_error < plain_error:
        raise AssertionError(
            f"residual flexibility did not improve L2 error "
            f"({plain_error:.3e} -> {corrected_error:.3e})"
        )
    return {
        "truncated_modes": truncated.n_modes,
        "plain_relative_l2": plain_error,
        "corrected_relative_l2": corrected_error,
        "improvement_factor": plain_error / corrected_error,
    }


def probe_ctetra10_bdf_roundtrip() -> dict[str, Any]:
    from femtools.fea.elements import tet10  # noqa: F401
    from femtools.io.bdf import read_bdf, write_bdf

    model = _tet10_model("ctetra10-bdf-boundary-probe")
    with TemporaryDirectory(prefix="femtools-ctetra10-bdf-probe-") as tmp:
        path = Path(tmp) / "tet10.bdf"
        try:
            write_bdf(model, path)
        except Exception as exc:
            message = str(exc).upper()
            if "TET10" in message and "BDF" in message and "MAPPING" in message:
                raise ProbeUnavailable("CTETRA10 BDF writer is not available") from exc
            raise
        text = path.read_text(encoding="utf-8").upper()
        loaded = read_bdf(path)

    if "CTETRA" not in text:
        raise AssertionError("TET10 BDF export contains no CTETRA card")
    element = loaded.elements.get(1)
    if element is None:
        raise AssertionError("CTETRA10 BDF round-trip lost element 1")
    if str(element.type).upper() == "TET4" and tuple(element.nodes) == (1, 2, 3, 4):
        raise ProbeUnavailable("CTETRA10 BDF reader still drops midside nodes")
    if str(element.type).upper() != "TET10" or tuple(element.nodes) != tuple(range(1, 11)):
        raise AssertionError(
            f"CTETRA10 round-trip produced {element.type} connectivity {element.nodes}"
        )
    return {
        "nodes": len(loaded.nodes),
        "element_type": str(element.type),
        "connectivity": list(element.nodes),
    }


def probe_read_pch_stress() -> dict[str, Any]:
    from femtools.io.pch import read_pch_stress

    expected_ids = [11, 22]
    expected = np.array(
        [
            [100.0, 200.0, 300.0, 10.0, 20.0, 30.0],
            [-40.0, 50.0, -60.0, 7.0, -8.0, 9.0],
        ]
    )
    punch = """\
$TITLE = FEMTOOLS STRESS BOUNDARY PROBE
$ELEMENT STRESSES
$REAL OUTPUT
$SUBCASE ID = 1
      11  1.000000E+02  2.000000E+02  3.000000E+02
-CONT-   1.000000E+01  2.000000E+01  3.000000E+01
      22 -4.000000E+01  5.000000E+01 -6.000000E+01
-CONT-   7.000000E+00 -8.000000E+00  9.000000E+00
"""
    with TemporaryDirectory(prefix="femtools-pch-stress-probe-") as tmp:
        path = Path(tmp) / "stress.pch"
        path.write_text(punch, encoding="utf-8")
        loaded = read_pch_stress(path)

    ids = getattr(loaded, "element_ids", getattr(loaded, "ids", None))
    values = getattr(
        loaded,
        "stress",
        getattr(loaded, "stresses", getattr(loaded, "values", getattr(loaded, "data", None))),
    )
    if ids is None or values is None:
        raise AssertionError("punch stress result lacks element_ids/stress fields")
    actual_ids = [int(value) for value in np.asarray(ids).reshape(-1)]
    actual = np.asarray(values, dtype=float)
    if actual.ndim == 3 and actual.shape[-1] == 1:
        actual = actual[..., 0]
    error = (
        float(np.max(np.abs(actual - expected)))
        if actual.shape == expected.shape
        else float("inf")
    )
    if actual_ids != expected_ids or error > 1.0e-12:
        raise AssertionError(
            f"punch stresses returned ids {actual_ids}, shape {actual.shape}, error {error:.3e}"
        )
    return {
        "element_ids": actual_ids,
        "shape": list(actual.shape),
        "max_absolute_error": error,
    }


PROBES: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("core_model", probe_core_model),
    ("mac", probe_mac),
    ("eigen", probe_eigen),
    ("modal_frf", probe_modal_frf),
    ("mock_unv", probe_mock_unv),
    ("guyan", probe_guyan),
    ("h1", probe_h1),
    ("read_inp", probe_read_inp),
    ("read_k", probe_read_k),
    ("nmd", probe_nmd),
    ("ssi_data", probe_ssi_data),
    ("parameter_covariance", probe_parameter_covariance),
    ("modal_strain_energy", probe_modal_strain_energy),
    ("recover_stress", probe_recover_stress),
    ("apply_rbe3", probe_apply_rbe3),
    ("average_nodal", probe_average_nodal),
    ("write_cdb", probe_write_cdb),
    ("write_k", probe_write_k),
    ("map_nearest_nodes", probe_map_nearest_nodes),
    ("topometry_optimize", probe_topometry_optimize),
    ("nastran_punch_driver", probe_nastran_punch_driver),
    ("ansys_cdb_driver", probe_ansys_cdb_driver),
    ("abaqus_inp_driver", probe_abaqus_inp_driver),
    ("dump_frf", probe_dump_frf),
    ("plot_stress", probe_plot_stress),
    ("update_from_static", probe_update_from_static),
    ("mapped_mode_matrix", probe_mapped_mode_matrix),
    ("apply_mpc", probe_apply_mpc),
    ("static_stress_response", probe_static_stress_response),
    ("dump_psd", probe_dump_psd),
    ("mapped_mac", probe_mapped_mac),
    ("nastran_static", probe_nastran_static),
    ("tet10_patch", probe_tet10_patch),
    ("recover_spr", probe_recover_spr),
    ("era", probe_era),
    ("expanded_mac", probe_expanded_mac),
    ("residual_flexibility", probe_residual_flexibility),
    ("ctetra10_bdf_roundtrip", probe_ctetra10_bdf_roundtrip),
    ("read_pch_stress", probe_read_pch_stress),
)


def main() -> int:
    results: list[dict[str, Any]] = []
    failures = 0
    for name, probe in PROBES:
        try:
            details = probe()
        except (ModuleNotFoundError, ImportError, ProbeUnavailable) as exc:
            results.append({"probe": name, "status": "skip", "reason": str(exc)})
        except Exception as exc:  # Deliberately summarize each independent boundary.
            failures += 1
            results.append(
                {
                    "probe": name,
                    "status": "fail",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            results.append({"probe": name, "status": "pass", "details": details})

    print(json.dumps(results, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

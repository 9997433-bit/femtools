#!/usr/bin/env python3
"""Probe frozen API boundaries without requiring every package module to exist.

Missing femtools modules are reported as skips and never produce a failing exit
status. Implemented modules are checked against deterministic contract cases.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
RNG = np.random.default_rng(0)


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
)


def main() -> int:
    results: list[dict[str, Any]] = []
    failures = 0
    for name, probe in PROBES:
        try:
            details = probe()
        except (ModuleNotFoundError, ImportError) as exc:
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

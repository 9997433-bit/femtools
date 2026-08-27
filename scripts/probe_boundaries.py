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
    from femtools.fea.assemble import assemble_km
    from femtools.fea.eigen import solve_modes

    from benchmarks.bench_eigen import build_axial_chain

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


PROBES: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("core_model", probe_core_model),
    ("mac", probe_mac),
    ("eigen", probe_eigen),
    ("modal_frf", probe_modal_frf),
    ("mock_unv", probe_mock_unv),
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

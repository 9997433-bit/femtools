"""Sparse eigen-solver benchmark on a constrained axial chain.

Run from the repository root, for example::

    python benchmarks/bench_eigen.py --sizes 32 128 512 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_axial_chain(n_elements: int) -> Any:
    """Build a stable BAR2 chain with one unconstrained DOF per free node."""
    if n_elements < 2:
        raise ValueError("n_elements must be at least 2")

    from femtools.core.model import FEModel

    model = FEModel(name=f"axial-chain-{n_elements}")
    for node_id in range(1, n_elements + 2):
        model.add_node(id=node_id, xyz=(float(node_id - 1), 0.0, 0.0))

    model.add_material(id=1, type="isotropic", E=210.0e9, nu=0.3, rho=7850.0)
    model.add_property(id=1, type="bar", material_id=1, A=1.0e-4)
    for element_id in range(1, n_elements + 1):
        model.add_element(
            id=element_id,
            type="BAR2",
            nodes=(element_id, element_id + 1),
            property_id=1,
        )

    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    transverse_and_rotations = (False, True, True, True, True, True)
    for node_id in range(2, n_elements + 2):
        model.add_spc(node_id=node_id, mask=transverse_and_rotations)
    return model


def run_case(n_elements: int, n_modes: int = 8, repeat: int = 3) -> dict[str, Any]:
    """Time ``solve_modes`` and return machine-readable measurements."""
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if repeat < 1:
        raise ValueError("repeat must be positive")

    from femtools.fea.eigen import solve_modes

    model = build_axial_chain(n_elements)
    requested_modes = min(n_modes, n_elements - 1)

    # Warm sparse imports, matrix assembly, and ARPACK before measuring.
    solve_modes(model, n_modes=requested_modes, shift=0.0)

    samples: list[float] = []
    modal = None
    for _ in range(repeat):
        started = time.perf_counter()
        modal = solve_modes(model, n_modes=requested_modes, shift=0.0)
        samples.append(time.perf_counter() - started)

    assert modal is not None
    frequencies = np.asarray(modal.freq_hz, dtype=float)
    if frequencies.shape != (requested_modes,):
        raise RuntimeError(
            f"solve_modes returned {frequencies.shape}, expected {(requested_modes,)}"
        )
    if not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) < 0.0):
        raise RuntimeError("solve_modes returned non-finite or unsorted frequencies")

    return {
        "benchmark": "eigen",
        "n_elements": n_elements,
        "n_modes": requested_modes,
        "seconds_min": min(samples),
        "seconds_median": statistics.median(samples),
        "first_frequency_hz": float(frequencies[0]),
        "last_frequency_hz": float(frequencies[-1]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = [run_case(size, args.modes, args.repeat) for size in args.sizes]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

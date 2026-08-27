"""Modal Assurance Criterion benchmark with deterministic orthonormal modes.

Run from the repository root, for example::

    python benchmarks/bench_mac.py --dofs 1000 10000 50000 --modes 16
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


def make_modes(n_dof: int, n_modes: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic, column-orthonormal modal basis."""
    if n_dof < n_modes:
        raise ValueError("n_dof must be greater than or equal to n_modes")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    rng = np.random.default_rng(seed)
    modes, _ = np.linalg.qr(rng.standard_normal((n_dof, n_modes)), mode="reduced")
    return modes


def run_case(
    n_dof: int,
    n_modes: int = 16,
    repeat: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Time a square MAC calculation and check its numerical invariants."""
    if repeat < 1:
        raise ValueError("repeat must be positive")

    from femtools.correlation.mac import mac_matrix

    modes = make_modes(n_dof, n_modes, seed)
    mac_matrix(modes, modes)  # Warm up imports and BLAS dispatch.

    samples: list[float] = []
    mac = None
    for _ in range(repeat):
        started = time.perf_counter()
        mac = np.asarray(mac_matrix(modes, modes))
        samples.append(time.perf_counter() - started)

    assert mac is not None
    if mac.shape != (n_modes, n_modes):
        raise RuntimeError(f"mac_matrix returned {mac.shape}, expected {(n_modes, n_modes)}")
    diagonal_error = float(np.max(np.abs(np.diag(mac) - 1.0)))
    off_diagonal = mac - np.diag(np.diag(mac))
    off_diagonal_max = float(np.max(np.abs(off_diagonal)))
    if diagonal_error > 1.0e-12 or off_diagonal_max > 1.0e-10:
        raise RuntimeError("self-MAC violated the orthonormal-basis contract")

    return {
        "benchmark": "mac",
        "n_dof": n_dof,
        "n_modes": n_modes,
        "seconds_min": min(samples),
        "seconds_median": statistics.median(samples),
        "diagonal_error": diagonal_error,
        "off_diagonal_max": off_diagonal_max,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dofs", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = [
        run_case(size, args.modes, args.repeat, args.seed) for size in args.dofs
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

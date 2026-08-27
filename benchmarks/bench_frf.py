"""Modal FRF benchmark over increasing frequency-grid sizes.

Run from the repository root, for example::

    python benchmarks/bench_frf.py --frequencies 256 2048 8192
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_modal_data(n_dof: int, n_modes: int, seed: int = 0) -> Any:
    """Build the documented ModalResult-shaped input without solving a model."""
    if n_dof < n_modes:
        raise ValueError("n_dof must be greater than or equal to n_modes")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")

    rng = np.random.default_rng(seed)
    modes, _ = np.linalg.qr(rng.standard_normal((n_dof, n_modes)), mode="reduced")
    freq_hz = np.geomspace(5.0, 2_000.0, n_modes)
    return SimpleNamespace(
        freq_hz=freq_hz,
        eigenvalues=np.square(2.0 * np.pi * freq_hz),
        modes=modes,
        generalized_mass=np.ones(n_modes),
    )


def response_array(result: Any) -> np.ndarray:
    """Extract the complex response array from common FRFResult containers."""
    if isinstance(result, np.ndarray):
        return result
    for field in ("H", "frf", "response", "values", "data"):
        if hasattr(result, field):
            candidate = np.asarray(getattr(result, field))
            if candidate.ndim == 3:
                return candidate
    raise TypeError("FRFResult does not expose its response array")


def run_case(
    n_frequency: int,
    n_dof: int = 256,
    n_modes: int = 32,
    n_channels: int = 4,
    repeat: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Time ``modal_frf`` for a deterministic synthetic modal basis."""
    if n_frequency < 2:
        raise ValueError("n_frequency must be at least 2")
    if not 1 <= n_channels <= n_dof:
        raise ValueError("n_channels must be between 1 and n_dof")
    if repeat < 1:
        raise ValueError("repeat must be positive")

    from femtools.dynamics.frf import modal_frf

    modal = make_modal_data(n_dof, n_modes, seed)
    channels = np.arange(n_channels, dtype=int)
    freq_hz = np.linspace(1.0, 2_200.0, n_frequency)
    damping = np.full(n_modes, 0.01)

    modal_frf(modal, channels, channels, freq_hz, damping)

    samples: list[float] = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = modal_frf(modal, channels, channels, freq_hz, damping)
        samples.append(time.perf_counter() - started)

    response = response_array(result)
    expected_shape = (n_channels, n_channels, n_frequency)
    if response.shape != expected_shape:
        raise RuntimeError(f"modal_frf returned {response.shape}, expected {expected_shape}")
    if not np.iscomplexobj(response) or not np.all(np.isfinite(response)):
        raise RuntimeError("modal_frf returned a non-complex or non-finite response")

    return {
        "benchmark": "frf",
        "n_frequency": n_frequency,
        "n_dof": n_dof,
        "n_modes": n_modes,
        "n_inputs": n_channels,
        "n_outputs": n_channels,
        "seconds_min": min(samples),
        "seconds_median": statistics.median(samples),
        "response_norm": float(np.linalg.norm(response)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies", nargs="+", type=int, default=[256, 2_048, 8_192]
    )
    parser.add_argument("--dofs", type=int, default=256)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = [
        run_case(
            size,
            n_dof=args.dofs,
            n_modes=args.modes,
            n_channels=args.channels,
            repeat=args.repeat,
            seed=args.seed,
        )
        for size in args.frequencies
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

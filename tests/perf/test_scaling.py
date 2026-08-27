"""Opt-in scaling smoke tests for the principal numerical kernels."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import pytest

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        os.environ.get("FEMTOOLS_PERF") != "1",
        reason="set FEMTOOLS_PERF=1 to run performance tests",
    ),
]


def _check_growth(
    results: Sequence[dict[str, Any]],
    work_key: str,
    *,
    multiplier: float = 12.0,
) -> None:
    """Reject catastrophic growth while tolerating noisy shared CI workers."""
    assert len(results) >= 2
    first, last = results[0], results[-1]
    first_time = max(float(first["seconds_median"]), 1.0e-6)
    last_time = float(last["seconds_median"])
    work_ratio = float(last[work_key]) / float(first[work_key])
    assert first_time > 0.0
    assert last_time > 0.0
    assert last_time / first_time <= multiplier * work_ratio + 2.0


def test_mac_scales_with_dof_count(record_property: Any) -> None:
    from benchmarks.bench_mac import run_case

    results = [
        run_case(n_dof, n_modes=12, repeat=3)
        for n_dof in (10_000, 40_000)
    ]
    _check_growth(results, "n_dof")
    record_property("mac_scaling", json.dumps(results, sort_keys=True))


def test_modal_frf_scales_with_frequency_count(record_property: Any) -> None:
    from benchmarks.bench_frf import run_case

    results = [
        run_case(n_frequency, n_dof=128, n_modes=20, n_channels=4, repeat=3)
        for n_frequency in (256, 4_096)
    ]
    _check_growth(results, "n_frequency")
    record_property("frf_scaling", json.dumps(results, sort_keys=True))


def test_sparse_eigen_scales_with_chain_size(record_property: Any) -> None:
    from benchmarks.bench_eigen import run_case

    results = [
        run_case(n_elements, n_modes=6, repeat=2)
        for n_elements in (32, 128)
    ]
    _check_growth(results, "n_elements")
    record_property("eigen_scaling", json.dumps(results, sort_keys=True))

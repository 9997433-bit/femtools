"""Small, fast SIMP topology-optimization smoke test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

topology_module = pytest.importorskip("femtools.optimization.topology")


def _density_array(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, tuple) and result:
        return _density_array(result[0])
    for attribute in ("density", "densities", "x", "design"):
        if hasattr(result, attribute):
            return np.asarray(getattr(result, attribute))
    raise AssertionError("topology_simp must expose the optimized element densities")


def test_topology_simp_small_mesh_smoke() -> None:
    nelx, nely, volume_fraction = 8, 4, 0.5

    result = topology_module.topology_simp(
        nelx=nelx,
        nely=nely,
        volfrac=volume_fraction,
        penal=3.0,
        rmin=1.5,
        max_iter=4,
    )
    density = np.asarray(_density_array(result), dtype=float)

    assert density.size == nelx * nely
    assert np.all(np.isfinite(density))
    assert np.all(density >= -1.0e-12)
    assert np.all(density <= 1.0 + 1.0e-12)
    assert density.mean() <= volume_fraction + 0.02

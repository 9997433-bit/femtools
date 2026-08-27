"""Round-trip preservation tests for the three contracted model formats."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _assert_basic_model_equal(
    expected: Any,
    actual: Any,
    *,
    elements: bool = True,
    name: bool = True,
) -> None:
    if name:
        assert actual.name == expected.name
    assert set(actual.nodes) == set(expected.nodes)
    for node_id in expected.nodes:
        np.testing.assert_allclose(
            actual.nodes[node_id].xyz,
            expected.nodes[node_id].xyz,
            rtol=0.0,
            atol=1.0e-12,
        )
    if elements:
        assert set(actual.elements) == set(expected.elements)
        for element_id in expected.elements:
            assert tuple(actual.elements[element_id].nodes) == tuple(
                expected.elements[element_id].nodes
            )


def _roundtrip(
    writer: Callable[[Any, Path], Any],
    reader: Callable[[Path], Any],
    model: Any,
    path: Path,
) -> Any:
    writer(model, path)
    assert path.exists()
    loaded = reader(path)
    return getattr(loaded, "model", loaded)


def test_project_roundtrip_preserves_model(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    project_module = pytest.importorskip("femtools.io.project")
    model, _ = axial_bar
    path = tmp_path / "axial.ftproj"

    loaded = _roundtrip(
        project_module.save_project,
        project_module.load_project,
        model,
        path,
    )

    _assert_basic_model_equal(model, loaded)
    assert set(loaded.materials) == set(model.materials)
    assert set(loaded.properties) == set(model.properties)
    assert len(loaded.spcs) == len(model.spcs)


def test_bdf_roundtrip_preserves_bar_connectivity(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    bdf_module = pytest.importorskip("femtools.io.bdf")
    model, _ = axial_bar
    path = tmp_path / "axial.bdf"

    loaded = _roundtrip(bdf_module.write_bdf, bdf_module.read_bdf, model, path)

    _assert_basic_model_equal(model, loaded, name=False)


def test_unv_roundtrip_preserves_node_dataset(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    unv_module = pytest.importorskip("femtools.io.unv")
    model, _ = axial_bar
    path = tmp_path / "axial.unv"

    loaded = _roundtrip(unv_module.write_unv, unv_module.read_unv, model, path)

    # Round 1 UNV explicitly contracts node datasets; element dataset 2412 is not included.
    _assert_basic_model_equal(model, loaded, elements=False)

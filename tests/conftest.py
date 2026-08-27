"""Shared deterministic finite-element models for the public API tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Keep the suite runnable from an unpacked source tree as well as an installed wheel.
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


@pytest.fixture
def axial_bar() -> tuple[Any, dict[str, float]]:
    """A one-element bar with exactly one unconstrained axial DOF."""
    model_module = pytest.importorskip("femtools.core.model")
    model = model_module.FEModel(name="axial-bar-golden")

    data = {"E": 210.0e9, "rho": 7800.0, "A": 2.5e-4, "L": 1.7}
    model.add_node(id=1, xyz=(0.0, 0.0, 0.0))
    model.add_node(id=2, xyz=(data["L"], 0.0, 0.0))
    model.add_material(
        id=1,
        type="isotropic",
        E=data["E"],
        nu=0.3,
        rho=data["rho"],
    )
    model.add_property(id=1, type="bar", material_id=1, A=data["A"])
    model.add_element(id=1, type="BAR2", nodes=(1, 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    model.add_spc(node_id=2, mask=(False, True, True, True, True, True))
    return model, data


@pytest.fixture
def cantilever() -> tuple[Any, dict[str, float]]:
    """A slender circular Euler beam represented by sixteen BEAM2 elements."""
    model_module = pytest.importorskip("femtools.core.model")
    model = model_module.FEModel(name="cantilever-golden")

    data = {
        "E": 70.0e9,
        "rho": 2700.0,
        "L": 2.0,
        "A": 8.0e-4,
        "I": 5.0e-8,
        "J": 1.0e-7,
        "n_elements": 16,
    }
    for index in range(data["n_elements"] + 1):
        x = data["L"] * index / data["n_elements"]
        model.add_node(id=index + 1, xyz=(x, 0.0, 0.0))

    model.add_material(
        id=1,
        type="isotropic",
        E=data["E"],
        nu=0.3,
        rho=data["rho"],
    )
    model.add_property(
        id=1,
        type="beam",
        material_id=1,
        A=data["A"],
        Iy=data["I"],
        Iz=data["I"],
        J=data["J"],
    )
    for index in range(data["n_elements"]):
        model.add_element(
            id=index + 1,
            type="BEAM2",
            nodes=(index + 1, index + 2),
            property_id=1,
        )
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model, data


def positive_frequencies(modal: Any, count: int | None = None) -> np.ndarray:
    """Return finite, positive modal frequencies from a ModalResult."""
    frequencies = np.asarray(modal.freq_hz, dtype=float)
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 1.0e-9)]
    frequencies.sort()
    return frequencies if count is None else frequencies[:count]

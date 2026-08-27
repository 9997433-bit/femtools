"""Cross-check modal superposition against direct dynamic stiffness."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

frf_module = pytest.importorskip("femtools.dynamics.frf")


def _frf_array(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result
    for attribute in ("H", "frf", "data"):
        if hasattr(result, attribute):
            return np.asarray(getattr(result, attribute))
    raise AssertionError("FRFResult must expose its complex response array")


def _direct_response(
    stiffness: sparse.csr_matrix,
    mass: sparse.csr_matrix,
    damping_matrix: sparse.csr_matrix,
    inputs: list[int],
    outputs: list[int],
    frequencies: np.ndarray,
    modal_damping: np.ndarray,
) -> Any:
    """Accommodate either an explicit C matrix or the documented damping argument."""
    parameters = inspect.signature(frf_module.direct_frf).parameters
    explicit_damping_matrix = "C" in parameters or "c" in parameters
    values: dict[str, Any] = {
        "K": stiffness,
        "k": stiffness,
        "M": mass,
        "m": mass,
        "C": damping_matrix,
        "c": damping_matrix,
        "inputs": inputs,
        "outputs": outputs,
        "freq_hz": frequencies,
        "frequencies": frequencies,
        "damping": None if explicit_damping_matrix else modal_damping,
    }
    kwargs = {name: values[name] for name in parameters if name in values}
    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    if not required.issubset(kwargs):
        pytest.fail(f"Unsupported direct_frf signature: {inspect.signature(frf_module.direct_frf)}")
    return frf_module.direct_frf(**kwargs)


def test_modal_and_direct_frf_agree_for_complete_basis() -> None:
    rng = np.random.default_rng(9182)
    n_modes = 20
    modes, _ = np.linalg.qr(rng.standard_normal((n_modes, n_modes)))
    natural_hz = np.linspace(25.0, 250.0, n_modes)
    omega = 2.0 * np.pi * natural_hz
    modal_damping = np.full(n_modes, 0.0075)

    mass = sparse.eye(n_modes, format="csr")
    stiffness = sparse.csr_matrix((modes * omega**2) @ modes.T)
    damping_matrix = sparse.csr_matrix((modes * (2.0 * modal_damping * omega)) @ modes.T)
    modal = SimpleNamespace(
        freq_hz=natural_hz,
        eigenvalues=omega**2,
        modes=modes,
        generalized_mass=np.ones(n_modes),
    )
    frequencies = np.linspace(0.2 * natural_hz[-1], 0.8 * natural_hz[-1], 240)
    inputs = [0, 7]
    outputs = [3, 12]

    modal_result = frf_module.modal_frf(
        modal,
        inputs=inputs,
        outputs=outputs,
        freq_hz=frequencies,
        damping=modal_damping,
    )
    direct_result = _direct_response(
        stiffness,
        mass,
        damping_matrix,
        inputs,
        outputs,
        frequencies,
        modal_damping,
    )
    modal_array = _frf_array(modal_result)
    direct_array = _frf_array(direct_result)

    assert modal_array.shape == (len(outputs), len(inputs), frequencies.size)
    assert direct_array.shape == modal_array.shape
    relative_l2 = np.linalg.norm(modal_array - direct_array) / np.linalg.norm(direct_array)
    assert relative_l2 < 0.05

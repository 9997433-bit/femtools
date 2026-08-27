"""Cross-check modal superposition against direct dynamic stiffness."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from femtools.dynamics import frf as frf_module
from femtools.fea.assemble import AssemblyResult, assemble_km
from femtools.fea.eigen import solve_modes


def _frf_array(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result
    for attribute in ("H", "frf", "data"):
        if hasattr(result, attribute):
            return np.asarray(getattr(result, attribute))
    raise AssertionError("FRFResult must expose its complex response array")


def _free_index(assembly: AssemblyResult, node_id: int, component: int) -> int:
    global_index = assembly.dof_map.index(node_id, component)
    free_position = np.flatnonzero(assembly.free_dof == global_index)
    assert free_position.size == 1
    return int(free_position[0])


def test_truncated_modal_frf_matches_direct_in_retained_band(
    cantilever: tuple[object, dict[str, float]],
) -> None:
    model, data = cantilever
    n_retained = 20
    assembly = assemble_km(model)
    solved = solve_modes(model, n_modes=n_retained + 1, assembly=assembly)
    retained = SimpleNamespace(
        freq_hz=solved.freq_hz[:n_retained],
        eigenvalues=solved.eigenvalues[:n_retained],
        modes=solved.modes[assembly.free_dof, :n_retained],
        generalized_mass=solved.generalized_mass[:n_retained],
    )

    # CONTRACT_API defines fmax as the last retained mode, not the parent system's
    # highest mode. The direct solve still contains every free physical DOF.
    fmax = float(retained.freq_hz[-1])
    frequencies = np.linspace(0.2 * fmax, 0.8 * fmax, 300)
    omega_first, omega_last = 2.0 * np.pi * retained.freq_hz[[0, -1]]
    zeta = 0.01
    damping = {
        "alpha": 2.0 * zeta * omega_first * omega_last / (omega_first + omega_last),
        "beta": 2.0 * zeta / (omega_first + omega_last),
    }
    tip_node = int(data["n_elements"]) + 1
    mid_node = int(data["n_elements"]) // 2 + 1
    inputs = [_free_index(assembly, tip_node, 1)]
    outputs = [
        _free_index(assembly, tip_node, 1),
        _free_index(assembly, mid_node, 1),
    ]

    modal_result = frf_module.modal_frf(
        retained,
        inputs=inputs,
        outputs=outputs,
        freq_hz=frequencies,
        damping=damping,
    )
    direct_result = frf_module.direct_frf(
        assembly.Kff,
        assembly.Mff,
        inputs=inputs,
        outputs=outputs,
        freq_hz=frequencies,
        damping=damping,
    )
    modal_array = _frf_array(modal_result)
    direct_array = _frf_array(direct_result)

    assert n_retained < assembly.Kff.shape[0]
    assert solved.freq_hz[n_retained] > fmax
    assert modal_array.shape == (len(outputs), len(inputs), frequencies.size)
    assert direct_array.shape == modal_array.shape
    relative_l2 = np.linalg.norm(modal_array - direct_array, axis=(1, 2))
    relative_l2 /= np.linalg.norm(direct_array, axis=(1, 2))
    assert np.all(relative_l2 < 0.05), relative_l2

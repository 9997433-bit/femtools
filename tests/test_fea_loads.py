"""FEA load-vector construction from ``FEModel.Load`` records."""

from __future__ import annotations

import numpy as np
import pytest

from femtools.fea import assemble_km, build_load_vector, solve_static


def test_build_load_vector_reads_model_force_and_moment(
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, data = axial_bar
    force = 1.25e3
    model.add_load(node_id=2, force=(force, 0.0, 0.0), moment=(0.0, 0.0, 4.0))

    asm = assemble_km(model)
    f = build_load_vector(None, asm.dof_map, model=model)
    ux = asm.dof_map.index(2, 0)
    rz = asm.dof_map.index(2, 5)
    assert f[ux] == pytest.approx(force)
    assert f[rz] == pytest.approx(4.0)
    assert np.count_nonzero(f) == 2


def test_solve_static_uses_model_loads_for_axial_bar(
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, data = axial_bar
    force = 2.0e3
    model.add_load(node_id=2, force=(force, 0.0, 0.0))

    static = solve_static(model)
    expected = force * data["L"] / (data["E"] * data["A"])
    assert static.u[static.dof_map.index(2, 0)] == pytest.approx(expected, rel=1e-12, abs=0.0)

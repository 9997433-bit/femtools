"""Round 5 (R5-O1) regressions for the FEA kernel.

Two findings from the round-5 audit of ``femtools.fea``:

* ``solve_static(enforced=...)`` silently corrupted the whole displacement
  field when the driven DOF was one the assembler had left free;
* a flat shell mesh whose normal is not a global axis keeps a fictitious
  drilling mechanism, which now raises a warning instead of surfacing as an
  unexplained seventh rigid body mode.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from femtools.fea import assemble_km, solve_static
from femtools.fea.verification import (
    shell_drilling_orientation_gap,
    shell_plate,
)

SPRING_K = 1000.0


def spring_chain() -> dict:
    """Three equal springs in series along x, node 1 grounded."""
    return {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3, 4)},
        "elements": {
            i: {"type": "SPRING", "nodes": (i, i + 1), "k": SPRING_K, "c1": 0}
            for i in (1, 2, 3)
        },
        "materials": {},
        "properties": {},
        "spcs": [{"node_id": 1, "dofs": (0,)}],
    }


def axial(u: np.ndarray, assembly) -> np.ndarray:
    return np.array([u[assembly.dof_map.index(nid, 0)] for nid in (1, 2, 3, 4)])


def test_enforced_displacement_on_an_unconstrained_dof() -> None:
    """Driving a free DOF must hold it *and* keep the rest in equilibrium.

    Node 3 is pulled to 30 mm with no external load, so the chain is a linear
    ramp up to node 3 and rigid beyond it.  Before the fix the driven DOF was
    also solved as if free and the enforced value was written over the answer,
    which left every other DOF at zero.
    """
    model = spring_chain()
    asm = assemble_km(model)
    u = solve_static(model, {}, assembly=asm, enforced={(3, 0): 0.03})

    np.testing.assert_allclose(axial(u, asm), [0.0, 0.015, 0.03, 0.03], atol=1.0e-12)


def test_enforced_zero_holds_a_free_dof_at_zero() -> None:
    model = spring_chain()
    asm = assemble_km(model)
    u = solve_static(model, {(4, 0): 60.0}, assembly=asm, enforced={(3, 0): 0.0})

    np.testing.assert_allclose(axial(u, asm), [0.0, 0.0, 0.0, 0.06], atol=1.0e-12)


def test_enforced_free_dof_reports_its_reaction() -> None:
    """The force needed to hold the driven DOF, and global equilibrium."""
    model = spring_chain()
    asm = assemble_km(model)
    u, r = solve_static(
        model, {}, assembly=asm, enforced={(3, 0): 0.03}, return_reactions=True
    )

    np.testing.assert_allclose(axial(r, asm), [-15.0, 0.0, 15.0, 0.0], atol=1.0e-9)
    assert r.sum() == pytest.approx(0.0, abs=1.0e-9)
    residual = asm.K @ u
    still_free = np.setdiff1d(asm.free_dof, [asm.dof_map.index(3, 0)])
    np.testing.assert_allclose(residual[still_free], 0.0, atol=1.0e-9)


def test_enforced_dof_across_multiple_load_cases() -> None:
    model = spring_chain()
    asm = assemble_km(model)
    f = np.zeros((asm.n_dof, 2))
    f[asm.dof_map.index(4, 0), 0] = 30.0
    f[asm.dof_map.index(2, 0), 1] = 30.0

    both = solve_static(model, f, assembly=asm, enforced={(3, 0): 0.03})
    first = solve_static(model, {(4, 0): 30.0}, assembly=asm, enforced={(3, 0): 0.03})
    second = solve_static(model, {(2, 0): 30.0}, assembly=asm, enforced={(3, 0): 0.03})

    np.testing.assert_allclose(both[:, 0], first, atol=1.0e-12)
    np.testing.assert_allclose(both[:, 1], second, atol=1.0e-12)


def test_enforced_on_a_constrained_dof_is_unchanged() -> None:
    """The pre-existing path: every enforced DOF is already SPC'd."""
    model = spring_chain()
    asm = assemble_km(model)
    u = solve_static(model, {}, assembly=asm, enforced={(1, 0): 0.009})

    np.testing.assert_allclose(axial(u, asm), 0.009, atol=1.0e-12)


@pytest.mark.parametrize("etype", ["QUAD4", "TRIA3"])
def test_axis_aligned_flat_shell_has_no_drilling_mechanism(etype: str) -> None:
    gap = shell_drilling_orientation_gap(etype)

    assert gap["aligned_zero_modes"] == 6.0
    assert gap["aligned_drilling_dof"] > 0.0
    assert gap["aligned_warned"] == 0.0


@pytest.mark.parametrize("etype", ["QUAD4", "TRIA3"])
def test_oblique_flat_shell_mechanism_is_reported(etype: str) -> None:
    """The mechanism cannot be eliminated DOF-wise, so it must at least warn.

    Only the count of zero frequencies differs: the elastic spectrum of the
    tilted plate is the aligned one to round-off, which is what makes the extra
    mode identifiable as fictitious rather than as a modelling error.
    """
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_zero_modes"] == gap["aligned_zero_modes"] + 1.0
    assert gap["oblique_drilling_dof"] == 0.0
    assert gap["oblique_warned"] == 1.0
    assert gap["oblique_first_elastic_hz"] == pytest.approx(
        gap["aligned_first_elastic_hz"], rel=1.0e-9
    )


def test_curved_and_clamped_shells_do_not_warn() -> None:
    """No false positives: the warning is for a retained *flat* mechanism only."""
    rotation = np.linalg.qr(
        np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
    )[0]
    quiet = {
        "aligned": shell_plate(3, 3),
        "aligned x-z plane": shell_plate(
            3, 3, rotation=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        ),
        "oblique but clamped": shell_plate(3, 3, rotation=rotation, clamped_edge=True),
        "folded": _folded_shell(),
    }
    for label, model in quiet.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assemble_km(model)
        assert not caught, f"{label} warned: {[str(c.message) for c in caught]}"


def _folded_shell(angle_deg: float = 35.0) -> dict:
    """Two flat panels meeting along a ridge: no common normal, no mechanism."""
    model = shell_plate(4, 2, side=2.0)
    theta = np.radians(angle_deg)
    for node in model["nodes"].values():
        x, y, _z = node["xyz"]
        if x > 1.0:
            node["xyz"] = (1.0 + (x - 1.0) * np.cos(theta), y, (x - 1.0) * np.sin(theta))
    return model

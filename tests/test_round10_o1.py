"""Round 10 (R10-O1): the quadratic tetrahedron and Zienkiewicz-Zhu patch recovery.

Two things land here and they are deliberately independent of each other.

``TET10`` is the ten-node quadratic tetrahedron -- four corners, one node at
the middle of each of the six edges, quadratic shape functions in volume
coordinates and therefore a *linear* strain field (Zienkiewicz & Taylor, *The
Finite Element Method*, 6th ed., §4/§9; Bathe, *Finite Element Procedures*,
§5.3; Cook et al., *Concepts and Applications of Finite Element Analysis*, 4th
ed., §6.4).  It is integrated on the classical four-point symmetric rule, which
is exact for a straight-edged element and is also the smallest rule that leaves
the 30x30 stiffness with rank 24, i.e. with six rigid body modes and no
mechanism.  The consistent mass needs more than that -- ``N_i N_j`` is quartic
-- and gets Keast's fifteen-point degree-five rule.

``recover_spr`` is superconvergent patch recovery (Zienkiewicz, O.C. and Zhu,
J.Z., *The superconvergent patch recovery and a posteriori error estimates.
Part 1: The recovery technique*, IJNME **33**\\ (7), 1992, pp. 1331-1364): a
linear polynomial fitted by least squares over the patch of elements meeting at
a node, sampled at their centroids -- the superconvergent points of a linear
element (Barlow, IJNME 10, 1976) -- and evaluated at the node.  It is a *new*
function; ``average_nodal`` stays the ``1 / n_a`` arithmetic mean it has always
been, and the last section of this module re-measures that as well as the HEX8,
MITC4 and tilted-shell goldens that must not move.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import femtools.fea as fea
from femtools.core.model import ELEMENT_NODE_COUNTS, FEModel
from femtools.fea import (
    assemble_km,
    available_elements,
    element_matrices,
    element_spec,
    solve_modes,
    solve_static,
)
from femtools.fea.elements import REGISTRY, element_info, tet10
from femtools.fea.elements.solid import TET10_CENTROID, TET10_EDGES, _tet10_shape, tet10_gradient
from femtools.fea.materials import solid_D
from femtools.fea.quadrature import tet_rule
from femtools.fea.recover import (
    NodalStressResult,
    StressResult,
    average_nodal,
    recover_spr,
    recover_strain,
    recover_stress,
)
from femtools.fea.verification import (
    PATCH_GRADIENT,
    PATCH_TYPES,
    hex8_bending_ratio,
    hex8_patch_test_error,
    hex8_rigid_body_frequencies,
    hex_cantilever,
    shell_drilling_orientation_gap,
    shell_plate,
    stress_patch_error,
    tet10_rigid_body_frequencies,
    tet_bending_ratio,
    tet_cantilever,
    tet_patch_model,
)

E, NU, RHO = 1.0e7, 0.3, 2700.0

#: An irregular tetrahedron: no right angles, no equal edges, nothing that
#: could make a wrong shape function look right by symmetry.
CORNERS = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.2, 1.1, 0.0], [0.4, 0.3, 1.2]])


def tet10_nodes(corners: np.ndarray = CORNERS) -> np.ndarray:
    """The ten node positions of a straight-edged ``TET10`` on *corners*."""
    corners = np.asarray(corners, dtype=float)
    midsides = [0.5 * (corners[a] + corners[b]) for a, b in TET10_EDGES]
    return np.vstack([corners, np.array(midsides)])


def one_tet10(points: np.ndarray | None = None, *, spcs: list | None = None) -> dict:
    """A single free-free ``TET10`` as a plain-dictionary model."""
    xyz = tet10_nodes() if points is None else np.asarray(points, dtype=float)
    return {
        "nodes": {i + 1: {"xyz": tuple(p)} for i, p in enumerate(xyz)},
        "elements": {1: {"type": "TET10", "property_id": 1, "nodes": tuple(range(1, 11))}},
        "materials": {1: {"E": E, "nu": NU, "rho": RHO}},
        "properties": {1: {"type": "solid", "material_id": 1}},
        "spcs": [] if spcs is None else spcs,
    }


def tet_volume(corners: np.ndarray = CORNERS) -> float:
    return abs(float(np.linalg.det(np.column_stack([np.ones(4), corners])))) / 6.0


# ----------------------------------------------------------------------
# TET10: the frozen entry point
# ----------------------------------------------------------------------


def test_the_frozen_import_path_resolves() -> None:
    """``from femtools.fea.elements import tet10``, and the registry entry."""
    from femtools.fea.elements.solid import tet10 as from_module

    assert from_module is tet10
    assert element_spec("TET10").builder is tet10
    assert "TET10" in available_elements()
    assert "TET10" in element_info()


def test_the_registry_entry_says_what_the_element_is() -> None:
    spec = element_spec("TET10")

    assert spec.name == "TET10"
    assert spec.n_nodes == (10,)
    assert spec.family == "solid"
    # A solid node carries the three translations and nothing else.
    assert spec.dofs_per_node == (0, 1, 2)
    assert spec.accepts(10)
    assert not spec.accepts(4)

    # The public card names for the same element resolve to it, and the
    # four-node ``CTETRA`` is emphatically still the linear tetrahedron.
    for alias in ("CTETRA10", "TETRA10", "C3D10"):
        assert element_spec(alias) is spec
    assert element_spec("CTETRA").name == "TET4"
    assert element_spec("TET").name == "TET4"
    assert REGISTRY["TET4"].n_nodes == (4,)
    # The model database was seeded with the same node count.
    assert ELEMENT_NODE_COUNTS["TET10"] == (10,)


def test_a_tet10_given_the_wrong_number_of_nodes_is_refused() -> None:
    model = one_tet10()
    model["elements"][1] = {"type": "TET10", "property_id": 1, "nodes": tuple(range(1, 5))}

    with pytest.raises(ValueError, match="expected one of"):
        element_matrices(model, 1, model["elements"][1])


# ----------------------------------------------------------------------
# TET10: the shape functions
# ----------------------------------------------------------------------


def natural_samples(seed: int = 5, n: int = 40) -> np.ndarray:
    """Points inside the reference tetrahedron, plus its corners and midsides."""
    rng = np.random.default_rng(seed)
    fixed = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    fixed += [(0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), TET10_CENTROID]
    drawn = rng.dirichlet(np.ones(4), size=n)[:, 1:]
    return np.vstack([np.array(fixed), drawn])


def test_the_shape_functions_are_a_partition_of_unity() -> None:
    """``sum N_i = 1`` and ``sum dN_i = 0`` everywhere inside the element."""
    for point in natural_samples():
        n, dn = _tet10_shape(*point)
        assert n.shape == (10,)
        assert dn.shape == (10, 3)
        assert n.sum() == pytest.approx(1.0, abs=1.0e-14)
        np.testing.assert_allclose(dn.sum(axis=0), np.zeros(3), atol=1.0e-14)


def test_each_shape_function_is_one_at_its_own_node_and_zero_at_the_others() -> None:
    """The Kronecker delta property, corners and midsides alike."""
    natural = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }
    for e, (a, b) in enumerate(TET10_EDGES):
        natural[4 + e] = tuple(
            (0.5 * (np.asarray(natural[a]) + np.asarray(natural[b]))).tolist()
        )

    got = np.array([_tet10_shape(*natural[i])[0] for i in range(10)])
    np.testing.assert_allclose(got, np.eye(10), atol=1.0e-14)

    # ... and the midside nodes really are between the corners the public card
    # layouts put them between: 5..10 bisect 1-2, 2-3, 3-1, 1-4, 2-4, 3-4.
    assert TET10_EDGES == ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def test_the_element_reproduces_any_linear_field_exactly() -> None:
    """Quadratic contains linear, which is why the patch test can be exact.

    Interpolating an exactly linear field from the ten nodes must return that
    field at every point of the element, and its gradient must be the constant
    the field was built with -- read through the isoparametric map, so the
    geometry interpolation is checked at the same time.
    """
    xyz = tet10_nodes()
    gradient = PATCH_GRADIENT
    offset = np.array([3.0e-4, -1.0e-4, 7.0e-5])
    nodal = (gradient @ xyz.T).T + offset

    for point in natural_samples():
        n, _dn = _tet10_shape(*point)
        position = n @ xyz
        np.testing.assert_allclose(n @ nodal, gradient @ position + offset, atol=1.0e-16)
        grad, det = tet10_gradient(xyz, point)
        assert det > 0.0
        np.testing.assert_allclose(grad.T @ nodal, gradient.T, atol=1.0e-12)


def test_the_stiffness_quadrature_is_exact_for_a_straight_edged_element() -> None:
    """Four points, and the same matrix a far finer rule gives.

    A straight-edged tetrahedron has a constant Jacobian, so ``B`` is linear in
    the natural coordinates and ``B^T D B`` is quadratic: the degree-two rule
    is not an approximation here, it is the integral.
    """
    points, weights = tet_rule(2)
    assert points.shape == (4, 3)
    assert weights.sum() == pytest.approx(1.0 / 6.0, rel=1.0e-15)

    xyz = tet10_nodes()
    D = solid_D(fea.MaterialData(E=E, nu=NU))
    reference = np.zeros((30, 30))
    for point, weight in zip(*tet_rule(5), strict=True):
        grad, det = tet10_gradient(xyz, point)
        B = np.zeros((6, 30))
        for i in range(10):
            gx, gy, gz = grad[i]
            B[0, 3 * i], B[1, 3 * i + 1], B[2, 3 * i + 2] = gx, gy, gz
            B[3, 3 * i], B[3, 3 * i + 1] = gy, gx
            B[4, 3 * i + 1], B[4, 3 * i + 2] = gz, gy
            B[5, 3 * i], B[5, 3 * i + 2] = gz, gx
        reference += weight * abs(det) * (B.T @ D @ B)

    model = one_tet10()
    shipped = np.asarray(element_matrices(model, 1, model["elements"][1]).k, dtype=float)
    assert np.max(np.abs(shipped - reference)) / np.max(np.abs(shipped)) < 1.0e-13


def test_the_quadrature_rules_integrate_the_degree_they_claim() -> None:
    """``tet_rule(2)`` is exact to degree 2, ``tet_rule(5)`` to degree 5."""

    def exact(exponents: tuple[int, ...]) -> float:
        return math.prod(math.factorial(k) for k in exponents) / math.factorial(
            sum(exponents) + 3
        )

    for order, degree, n_points in ((1, 1, 1), (2, 2, 4), (5, 5, 15)):
        points, weights = tet_rule(order)
        assert weights.size == n_points
        assert np.all(weights > 0.0)
        assert weights.sum() == pytest.approx(1.0 / 6.0, rel=1.0e-15)
        L = np.column_stack([1.0 - points.sum(axis=1), points])
        for exponents in np.ndindex(*(degree + 1,) * 4):
            if sum(exponents) > degree:
                continue
            got = float((weights * np.prod(L ** np.array(exponents), axis=1)).sum())
            assert got == pytest.approx(exact(exponents), abs=1.0e-15)


# ----------------------------------------------------------------------
# TET10: the gate cases
# ----------------------------------------------------------------------


def test_the_tet10_constant_strain_patch_test_is_exact() -> None:
    """The Round 10 gate: patch error at or below 1e-12, stress and strain.

    Four ``TET10`` fill one irregular outer tetrahedron; the outer surface --
    corners *and* the midsides of the six outer edges -- is driven with an
    exact linear displacement field and five nodes are left free.  Every
    element must then report the analytic constant state, and so must every
    node after either smoothing step.
    """
    error = stress_patch_error("TET10")

    assert error["elements"] == 4.0
    assert error["nodes"] == 15.0
    assert error["displacement"] < 1.0e-12
    assert error["stress"] < 1.0e-12
    assert error["strain"] < 1.0e-12
    assert error["nodal"] < 1.0e-12
    assert error["spr"] < 1.0e-12


def test_the_tet10_patch_mesh_is_the_shape_the_case_claims() -> None:
    """Five free nodes, not one: the midsides of the interior edges are free too."""
    model, coords, boundary = tet_patch_model("TET10")

    assert len(model["elements"]) == 4
    assert all(element["type"] == "TET10" for element in model["elements"].values())
    assert all(len(element["nodes"]) == 10 for element in model["elements"].values())
    # 5 corners + 10 edges of the four-tetrahedron block = 15 nodes.
    assert len(coords) == 15
    assert len(boundary) == 10
    # The free nodes are the enclosed corner and the four midsides of the
    # edges that run out to it -- everything not on the outer surface.
    free = sorted(set(coords) - set(boundary))
    assert len(free) == 5
    assert free[0] == 5
    for nid in free[1:]:
        assert min(np.linalg.norm(coords[nid] - 0.5 * (coords[5] + coords[c])) for c in
                   (1, 2, 3, 4)) < 1.0e-15

    # Every midside node really sits between its two corners.
    for element in model["elements"].values():
        conn = element["nodes"]
        for e, (a, b) in enumerate(TET10_EDGES):
            np.testing.assert_allclose(
                coords[conn[4 + e]], 0.5 * (coords[conn[a]] + coords[conn[b]]), atol=1.0e-15
            )

    # ... and the same builder still produces the TET4 patch it always did.
    linear, linear_coords, linear_boundary = tet_patch_model("TET4")
    assert len(linear["elements"]) == 4
    assert len(linear_coords) == 5
    assert linear_boundary == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="expected TET4 or TET10"):
        tet_patch_model("HEX20")


@pytest.mark.parametrize("etype", PATCH_TYPES)
def test_every_patch_type_including_tet10_is_still_exact(etype: str) -> None:
    """TET10 joined :data:`PATCH_TYPES`; nothing already in it moved."""
    error = stress_patch_error(etype)

    assert error["elements"] >= 3.0
    assert error["displacement"] < 1.0e-12
    assert error["stress"] < 1.0e-12
    assert error["strain"] < 1.0e-12
    assert error["nodal"] < 1.0e-12
    assert "TET10" in PATCH_TYPES
    assert PATCH_TYPES == ("BAR2", "BEAM2", "TRIA3", "QUAD4", "TET4", "TET10", "HEX8")


def test_a_free_free_tet10_has_exactly_six_rigid_body_modes() -> None:
    """The other Round 10 gate, on one element and on a small mesh.

    Six zero eigenvalues and a strictly positive seventh.  The four-point rule
    is the minimum that can manage it -- four points times six strain
    components is exactly the rank 24 the element needs -- so a quadrature
    reduced one point further would show up here as a mechanism.
    """
    model = one_tet10()
    k = np.asarray(element_matrices(model, 1, model["elements"][1]).k, dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (k + k.T))

    assert k.shape == (30, 30)
    assert np.count_nonzero(np.abs(eigenvalues) < 1.0e-8 * eigenvalues.max()) == 6
    assert eigenvalues[6] > 1.0e-6 * eigenvalues.max()
    assert eigenvalues.min() > -1.0e-8 * eigenvalues.max()

    frequencies = np.asarray(solve_modes(model, n_modes=10).freq_hz, dtype=float)
    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0

    # ... and on the four-element patch block, where the same statement has to
    # survive assembly and the shared midside nodes.
    block = tet10_rigid_body_frequencies()
    assert np.count_nonzero(block < 1.0e-6) == 6
    assert block[6] > 1.0


def test_the_six_null_modes_are_the_rigid_body_motions_themselves() -> None:
    """Not merely six zeros: translations and rotations carry no energy."""
    model = one_tet10()
    xyz = tet10_nodes()
    k = np.asarray(element_matrices(model, 1, model["elements"][1]).k, dtype=float)
    scale = float(np.abs(k).max())

    modes = np.zeros((30, 6))
    for i, point in enumerate(xyz):
        for c in range(3):
            modes[3 * i + c, c] = 1.0
            axis = np.zeros(3)
            axis[c] = 1.0
            modes[3 * i : 3 * i + 3, 3 + c] = np.cross(axis, point)
    assert np.max(np.abs(k @ modes)) / scale < 1.0e-14


def test_the_consistent_mass_carries_the_exact_volume() -> None:
    """``rho V`` in every direction, and a positive definite matrix."""
    model = one_tet10()
    matrices = element_matrices(model, 1, model["elements"][1])
    m = np.asarray(matrices.m, dtype=float)
    volume = tet_volume()

    assert m.shape == (30, 30)
    for direction in range(3):
        block = m[direction::3, direction::3]
        assert block.sum() == pytest.approx(RHO * volume, rel=1.0e-13)
    assert np.linalg.eigvalsh(0.5 * (m + m.T)).min() > 0.0
    np.testing.assert_allclose(m, m.T, atol=1.0e-18 * np.abs(m).max())


def test_the_lumped_mass_is_hrz_scaled_and_stays_positive() -> None:
    """Row-sum lumping is not available on a quadratic element; HRZ is.

    ``\\int N dV`` of a corner function of the ten-node tetrahedron is
    ``-V/20`` -- negative -- so the classical row sums would hand four nodes a
    negative mass.  Hinton, Rock & Zienkiewicz (1976) scale the diagonal of
    the consistent matrix instead, which is positive everywhere and still
    carries the exact total mass.
    """
    model = one_tet10()
    consistent = np.asarray(element_matrices(model, 1, model["elements"][1]).m, dtype=float)
    lumped = np.asarray(
        element_matrices(model, 1, model["elements"][1], lumped_mass=True).m, dtype=float
    )
    volume = tet_volume()

    assert np.count_nonzero(lumped - np.diag(np.diag(lumped))) == 0
    assert np.diag(lumped).min() > 0.0
    assert lumped[0::3, 0::3].sum() == pytest.approx(RHO * volume, rel=1.0e-13)
    # The row sums the naive lumping would have used really are negative.
    row_sums = consistent[0::3, 0::3].sum(axis=1)
    assert row_sums[:4].max() < 0.0
    np.testing.assert_allclose(row_sums[:4], np.full(4, -RHO * volume / 20.0), rtol=1.0e-12)
    # HRZ is a rescaling of the diagonal, so the ratios of the diagonal stay.
    ratio = np.diag(lumped)[0::3] / np.diag(consistent)[0::3]
    np.testing.assert_allclose(ratio, np.full(10, ratio[0]), rtol=1.0e-13)


def test_a_collapsed_or_folded_tet10_is_refused() -> None:
    """Bad geometry is rejected rather than integrated into a plausible answer."""
    coplanar = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    flat = tet10_nodes(coplanar)
    with pytest.raises(ValueError, match="degenerate TET10"):
        element_matrices(one_tet10(flat), 1, one_tet10(flat)["elements"][1])

    # A midside node pushed far past the quarter point of its edge folds the
    # element through itself; the Jacobian changes sign inside it.
    folded = tet10_nodes()
    folded[4] = CORNERS[0] + 2.5 * (CORNERS[1] - CORNERS[0])
    with pytest.raises(ValueError, match="folded TET10"):
        element_matrices(one_tet10(folded), 1, one_tet10(folded)["elements"][1])

    # A consistently reversed ordering is not an error: it integrates the same.
    reversed_corners = CORNERS[[0, 2, 1, 3]]
    straight = element_matrices(one_tet10(), 1, one_tet10()["elements"][1]).k
    swapped = element_matrices(
        one_tet10(tet10_nodes(reversed_corners)),
        1,
        one_tet10()["elements"][1],
    ).k
    assert float(np.abs(swapped).max()) == pytest.approx(float(np.abs(straight).max()), rel=0.4)


# ----------------------------------------------------------------------
# TET10: stress recovery and what the element is for
# ----------------------------------------------------------------------


def test_tet10_stress_is_recovered_at_the_centroid() -> None:
    """One value per element, at the centroid, and it is the Gauss mean.

    A quadratic tetrahedron has a linear strain field of its own, so the
    centroid value is a choice rather than the only one available.  With
    straight edges it is exactly the average of the four Gauss point values
    the stiffness was integrated with, which is what makes it the right single
    number to report and the right sample for :func:`recover_spr`.
    """
    model, tip, loads = tet_cantilever(3, 1, 1, etype="TET10")
    asm = assemble_km(model)
    u = np.asarray(solve_static(model, loads, assembly=asm))
    result = recover_stress(model, u, assembly=asm)

    assert len(result) == len(model["elements"])
    assert set(result.etypes) == {"TET10"}
    assert result.location == "centroid"
    assert not result.skipped
    np.testing.assert_allclose(result.frame, np.broadcast_to(np.eye(3), result.frame.shape))

    index = fea.element_matrices.__globals__["ModelIndex"].build(model)
    D = solid_D(fea.MaterialData(E=E, nu=NU))
    for row, eid in enumerate(result.element_ids):
        conn = model["elements"][eid]["nodes"]
        xyz = np.array([index.xyz(nid) for nid in conn])
        disp = np.concatenate([u[asm.dof_map.node_dofs(nid)][:3] for nid in conn])
        gauss = np.zeros(6)
        for point in tet_rule(2)[0]:
            grad, _det = tet10_gradient(xyz, point)
            B = np.zeros((6, 30))
            for i in range(10):
                gx, gy, gz = grad[i]
                B[0, 3 * i], B[1, 3 * i + 1], B[2, 3 * i + 2] = gx, gy, gz
                B[3, 3 * i], B[3, 3 * i + 1] = gy, gx
                B[4, 3 * i + 1], B[4, 3 * i + 2] = gz, gy
                B[5, 3 * i], B[5, 3 * i + 2] = gz, gx
            gauss += B @ disp
        gauss /= 4.0
        np.testing.assert_allclose(result.strain[row], gauss, rtol=1.0e-11, atol=1.0e-18)
        np.testing.assert_allclose(result.stress[row], D @ result.strain[row], rtol=1.0e-12)
        np.testing.assert_allclose(result.centroid[row], xyz[:4].mean(axis=0), atol=1.0e-13)

    # ``recover_strain`` is the same computation under a name that reads better.
    np.testing.assert_array_equal(
        recover_strain(model, u, assembly=asm).strain, result.strain
    )


def test_the_quadratic_tetrahedron_can_bend_and_the_linear_one_cannot() -> None:
    """What TET10 is for, measured against the Timoshenko tip deflection.

    The same tetrahedral mesh of the same cantilever, read once as ``TET4`` and
    once as ``TET10``.  The constant-strain tetrahedron reaches barely a fifth
    of the reference -- a tet mesh of it is not a bending model -- while the
    quadratic one lands within three percent of it.
    """
    linear = tet_bending_ratio("TET4")
    quadratic = tet_bending_ratio("TET10")

    assert linear == pytest.approx(0.2194, abs=0.005)
    assert quadratic == pytest.approx(0.9757, abs=0.005)
    assert quadratic > 0.95
    # ... and still below the hexahedral answer, which it should be: the brick
    # has eight times fewer elements and its own anti-locking treatment.
    assert quadratic < hex8_bending_ratio()

    # The mesh really is the same mesh, only its elements are read differently.
    coarse, _tip, _loads = tet_cantilever(3, 1, 1, etype="TET4")
    fine, _tip10, _loads10 = tet_cantilever(3, 1, 1, etype="TET10")
    assert len(fine["elements"]) == len(coarse["elements"]) == 3 * 6
    assert set(coarse["nodes"]) <= set(fine["nodes"])
    for nid in coarse["nodes"]:
        assert fine["nodes"][nid]["xyz"] == coarse["nodes"][nid]["xyz"]
    with pytest.raises(ValueError, match="expected TET4 or TET10"):
        tet_cantilever(2, 1, 1, etype="HEX8")


def test_the_core_model_accepts_a_ten_node_tetra_and_the_kernel_solves_it() -> None:
    """The same element through :class:`femtools.core.model.FEModel`."""
    model = FEModel(name="tet10")
    xyz = tet10_nodes()
    for i, point in enumerate(xyz):
        model.add_node(i + 1, tuple(float(v) for v in point))
    model.add_material(1, E=E, nu=NU, rho=RHO)
    model.add_property(1, type="solid", material_id=1)
    model.add_element(1, "TET10", tuple(range(1, 11)), property_id=1)

    assert model.elements[1].type == "TET10"
    assert len(model.elements[1].nodes) == 10
    frequencies = np.asarray(solve_modes(model, n_modes=8).freq_hz, dtype=float)
    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0


# ----------------------------------------------------------------------
# recover_spr: the frozen entry point
# ----------------------------------------------------------------------


def test_the_spr_import_path_and_signature_are_frozen() -> None:
    """``from femtools.fea.recover import recover_spr``, and the re-export."""
    from femtools.fea.recover import recover_spr as from_module

    assert from_module is recover_spr
    assert fea.recover_spr is recover_spr
    assert "recover_spr" in fea.__all__

    parameters = inspect.signature(recover_spr).parameters
    assert list(parameters) == ["stress", "model", "nodes", "index"]
    assert parameters["stress"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("nodes", "index"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
    # The same two positional arguments as ``average_nodal``, so the one can be
    # swapped for the other.
    assert list(inspect.signature(average_nodal).parameters)[:2] == ["stress", "model"]


def test_recover_spr_takes_the_result_of_recover_stress() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    with pytest.raises(TypeError, match="StressResult"):
        recover_spr(np.zeros((3, 6)), model)
    with pytest.raises(TypeError, match="StressResult"):
        recover_spr(average_nodal(stress, model), model)


# ----------------------------------------------------------------------
# recover_spr: the gate cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("etype", ["BAR2", "TRIA3", "QUAD4", "TET4", "TET10", "HEX8"])
def test_a_constant_stress_patch_comes_through_spr_exactly(etype: str) -> None:
    """The Round 10 gate: SPR is exact on a constant field at every node.

    The polynomial fitted over a patch of identical values is that value, so
    there is nothing for the fit to get wrong -- interior nodes, boundary
    nodes, corner nodes with a single element, all of them.
    """
    error = stress_patch_error(etype)

    assert error["spr"] < 1.0e-12
    assert error["nodal"] < 1.0e-12
    assert error["nodes"] >= 4.0


def test_the_spr_result_is_a_nodal_stress_result_on_the_same_nodes() -> None:
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)

    patch = recover_spr(stress, model)
    average = average_nodal(stress, model)

    assert isinstance(patch, NodalStressResult)
    assert patch.node_ids == average.node_ids
    assert patch.element_ids == average.element_ids
    assert patch.location == "spr"
    assert average.location == "node"
    np.testing.assert_array_equal(patch.count, average.count)
    np.testing.assert_allclose(patch.xyz, average.xyz)
    assert patch.stress.shape == (len(patch), 6)
    assert patch.strain.shape == (len(patch), 6)
    assert patch.components == average.components

    # ``patch_terms`` is the SPR-only field and the average leaves it empty.
    assert patch.patch_terms.shape == (len(patch),)
    assert set(np.unique(patch.patch_terms)) <= {1, 2, 3, 4}
    assert average.patch_terms.size == 0

    # The node filters behave like the average's.
    wanted = patch.node_ids[:3]
    assert recover_spr(stress, model, nodes=wanted).node_ids == wanted
    assert recover_spr(stress, model, nodes=lambda nid: nid == wanted[0]).n_nodes == 1
    assert recover_spr(stress, model, nodes=()).n_nodes == 0
    np.testing.assert_allclose(
        recover_spr(stress, model, nodes=wanted).stress,
        patch.stress[: len(wanted)],
        rtol=1.0e-14,
    )


def linear_field_samples(model: dict, centroids: np.ndarray) -> tuple[StressResult, np.ndarray]:
    """A ``StressResult`` whose element values are an exact linear field of ``x``.

    Synthetic on purpose: the question is whether the *recovery* reproduces a
    linear field, which a real solve would only supply approximately.
    """
    gradient = np.arange(1.0, 19.0).reshape(6, 3) * 0.37
    offset = np.arange(1.0, 7.0)
    values = np.asarray(centroids, dtype=float) @ gradient.T + offset
    element_ids = list(model["elements"])
    synthetic = StressResult(
        element_ids=element_ids,
        etypes=[element["type"] for element in model["elements"].values()],
        stress=values.copy(),
        strain=1.0e-6 * values.copy(),
        frame=np.broadcast_to(np.eye(3), (len(element_ids), 3, 3)).copy(),
        centroid=np.asarray(centroids, dtype=float).copy(),
    )
    return synthetic, np.hstack([gradient, offset[:, None]])


def test_spr_reproduces_a_linear_stress_field_where_the_average_cannot() -> None:
    """The property that makes SPR worth having, and the one the mean lacks.

    The element values are made an exact linear function of the sampling point.
    Wherever the patch can carry the full linear polynomial -- the interior
    nodes of a solid mesh, which see elements on every side of them -- the
    fit returns the analytic value at the node to round-off.  The arithmetic
    mean of the same numbers does not: it is the value at the *centroid of the
    patch*, and on a lopsided patch that is somewhere else.
    """
    for etype in ("TET4", "TET10"):
        model, coords, _boundary = tet_patch_model(etype)
        centroids = np.array(
            [
                np.mean([coords[nid] for nid in element["nodes"][:4]], axis=0)
                for element in model["elements"].values()
            ]
        )
        synthetic, fit = linear_field_samples(model, centroids)
        gradient, offset = fit[:, :3], fit[:, 3]

        patch = recover_spr(synthetic, model)
        average = average_nodal(synthetic, model)
        exact = patch.xyz @ gradient.T + offset
        scale = float(np.max(np.abs(exact)))
        full = patch.patch_terms == 4

        # The enclosed node is the one with elements on every side of it.
        assert full.sum() == 1
        assert patch.node_ids[int(np.flatnonzero(full)[0])] == 5
        assert np.max(np.abs(patch.stress[full] - exact[full])) / scale < 1.0e-14
        np.testing.assert_allclose(
            patch.strain[full], 1.0e-6 * exact[full], rtol=1.0e-11, atol=1.0e-18
        )
        # The mean is not merely less accurate, it is wrong by a visible amount:
        # the four tetrahedra are of different sizes, so the centroid of the
        # patch is not the node.
        assert np.max(np.abs(average.stress[full] - exact[full])) / scale > 1.0e-3


def test_spr_lowers_the_polynomial_order_rather_than_extrapolating_wildly() -> None:
    """A patch that cannot carry a linear fit gets fewer terms, not a wild one.

    A node on the surface of a solid mesh sees sampling points on one side of
    itself only, and a corner node sees a single element.  Both are reported
    through ``patch_terms``, and neither may turn round-off into a visible
    error -- which is exactly what an unguarded exactly-determined
    extrapolation over a short baseline does.
    """
    model, _tip, loads = hex_cantilever(4, 2, 2)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    patch = recover_spr(stress, model)
    average = average_nodal(stress, model)

    by_count = {int(c): int(t) for c, t in zip(patch.count, patch.patch_terms, strict=True)}
    # One element, one term: the fit is the value, which is also the average.
    assert by_count[1] == 1
    single = patch.count == 1
    np.testing.assert_allclose(patch.stress[single], average.stress[single], rtol=1.0e-12)
    # Four bricks meeting on a face plane give four coplanar samples, so the
    # through-thickness term is not there to be fitted.
    assert by_count[4] == 3
    assert by_count[8] == 4
    assert patch.patch_terms.max() == 4

    # No node may be further from the element values than the spread of its
    # own patch allows; SPR is a smoothing step, not an extrapolator.
    reach = float(np.max(np.abs(stress.stress_basic)))
    assert np.max(np.abs(patch.stress)) < 2.0 * reach


def test_spr_is_not_the_average_and_the_average_is_still_one_over_n() -> None:
    """``average_nodal`` keeps its ``1 / n_a`` weighting, entry for entry."""
    model, _tip, loads = hex_cantilever(3, 1, 1)
    asm = assemble_km(model)
    stress = recover_stress(model, solve_static(model, loads, assembly=asm), assembly=asm)
    average = average_nodal(stress, model)
    patch = recover_spr(stress, model)

    incident: dict[int, list[int]] = {}
    for i, eid in enumerate(stress.element_ids):
        for nid in model["elements"][eid]["nodes"]:
            incident.setdefault(nid, []).append(i)
    for nid, rows in incident.items():
        np.testing.assert_allclose(
            average.stress[average.index_of(nid)],
            stress.stress_basic[rows].mean(axis=0),
            rtol=1.0e-14,
        )
    # ... and the patch recovery genuinely says something else on this mesh.
    assert np.max(np.abs(patch.stress - average.stress)) > 0.0


def test_spr_reads_the_tet10_centroid_samples() -> None:
    """TET10 is not skipped: its centroid values are the patch samples.

    The superconvergent points of a quadratic tetrahedron are its four Gauss
    points rather than its centroid, so this is a documented departure from the
    1992 paper -- an honest least-squares patch fit, constant-stress exact, but
    not a formally superconvergent one.
    """
    model, _coords, _boundary = tet_patch_model("TET10")
    asm = assemble_km(model)
    enforced = {
        (nid, comp): float((PATCH_GRADIENT @ np.asarray(node["xyz"]))[comp])
        for nid, node in model["nodes"].items()
        for comp in range(3)
        if nid in _boundary
    }
    model["spcs"] = [{"node_id": nid, "dofs": (0, 1, 2)} for nid in _boundary]
    asm = assemble_km(model)
    u = np.asarray(solve_static(model, {}, assembly=asm, enforced=enforced))
    stress = recover_stress(model, u, assembly=asm)
    patch = recover_spr(stress, model)

    assert set(stress.etypes) == {"TET10"}
    assert len(patch) == 15
    assert not stress.skipped
    # The samples the fit used are the centroids the recovery reported.
    np.testing.assert_allclose(
        stress.centroid,
        np.array(
            [
                np.mean([model["nodes"][nid]["xyz"] for nid in element["nodes"][:4]], axis=0)
                for element in model["elements"].values()
            ]
        ),
        atol=1.0e-13,
    )
    reference = stress.stress_basic.mean(axis=0)
    assert np.max(np.abs(patch.stress - reference)) / np.max(np.abs(reference)) < 1.0e-12


# ----------------------------------------------------------------------
# goldens that must not move
# ----------------------------------------------------------------------


def test_hex8_cantilever_tip_ratio_golden() -> None:
    """The Wilson/Taylor incompatible-mode default was not retuned."""
    ratio = hex8_bending_ratio()

    assert ratio >= 0.98
    assert ratio == pytest.approx(0.9854730473, rel=1.0e-6)


def test_hex8_patch_and_rigid_body_goldens() -> None:
    frequencies = hex8_rigid_body_frequencies()

    assert np.count_nonzero(frequencies < 1.0e-6) == 6
    assert frequencies[6] > 1.0
    assert hex8_patch_test_error() < 1.0e-10


@pytest.mark.parametrize("etype", ["QUAD4", "TRIA3"])
def test_tilted_shell_still_has_six_rigid_body_modes(etype: str) -> None:
    gap = shell_drilling_orientation_gap(etype)

    assert gap["oblique_zero_modes"] == 6.0
    assert gap["aligned_zero_modes"] == 6.0
    assert gap["oblique_warned"] == 0.0
    assert gap["oblique_first_elastic_hz"] == pytest.approx(
        gap["aligned_first_elastic_hz"], rel=1.0e-9
    )


@pytest.mark.parametrize("thickness", [1.0e-2, 1.0e-4])
def test_mitc4_thin_plate_does_not_shear_lock(thickness: float) -> None:
    modulus, nu, side = 70.0e9, 0.3, 1.0
    model = shell_plate(8, 8, side=side, thickness=thickness, E=modulus, nu=nu, clamped_edge=True)
    asm = assemble_km(model)
    tip = [nid for nid, node in model["nodes"].items() if abs(node["xyz"][0] - side) < 1.0e-12]
    u = solve_static(model, {(nid, 2): 1.0 / len(tip) for nid in tip}, assembly=asm)
    deflection = float(np.mean([u[asm.dof_map.index(nid, 2)] for nid in tip]))

    rigidity = modulus * thickness**3 / (12.0 * (1.0 - nu**2))
    assert deflection / (side**3 / (3.0 * rigidity * side)) == pytest.approx(1.025, abs=0.01)


def test_enforced_displacement_still_solves() -> None:
    """``solve_static(enforced=)`` is untouched, on springs and on a TET10."""
    springs = {
        "nodes": {i: {"xyz": (float(i - 1), 0.0, 0.0)} for i in (1, 2, 3)},
        "elements": {
            i: {"type": "SPRING", "nodes": (i, i + 1), "k": 1000.0, "c1": 0} for i in (1, 2)
        },
        "materials": {},
        "properties": {},
        "spcs": [{"node_id": 1, "dofs": (0,)}],
    }
    asm = assemble_km(springs)
    u = solve_static(springs, {}, assembly=asm, enforced={(3, 0): 0.03})
    np.testing.assert_allclose(
        [u[asm.dof_map.index(nid, 0)] for nid in (1, 2, 3)], [0.0, 0.015, 0.03], atol=1.0e-12
    )

    # The same mechanism is what drives the TET10 patch test.
    model, coords, boundary = tet_patch_model("TET10")
    model["spcs"] = [{"node_id": nid, "dofs": (0, 1, 2)} for nid in boundary]
    enforced = {
        (nid, comp): float((PATCH_GRADIENT @ coords[nid])[comp])
        for nid in boundary
        for comp in range(3)
    }
    solid = assemble_km(model)
    field = np.asarray(solve_static(model, {}, assembly=solid, enforced=enforced))
    want = np.array([PATCH_GRADIENT @ coords[nid] for nid in coords])
    got = np.array([field[solid.dof_map.node_dofs(nid)][:3] for nid in coords])
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1.0e-12

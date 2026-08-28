"""Optimization: sizing, shape, 2-D SIMP topology, and design of experiments.

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.optimization.size import size_optimize
    from femtools.optimization.shape import shape_optimize
    from femtools.optimization.topology import topology_simp
    from femtools.optimization.topometry import topometry_optimize
    from femtools.optimization.doe import latin_hypercube, full_factorial
    from femtools.optimization.surrogate import fit_rsm, predict_rsm
    from femtools.optimization.multi import pareto_weighted
"""

from __future__ import annotations

from .doe import (
    central_composite,
    discrepancy,
    full_factorial,
    latin_hypercube,
    maximin_distance,
    random_sampling,
    scale_to_bounds,
    sobol,
)
from .multi import (
    ParetoResult,
    crowding_distance,
    hypervolume,
    non_dominated_sort,
    pareto_front,
    pareto_weighted,
    simplex_lattice,
)
from .shape import ShapeResult, element_size_ratios, shape_optimize
from .size import (
    Constraint,
    OptimizationResult,
    finite_difference_gradient,
    size_optimize,
)
from .surrogate import RSMFit, design_matrix, fit_rsm, predict_rsm, rsm_terms
from .topology import TopologyResult, element_stiffness_q4, topology_simp
from .topometry import TopometryResult, topometry_optimize

__all__ = [
    "size_optimize",
    "OptimizationResult",
    "Constraint",
    "finite_difference_gradient",
    # shape
    "shape_optimize",
    "ShapeResult",
    "element_size_ratios",
    "topology_simp",
    "TopologyResult",
    "element_stiffness_q4",
    # topometry (element-wise sizing on an existing mesh)
    "topometry_optimize",
    "TopometryResult",
    "latin_hypercube",
    "full_factorial",
    "random_sampling",
    "sobol",
    "central_composite",
    "scale_to_bounds",
    "maximin_distance",
    "discrepancy",
    # response surfaces
    "fit_rsm",
    "predict_rsm",
    "RSMFit",
    "rsm_terms",
    "design_matrix",
    # multi-objective
    "pareto_weighted",
    "ParetoResult",
    "pareto_front",
    "non_dominated_sort",
    "crowding_distance",
    "simplex_lattice",
    "hypervolume",
]

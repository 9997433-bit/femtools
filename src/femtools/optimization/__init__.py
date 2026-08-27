"""Optimization: sizing (SLSQP), 2-D SIMP topology, and design of experiments.

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.optimization.size import size_optimize
    from femtools.optimization.topology import topology_simp
    from femtools.optimization.doe import latin_hypercube, full_factorial
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
from .size import (
    Constraint,
    OptimizationResult,
    finite_difference_gradient,
    size_optimize,
)
from .topology import TopologyResult, element_stiffness_q4, topology_simp

__all__ = [
    "size_optimize",
    "OptimizationResult",
    "Constraint",
    "finite_difference_gradient",
    "topology_simp",
    "TopologyResult",
    "element_stiffness_q4",
    "latin_hypercube",
    "full_factorial",
    "random_sampling",
    "sobol",
    "central_composite",
    "scale_to_bounds",
    "maximin_distance",
    "discrepancy",
]

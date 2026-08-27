"""Model updating: sensitivity analysis, WLS/Bayesian updating, force identification.

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.updating.sensitivity import sensitivity_matrix
    from femtools.updating.updater import update_model, UpdateResult
    from femtools.updating.force_id import identify_harmonic_forces
    from femtools.updating.frf_updating import update_from_frf
    from femtools.updating.selection import select_parameters
    from femtools.updating.uq import parameter_covariance, monte_carlo_update
"""

from __future__ import annotations

from .force_id import ForceIdResult, identify_harmonic_forces, tikhonov_solve
from .frf_updating import (
    frf_residual,
    frf_sample_function,
    modal_frf_samples,
    update_from_frf,
)
from .parameters import (
    Parameter,
    ParameterSet,
    apply_parameters,
    as_parameters,
    parameter_bounds,
)
from .reference import (
    AxialBarModel,
    BeamModel,
    ReferenceModel,
    TwoDOFModel,
    analytical_axial_frequencies,
    analytical_cantilever_frequencies,
    make_updating_testcase,
)
from .responses import (
    ResponseSpec,
    frf_response_function,
    have_fea,
    mac_vector,
    modal_response_function,
    pair_by_mac,
)
from .selection import ParameterSelection, parameter_correlation, select_parameters
from .sensitivity import (
    SensitivityResult,
    analytic_frequency_sensitivity,
    eigenvector_sensitivity,
    finite_difference_jacobian,
    relative_sensitivity,
    sensitivity_matrix,
)
from .updater import UpdateOptions, UpdateResult, update_model
from .uq import UQResult, monte_carlo_update, parameter_covariance

__all__ = [
    # sensitivity
    "sensitivity_matrix",
    "SensitivityResult",
    "finite_difference_jacobian",
    "analytic_frequency_sensitivity",
    "eigenvector_sensitivity",
    "relative_sensitivity",
    # parameter selection
    "select_parameters",
    "ParameterSelection",
    "parameter_correlation",
    # updating
    "update_model",
    "UpdateResult",
    "UpdateOptions",
    # uncertainty quantification
    "parameter_covariance",
    "monte_carlo_update",
    "UQResult",
    # FRF-based updating
    "update_from_frf",
    "frf_sample_function",
    "modal_frf_samples",
    "frf_residual",
    # force identification
    "identify_harmonic_forces",
    "ForceIdResult",
    "tikhonov_solve",
    # parameters / responses
    "Parameter",
    "ParameterSet",
    "as_parameters",
    "apply_parameters",
    "parameter_bounds",
    "ResponseSpec",
    "modal_response_function",
    "frf_response_function",
    "mac_vector",
    "pair_by_mac",
    "have_fea",
    # reference models
    "ReferenceModel",
    "TwoDOFModel",
    "AxialBarModel",
    "BeamModel",
    "analytical_axial_frequencies",
    "analytical_cantilever_frequencies",
    "make_updating_testcase",
]

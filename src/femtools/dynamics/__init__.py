"""femtools.dynamics — forced response, damping, substructuring and modification.

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.dynamics.frf import modal_frf, direct_frf
    from femtools.dynamics.harmonic import harmonic_response
    from femtools.dynamics.mba import modal_based_assembly
    from femtools.dynamics.craig_bampton import craig_bampton
    from femtools.dynamics.time_domain import time_history
    from femtools.dynamics.residuals import residual_vectors
    from femtools.dynamics.frf import verify_modal_vs_direct, retained_band

``verify_modal_vs_direct`` is the modal-vs-direct acceptance check; on a truncated basis
it anchors the 0.2-0.8 fmax band on the last retained mode (``retained_band``).

Everything is also re-exported from this package. Modal input is duck-typed: any object
exposing ``freq_hz`` and ``modes`` works, including ``femtools.fea.eigen.ModalResult``.
"""

from __future__ import annotations

from .craig_bampton import CraigBamptonResult, craig_bampton
from .damping import (
    CombinedDamping,
    DampingModel,
    ModalDamping,
    NoDamping,
    RayleighDamping,
    StructuralDamping,
    ViscousDamping,
    as_damping,
    rayleigh_coefficients,
)
from .fba import frf_based_assembly
from .frf import (
    FRFResult,
    direct_frf,
    modal_frf,
    retained_band,
    retained_band_lines,
    retained_fmax_hz,
    verify_modal_vs_direct,
)
from .harmonic import HarmonicResult, harmonic_response
from .mba import (
    AssemblyResult,
    MassModification,
    ModalComponent,
    ModificationResult,
    SpringModification,
    modal_based_assembly,
    structural_dynamic_modification,
)
from .modal import ModalModel, as_modal
from .residuals import ResidualVectorResult, residual_vectors
from .synthetic import SyntheticTest, synthetic_frf, synthetic_time_response
from .time_domain import TimeHistoryResult, time_history

__all__ = [
    "AssemblyResult",
    "CombinedDamping",
    "CraigBamptonResult",
    "DampingModel",
    "FRFResult",
    "HarmonicResult",
    "MassModification",
    "ModalComponent",
    "ModalDamping",
    "ModalModel",
    "ModificationResult",
    "NoDamping",
    "RayleighDamping",
    "ResidualVectorResult",
    "SpringModification",
    "StructuralDamping",
    "SyntheticTest",
    "TimeHistoryResult",
    "ViscousDamping",
    "as_damping",
    "as_modal",
    "craig_bampton",
    "direct_frf",
    "frf_based_assembly",
    "harmonic_response",
    "modal_based_assembly",
    "modal_frf",
    "rayleigh_coefficients",
    "residual_vectors",
    "retained_band",
    "retained_band_lines",
    "retained_fmax_hz",
    "structural_dynamic_modification",
    "synthetic_frf",
    "synthetic_time_response",
    "time_history",
    "verify_modal_vs_direct",
]

"""femtools.dynamics — forced response, damping, substructuring and modification.

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.dynamics.frf import modal_frf, direct_frf
    from femtools.dynamics.harmonic import harmonic_response
    from femtools.dynamics.mba import modal_based_assembly
    from femtools.dynamics.craig_bampton import craig_bampton
    from femtools.dynamics.cms_free import rubin, macneal, free_interface_assembly
    from femtools.dynamics.time_domain import time_history
    from femtools.dynamics.residuals import residual_vectors
    from femtools.dynamics.random import psd_response
    from femtools.dynamics.superelement import dump_cms, load_cms
    from femtools.dynamics.frf import verify_modal_vs_direct, retained_band
    from femtools.dynamics.energy import modal_strain_energy, modal_kinetic_energy

``verify_modal_vs_direct`` is the modal-vs-direct acceptance check; on a truncated basis
it anchors the 0.2-0.8 fmax band on the last retained mode (``retained_band``).

Everything is also re-exported from this package. Modal input is duck-typed: any object
exposing ``freq_hz`` and ``modes`` works, including ``femtools.fea.eigen.ModalResult``.

Physical input is duck-typed in the same spirit. ``direct_frf``, ``harmonic_response`` and
``verify_modal_vs_direct`` are defined on ``(K, M)`` and keep that signature, but a single
first argument may also be an ``AssemblyResult``, a ``ModalResult`` carrying one, or a model
database, which is assembled here and reduced to its free partition
(``femtools.dynamics.system.as_system``). :mod:`femtools.fea` is imported only inside those
branches, so the package still runs on nothing but numpy/scipy.
"""

from __future__ import annotations

from .cms_free import (
    FreeCMSComponent,
    FreeCMSResult,
    FreeInterfaceAssembly,
    free_interface_assembly,
    macneal,
    rubin,
)
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
from .energy import (
    ElementEnergy,
    element_modal_energy,
    modal_kinetic_energy,
    modal_strain_energy,
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
from .random import PSDResult, miles_rms, psd_response
from .residuals import ResidualVectorResult, residual_vectors
from .superelement import Superelement, dump_cms, load_cms
from .synthetic import SyntheticTest, synthetic_frf, synthetic_time_response
from .system import SystemMatrices, as_system
from .time_domain import TimeHistoryResult, time_history

__all__ = [
    "AssemblyResult",
    "CombinedDamping",
    "CraigBamptonResult",
    "DampingModel",
    "ElementEnergy",
    "FRFResult",
    "FreeCMSComponent",
    "FreeCMSResult",
    "FreeInterfaceAssembly",
    "HarmonicResult",
    "MassModification",
    "ModalComponent",
    "ModalDamping",
    "ModalModel",
    "ModificationResult",
    "NoDamping",
    "PSDResult",
    "RayleighDamping",
    "ResidualVectorResult",
    "SpringModification",
    "StructuralDamping",
    "Superelement",
    "SyntheticTest",
    "SystemMatrices",
    "TimeHistoryResult",
    "ViscousDamping",
    "as_damping",
    "as_modal",
    "as_system",
    "craig_bampton",
    "direct_frf",
    "dump_cms",
    "element_modal_energy",
    "free_interface_assembly",
    "frf_based_assembly",
    "harmonic_response",
    "load_cms",
    "macneal",
    "miles_rms",
    "modal_based_assembly",
    "modal_frf",
    "modal_kinetic_energy",
    "modal_strain_energy",
    "psd_response",
    "rayleigh_coefficients",
    "residual_vectors",
    "retained_band",
    "rubin",
    "retained_band_lines",
    "retained_fmax_hz",
    "structural_dynamic_modification",
    "synthetic_frf",
    "synthetic_time_response",
    "time_history",
    "verify_modal_vs_direct",
]

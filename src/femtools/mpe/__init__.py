"""Modal parameter estimation (EMA / OMA).

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.mpe.p_lscf import poly_lscf     # PolyMAX-class, FRF based
    from femtools.mpe.fdd import fdd, efdd        # output-only, frequency domain
    from femtools.mpe.lsce import lsce            # time domain (IRF based)
"""

from __future__ import annotations

from .common import (
    ModalParameterResult,
    Pole,
    StabilizationDiagram,
    lsfd,
    poles_from_roots,
    select_physical_poles,
    stabilization_diagram,
    synthesize_frf,
)
from .fdd import cross_spectral_density, efdd, fdd
from .lsce import irf_from_frf, lsce
from .p_lscf import poly_lscf, polymax
from .synthetic import SyntheticModal, synthetic_frf, synthetic_response

__all__ = [
    "poly_lscf",
    "polymax",
    "fdd",
    "efdd",
    "cross_spectral_density",
    "lsce",
    "irf_from_frf",
    "ModalParameterResult",
    "Pole",
    "StabilizationDiagram",
    "stabilization_diagram",
    "select_physical_poles",
    "poles_from_roots",
    "lsfd",
    "synthesize_frf",
    "synthetic_frf",
    "synthetic_response",
    "SyntheticModal",
]

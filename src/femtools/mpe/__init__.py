"""Modal parameter estimation (EMA / OMA).

Public entry points (see ``docs/CONTRACT_API.md``)::

    from femtools.mpe.p_lscf import poly_lscf     # PolyMAX-class, FRF based
    from femtools.mpe.fdd import fdd, efdd        # output-only, frequency domain
    from femtools.mpe.lsce import lsce            # time domain (IRF based)
    from femtools.mpe.ssi import ssi_cov, ssi_data  # output-only, subspace
    from femtools.mpe.frf_estimation import estimate_h1, estimate_h2, coherence
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
from .frf_estimation import (
    FRFEstimate,
    SpectralMatrices,
    coherence,
    estimate_frf,
    estimate_h1,
    estimate_h2,
    multiple_coherence,
    welch_spectra,
)
from .lsce import irf_from_frf, lsce
from .p_lscf import poly_lscf, polymax
from .ssi import block_hankel, block_toeplitz, output_covariances, ssi_cov, ssi_data
from .synthetic import SyntheticModal, synthetic_frf, synthetic_response

__all__ = [
    "poly_lscf",
    "polymax",
    "fdd",
    "efdd",
    "cross_spectral_density",
    "lsce",
    "irf_from_frf",
    "ssi_cov",
    "ssi_data",
    "output_covariances",
    "block_toeplitz",
    "block_hankel",
    "estimate_h1",
    "estimate_h2",
    "estimate_frf",
    "coherence",
    "multiple_coherence",
    "welch_spectra",
    "FRFEstimate",
    "SpectralMatrices",
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

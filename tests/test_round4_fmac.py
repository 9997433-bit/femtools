"""Scale/phase invariance checks for frequency-domain MAC."""

from __future__ import annotations

import numpy as np
import pytest


def _scalar(value: object) -> float:
    array = np.asarray(value, dtype=float).squeeze()
    assert array.ndim == 0
    return float(array)


def test_fmac_is_phase_invariant_and_rejects_orthogonal_frfs() -> None:
    mac_module = pytest.importorskip("femtools.correlation.mac")
    fmac = getattr(mac_module, "fmac", None)
    if fmac is None:
        pytest.skip("frequency-domain MAC API is not available")

    reference = np.array([1.0 + 2.0j, 2.0 - 1.0j, -0.5 + 0.25j, 0.75j])
    scaled = -2.5j * reference
    orthogonal = np.array(
        [reference[1].conjugate(), -reference[0].conjugate(), 0.0j, 0.0j]
    )

    assert _scalar(fmac(reference, scaled)) == pytest.approx(1.0, abs=1.0e-13)
    assert _scalar(fmac(reference, orthogonal)) == pytest.approx(0.0, abs=1.0e-13)

"""Nastran punch eigenvalue/mode-shape round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from femtools.core.results import ModalResult


def test_pch_modal_roundtrip_preserves_eigenpairs(tmp_path: Path) -> None:
    pch = pytest.importorskip("femtools.io.pch")
    read_pch = getattr(pch, "read_pch", None)
    write_pch = getattr(pch, "write_pch", None)
    if read_pch is None or write_pch is None:
        pytest.skip("PCH read/write API is not available")

    frequency = np.array([3.25, 11.5])
    modes = np.array(
        [
            [1.0, -0.2],
            [0.1, 0.8],
            [-0.3, 0.5],
            [0.01, -0.04],
            [0.02, 0.03],
            [-0.05, 0.06],
            [0.4, 1.0],
            [-0.7, 0.2],
            [0.6, -0.1],
            [-0.02, 0.01],
            [0.03, -0.02],
            [0.04, 0.05],
        ]
    )
    source = ModalResult(
        freq_hz=frequency,
        eigenvalues=(2.0 * np.pi * frequency) ** 2,
        modes=modes,
        generalized_mass=np.array([1.0, 1.0]),
        dof_index=tuple((node, dof) for node in (10, 20) for dof in range(6)),
    )
    path = tmp_path / "modes.pch"

    write_pch(path=path, modal=source)
    loaded = read_pch(path)
    actual = getattr(loaded, "modal", loaded)

    assert path.read_text(encoding="utf-8").strip()
    np.testing.assert_allclose(actual.eigenvalues, source.eigenvalues, rtol=2.0e-6)
    np.testing.assert_allclose(actual.freq_hz, source.freq_hz, rtol=2.0e-6)
    np.testing.assert_allclose(actual.modes, source.modes, rtol=2.0e-6, atol=1.0e-10)
    assert actual.dof_index == source.dof_index

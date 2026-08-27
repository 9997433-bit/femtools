"""Mode pairing must recover a known permutation."""

from __future__ import annotations

from typing import Any

import numpy as np

from femtools.correlation import pairing as pairing_module


def _pairs_array(result: Any) -> np.ndarray:
    if hasattr(result, "as_pairs"):
        return np.asarray(result.as_pairs())
    for attribute in ("pairs", "assignments", "indices"):
        if hasattr(result, attribute):
            return _pairs_array(getattr(result, attribute))

    if isinstance(result, list) and result and hasattr(result[0], "index_a"):
        return np.array([(pair.index_a, pair.index_b) for pair in result])

    if isinstance(result, tuple):
        if len(result) >= 2:
            first = np.asarray(result[0])
            second = np.asarray(result[1])
            if first.ndim == second.ndim == 1 and first.size == second.size:
                return np.column_stack((first, second))
        if result:
            return _pairs_array(result[0])

    array = np.asarray(result)
    if array.ndim == 1:
        return np.column_stack((np.arange(array.size), array))
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, :2]
    raise AssertionError("pair_modes must return reference/candidate index pairs")


def test_pair_modes_recovers_permuted_and_sign_flipped_modes() -> None:
    rng = np.random.default_rng(735)
    reference, _ = np.linalg.qr(rng.standard_normal((15, 6)))
    permutation = np.array([4, 1, 5, 0, 3, 2])
    signs = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    candidate = reference[:, permutation] * signs

    result = pairing_module.pair_modes(reference, candidate)
    pairs = _pairs_array(result).astype(int)
    mapping = {reference_index: candidate_index for reference_index, candidate_index in pairs}
    expected = {mode: int(np.flatnonzero(permutation == mode)[0]) for mode in range(6)}

    assert mapping == expected

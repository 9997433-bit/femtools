"""Shared behaviour for pretest result objects.

Selection results carry a primary list of ids (mode indices, sensor DOFs)
plus diagnostic arrays.  The mixin makes the object behave like that id list
— ``np.asarray(result)``, ``len(result)``, iteration and indexing all address
the ids — while the extra information stays reachable as attributes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["IdSequenceMixin"]


class IdSequenceMixin:
    """Make a result object behave like its primary id array."""

    #: Name of the attribute holding the primary id array.
    _id_field: str = "ids"

    @property
    def _ids(self) -> NDArray[Any]:
        return np.asarray(getattr(self, self._id_field))

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[Any]:
        arr = self._ids
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __len__(self) -> int:
        return int(self._ids.size)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._ids.tolist())

    def __getitem__(self, item: Any) -> Any:
        return self._ids[item]

    def __contains__(self, item: Any) -> bool:
        return bool(np.isin(item, self._ids).any())

    def tolist(self) -> list[Any]:
        return list(self._ids.tolist())

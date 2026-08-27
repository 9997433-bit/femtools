"""Accept both write_*(path, model) and write_*(model, path) call conventions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from femtools.core.model import FEModel


def coerce_path_model(first: Any, second: Any) -> tuple[Path, FEModel]:
    """Return ``(path, model)`` regardless of argument order."""
    if isinstance(first, FEModel) and isinstance(second, (str, Path)):
        return Path(second), first
    if isinstance(second, FEModel) and isinstance(first, (str, Path)):
        return Path(first), second
    raise TypeError(
        "expected (path, model) or (model, path); "
        f"got {type(first).__name__} and {type(second).__name__}"
    )

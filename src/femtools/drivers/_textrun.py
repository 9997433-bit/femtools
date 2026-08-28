"""Shared plumbing for the text-only external solver drivers (Round 8).

:class:`~femtools.drivers.ansys.AnsysCdbDriver` and
:class:`~femtools.drivers.abaqus.AbaqusInpDriver` differ only in their
input translator, launch command line and native *binary* result format
(RST / ODB -- both N/A in femtools by design).  Everything else -- PATH
resolution with alias fallbacks, the ``subprocess`` launch with the
Round-7 :class:`~femtools.core.errors.SolverError` conventions (missing
executable / non-zero exit / timeout / no result), and reading modal
results back from ``.pch`` / ``.unv`` TEXT files -- lives here so the
three drivers cannot drift apart.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import SolverError
from ..io.pch import read_pch
from ..io.unv import read_unv

if TYPE_CHECKING:
    from ..core.results import ModalResult

#: Text result suffixes the drivers read back, in preference order.
TEXT_RESULT_SUFFIXES = (".pch", ".unv")


def resolve_executable(explicit: str | None, defaults: Sequence[str]) -> str | None:
    """First launcher found on the PATH: ``explicit`` when given, else the
    first hit among the ``defaults`` alias names.  Never raises."""
    for candidate in (explicit,) if explicit else defaults:
        hit = shutil.which(candidate)
        if hit is not None:
            return hit
    return None


def sanitized_stem(name: str) -> str:
    """Model name reduced to a safe job/file stem (Round-7 convention)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "model"


def run_text_solver(
    solver: str,
    cmd: list[str],
    input_file: Path,
    timeout: float | None,
    binary_suffix: str,
    binary_name: str,
) -> Path:
    """Launch ``cmd`` next to ``input_file`` and locate the text result.

    Raises :class:`~femtools.core.errors.SolverError` when the process
    exits non-zero, exceeds ``timeout`` (seconds) or leaves neither a
    ``.pch`` nor a ``.unv`` file next to the input deck.  A run that only
    produced the solver's native binary result (``binary_suffix``) gets a
    dedicated hint: femtools reads text results only.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=input_file.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SolverError(
            f"{solver} run on {input_file.name} exceeded the {timeout} s timeout"
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise SolverError(
            f"{solver} exited with status {proc.returncode} on {input_file.name}"
            + (f": {detail[:500]}" if detail else "")
        )
    for suffix in TEXT_RESULT_SUFFIXES:
        result = input_file.with_suffix(suffix)
        if result.is_file():
            return result
    hint = ""
    if input_file.with_suffix(binary_suffix).is_file():
        hint = (
            f" (only a {binary_suffix} was produced; {binary_name} binary results are "
            "N/A in femtools by design -- have the run export text modes instead)"
        )
    raise SolverError(
        f"{solver} run finished but produced no text modal result "
        f"({input_file.stem}.pch or {input_file.stem}.unv) next to the input deck{hint}"
    )


def read_text_modal(
    driver: str, result_file: str | Path, binary_suffix: str, binary_name: str
) -> ModalResult:
    """Modal result from a ``.pch`` / ``.unv`` TEXT file.

    A path with the solver's native binary suffix raises
    :class:`~femtools.core.errors.SolverError` naming that format as N/A
    (femtools ships no proprietary binary parsers); any other suffix
    raises with the list of supported text formats.
    """
    result_file = Path(result_file)
    suffix = result_file.suffix.lower()
    if suffix == binary_suffix:
        raise SolverError(
            f"{binary_name} binary results ({result_file.name}) are N/A: femtools ships "
            f"no {binary_name} parser by design -- export and read text modal output "
            "(.pch or .unv) instead"
        )
    if suffix == ".pch":
        return read_pch(result_file)
    if suffix in (".unv", ".uff"):
        data = read_unv(result_file)
        if data.modal is None:
            raise SolverError(f"{result_file.name} contains no modal data (no dataset 55)")
        return data.modal
    raise SolverError(
        f"{driver} cannot read {result_file.name}: only .pch and .unv text results are supported"
    )

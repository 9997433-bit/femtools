"""Nastran SOL 103 driver over text files: BDF in, punch out.

:class:`NastranPunchDriver` is the concrete
:class:`~femtools.drivers.base.SolverDriver` for a locally installed
Nastran (MSC/NX/MYSTRAN-style command line: ``nastran jobname``).  It is
built entirely from femtools translators and the standard library --
femtools still ships **no** proprietary binary parsers (no OP2, by
design; the punch file covers the modal exchange loop):

* :meth:`~NastranPunchDriver.write_input` writes the model with
  :func:`femtools.io.write_bdf` and splices a SOL 103 executive /
  case-control section around the bulk data (``METHOD = 1``,
  ``DISPLACEMENT(PUNCH) = ALL``, an ``EIGRL`` card requesting
  ``n_modes`` modes).
* :meth:`~NastranPunchDriver.run` launches the executable found via
  :func:`shutil.which` and raises
  :class:`~femtools.core.errors.SolverError` when it is missing, exits
  non-zero, times out or produces no ``.pch`` file.
* :meth:`~NastranPunchDriver.read_modal` reads the punch file back with
  :func:`femtools.io.read_pch`.

Nothing here requires Nastran to be installed:
:meth:`~NastranPunchDriver.is_available` probes the PATH and the test
suite exercises the loop with a stub executable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import SolverError
from ..io.bdf import write_bdf
from ..io.pch import read_pch

if TYPE_CHECKING:
    from ..core.model import FEModel
    from ..core.results import ModalResult

__all__ = ["NastranPunchDriver"]


class NastranPunchDriver:
    """SOL 103 normal modes through a local Nastran, results via punch.

    Implements the :class:`~femtools.drivers.base.SolverDriver` protocol
    (structurally; ``isinstance`` checks pass at runtime).

    Parameters
    ----------
    executable:
        Name (resolved on the PATH) or path of the Nastran launcher.
    n_modes:
        Number of modes requested on the EIGRL card (ND field).
    """

    name = "nastran-punch"

    def __init__(self, executable: str = "nastran", n_modes: int = 10) -> None:
        self.executable = str(executable)
        self.n_modes = int(n_modes)
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")

    def __repr__(self) -> str:
        return (
            f"NastranPunchDriver(executable={self.executable!r}, n_modes={self.n_modes})"
        )

    # -- SolverDriver protocol -------------------------------------------------
    def is_available(self) -> bool:
        """Whether the executable resolves on the PATH.  Never raises."""
        return shutil.which(self.executable) is not None

    def write_input(self, model: FEModel, workdir: str | Path) -> Path:
        """Write ``<model.name>.bdf`` into ``workdir`` as a complete SOL 103 deck.

        The bulk data section comes from :func:`femtools.io.write_bdf`
        (same translation losses, same aggregated warnings); the
        executive/case control requests punched displacements for every
        grid and ``n_modes`` Lanczos modes (EIGRL SID 1).
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in model.name) or "model"
        deck = workdir / f"{stem}.bdf"
        write_bdf(deck, model)
        bulk = deck.read_text(encoding="utf-8").splitlines()
        out = [
            "$ femtools NastranPunchDriver -- SOL 103 normal modes, punch output",
            "SOL 103",
            "CEND",
            f"TITLE = {(model.name or 'femtools model').upper()}",
            "ECHO = NONE",
            "METHOD = 1",
            "DISPLACEMENT(PUNCH) = ALL",
        ]
        for line in bulk:
            out.append(line)
            if line.upper().lstrip().startswith("BEGIN") and "BULK" in line.upper():
                out.append(f"{'EIGRL':<8s}{1:>8d}{'':16s}{self.n_modes:>8d}")
        deck.write_text("\n".join(out) + "\n", encoding="utf-8")
        return deck

    def run(self, input_file: str | Path, timeout: float | None = None) -> Path:
        """Execute Nastran on ``input_file`` and return the punch file path.

        Raises :class:`~femtools.core.errors.SolverError` when the
        executable is not available, exits with a non-zero status, exceeds
        ``timeout`` (seconds) or leaves no ``.pch`` next to the input deck.
        """
        input_file = Path(input_file)
        exe = shutil.which(self.executable)
        if exe is None:
            raise SolverError(
                f"Nastran executable {self.executable!r} not found on the PATH "
                "(NastranPunchDriver.is_available() is False)"
            )
        try:
            proc = subprocess.run(
                [exe, str(input_file)],
                cwd=input_file.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SolverError(
                f"Nastran run on {input_file.name} exceeded the {timeout} s timeout"
            ) from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise SolverError(
                f"Nastran exited with status {proc.returncode} on {input_file.name}"
                + (f": {detail[:500]}" if detail else "")
            )
        result = input_file.with_suffix(".pch")
        if not result.is_file():
            raise SolverError(
                f"Nastran run finished but produced no punch file at {result} "
                "(is DISPLACEMENT(PUNCH) supported by this executable?)"
            )
        return result

    def read_modal(self, result_file: str | Path) -> ModalResult:
        """Read frequencies and real mode shapes from a punch file."""
        return read_pch(result_file)

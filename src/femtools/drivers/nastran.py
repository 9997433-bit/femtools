"""Nastran SOL 103 / SOL 101 driver over text files: BDF in, punch out.

:class:`NastranPunchDriver` is the concrete
:class:`~femtools.drivers.base.SolverDriver` for a locally installed
Nastran (MSC/NX/MYSTRAN-style command line: ``nastran jobname``).  It is
built entirely from femtools translators and the standard library --
femtools still ships **no** proprietary binary parsers (no OP2, by
design; the punch file covers both exchange loops):

* :meth:`~NastranPunchDriver.write_input` writes the model with
  :func:`femtools.io.write_bdf` and splices an executive / case-control
  section around the bulk data.  The default (``sol=103``, unchanged
  since Round 7) requests punched displacements and ``n_modes`` Lanczos
  modes (``METHOD = 1``, ``DISPLACEMENT(PUNCH) = ALL``, an ``EIGRL``
  card); ``sol=101`` emits a linear-static case control instead
  (``SPC``/``LOAD`` set selection plus the same
  ``DISPLACEMENT(PUNCH) = ALL`` request, no ``EIGRL``).
* :meth:`~NastranPunchDriver.run` launches the executable found via
  :func:`shutil.which` and raises
  :class:`~femtools.core.errors.SolverError` when it is missing, exits
  non-zero, times out or produces no ``.pch`` file.
* :meth:`~NastranPunchDriver.read_modal` reads a SOL 103 punch back with
  :func:`femtools.io.read_pch`; :meth:`~NastranPunchDriver.read_static`
  reads a SOL 101 punch with :func:`femtools.io.read_pch_static`;
  :meth:`~NastranPunchDriver.read_stress` (Round 10) reads the
  ``$STRESSES`` blocks a ``write_input(..., stress=True)`` deck requests
  with :func:`femtools.io.read_pch_stress`.  A ``.op2`` path raises
  :class:`~femtools.core.errors.SolverError` naming OP2 as N/A.

Nothing here requires Nastran to be installed:
:meth:`~NastranPunchDriver.is_available` probes the PATH and the test
suite exercises both loops with stub executables.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import SolverError
from ..io.bdf import write_bdf
from ..io.pch import read_pch, read_pch_static, read_pch_stress

if TYPE_CHECKING:
    from ..core.model import FEModel
    from ..core.results import ModalResult, StaticResult
    from ..io.pch import PchStressResult

__all__ = ["NastranPunchDriver"]


class NastranPunchDriver:
    """SOL 103 modes (default) or SOL 101 statics through a local Nastran,
    results via punch.

    Implements the :class:`~femtools.drivers.base.SolverDriver` protocol
    (structurally; ``isinstance`` checks pass at runtime).

    Parameters
    ----------
    executable:
        Name (resolved on the PATH) or path of the Nastran launcher.
    n_modes:
        Number of modes requested on the EIGRL card (ND field, SOL 103
        decks only).
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

    def write_input(
        self, model: FEModel, workdir: str | Path, *, sol: int = 103, stress: bool = False
    ) -> Path:
        """Write ``<model.name>.bdf`` into ``workdir`` as a complete deck.

        The bulk data section comes from :func:`femtools.io.write_bdf`
        (same translation losses, same aggregated warnings).  With the
        default ``sol=103`` the executive/case control requests punched
        displacements for every grid and ``n_modes`` Lanczos modes (EIGRL
        SID 1) -- byte-identical to the Round-7 deck.  With ``sol=101``
        it requests a linear static solution instead: ``SPC``/``LOAD``
        select the model's constraint and load sets (the lowest set id
        each, matching the femtools default of 1; multiple set ids warn)
        and ``DISPLACEMENT(PUNCH) = ALL`` punches the displacements
        :meth:`read_static` reads back.  ``stress=True`` (Round 10)
        additionally requests ``STRESS(PUNCH) = ALL``, the element
        stresses :meth:`read_stress` reads back; the default deck is
        unchanged.  Any other ``sol`` raises :class:`ValueError`.
        """
        if sol not in (101, 103):
            raise ValueError(f"sol must be 101 (statics) or 103 (normal modes), got {sol}")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in model.name) or "model"
        deck = workdir / f"{stem}.bdf"
        write_bdf(deck, model)
        bulk = deck.read_text(encoding="utf-8").splitlines()
        title = f"TITLE = {(model.name or 'femtools model').upper()}"
        if sol == 103:
            out = [
                "$ femtools NastranPunchDriver -- SOL 103 normal modes, punch output",
                "SOL 103",
                "CEND",
                title,
                "ECHO = NONE",
                "METHOD = 1",
                "DISPLACEMENT(PUNCH) = ALL",
            ]
        else:
            out = [
                "$ femtools NastranPunchDriver -- SOL 101 linear statics, punch output",
                "SOL 101",
                "CEND",
                title,
                "ECHO = NONE",
                f"SPC = {self._case_set(model, 'spc')}",
                f"LOAD = {self._case_set(model, 'load')}",
                "DISPLACEMENT(PUNCH) = ALL",
            ]
        if stress:
            out.append("STRESS(PUNCH) = ALL")
        for line in bulk:
            out.append(line)
            if (
                sol == 103
                and line.upper().lstrip().startswith("BEGIN")
                and "BULK" in line.upper()
            ):
                out.append(f"{'EIGRL':<8s}{1:>8d}{'':16s}{self.n_modes:>8d}")
        deck.write_text("\n".join(out) + "\n", encoding="utf-8")
        return deck

    @staticmethod
    def _case_set(model: FEModel, kind: str) -> int:
        """Set id the SOL 101 case control selects: the lowest SPC/LOAD set
        id in the model (``write_bdf`` maps SPC sids < 1 to 1), default 1.
        A model with several set ids gets a warning -- one deck selects
        exactly one set of each kind."""
        if kind == "spc":
            sids = sorted({spc.sid if spc.sid > 0 else 1 for spc in model.spcs})
        else:
            sids = sorted({load.sid for load in model.loads})
        if not sids:
            return 1
        if len(sids) > 1:
            warnings.warn(
                f"NastranPunchDriver: model has {kind.upper()} set ids {sids}; "
                f"the SOL 101 case control selects only {kind.upper()} = {sids[0]}",
                UserWarning,
                stacklevel=3,
            )
        return sids[0]

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
            hint = (
                " (only a .op2 was produced; OP2 binary results are N/A in femtools "
                "by design -- request DISPLACEMENT(PUNCH) text output instead)"
                if input_file.with_suffix(".op2").is_file()
                else " (is DISPLACEMENT(PUNCH) supported by this executable?)"
            )
            raise SolverError(
                f"Nastran run finished but produced no punch file at {result}{hint}"
            )
        return result

    def read_modal(self, result_file: str | Path) -> ModalResult:
        """Read frequencies and real mode shapes from a punch file."""
        return read_pch(result_file)

    def read_static(self, result_file: str | Path) -> StaticResult:
        """Read static displacements from a SOL 101 punch file.

        Returns the :class:`~femtools.core.results.StaticResult` of
        :func:`femtools.io.read_pch_static` (one column per subcase).  A
        ``.op2`` path raises :class:`~femtools.core.errors.SolverError`:
        OP2 binary results are N/A in femtools by design -- request
        ``DISPLACEMENT(PUNCH)`` text output instead.
        """
        result_file = Path(result_file)
        if result_file.suffix.lower() == ".op2":
            raise SolverError(
                f"OP2 binary results ({result_file.name}) are N/A: femtools ships no "
                "OP2 parser by design -- request DISPLACEMENT(PUNCH) text output and "
                "read the .pch instead"
            )
        return read_pch_static(result_file)

    def read_stress(self, result_file: str | Path) -> PchStressResult:
        """Read element stresses from a punch file (Round 10).

        Returns the :class:`~femtools.io.pch.PchStressResult` of
        :func:`femtools.io.read_pch_stress` (one Voigt slab per subcase;
        request the block with ``write_input(..., stress=True)``).  A
        ``.op2`` path raises :class:`~femtools.core.errors.SolverError`:
        OP2 binary results are N/A in femtools by design -- request
        ``STRESS(PUNCH)`` text output instead.
        """
        result_file = Path(result_file)
        if result_file.suffix.lower() == ".op2":
            raise SolverError(
                f"OP2 binary results ({result_file.name}) are N/A: femtools ships no "
                "OP2 parser by design -- request STRESS(PUNCH) text output and "
                "read the .pch instead"
            )
        return read_pch_stress(result_file)

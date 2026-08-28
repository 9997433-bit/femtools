"""Round 9 io: Nastran text SOL 101 static punch (still no OP2).

Acceptance (R9-F2):

* ``read_pch_static`` parses public punch ``$DISPLACEMENTS`` blocks into a
  ``StaticResult`` (one column per ``$SUBCASE``); eigenvector / complex /
  other blocks are skipped with warnings, never an error; ``read_pch``
  keeps its Round-4 modal behaviour on the same files;
* ``NastranPunchDriver.write_input(..., sol=101)`` emits a public SOL 101
  case control requesting ``DISPLACEMENT(PUNCH) = ALL``; the default
  deck stays the byte-identical Round-7 SOL 103 layout;
* ``NastranPunchDriver.read_static`` reads that punch end to end with a
  stub executable only (no Nastran installation); missing executable /
  non-zero exit / timeout raise ``SolverError``; ``.op2`` paths raise
  ``SolverError`` naming OP2 as N/A (femtools ships no binary parsers).
"""

from __future__ import annotations

import stat
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from femtools.core.errors import SolverError
from femtools.core.model import FEModel
from femtools.core.results import ModalResult, StaticResult
from femtools.drivers import NastranPunchDriver, SolverDriver
from femtools.io import read_bdf, read_pch, read_pch_static, write_pch
from femtools.io.pch import PchError

STEEL = {"E": 2.1e11, "nu": 0.3, "rho": 7850.0}

# ---------------------------------------------------------------------------
# fixtures: public punch layout, written by hand (not by femtools itself)
# ---------------------------------------------------------------------------


def _seq(bodies: list[str]) -> str:
    """72-column bodies + the punch sequence number in columns 73-80."""
    return "\n".join(f"{b:<72.72s}{i + 1:>8d}" for i, b in enumerate(bodies)) + "\n"


def _static_bodies(
    cases: Sequence[tuple[int, dict[int, Sequence[float]]]],
    output_marker: str = "$REAL OUTPUT",
) -> list[str]:
    """Header + point lines of SOL 101 ``$DISPLACEMENTS`` blocks."""
    bodies: list[str] = []
    for subcase, points in cases:
        bodies += [
            "$TITLE   = FEMTOOLS STATIC FIXTURE",
            "$SUBTITLE= SOL 101",
            "$LABEL   =",
            "$DISPLACEMENTS",
            output_marker,
            f"$SUBCASE ID = {subcase:11d}",
        ]
        for nid in sorted(points):
            v = list(points[nid])
            bodies.append(f"{nid:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in v[:3]))
            bodies.append(f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in v[3:6]))
    return bodies


def _static_punch(
    cases: Sequence[tuple[int, dict[int, Sequence[float]]]],
    output_marker: str = "$REAL OUTPUT",
) -> str:
    return _seq(_static_bodies(cases, output_marker))


_CASE1 = {
    1: [0.0] * 6,
    2: [4.7619e-05, 0.0, -1.25e-03, 0.0, 2.5e-04, 0.0],
}
_CASE2 = {
    1: [0.0] * 6,
    2: [-9.5238e-05, 3.0e-04, 0.0, 1.0e-05, 0.0, 0.0],
}


def _modal_fixture() -> ModalResult:
    freq = np.array([10.0, 20.0])
    modes = np.zeros((12, 2))
    modes[6, 0] = 1.0
    modes[7, 1] = 1.0
    return ModalResult(
        freq_hz=freq,
        eigenvalues=(2.0 * np.pi * freq) ** 2,
        modes=modes,
        generalized_mass=np.ones(2),
        dof_index=tuple((n, d) for n in (1, 2) for d in range(6)),
    )


def _row(res: StaticResult, node: int, dof: int) -> int:
    assert res.dof_index is not None
    return res.dof_index.index((node, dof))


# ---------------------------------------------------------------------------
# read_pch_static: single and multiple subcases
# ---------------------------------------------------------------------------


def test_static_pch_single_subcase(tmp_path: Path) -> None:
    path = tmp_path / "static.pch"
    path.write_text(_static_punch([(1, _CASE1)]), encoding="utf-8")
    res = read_pch_static(path)
    assert isinstance(res, StaticResult)
    assert res.u.ndim == 1 and res.n_dof == 12
    assert res.load_case == 1
    assert res.dof_index == tuple((n, d) for n in (1, 2) for d in range(6))
    np.testing.assert_allclose(res.u[_row(res, 1, 0) : _row(res, 1, 5) + 1], 0.0)
    assert res.u[_row(res, 2, 0)] == pytest.approx(4.7619e-05)
    assert res.u[_row(res, 2, 2)] == pytest.approx(-1.25e-03)
    assert res.u[_row(res, 2, 4)] == pytest.approx(2.5e-04)


def test_static_pch_multiple_subcases_stack_columns(tmp_path: Path) -> None:
    path = tmp_path / "static2.pch"
    path.write_text(_static_punch([(1, _CASE1), (2, _CASE2)]), encoding="utf-8")
    res = read_pch_static(path)
    assert res.u.shape == (12, 2)
    assert res.load_case == (1, 2)
    assert res.u[_row(res, 2, 0), 0] == pytest.approx(4.7619e-05)
    assert res.u[_row(res, 2, 0), 1] == pytest.approx(-9.5238e-05)
    assert res.u[_row(res, 2, 1), 1] == pytest.approx(3.0e-04)


def test_static_pch_repeated_subcase_without_new_header(tmp_path: Path) -> None:
    # some punches repeat only the $SUBCASE line between subcases
    bodies = _static_bodies([(1, _CASE1)])
    bodies.append(f"$SUBCASE ID = {2:11d}")
    for nid in sorted(_CASE2):
        v = list(_CASE2[nid])
        bodies.append(f"{nid:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in v[:3]))
        bodies.append(f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in v[3:6]))
    path = tmp_path / "compact.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    res = read_pch_static(path)
    assert res.u.shape == (12, 2)
    assert res.load_case == (1, 2)
    assert res.u[_row(res, 2, 1), 1] == pytest.approx(3.0e-04)


def test_static_pch_without_subcase_headers_numbers_cases(tmp_path: Path) -> None:
    bodies = [
        "$DISPLACEMENTS",
        f"{5:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (1.0, 2.0, 3.0)),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in (4.0, 5.0, 6.0)),
    ]
    path = tmp_path / "bare.pch"
    path.write_text("\n".join(bodies) + "\n", encoding="utf-8")  # no sequence numbers
    res = read_pch_static(path)
    assert res.load_case == 1
    np.testing.assert_allclose(res.u, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# read_pch_static: tolerance (scalar points, D exponents, uneven subcases)
# ---------------------------------------------------------------------------


def test_static_pch_scalar_point_and_fortran_exponent(tmp_path: Path) -> None:
    bodies = [
        "$DISPLACEMENTS",
        "$REAL OUTPUT",
        f"$SUBCASE ID = {1:11d}",
        f"{1:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (0.0, 0.0, 0.0)),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in (0.0, 0.0, 0.0)),
        f"{7:>10d}{'S':>8s}      1.500000D-02",  # scalar point, Fortran D exponent
    ]
    path = tmp_path / "scalar.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    res = read_pch_static(path)
    assert res.dof_index is not None and (7, 0) in res.dof_index
    assert res.u[_row(res, 7, 0)] == pytest.approx(1.5e-02)
    assert res.n_dof == 7  # 6 grid DOFs + 1 scalar DOF


def test_static_pch_uneven_subcases_zero_fill_with_warning(tmp_path: Path) -> None:
    case2 = {2: _CASE2[2]}  # node 1 missing from the second subcase
    path = tmp_path / "uneven.pch"
    path.write_text(_static_punch([(1, _CASE1), (2, case2)]), encoding="utf-8")
    with pytest.warns(UserWarning, match="zero-filled"):
        res = read_pch_static(path)
    assert res.u.shape == (12, 2)
    np.testing.assert_allclose(res.u[_row(res, 1, 0), 1], 0.0)
    assert res.u[_row(res, 2, 1), 1] == pytest.approx(3.0e-04)


# ---------------------------------------------------------------------------
# read_pch_static: skipped blocks and errors
# ---------------------------------------------------------------------------


def test_static_pch_skips_eigenvector_blocks_with_warning(tmp_path: Path) -> None:
    modal = tmp_path / "modal.pch"
    write_pch(modal, _modal_fixture())
    mixed = tmp_path / "mixed.pch"
    mixed.write_text(
        modal.read_text(encoding="utf-8") + _static_punch([(1, _CASE1)]), encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="eigenvector"):
        res = read_pch_static(mixed)
    assert res.u.ndim == 1
    assert res.u[_row(res, 2, 2)] == pytest.approx(-1.25e-03)


def test_read_pch_on_mixed_file_keeps_modal_and_hints_static(tmp_path: Path) -> None:
    # the Round-4 modal reader is untouched: it still skips $DISPLACEMENTS
    # (now pointing at read_pch_static) and returns the eigenpairs
    modal = tmp_path / "modal.pch"
    write_pch(modal, _modal_fixture())
    mixed = tmp_path / "mixed.pch"
    mixed.write_text(
        modal.read_text(encoding="utf-8") + _static_punch([(1, _CASE1)]), encoding="utf-8"
    )
    with pytest.warns(UserWarning, match=r"DISPLACEMENTS.*read_pch_static"):
        out = read_pch(mixed)
    np.testing.assert_allclose(out.freq_hz, [10.0, 20.0], rtol=1.0e-6)
    assert out.modes.shape == (12, 2)


def test_static_pch_complex_output_skipped(tmp_path: Path) -> None:
    complex_only = tmp_path / "complex.pch"
    complex_only.write_text(
        _static_punch([(1, _CASE1)], output_marker="$REAL-IMAGINARY OUTPUT"),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="complex"), pytest.raises(PchError):
        read_pch_static(complex_only)

    # a real block after the complex one is still read
    both = tmp_path / "both.pch"
    both.write_text(
        _static_punch([(1, _CASE1)], output_marker="$REAL-IMAGINARY OUTPUT")
        + _static_punch([(2, _CASE2)]),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="complex"):
        res = read_pch_static(both)
    assert res.u.ndim == 1 and res.load_case == 2
    assert res.u[_row(res, 2, 1)] == pytest.approx(3.0e-04)


def test_static_pch_skips_other_blocks_with_warning(tmp_path: Path) -> None:
    bodies = _static_bodies([(1, _CASE1)])
    bodies += [
        "$SPCF",
        f"{1:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (9.0, 9.0, 9.0)),
    ]
    path = tmp_path / "spcf.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    with pytest.warns(UserWarning, match="SPCF"):
        res = read_pch_static(path)
    assert res.u[_row(res, 1, 0)] == pytest.approx(0.0)  # SPCF data not mixed in


def test_static_pch_no_displacements_raises(tmp_path: Path) -> None:
    modal = tmp_path / "modal.pch"
    write_pch(modal, _modal_fixture())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(PchError, match=r"no real static \$DISPLACEMENTS"):
            read_pch_static(modal)
    empty = tmp_path / "empty.pch"
    empty.write_text("$TITLE   = NOTHING HERE\n", encoding="utf-8")
    with pytest.raises(PchError):
        read_pch_static(empty)


def test_static_pch_malformed_lines_raise(tmp_path: Path) -> None:
    cont_first = tmp_path / "cont.pch"
    cont_first.write_text(
        "$DISPLACEMENTS\n-CONT-            1.0\n", encoding="utf-8"
    )
    with pytest.raises(PchError, match="-CONT-"):
        read_pch_static(cont_first)
    garbage = tmp_path / "garbage.pch"
    garbage.write_text(
        "$DISPLACEMENTS\n         1       G      not-a-number\n", encoding="utf-8"
    )
    with pytest.raises(PchError, match="cannot parse"):
        read_pch_static(garbage)


# ---------------------------------------------------------------------------
# NastranPunchDriver.write_input: SOL 101 deck (default stays SOL 103)
# ---------------------------------------------------------------------------


def _static_model() -> FEModel:
    m = FEModel(name="tiny truss")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="bar", material_id=1, A=1.0e-4)
    m.add_element(1, "BAR2", (1, 2), property_id=1)
    m.add_spc(1, [True] * 6)
    m.add_load(2, force=(100.0, 0.0, 0.0))
    return m


def test_write_input_sol101_deck_layout(tmp_path: Path) -> None:
    driver = NastranPunchDriver()
    deck = driver.write_input(_static_model(), tmp_path / "job", sol=101)
    assert deck.suffix == ".bdf" and deck.is_file()
    lines = deck.read_text(encoding="utf-8").splitlines()
    assert "SOL 101" in lines
    assert "CEND" in lines
    assert "SPC = 1" in lines
    assert "LOAD = 1" in lines
    assert "DISPLACEMENT(PUNCH) = ALL" in lines
    # no modal cards in a static deck
    assert "METHOD = 1" not in lines
    assert not any(ln.startswith("EIGRL") for ln in lines)
    # the deck is still valid bulk data for femtools' own reader
    loaded = read_bdf(deck)
    assert set(loaded.nodes) == {1, 2}
    assert len(loaded.loads) == 1 and loaded.loads[0].sid == 1


def test_write_input_default_is_the_round7_sol103_deck(tmp_path: Path) -> None:
    model = _static_model()
    driver = NastranPunchDriver(n_modes=7)
    default = driver.write_input(model, tmp_path / "a")
    explicit = driver.write_input(model, tmp_path / "b", sol=103)
    assert default.read_text() == explicit.read_text()
    lines = default.read_text(encoding="utf-8").splitlines()
    assert "SOL 103" in lines
    assert "METHOD = 1" in lines
    assert "DISPLACEMENT(PUNCH) = ALL" in lines
    bulk_at = next(i for i, ln in enumerate(lines) if ln.startswith("BEGIN BULK"))
    assert lines[bulk_at + 1].startswith("EIGRL")
    static = driver.write_input(model, tmp_path / "c", sol=101)
    assert static.read_text() != default.read_text()


def test_write_input_rejects_unknown_sol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sol"):
        NastranPunchDriver().write_input(_static_model(), tmp_path, sol=111)


def test_write_input_sol101_multiple_load_sets_warn(tmp_path: Path) -> None:
    model = _static_model()
    model.add_load(2, force=(0.0, 50.0, 0.0), sid=5)
    with pytest.warns(UserWarning, match="LOAD"):
        deck = NastranPunchDriver().write_input(model, tmp_path, sol=101)
    assert "LOAD = 1" in deck.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# NastranPunchDriver.read_static: full loop with stub executables only
# ---------------------------------------------------------------------------


def _stub_exe(tmp_path: Path, name: str, script: str) -> Path:
    exe = tmp_path / name
    exe.write_text(script, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _static_fixture_pch(tmp_path: Path) -> Path:
    fixture = tmp_path / "static-fixture.pch"
    fixture.write_text(_static_punch([(1, _CASE1)]), encoding="utf-8")
    return fixture


def test_driver_still_conforms_to_solver_driver_protocol() -> None:
    driver = NastranPunchDriver()
    assert isinstance(driver, SolverDriver)
    assert callable(driver.read_static)


def test_driver_static_full_loop_with_stub_executable(tmp_path: Path) -> None:
    fixture = _static_fixture_pch(tmp_path)
    exe = _stub_exe(
        tmp_path,
        "fake-nastran",
        f'#!/bin/sh\ncp "{fixture}" "$(dirname "$1")/$(basename "$1" .bdf).pch"\n',
    )
    driver = NastranPunchDriver(executable=str(exe))
    assert driver.is_available() is True
    deck = driver.write_input(_static_model(), tmp_path / "run", sol=101)
    result = driver.run(deck)
    assert result.suffix == ".pch" and result.parent == deck.parent
    res = driver.read_static(result)
    assert isinstance(res, StaticResult)
    assert res.load_case == 1 and res.u.ndim == 1
    assert res.u[_row(res, 2, 0)] == pytest.approx(4.7619e-05)
    np.testing.assert_allclose(res.u[_row(res, 1, 0)], 0.0)


def test_driver_static_missing_executable(tmp_path: Path) -> None:
    driver = NastranPunchDriver(executable="definitely-not-a-nastran-binary")
    assert driver.is_available() is False
    deck = driver.write_input(_static_model(), tmp_path, sol=101)
    with pytest.raises(SolverError, match="not found"):
        driver.run(deck)


def test_driver_static_run_failures_raise_solver_error(tmp_path: Path) -> None:
    deck = NastranPunchDriver().write_input(_static_model(), tmp_path, sol=101)
    failing = _stub_exe(tmp_path, "failing", "#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(SolverError, match="status 3"):
        NastranPunchDriver(executable=str(failing)).run(deck)
    silent = _stub_exe(tmp_path, "silent", "#!/bin/sh\nexit 0\n")
    with pytest.raises(SolverError, match="no punch file"):
        NastranPunchDriver(executable=str(silent)).run(deck)
    sleepy = _stub_exe(tmp_path, "sleepy", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(SolverError, match="timeout"):
        NastranPunchDriver(executable=str(sleepy)).run(deck, timeout=0.2)


def test_driver_run_with_only_op2_output_names_op2_as_na(tmp_path: Path) -> None:
    op2_only = _stub_exe(
        tmp_path,
        "op2-only",
        '#!/bin/sh\ntouch "$(dirname "$1")/$(basename "$1" .bdf).op2"\n',
    )
    driver = NastranPunchDriver(executable=str(op2_only))
    deck = driver.write_input(_static_model(), tmp_path / "run", sol=101)
    with pytest.raises(SolverError, match=r"no punch file.*OP2.*N/A"):
        driver.run(deck)


def test_driver_read_static_op2_path_is_na(tmp_path: Path) -> None:
    driver = NastranPunchDriver()
    op2 = tmp_path / "job.op2"
    op2.write_bytes(b"\x00\x01binary")
    with pytest.raises(SolverError, match=r"OP2.*N/A"):
        driver.read_static(op2)
    # the message names OP2 even if the file does not exist
    with pytest.raises(SolverError, match="OP2"):
        driver.read_static("missing.op2")


def test_driver_read_static_from_modal_only_punch_raises(tmp_path: Path) -> None:
    modal = tmp_path / "modal.pch"
    write_pch(modal, _modal_fixture())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(PchError, match="no real static"):
            NastranPunchDriver().read_static(modal)

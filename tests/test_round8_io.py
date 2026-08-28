"""Round 8 io: BDF RBE3 parse/emit + ANSYS/Abaqus text drivers.

Acceptance (R8-F2):

* ``read_bdf`` parses public RBE3 cards (refgrid / refc / WT,C,G lists,
  THRU) into ``FEModel.add_rbe3``; unsupported/degraded fields raise one
  aggregated ``UserWarning`` per file; ``write_bdf`` emits RBE3 cards
  that round-trip; RBE2 and INCLUDE behaviour are untouched (pinned by
  ``test_round7_io.py``);
* ``AnsysCdbDriver`` / ``AbaqusInpDriver`` implement ``SolverDriver``
  end to end without any solver installation (stub shell executables
  only, Round-7 convention): ``write_input`` is exactly the
  ``write_cdb`` / ``write_inp`` translator, availability comes from
  ``shutil.which`` (``ansys``/``mapdl`` aliases), ``run`` raises
  ``SolverError`` on missing executable / non-zero exit / timeout / no
  text result, and ``read_modal`` reads ``.pch``/``.unv`` text only --
  ``.rst`` / ``.odb`` binary paths raise ``SolverError`` naming the
  format as N/A.
"""

from __future__ import annotations

import stat
import warnings
from pathlib import Path

import numpy as np
import pytest

from femtools.core.errors import SolverError
from femtools.core.model import FEModel
from femtools.core.results import ModalResult
from femtools.drivers import SolverDriver
from femtools.drivers.abaqus import AbaqusInpDriver
from femtools.drivers.ansys import AnsysCdbDriver
from femtools.io import read_cdb, read_inp, write_cdb, write_inp, write_pch, write_unv
from femtools.io.bdf import BdfError, read_bdf, write_bdf

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_GRIDS = (
    "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,2.,0.,0.\nGRID,4,,3.,0.,0.\nGRID,10,,1.,1.,0.\n"
)


def _read(tmp_path: Path, bulk: str, name: str = "deck.bdf") -> FEModel:
    deck = tmp_path / name
    deck.write_text(f"BEGIN BULK\n{_GRIDS}{bulk}ENDDATA\n", encoding="utf-8")
    return read_bdf(deck)


def _read_catching(tmp_path: Path, bulk: str) -> tuple[FEModel, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _read(tmp_path, bulk)
    return model, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# read_bdf: RBE3
# ---------------------------------------------------------------------------


def test_bdf_rbe3_parses_into_model_table(tmp_path: Path) -> None:
    m, texts = _read_catching(
        tmp_path,
        "RBE3,21,,10,123,1.,123,1,2,+\n+,3\n"  # free-field continuation
        "RBE3,22,,10,12,2.,12,1,4\n",
    )
    assert not any("RBE3" in t for t in texts)  # clean cards: no warning
    assert len(m.rbe3) == 2 and not m.rbe2
    first = m.rbe3[0]
    assert (first.id, first.dependent, first.independents) == (21, 10, (1, 2, 3))
    assert first.components == (1, 2, 3)
    assert first.independent_components == (1, 2, 3)
    assert first.weights == (1.0, 1.0, 1.0)
    second = m.rbe3[1]
    assert (second.dependent, second.independents) == (10, (1, 4))
    assert second.components == (1, 2)
    assert second.independent_components == (1, 2)
    assert second.weights == (2.0, 2.0)


def test_bdf_rbe3_weight_groups_and_thru(tmp_path: Path) -> None:
    m = _read(tmp_path, "RBE3,23,,10,123,1.,123,1,THRU,3,+\n+,2.5,123,4\n")
    rbe = m.rbe3[0]
    assert rbe.independents == (1, 2, 3, 4)
    assert rbe.weights == (1.0, 1.0, 1.0, 2.5)


def test_bdf_rbe3_small_field_card(tmp_path: Path) -> None:
    deck = tmp_path / "small.bdf"
    deck.write_text(
        "BEGIN BULK\n"
        + _GRIDS
        + f"{'RBE3':<8s}{24:>8d}{'':8s}{10:>8d}{'123':>8s}{'1.':>8s}{'123':>8s}"
        + f"{1:>8d}{2:>8d}+\n"
        + f"{'+':<8s}{3:>8d}\n"
        + "ENDDATA\n",
        encoding="utf-8",
    )
    m = read_bdf(deck)
    assert m.rbe3[0].independents == (1, 2, 3)
    assert m.rbe3[0].weights == (1.0, 1.0, 1.0)


def test_bdf_rbe3_unsupported_fields_one_aggregated_warning(tmp_path: Path) -> None:
    # UM override, ALPHA tail and REFC beyond the independent components on
    # one deck: exactly ONE UserWarning mentioning RBE3, data degraded as
    # documented (femtools' RBE3 is a component-wise weighted average).
    m, texts = _read_catching(
        tmp_path,
        "RBE3,30,,10,123456,1.,123,1,2,+\n+,UM,1,123,+\n+,ALPHA,6.5-6\n"
        "RBE3,31,,10,123,1.,123,1,2,+\n+,2.,12,3\n",  # mixed Ci lists
    )
    assert sum("RBE3" in t for t in texts) == 1
    note = next(t for t in texts if "RBE3" in t)
    assert "UM" in note and "ALPHA" in note and "456" in note and "mixed" in note
    assert len(m.rbe3) == 2
    assert m.rbe3[0].components == (1, 2, 3)  # 456 dropped
    assert m.rbe3[0].independents == (1, 2)  # UM/ALPHA fields not read as grids
    assert m.rbe3[1].independent_components == (1, 2, 3)  # first group's list
    assert m.rbe3[1].weights == (1.0, 1.0, 2.0)


def test_bdf_rbe3_round7_pin_deck_still_warns(tmp_path: Path) -> None:
    # the exact deck test_round7_io pins: REFC=123456 from translation-only
    # independents must keep warning (and now parses the supported subset)
    deck = tmp_path / "rbe3.bdf"
    deck.write_text(
        "BEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nRBE3,7,,1,123456,1.,123,2\nENDDATA\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="RBE3"):
        m = read_bdf(deck)
    assert not m.rbe2
    assert len(m.rbe3) == 1
    assert m.rbe3[0].components == (1, 2, 3)
    assert m.rbe3[0].independents == (2,)


def test_bdf_rbe3_malformed_cards_raise(tmp_path: Path) -> None:
    with pytest.raises(BdfError, match="REFC"):
        _read(tmp_path, "RBE3,40,,10,127,1.,123,1\n")
    with pytest.raises(BdfError, match="Ci must be digits"):
        _read(tmp_path, "RBE3,41,,10,123,1.,9,1\n")
    with pytest.raises(BdfError, match="positive"):
        _read(tmp_path, "RBE3,42,,10,123,-1.,123,1\n")
    with pytest.raises(BdfError, match="before the first"):
        _read(tmp_path, "RBE3,43,,10,123,1,2\n")
    with pytest.raises(BdfError, match="no independent grids"):
        _read(tmp_path, "RBE3,44,,10,123\n")
    with pytest.raises(BdfError, match="THRU"):
        _read(tmp_path, "RBE3,45,,10,123,1.,123,THRU,3\n")


def test_bdf_rbe3_write_read_roundtrip(tmp_path: Path) -> None:
    src = FEModel(name="rt")
    for i in (1, 2, 3, 4, 10, 11):
        src.add_node(i, (float(i), 0.0, 0.0))
    src.add_rbe3(60, dependent=10, independents=(1, 2, 3))  # default comps, no weights
    src.add_rbe3(
        61,
        dependent=11,
        independents=(1, 2, 3, 4),
        components=(1, 3),
        independent_components=(1, 2, 3),
        weights=(1.0, 1.0, 2.5, 0.5),
    )
    path = tmp_path / "rt.bdf"
    write_bdf(path, src)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = read_bdf(path)
    assert not any("RBE3" in str(w.message) for w in caught)
    assert len(loaded.rbe3) == 2
    a, b = loaded.rbe3
    assert (a.id, a.dependent, a.independents) == (60, 10, (1, 2, 3))
    assert a.components == (1, 2, 3) and a.independent_components == (1, 2, 3)
    assert a.weights == (1.0, 1.0, 1.0)  # implicit equal weights become explicit
    assert (b.id, b.dependent, b.independents) == (61, 11, (1, 2, 3, 4))
    assert b.components == (1, 3) and b.independent_components == (1, 2, 3)
    assert b.weights == (1.0, 1.0, 2.5, 0.5)


def test_bdf_rbe2_and_rbe3_coexist(tmp_path: Path) -> None:
    m = _read(
        tmp_path,
        "RBE2,50,1,123456,2,3\nRBE3,51,,10,123,1.,123,1,4\n",
    )
    assert len(m.rbe2) == 1 and len(m.rbe2[0].dependents) == 2
    assert len(m.rbe3) == 1 and m.rbe3[0].independents == (1, 4)
    out = tmp_path / "both.bdf"
    write_bdf(out, m)
    loaded = read_bdf(out)
    assert len(loaded.rbe2) == 1 and loaded.rbe2[0].dependents == (2, 3)
    assert len(loaded.rbe3) == 1 and loaded.rbe3[0].independents == (1, 4)


# ---------------------------------------------------------------------------
# drivers: shared fixtures (no ANSYS / Abaqus installation anywhere)
# ---------------------------------------------------------------------------

STEEL = {"E": 2.1e11, "nu": 0.3, "rho": 7850.0}


def _driver_model() -> FEModel:
    m = FEModel(name="tiny truss")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="bar", material_id=1, A=1.0e-4)
    m.add_element(1, "BAR2", (1, 2), property_id=1)
    m.add_spc(1, [True] * 6)
    return m


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


def _stub_exe(tmp_path: Path, name: str, script: str) -> Path:
    exe = tmp_path / name
    exe.write_text(script, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


_DRIVERS = pytest.mark.parametrize(
    "driver_cls, expected_name, binary",
    [(AnsysCdbDriver, "ansys-cdb", "RST"), (AbaqusInpDriver, "abaqus-inp", "ODB")],
    ids=["ansys", "abaqus"],
)


# ---------------------------------------------------------------------------
# drivers: contract, availability, read_modal (solver-independent parts)
# ---------------------------------------------------------------------------


@_DRIVERS
def test_driver_conforms_to_solver_driver_protocol(driver_cls, expected_name, binary) -> None:
    driver = driver_cls()
    assert isinstance(driver, SolverDriver)
    assert driver.name == expected_name


@_DRIVERS
def test_driver_missing_executable(driver_cls, expected_name, binary, tmp_path: Path) -> None:
    driver = driver_cls(executable="definitely-not-a-solver-binary")
    assert driver.is_available() is False
    deck = driver.write_input(_driver_model(), tmp_path)
    with pytest.raises(SolverError, match="not found"):
        driver.run(deck)


@_DRIVERS
def test_driver_read_modal_from_pch_and_unv_text(
    driver_cls, expected_name, binary, tmp_path: Path
) -> None:
    driver = driver_cls()
    modal = _modal_fixture()
    pch = tmp_path / "fixture.pch"
    write_pch(pch, modal)
    out = driver.read_modal(pch)
    np.testing.assert_allclose(out.freq_hz, [10.0, 20.0], rtol=1.0e-6)
    assert out.modes.shape == (12, 2)
    unv = tmp_path / "fixture.unv"
    write_unv(unv, modal=modal)
    out = driver.read_modal(unv)
    np.testing.assert_allclose(out.freq_hz, [10.0, 20.0], rtol=1.0e-6)


@_DRIVERS
def test_driver_read_modal_binary_result_is_na(
    driver_cls, expected_name, binary, tmp_path: Path
) -> None:
    driver = driver_cls()
    suffix = f".{binary.lower()}"
    binary_file = tmp_path / f"job{suffix}"
    binary_file.write_bytes(b"\x00\x01binary")
    with pytest.raises(SolverError, match=rf"{binary}.*N/A"):
        driver.read_modal(binary_file)
    # the message names the binary format even if the file does not exist
    with pytest.raises(SolverError, match=binary):
        driver.read_modal(f"missing{suffix}")


@_DRIVERS
def test_driver_read_modal_rejects_other_files(
    driver_cls, expected_name, binary, tmp_path: Path
) -> None:
    driver = driver_cls()
    listing = tmp_path / "job.out"
    listing.write_text("solver listing\n", encoding="utf-8")
    with pytest.raises(SolverError, match=r"only \.pch and \.unv"):
        driver.read_modal(listing)
    empty_unv = tmp_path / "empty.unv"
    write_unv(empty_unv, model=_driver_model())  # a model, but no dataset 55
    with pytest.raises(SolverError, match="no modal data"):
        driver.read_modal(empty_unv)


# ---------------------------------------------------------------------------
# AnsysCdbDriver
# ---------------------------------------------------------------------------


def test_ansys_write_input_is_exactly_write_cdb(tmp_path: Path) -> None:
    model = _driver_model()
    deck = AnsysCdbDriver().write_input(model, tmp_path / "job")
    assert deck.name == "tiny_truss.cdb" and deck.is_file()
    direct = tmp_path / "direct.cdb"
    write_cdb(direct, model)
    assert deck.read_text() == direct.read_text()
    loaded = read_cdb(deck)
    assert set(loaded.nodes) == {1, 2}
    assert loaded.elements[1].type == "BAR2"


def test_ansys_is_available_probes_ansys_and_mapdl_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert AnsysCdbDriver().is_available() is False
    _stub_exe(tmp_path, "mapdl", "#!/bin/sh\nexit 0\n")
    assert AnsysCdbDriver().is_available() is True  # second alias
    _stub_exe(tmp_path, "ansys", "#!/bin/sh\nexit 0\n")
    assert AnsysCdbDriver().is_available() is True  # first alias
    assert AnsysCdbDriver(executable="mapdl241").is_available() is False


def test_ansys_full_loop_with_stub_executable(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.pch"
    write_pch(fixture, _modal_fixture())
    # MAPDL batch line: <exe> -b -i <deck> -o <out>; the deck path is $3
    exe = _stub_exe(
        tmp_path,
        "fake-mapdl",
        f'#!/bin/sh\ncp "{fixture}" "${{3%.cdb}}.pch"\n',
    )
    driver = AnsysCdbDriver(executable=str(exe))
    assert driver.is_available() is True
    deck = driver.write_input(_driver_model(), tmp_path / "run")
    result = driver.run(deck)
    assert result.suffix == ".pch" and result.parent == deck.parent
    modal = driver.read_modal(result)
    np.testing.assert_allclose(modal.freq_hz, [10.0, 20.0], rtol=1.0e-6)
    assert modal.modes.shape == (12, 2)


def test_ansys_run_failures_raise_solver_error(tmp_path: Path) -> None:
    deck = AnsysCdbDriver().write_input(_driver_model(), tmp_path)
    failing = _stub_exe(tmp_path, "failing", "#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(SolverError, match="status 3"):
        AnsysCdbDriver(executable=str(failing)).run(deck)
    silent = _stub_exe(tmp_path, "silent", "#!/bin/sh\nexit 0\n")
    with pytest.raises(SolverError, match="no text modal result"):
        AnsysCdbDriver(executable=str(silent)).run(deck)
    sleepy = _stub_exe(tmp_path, "sleepy", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(SolverError, match="timeout"):
        AnsysCdbDriver(executable=str(sleepy)).run(deck, timeout=0.2)


def test_ansys_run_with_only_rst_output_names_rst_as_na(tmp_path: Path) -> None:
    rst_only = _stub_exe(tmp_path, "rst-only", '#!/bin/sh\ntouch "${3%.cdb}.rst"\n')
    driver = AnsysCdbDriver(executable=str(rst_only))
    deck = driver.write_input(_driver_model(), tmp_path / "run")
    with pytest.raises(SolverError, match="RST.*N/A"):
        driver.run(deck)


# ---------------------------------------------------------------------------
# AbaqusInpDriver
# ---------------------------------------------------------------------------


def test_abaqus_write_input_is_exactly_write_inp(tmp_path: Path) -> None:
    model = _driver_model()
    deck = AbaqusInpDriver().write_input(model, tmp_path / "job")
    assert deck.name == "tiny_truss.inp" and deck.is_file()
    direct = tmp_path / "direct.inp"
    write_inp(direct, model)
    assert deck.read_text() == direct.read_text()
    loaded = read_inp(deck)
    assert set(loaded.nodes) == {1, 2}
    assert loaded.elements[1].type == "BAR2"


def test_abaqus_full_loop_with_stub_executable(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.pch"
    write_pch(fixture, _modal_fixture())
    # Abaqus line: <exe> job=<stem> input=<deck name> interactive, cwd = deck dir
    exe = _stub_exe(
        tmp_path,
        "fake-abaqus",
        f'#!/bin/sh\nn="${{2#input=}}"\ncp "{fixture}" "${{n%.inp}}.pch"\n',
    )
    driver = AbaqusInpDriver(executable=str(exe))
    assert driver.is_available() is True
    deck = driver.write_input(_driver_model(), tmp_path / "run")
    result = driver.run(deck)
    assert result.suffix == ".pch" and result.parent == deck.parent
    modal = driver.read_modal(result)
    np.testing.assert_allclose(modal.freq_hz, [10.0, 20.0], rtol=1.0e-6)
    assert modal.modes.shape == (12, 2)


def test_abaqus_run_failures_raise_solver_error(tmp_path: Path) -> None:
    deck = AbaqusInpDriver().write_input(_driver_model(), tmp_path)
    failing = _stub_exe(tmp_path, "failing", "#!/bin/sh\necho boom >&2\nexit 2\n")
    with pytest.raises(SolverError, match="status 2"):
        AbaqusInpDriver(executable=str(failing)).run(deck)
    silent = _stub_exe(tmp_path, "silent", "#!/bin/sh\nexit 0\n")
    with pytest.raises(SolverError, match="no text modal result"):
        AbaqusInpDriver(executable=str(silent)).run(deck)
    sleepy = _stub_exe(tmp_path, "sleepy", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(SolverError, match="timeout"):
        AbaqusInpDriver(executable=str(sleepy)).run(deck, timeout=0.2)


def test_abaqus_run_with_only_odb_output_names_odb_as_na(tmp_path: Path) -> None:
    odb_only = _stub_exe(
        tmp_path, "odb-only", '#!/bin/sh\nn="${2#input=}"\ntouch "${n%.inp}.odb"\n'
    )
    driver = AbaqusInpDriver(executable=str(odb_only))
    deck = driver.write_input(_driver_model(), tmp_path / "run")
    with pytest.raises(SolverError, match="ODB.*N/A"):
        driver.run(deck)

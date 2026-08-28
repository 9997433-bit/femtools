"""Round 7 io: write_cdb / write_k round-trips, BDF INCLUDE + RBE2, Nastran driver.

Acceptance (R7-F2):

* the Round-6 HEX8 cube, QUAD4 plate and BEAM2 line written as CDB (via
  ``write_cdb``) and as K (via ``write_k``) must ``read_*`` back and
  ``assemble_km`` without crash, round-tripping nodes / connectivity /
  E / nu / rho / thickness / section values;
* ``read_bdf`` follows INCLUDE (relative to the including file, max depth
  8, cycle-safe) and parses RBE2 into ``FEModel.add_rbe2``;
* ``NastranPunchDriver`` implements ``SolverDriver`` end to end without a
  Nastran installation (stub executable only).
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
from femtools.drivers import NastranPunchDriver, SolverDriver
from femtools.fea import assemble_km
from femtools.io import read_bdf, read_cdb, read_k, write_bdf, write_cdb, write_k, write_pch
from femtools.io.bdf import BdfError

# ---------------------------------------------------------------------------
# reference models (the Round-6 acceptance trio, built on the core database)
# ---------------------------------------------------------------------------

_CUBE_XYZ = [
    (1, 0.0, 0.0, 0.0),
    (2, 1.0, 0.0, 0.0),
    (3, 1.0, 1.0, 0.0),
    (4, 0.0, 1.0, 0.0),
    (5, 0.0, 0.0, 1.0),
    (6, 1.0, 0.0, 1.0),
    (7, 1.0, 1.0, 1.0),
    (8, 0.0, 1.0, 1.0),
]

STEEL = {"E": 2.1e11, "nu": 0.3, "rho": 7850.0}


def _cube_model() -> FEModel:
    m = FEModel(name="cube")
    for nid, x, y, z in _CUBE_XYZ:
        m.add_node(nid, (x, y, z))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"], name="STEEL")
    m.add_property(1, type="solid", material_id=1)
    m.add_element(1, "HEX8", (1, 2, 3, 4, 5, 6, 7, 8), property_id=1)
    for nid in (1, 2, 3, 4):
        m.add_spc(nid, [True] * 6)
    return m


def _plate_model() -> FEModel:
    m = FEModel(name="plate")
    for i, (x, y) in enumerate(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)], start=1
    ):
        m.add_node(i, (x, y, 0.0))
    m.add_material(1, E=7.0e10, nu=0.33, rho=2700.0, name="ALU")
    m.add_property(1, type="shell", material_id=1, t=0.002)
    m.add_element(1, "QUAD4", (1, 2, 5, 4), property_id=1)
    m.add_element(2, "QUAD4", (2, 3, 6, 5), property_id=1)
    m.add_spc(1, [True] * 6)
    m.add_spc(4, [True] * 6)
    return m


def _beam_model() -> FEModel:
    m = FEModel(name="beamline")
    for i in range(4):
        m.add_node(i + 1, (0.5 * i, 0.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"], name="STEEL")
    m.add_property(1, type="beam", material_id=1, A=1.0e-4, Iy=1.0e-8, Iz=2.0e-8, J=3.0e-8)
    for i in range(3):
        m.add_element(i + 1, "BEAM2", (i + 1, i + 2), property_id=1, orientation=(0.0, 0.0, 1.0))
    m.add_spc(1, [True] * 6)
    return m


def _assert_assembles(model: FEModel) -> None:
    asm = assemble_km(model)
    assert asm.K.shape == (model.ndof, model.ndof)
    assert asm.M.shape == asm.K.shape
    assert not asm.skipped_elements
    assert asm.n_free > 0
    kd = asm.K.diagonal()
    md = asm.M.diagonal()
    assert np.all(kd >= 0.0) and kd[asm.free_dof].sum() > 0.0
    assert np.all(md >= 0.0) and md[asm.free_dof].sum() > 0.0


def _assert_roundtrip(source: FEModel, loaded: FEModel) -> None:
    """Nodes / connectivity / material / section round-trip checks.

    ``loaded`` may carry extra nodes (materialized beam orientation
    nodes), always numbered after the source nodes.
    """
    assert set(source.nodes) <= set(loaded.nodes)
    extras = set(loaded.nodes) - set(source.nodes)
    assert all(nid > max(source.nodes) for nid in extras)
    for nid in source.nodes:
        np.testing.assert_allclose(loaded.nodes[nid].xyz, source.nodes[nid].xyz, atol=1.0e-12)
    assert set(loaded.elements) == set(source.elements)
    for eid, el in source.elements.items():
        out = loaded.elements[eid]
        assert out.type == el.type
        assert out.nodes == el.nodes
        if el.orientation is not None:
            np.testing.assert_allclose(out.orientation, el.orientation, atol=1.0e-12)
        src_m, out_m = source.element_material(eid), loaded.element_material(eid)
        assert out_m.E == pytest.approx(src_m.E)
        assert out_m.nu == pytest.approx(src_m.nu)
        assert out_m.rho == pytest.approx(src_m.rho)
        src_p, out_p = source.element_property(eid), loaded.element_property(eid)
        assert out_p.type == src_p.type
        for fieldname in ("t", "A", "Iy", "Iz", "J"):
            v = getattr(src_p, fieldname)
            if v is not None:
                assert getattr(out_p, fieldname) == pytest.approx(v)
    src_spc = {(s.node_id, s.mask) for s in source.spcs}
    out_spc = {(s.node_id, s.mask) for s in loaded.spcs}
    assert src_spc == out_spc
    _assert_assembles(loaded)


# ---------------------------------------------------------------------------
# write_cdb
# ---------------------------------------------------------------------------


def test_cdb_write_read_roundtrip_cube_plate_beam(tmp_path: Path) -> None:
    for build in (_cube_model, _plate_model, _beam_model):
        source = build()
        path = tmp_path / f"{source.name}.cdb"
        write_cdb(path, source)
        loaded = read_cdb(path)
        assert loaded.name == source.name  # read_cdb names the model after the file
        _assert_roundtrip(source, loaded)


def test_cdb_write_accepts_both_argument_orders(tmp_path: Path) -> None:
    model = _cube_model()
    write_cdb(model, tmp_path / "a.cdb")
    write_cdb(tmp_path / "b.cdb", model)
    assert (tmp_path / "a.cdb").read_text() == (tmp_path / "b.cdb").read_text()


def test_cdb_writer_shares_beam_orientation_nodes(tmp_path: Path) -> None:
    m = FEModel(name="fan")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_node(3, (0.0, 1.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="beam", material_id=1, A=1e-4, Iy=1e-8, Iz=1e-8, J=2e-8)
    # both beams leave node 1 with the same orientation -> one shared K node
    m.add_element(1, "BEAM2", (1, 2), property_id=1, orientation=(0.0, 0.0, 1.0))
    m.add_element(2, "BEAM2", (1, 3), property_id=1, orientation=(0.0, 0.0, 1.0))
    path = tmp_path / "fan.cdb"
    write_cdb(path, m)
    loaded = read_cdb(path)
    assert set(loaded.nodes) == {1, 2, 3, 4}  # exactly one extra node
    np.testing.assert_allclose(loaded.nodes[4].xyz, (0.0, 0.0, 1.0))
    for eid in (1, 2):
        np.testing.assert_allclose(loaded.elements[eid].orientation, (0.0, 0.0, 1.0))


def test_cdb_writer_mass_and_spring_values_roundtrip(tmp_path: Path) -> None:
    m = FEModel(name="lumped")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_property(1, type="lumped", m=2.5)
    m.add_property(2, type="lumped", k=1.0e6)
    m.add_element(1, "MASS", (1,), property_id=1)
    m.add_element(2, "SPRING", (1, 2), property_id=2)
    path = tmp_path / "lumped.cdb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lumped elements carry mat id 0 on read
        write_cdb(path, m)
        loaded = read_cdb(path)
    assert loaded.elements[1].type == "MASS"
    assert loaded.element_property(1).m == pytest.approx(2.5)
    assert loaded.elements[2].type == "SPRING"
    assert loaded.element_property(2).k == pytest.approx(1.0e6)


def test_cdb_writer_documented_losses_warn(tmp_path: Path) -> None:
    m = _cube_model()
    m.add_property(9, type="lumped", c=100.0)
    m.add_element(9, "DAMPER", (1, 5), dofs=(2, 2), property_id=9)
    m.add_rbe2(1, independent=5, dependents=(6, 7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_cdb(tmp_path / "loss.cdb", m)
    texts = [str(w.message) for w in caught]
    assert sum("DAMPER" in t for t in texts) == 1
    assert sum("RBE2" in t for t in texts) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = read_cdb(tmp_path / "loss.cdb")
    assert 9 not in loaded.elements  # damper dropped, everything else kept
    assert set(loaded.elements) == {1}


# ---------------------------------------------------------------------------
# write_k
# ---------------------------------------------------------------------------


def test_k_write_read_roundtrip_cube_plate_beam(tmp_path: Path) -> None:
    for build in (_cube_model, _plate_model, _beam_model):
        source = build()
        path = tmp_path / f"{source.name}.k"
        write_k(path, source)
        loaded = read_k(path)
        assert loaded.name == source.name  # *TITLE carries the model name
        _assert_roundtrip(source, loaded)
        # the keyword format preserves property ids and material names
        for eid in source.elements:
            assert loaded.elements[eid].property_id == source.elements[eid].property_id
            assert loaded.element_material(eid).name == source.element_material(eid).name


def test_k_write_accepts_both_argument_orders(tmp_path: Path) -> None:
    model = _plate_model()
    write_k(model, tmp_path / "a.k")
    write_k(tmp_path / "b.k", model)
    assert (tmp_path / "a.k").read_text() == (tmp_path / "b.k").read_text()


def test_k_writer_documented_losses_warn(tmp_path: Path) -> None:
    m = _plate_model()
    m.add_property(9, type="lumped", m=1.0)
    m.add_element(9, "MASS", (6,), property_id=9)
    m.add_load(6, force=(0.0, 0.0, -1.0))
    m.add_spc(5, [False, False, True, False, False, False], value=0.001)
    m.add_rbe2(1, independent=2, dependents=(3,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_k(tmp_path / "loss.k", m)
    texts = [str(w.message) for w in caught]
    assert sum("MASS element" in t for t in texts) == 1
    assert sum("lumped property" in t for t in texts) == 1
    assert sum("load" in t for t in texts) == 1
    assert sum("enforced SPC" in t for t in texts) == 1
    assert sum("RBE2" in t for t in texts) == 1
    loaded = read_k(tmp_path / "loss.k")
    assert 9 not in loaded.elements
    # the enforced SPC reads back as fixed at zero
    spc5 = [s for s in loaded.spcs if s.node_id == 5]
    assert len(spc5) == 1 and spc5[0].mask[2] and spc5[0].value == 0.0
    _assert_assembles(loaded)


def test_k_writer_truss2d_reads_back_as_bar2(tmp_path: Path) -> None:
    m = FEModel(name="truss")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="bar", material_id=1, A=3.0e-4)
    m.add_element(1, "TRUSS2D", (1, 2), property_id=1)
    with pytest.warns(UserWarning, match="TRUSS2D"):
        write_k(tmp_path / "truss.k", m)
    loaded = read_k(tmp_path / "truss.k")
    assert loaded.elements[1].type == "BAR2"
    assert loaded.element_property(1).A == pytest.approx(3.0e-4)


# ---------------------------------------------------------------------------
# acceptance: the same three tiny models through both new writers
# ---------------------------------------------------------------------------


def test_acceptance_cube_plate_beam_through_cdb_and_k(tmp_path: Path) -> None:
    """R7-F2 acceptance: HEX8 cube, QUAD4 plate, BEAM2 line written as CDB
    and as K read back and assemble through assemble_km."""
    for build in (_cube_model, _plate_model, _beam_model):
        source = build()
        cdb = tmp_path / f"{source.name}.cdb"
        write_cdb(cdb, source)
        _assert_assembles(read_cdb(cdb))
        k = tmp_path / f"{source.name}.k"
        write_k(k, source)
        _assert_assembles(read_k(k))


# ---------------------------------------------------------------------------
# read_bdf: INCLUDE
# ---------------------------------------------------------------------------


def test_bdf_include_resolves_relative_to_including_file(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "main.bdf").write_text(
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "INCLUDE 'sub/part.blk'\n"
        "MAT1,1,2.1+11,,0.3,7850.\n"
        "ENDDATA\n",
        encoding="utf-8",
    )
    # nested INCLUDE is relative to sub/, not to the top file; the bare
    # (unquoted) form and $ comments after the name are accepted
    (sub / "part.blk").write_text(
        'GRID,2,,1.,0.,0.\nINCLUDE "deeper.blk"\n', encoding="utf-8"
    )
    (sub / "deeper.blk").write_text(
        "INCLUDE tail.blk $ bare form\n", encoding="utf-8"
    )
    (sub / "tail.blk").write_text("GRID,3,,2.,0.,0.\n", encoding="utf-8")

    m = read_bdf(tmp_path / "main.bdf")
    assert set(m.nodes) == {1, 2, 3}
    assert 1 in m.materials  # cards after the INCLUDE are still read


def _include_chain(tmp_path: Path, depth: int) -> Path:
    top = tmp_path / "top.bdf"
    top.write_text("BEGIN BULK\nINCLUDE 'f1.blk'\nENDDATA\n", encoding="utf-8")
    for i in range(1, depth + 1):
        body = f"GRID,{i},,{float(i)},0.,0.\n"
        if i < depth:
            body += f"INCLUDE 'f{i + 1}.blk'\n"
        (tmp_path / f"f{i}.blk").write_text(body, encoding="utf-8")
    return top


def test_bdf_include_depth_eight_ok_nine_raises(tmp_path: Path) -> None:
    ok = tmp_path / "ok"
    ok.mkdir()
    m = read_bdf(_include_chain(ok, 8))
    assert set(m.nodes) == set(range(1, 9))

    deep = tmp_path / "deep"
    deep.mkdir()
    with pytest.raises(BdfError, match="deeper than 8"):
        read_bdf(_include_chain(deep, 9))


def test_bdf_include_cycle_detected(tmp_path: Path) -> None:
    (tmp_path / "a.bdf").write_text("BEGIN BULK\nINCLUDE 'b.blk'\n", encoding="utf-8")
    (tmp_path / "b.blk").write_text("INCLUDE 'a.bdf'\n", encoding="utf-8")
    with pytest.raises(BdfError, match="cycle"):
        read_bdf(tmp_path / "a.bdf")


def test_bdf_include_missing_file_raises(tmp_path: Path) -> None:
    (tmp_path / "main.bdf").write_text(
        "BEGIN BULK\nINCLUDE 'nope.blk'\n", encoding="utf-8"
    )
    with pytest.raises(BdfError, match="not found"):
        read_bdf(tmp_path / "main.bdf")


def test_bdf_include_in_case_control_is_not_expanded(tmp_path: Path) -> None:
    # INCLUDE lines in the skipped executive/case-control section are not
    # followed: a dangling name there must not raise
    (tmp_path / "cc.bdf").write_text(
        "SOL 103\nINCLUDE 'never-read.blk'\nCEND\nBEGIN BULK\n"
        "GRID,1,,0.,0.,0.\nENDDATA\n",
        encoding="utf-8",
    )
    m = read_bdf(tmp_path / "cc.bdf")
    assert set(m.nodes) == {1}


# ---------------------------------------------------------------------------
# read_bdf / write_bdf: RBE2
# ---------------------------------------------------------------------------


def test_bdf_rbe2_parses_into_model_table(tmp_path: Path) -> None:
    deck = tmp_path / "rbe2.bdf"
    deck.write_text(
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,2.,0.,0.\nGRID,4,,3.,0.,0.\n"
        "RBE2,99,1,123,2,3\n"
        "RBE2,100,4,123456,1\n"
        "ENDDATA\n",
        encoding="utf-8",
    )
    m = read_bdf(deck)
    assert len(m.rbe2) == 2
    first = m.rbe2[0]
    assert (first.id, first.independent, first.dependents) == (99, 1, (2, 3))
    assert first.components == (1, 2, 3)
    assert m.rbe2[1].components == (1, 2, 3, 4, 5, 6)


def test_bdf_rbe2_thru_and_alpha(tmp_path: Path) -> None:
    deck = tmp_path / "rbe2.bdf"
    deck.write_text(
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,2.,0.,0.\nGRID,4,,3.,0.,0.\n"
        "RBE2,7,1,123456,2,THRU,4,6.5-6\n"
        "ENDDATA\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="ALPHA"):
        m = read_bdf(deck)
    assert m.rbe2[0].dependents == (2, 3, 4)


def test_bdf_rbe2_malformed_cm_raises(tmp_path: Path) -> None:
    deck = tmp_path / "bad.bdf"
    deck.write_text(
        "BEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n"
        "RBE2,9,1,127,2\nENDDATA\n",
        encoding="utf-8",
    )
    with pytest.raises(BdfError, match="CM"):
        read_bdf(deck)


def test_bdf_rbe2_write_read_roundtrip(tmp_path: Path) -> None:
    src = FEModel(name="rt")
    for i in range(1, 5):
        src.add_node(i, (float(i), 0.0, 0.0))
    src.add_rbe2(50, independent=1, dependents=(2, 3), components=(1, 2, 3))
    src.add_rbe2(51, independent=4, dependents=(2,))
    path = tmp_path / "rt.bdf"
    write_bdf(path, src)
    loaded = read_bdf(path)
    assert len(loaded.rbe2) == 2
    assert loaded.rbe2[0].id == 50
    assert loaded.rbe2[0].dependents == (2, 3)
    assert loaded.rbe2[0].components == (1, 2, 3)
    assert loaded.rbe2[1].components == (1, 2, 3, 4, 5, 6)


def test_bdf_rbe3_still_warns_as_unsupported(tmp_path: Path) -> None:
    deck = tmp_path / "rbe3.bdf"
    deck.write_text(
        "BEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n"
        "RBE3,7,,1,123456,1.,123,2\nENDDATA\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="RBE3"):
        m = read_bdf(deck)
    assert not m.rbe2


# ---------------------------------------------------------------------------
# NastranPunchDriver (no Nastran installation required)
# ---------------------------------------------------------------------------


def _driver_model() -> FEModel:
    m = FEModel(name="tiny truss")
    m.add_node(1, (0.0, 0.0, 0.0))
    m.add_node(2, (1.0, 0.0, 0.0))
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="bar", material_id=1, A=1.0e-4)
    m.add_element(1, "BAR2", (1, 2), property_id=1)
    m.add_spc(1, [True] * 6)
    return m


def _stub_pch(tmp_path: Path) -> Path:
    freq = np.array([10.0, 20.0])
    modes = np.zeros((12, 2))
    modes[6, 0] = 1.0
    modes[7, 1] = 1.0
    modal = ModalResult(
        freq_hz=freq,
        eigenvalues=(2.0 * np.pi * freq) ** 2,
        modes=modes,
        generalized_mass=np.ones(2),
        dof_index=tuple((n, d) for n in (1, 2) for d in range(6)),
    )
    fixture = tmp_path / "fixture.pch"
    write_pch(fixture, modal)
    return fixture


def _stub_exe(tmp_path: Path, name: str, script: str) -> Path:
    exe = tmp_path / name
    exe.write_text(script, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def test_driver_conforms_to_solver_driver_protocol() -> None:
    driver = NastranPunchDriver()
    assert isinstance(driver, SolverDriver)
    assert driver.name == "nastran-punch"
    with pytest.raises(ValueError, match="n_modes"):
        NastranPunchDriver(n_modes=0)


def test_driver_write_input_is_a_sol103_punch_deck(tmp_path: Path) -> None:
    driver = NastranPunchDriver(n_modes=7)
    deck = driver.write_input(_driver_model(), tmp_path / "job")
    assert deck.suffix == ".bdf" and deck.is_file()
    lines = deck.read_text(encoding="utf-8").splitlines()
    assert "SOL 103" in lines
    assert "CEND" in lines
    assert "METHOD = 1" in lines
    assert "DISPLACEMENT(PUNCH) = ALL" in lines
    bulk_at = next(i for i, ln in enumerate(lines) if ln.startswith("BEGIN BULK"))
    assert lines[bulk_at + 1].startswith("EIGRL")
    assert "7" in lines[bulk_at + 1]
    # the deck is still valid bulk data for femtools' own reader
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = read_bdf(deck)
    assert set(loaded.nodes) == {1, 2}
    assert any("EIGRL" in str(w.message) for w in caught)  # aggregated, not an error


def test_driver_missing_executable(tmp_path: Path) -> None:
    driver = NastranPunchDriver(executable="definitely-not-a-nastran-binary")
    assert driver.is_available() is False
    deck = driver.write_input(_driver_model(), tmp_path)
    with pytest.raises(SolverError, match="not found"):
        driver.run(deck)


def test_driver_full_loop_with_stub_executable(tmp_path: Path) -> None:
    fixture = _stub_pch(tmp_path)
    exe = _stub_exe(
        tmp_path,
        "fake-nastran",
        f'#!/bin/sh\ncp "{fixture}" "$(dirname "$1")/$(basename "$1" .bdf).pch"\n',
    )
    driver = NastranPunchDriver(executable=str(exe), n_modes=2)
    assert driver.is_available() is True
    deck = driver.write_input(_driver_model(), tmp_path / "run")
    result = driver.run(deck)
    assert result.suffix == ".pch" and result.parent == deck.parent
    modal = driver.read_modal(result)
    np.testing.assert_allclose(modal.freq_hz, [10.0, 20.0], rtol=1.0e-6)
    assert modal.modes.shape == (12, 2)


def test_driver_run_failures_raise_solver_error(tmp_path: Path) -> None:
    deck = NastranPunchDriver().write_input(_driver_model(), tmp_path)
    failing = _stub_exe(tmp_path, "failing", "#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(SolverError, match="status 3"):
        NastranPunchDriver(executable=str(failing)).run(deck)
    silent = _stub_exe(tmp_path, "silent", "#!/bin/sh\nexit 0\n")
    with pytest.raises(SolverError, match="no punch file"):
        NastranPunchDriver(executable=str(silent)).run(deck)
    sleepy = _stub_exe(tmp_path, "sleepy", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(SolverError, match="timeout"):
        NastranPunchDriver(executable=str(sleepy)).run(deck, timeout=0.2)

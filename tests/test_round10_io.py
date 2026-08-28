"""Round 10 io: first-class CTETRA10 text I/O and punch ``$STRESSES``.

Acceptance (R10-F2):

* a 10-node ``CTETRA`` round-trips through ``read_bdf``/``write_bdf`` as a
  first-class ``TET10`` keeping all 10 node ids; the 4-node ``CTETRA``
  stays ``TET4``; the 20-node ``CHEXA`` still warns (one aggregated
  warning) and degrades to ``HEX8``;
* the same policy on the other text translators femtools already parses:
  the LS-DYNA ten-node ``*ELEMENT_SOLID`` form, the ANSYS SOLID92/187
  archive records and the Abaqus ``C3D10`` blocks keep their 10 nodes;
* ``read_pch_stress`` parses public punch ``$STRESSES`` /
  ``$ELEMENT STRESSES`` text (plain Voigt rows and the labeled solid
  ``CENTER`` layout) into element ids + Voigt tensors, skipping
  ``$EIGENVECTOR`` / ``$DISPLACEMENTS`` / complex blocks tolerantly;
* ``NastranPunchDriver.read_stress`` reads that punch end to end with a
  stub executable only (no Nastran installation); ``.op2`` paths raise
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
from femtools.drivers import NastranPunchDriver
from femtools.io import (
    PchStressResult,
    read_bdf,
    read_cdb,
    read_inp,
    read_k,
    read_pch_static,
    read_pch_stress,
    write_bdf,
    write_cdb,
    write_inp,
    write_k,
)
from femtools.io.pch import PchError

STEEL = {"E": 2.1e11, "nu": 0.3, "rho": 7850.0}

#: canonical 10-node tet: 4 corners then the 6 midsides (public CTETRA order)
TET10_XYZ = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]


def _tet10_model() -> FEModel:
    m = FEModel(name="tet10")
    for nid, xyz in enumerate(TET10_XYZ, start=1):
        m.add_node(nid, xyz)
    m.add_material(1, E=STEEL["E"], nu=STEEL["nu"], rho=STEEL["rho"])
    m.add_property(1, type="solid", material_id=1)
    m.add_element(1, "TET10", tuple(range(1, 11)), property_id=1)
    return m


def _grid_lines(n: int, xyz: Sequence[tuple[float, float, float]] | None = None) -> list[str]:
    coords = xyz if xyz is not None else [(float(i), 0.0, 0.0) for i in range(n)]
    return [f"GRID,{i + 1},0,{c[0]},{c[1]},{c[2]}" for i, c in enumerate(coords[:n])]


def _bdf(tmp_path: Path, name: str, body: list[str]) -> Path:
    deck = tmp_path / name
    deck.write_text(
        "\n".join(["BEGIN BULK", "MAT1,1,2.1+11,,0.3,7850.", "PSOLID,1,1", *body, "ENDDATA"])
        + "\n",
        encoding="utf-8",
    )
    return deck


# ---------------------------------------------------------------------------
# BDF: 10-node CTETRA is a first-class TET10
# ---------------------------------------------------------------------------


def test_bdf_ctetra10_reads_as_tet10_without_warning(tmp_path: Path) -> None:
    deck = _bdf(
        tmp_path,
        "tet10.bdf",
        [*_grid_lines(10, TET10_XYZ), "CTETRA,1,1,1,2,3,4,5,6", "+,7,8,9,10"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_bdf(deck)
    el = m.elements[1]
    assert el.type == "TET10"
    assert el.nodes == tuple(range(1, 11))  # all 10, card order
    assert not any("midside" in str(w.message) for w in caught)


def test_bdf_ctetra10_roundtrip_write_read(tmp_path: Path) -> None:
    out = tmp_path / "out.bdf"
    write_bdf(out, _tet10_model())
    text = out.read_text(encoding="utf-8")
    assert "CTETRA" in text
    back = read_bdf(out)
    el = back.elements[1]
    assert el.type == "TET10"
    assert el.nodes == tuple(range(1, 11))
    np.testing.assert_allclose(back.nodes[6].xyz, TET10_XYZ[5])


def test_bdf_ctetra4_stays_tet4(tmp_path: Path) -> None:
    deck = _bdf(tmp_path, "tet4.bdf", [*_grid_lines(4, TET10_XYZ), "CTETRA,1,1,1,2,3,4"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_bdf(deck)
    assert m.elements[1].type == "TET4"
    assert m.elements[1].nodes == (1, 2, 3, 4)
    assert not caught
    out = tmp_path / "tet4-out.bdf"
    write_bdf(out, m)
    assert read_bdf(out).elements[1].type == "TET4"


def test_bdf_partial_midside_ctetra_still_warns_and_drops(tmp_path: Path) -> None:
    # 6 of the 10 nodes present: not representable, degrade like before
    deck = _bdf(tmp_path, "partial.bdf", [*_grid_lines(6, TET10_XYZ), "CTETRA,1,1,1,2,3,4,5,6"])
    with pytest.warns(UserWarning, match="midside"):
        m = read_bdf(deck)
    assert m.elements[1].type == "TET4"
    assert m.elements[1].nodes == (1, 2, 3, 4)


def _chexa20_cards(eid: int) -> list[str]:
    """20-node CHEXA in free field with the required continuations."""
    return [
        f"CHEXA,{eid},1,1,2,3,4,5,6",
        "+,7,8,9,10,11,12,13,14",
        "+,15,16,17,18,19,20",
    ]


_HEX20_XYZ = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
] + [(0.5, 0.5, float(k)) for k in range(12)]  # fmt: skip


def test_bdf_chexa20_still_warns_and_becomes_hex8(tmp_path: Path) -> None:
    deck = _bdf(tmp_path, "hex20.bdf", [*_grid_lines(20, _HEX20_XYZ), *_chexa20_cards(1)])
    with pytest.warns(UserWarning, match=r"HEX20 -> HEX8.*midside"):
        m = read_bdf(deck)
    el = m.elements[1]
    assert el.type == "HEX8"
    assert el.nodes == tuple(range(1, 9))


def test_bdf_hex20_warning_is_aggregated(tmp_path: Path) -> None:
    deck = _bdf(
        tmp_path,
        "hex20x2.bdf",
        [*_grid_lines(20, _HEX20_XYZ), *_chexa20_cards(1), *_chexa20_cards(2)],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_bdf(deck)
    assert m.elements[1].type == "HEX8" and m.elements[2].type == "HEX8"
    drops = [w for w in caught if "HEX20 -> HEX8" in str(w.message)]
    assert len(drops) == 1  # one aggregated warning for both elements
    assert "2 element(s)" in str(drops[0].message)


# ---------------------------------------------------------------------------
# same policy on the other translators (LS-DYNA .k, ANSYS .cdb, Abaqus .inp)
# ---------------------------------------------------------------------------


def test_k_ten_node_solid_is_tet10(tmp_path: Path) -> None:
    node_rows = "\n".join(
        f"{i + 1}, {x}, {y}, {z}" for i, (x, y, z) in enumerate(TET10_XYZ)
    )
    deck = f"""\
*KEYWORD
*NODE
{node_rows}
*ELEMENT_SOLID
1, 1
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
*SECTION_SOLID
1, 1
*MAT_ELASTIC
1, 7850.0, 2.1e11, 0.3
*PART
tets
1, 1, 1
*END
"""
    p = tmp_path / "tet10.k"
    p.write_text(deck, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_k(p)
    el = m.elements[1]
    assert el.type == "TET10" and el.nodes == tuple(range(1, 11))
    assert not any("midside" in str(w.message) for w in caught)
    # write_k emits the documented two-line ten-node form and reads it back
    out = tmp_path / "out.k"
    write_k(out, m)
    back = read_k(out)
    assert back.elements[1].type == "TET10"
    assert back.elements[1].nodes == tuple(range(1, 11))


def test_cdb_solid187_is_tet10_and_roundtrips(tmp_path: Path) -> None:
    # hand-written archive: SOLID187 EBLOCK record spilling onto line 2
    node_rows = "\n".join(
        f"{i + 1:>9d}{0:>9d}{0:>9d}" + "".join(f"{v:21.13E}" for v in xyz)
        for i, xyz in enumerate(TET10_XYZ)
    )
    first = [1, 1, 1, 0, 0, 0, 0, 0, 10, 0, 1, *range(1, 9)]
    archive = f"""\
/PREP7
ET,1,SOLID187
MP,EX,1,2.1e11
MP,NUXY,1,0.3
MP,DENS,1,7850
NBLOCK,6,SOLID
(3i9,6e21.13e3)
{node_rows}
N,R5.3,LOC,       -1,
EBLOCK,19,SOLID,,1
(19i9)
{"".join(f"{v:>9d}" for v in first)}
{"".join(f"{v:>9d}" for v in (9, 10))}
{-1:>9d}
FINISH
"""
    p = tmp_path / "tet10.cdb"
    p.write_text(archive, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_cdb(p)
    el = m.elements[1]
    assert el.type == "TET10" and el.nodes == tuple(range(1, 11))
    assert not any("midside" in str(w.message) for w in caught)
    out = tmp_path / "out.cdb"
    write_cdb(out, m)
    assert "187" in out.read_text(encoding="utf-8")  # SOLID187 on write
    back = read_cdb(out)
    eid = next(iter(back.elements))
    assert back.elements[eid].type == "TET10"
    assert back.elements[eid].nodes == tuple(range(1, 11))


def test_cdb_degenerate_tet10_record_still_drops(tmp_path: Path) -> None:
    # collapsed midsides (duplicate ids) cannot stay quadratic
    m = _tet10_model()
    out = tmp_path / "degen.cdb"
    write_cdb(out, m)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("EBLOCK")) + 2
    lines[idx] = lines[idx].replace(f"{5:>9d}", f"{1:>9d}", 1)  # midside 5 -> corner 1
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="midside"):
        back = read_cdb(out)
    eid = next(iter(back.elements))
    assert back.elements[eid].type == "TET4"


def test_inp_c3d10_is_tet10_and_roundtrips(tmp_path: Path) -> None:
    m = _tet10_model()
    out = tmp_path / "tet10.inp"
    write_inp(out, m)
    assert "C3D10" in out.read_text(encoding="utf-8")
    back = read_inp(out)
    el = back.elements[1]
    assert el.type == "TET10" and el.nodes == tuple(range(1, 11))


# ---------------------------------------------------------------------------
# read_pch_stress fixtures (public punch layout, written by hand)
# ---------------------------------------------------------------------------


def _seq(bodies: list[str]) -> str:
    """72-column bodies + the punch sequence number in columns 73-80."""
    return "\n".join(f"{b:<72.72s}{i + 1:>8d}" for i, b in enumerate(bodies)) + "\n"


def _stress_rows(eid: int, voigt: Sequence[float]) -> list[str]:
    v = list(voigt)
    return [
        f"{eid:>10d}{'':8s}" + "".join(f"{x:18.6E}" for x in v[:3]),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in v[3:6]),
    ]


def _stress_bodies(
    cases: Sequence[tuple[int, dict[int, Sequence[float]]]],
    header: str = "$ELEMENT STRESSES",
    output_marker: str = "$REAL OUTPUT",
) -> list[str]:
    bodies: list[str] = []
    for subcase, elements in cases:
        bodies += [
            "$TITLE   = FEMTOOLS STRESS FIXTURE",
            "$SUBTITLE= SOL 101",
            "$LABEL   =",
            header,
            output_marker,
            f"$SUBCASE ID = {subcase:11d}",
            "$ELEMENT TYPE =          39  CTETRA",
        ]
        for eid in sorted(elements):
            bodies += _stress_rows(eid, elements[eid])
    return bodies


_SIG1 = {11: [1.0e2, 2.0e1, 3.0, 4.0, 5.0e-1, 6.0e-2], 12: [7.0e1, 0, 0, 0, 0, 0]}
_SIG2 = {11: [2.0e2, 4.0e1, 6.0, 8.0, 1.0, 1.2e-1], 12: [1.4e2, 0, 0, 0, 0, 0]}


def test_pch_stress_single_subcase(tmp_path: Path) -> None:
    path = tmp_path / "stress.pch"
    path.write_text(_seq(_stress_bodies([(1, _SIG1)])), encoding="utf-8")
    res = read_pch_stress(path)
    assert isinstance(res, PchStressResult)
    assert res.element_ids == (11, 12)
    assert res.stress.shape == (2, 6)
    assert res.load_case == 1 and res.n_cases == 1
    np.testing.assert_allclose(res.stress[res.index_of(11)], _SIG1[11])
    np.testing.assert_allclose(res.stress[res.index_of(12)], _SIG1[12])
    assert res.etypes == {11: "CTETRA", 12: "CTETRA"}
    # frame-independent scalar helper
    assert res.von_mises.shape == (2,)
    assert res.von_mises[res.index_of(12)] == pytest.approx(70.0)


def test_pch_stress_dollar_stresses_header(tmp_path: Path) -> None:
    path = tmp_path / "plain.pch"
    path.write_text(_seq(_stress_bodies([(1, _SIG1)], header="$STRESSES")), encoding="utf-8")
    res = read_pch_stress(path)
    assert res.element_ids == (11, 12)
    np.testing.assert_allclose(res.stress[0], _SIG1[11])


def test_pch_stress_multiple_subcases_stack_slabs(tmp_path: Path) -> None:
    path = tmp_path / "stress2.pch"
    path.write_text(_seq(_stress_bodies([(1, _SIG1), (2, _SIG2)])), encoding="utf-8")
    res = read_pch_stress(path)
    assert res.stress.shape == (2, 6, 2)
    assert res.load_case == (1, 2) and res.n_cases == 2
    np.testing.assert_allclose(res.stress[:, :, 1], 2.0 * res.stress[:, :, 0])
    assert res.von_mises.shape == (2, 2)


def test_pch_stress_labeled_solid_center_layout(tmp_path: Path) -> None:
    # the public solid layout: X/Y/Z/XY/YZ/ZX labels, CENTER group first,
    # per-corner repeats and principal values (A/B/C) ignored
    bodies = [
        "$ELEMENT STRESSES",
        "$REAL OUTPUT",
        "$SUBCASE ID =           1",
        "$ELEMENT TYPE =          39  CTETRA",
        f"{22:>10d}           0GRID CS  4 GP",
        "-CONT-  CENTER  X   1.829032E+02  XY  -9.212549E+00   A   1.925717E+02",
        "-CONT-          Y   1.093623E+02  YZ  -4.290556E+00   B   1.081840E+02",
        "-CONT-          Z   1.093812E+02  ZX   1.107610E+00   C   1.008947E+02",
        "-CONT-        4 X   9.999999E+09  XY   9.999999E+09   A   9.999999E+09",
    ]
    path = tmp_path / "solid.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    res = read_pch_stress(path)
    assert res.element_ids == (22,)
    np.testing.assert_allclose(
        res.stress[0],
        [1.829032e2, 1.093623e2, 1.093812e2, -9.212549, -4.290556, 1.10761],
        rtol=1.0e-6,
    )
    assert res.etypes == {22: "CTETRA"}


def test_pch_stress_short_and_fortran_exponent_rows(tmp_path: Path) -> None:
    bodies = [
        "$STRESSES",
        f"$SUBCASE ID = {1:11d}",
        f"{5:>10d}{'':8s}      1.500000D+02      2.500000E+01",  # 2 values, D exponent
    ]
    path = tmp_path / "short.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    res = read_pch_stress(path)
    np.testing.assert_allclose(res.stress[0], [150.0, 25.0, 0.0, 0.0, 0.0, 0.0])


def test_pch_stress_more_than_six_values_warns_and_truncates(tmp_path: Path) -> None:
    bodies = [
        "$STRESSES",
        f"$SUBCASE ID = {1:11d}",
        f"{7:>10d}{'':8s}" + "".join(f"{float(k):9.1E}" for k in range(1, 8)),  # 7 values
    ]
    path = tmp_path / "wide.pch"
    path.write_text(_seq(bodies), encoding="utf-8")
    with pytest.warns(UserWarning, match="more than 6"):
        res = read_pch_stress(path)
    np.testing.assert_allclose(res.stress[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_pch_stress_uneven_subcases_zero_fill_with_warning(tmp_path: Path) -> None:
    case2 = {11: _SIG2[11]}  # element 12 missing from the second subcase
    path = tmp_path / "uneven.pch"
    path.write_text(_seq(_stress_bodies([(1, _SIG1), (2, case2)])), encoding="utf-8")
    with pytest.warns(UserWarning, match="zero-filled"):
        res = read_pch_stress(path)
    assert res.stress.shape == (2, 6, 2)
    np.testing.assert_allclose(res.stress[res.index_of(12), :, 1], 0.0)


def test_pch_stress_skips_eigenvector_and_displacements(tmp_path: Path) -> None:
    disp = [
        "$DISPLACEMENTS",
        "$REAL OUTPUT",
        f"$SUBCASE ID = {1:11d}",
        f"{1:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (9.0, 9.0, 9.0)),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in (9.0, 9.0, 9.0)),
    ]
    eig = [
        "$EIGENVALUE =  3.947842E+05  MODE =     1",
        "$EIGENVECTOR",
        f"{1:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (1.0, 0.0, 0.0)),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in (0.0, 0.0, 0.0)),
    ]
    path = tmp_path / "mixed.pch"
    path.write_text(_seq(disp + eig + _stress_bodies([(1, _SIG1)])), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_pch_stress(path)
    texts = [str(w.message) for w in caught]
    assert any("eigenvector" in t for t in texts)
    assert any("DISPLACEMENTS" in t for t in texts)
    assert res.element_ids == (11, 12)
    np.testing.assert_allclose(res.stress[0], _SIG1[11])
    # and the static reader now hints at read_pch_stress on the same file
    with pytest.warns(UserWarning, match="read_pch_stress"):
        static = read_pch_static(path)
    assert static.u[0] == pytest.approx(9.0)


def test_pch_stress_complex_blocks_skipped(tmp_path: Path) -> None:
    complex_only = tmp_path / "complex.pch"
    complex_only.write_text(
        _seq(_stress_bodies([(1, _SIG1)], output_marker="$REAL-IMAGINARY OUTPUT")),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="complex"), pytest.raises(PchError):
        read_pch_stress(complex_only)
    both = tmp_path / "both.pch"
    both.write_text(
        _seq(
            _stress_bodies([(1, _SIG1)], output_marker="$REAL-IMAGINARY OUTPUT")
            + _stress_bodies([(2, _SIG2)])
        ),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="complex"):
        res = read_pch_stress(both)
    assert res.load_case == 2
    np.testing.assert_allclose(res.stress[res.index_of(11)], _SIG2[11])


def test_pch_stress_no_data_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pch"
    empty.write_text("$TITLE   = NOTHING HERE\n", encoding="utf-8")
    with pytest.raises(PchError, match=r"no real \$STRESSES"):
        read_pch_stress(empty)


def test_pch_stress_malformed_lines_raise(tmp_path: Path) -> None:
    cont_first = tmp_path / "cont.pch"
    cont_first.write_text("$STRESSES\n-CONT-            1.0\n", encoding="utf-8")
    with pytest.raises(PchError, match="-CONT-"):
        read_pch_stress(cont_first)
    garbage = tmp_path / "garbage.pch"
    garbage.write_text("$STRESSES\n        11        not-a-number\n", encoding="utf-8")
    with pytest.raises(PchError, match="cannot parse"):
        read_pch_stress(garbage)


def test_pch_stress_index_of_unknown_element_raises() -> None:
    res = PchStressResult(element_ids=(3,), stress=np.zeros((1, 6)))
    assert res.index_of(3) == 0
    with pytest.raises(KeyError):
        res.index_of(99)


# ---------------------------------------------------------------------------
# NastranPunchDriver: STRESS(PUNCH) request and read_stress (stub only)
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


def _stub_exe(tmp_path: Path, name: str, script: str) -> Path:
    exe = tmp_path / name
    exe.write_text(script, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def test_write_input_stress_request_is_opt_in(tmp_path: Path) -> None:
    driver = NastranPunchDriver()
    model = _static_model()
    plain = driver.write_input(model, tmp_path / "a", sol=101)
    assert "STRESS(PUNCH) = ALL" not in plain.read_text(encoding="utf-8")  # default deck
    deck = driver.write_input(model, tmp_path / "b", sol=101, stress=True)
    lines = deck.read_text(encoding="utf-8").splitlines()
    assert "STRESS(PUNCH) = ALL" in lines
    assert "DISPLACEMENT(PUNCH) = ALL" in lines  # still requested too


def test_driver_stress_full_loop_with_stub_executable(tmp_path: Path) -> None:
    fixture = tmp_path / "stress-fixture.pch"
    fixture.write_text(_seq(_stress_bodies([(1, _SIG1)])), encoding="utf-8")
    exe = _stub_exe(
        tmp_path,
        "fake-nastran",
        f'#!/bin/sh\ncp "{fixture}" "$(dirname "$1")/$(basename "$1" .bdf).pch"\n',
    )
    driver = NastranPunchDriver(executable=str(exe))
    deck = driver.write_input(_static_model(), tmp_path / "run", sol=101, stress=True)
    result = driver.run(deck)
    res = driver.read_stress(result)
    assert isinstance(res, PchStressResult)
    assert res.element_ids == (11, 12)
    np.testing.assert_allclose(res.stress[res.index_of(11)], _SIG1[11])


def test_driver_read_stress_op2_path_is_na(tmp_path: Path) -> None:
    driver = NastranPunchDriver()
    op2 = tmp_path / "job.op2"
    op2.write_bytes(b"\x00\x01binary")
    with pytest.raises(SolverError, match=r"OP2.*N/A"):
        driver.read_stress(op2)
    with pytest.raises(SolverError, match="OP2"):
        driver.read_stress("missing.op2")


def test_driver_read_stress_from_displacement_only_punch_raises(tmp_path: Path) -> None:
    bodies = [
        "$DISPLACEMENTS",
        f"$SUBCASE ID = {1:11d}",
        f"{1:>10d}{'G':>8s}" + "".join(f"{x:18.6E}" for x in (1.0, 2.0, 3.0)),
        f"{'-CONT-':<18s}" + "".join(f"{x:18.6E}" for x in (0.0, 0.0, 0.0)),
    ]
    punch = tmp_path / "disp.pch"
    punch.write_text(_seq(bodies), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(PchError, match=r"no real \$STRESSES"):
            NastranPunchDriver().read_stress(punch)

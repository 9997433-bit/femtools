"""Round 6 io: Abaqus .inp and LS-DYNA .k TEXT-subset translators.

Acceptance (R6-F2): a tiny HEX8 cube, a QUAD4 plate and a BEAM2 line
written as INP and as K must ``read_*`` into :class:`FEModel` and
``assemble_km`` without crash, round-tripping nodes / connectivity /
E / nu / rho / thickness as far as the subset allows.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from femtools.core.errors import FileFormatError
from femtools.core.model import FEModel
from femtools.core.sets import ElementSet, NodeSet
from femtools.fea import assemble_km
from femtools.io import read_inp, read_k, write_inp
from femtools.io.inp import InpError
from femtools.io.kfile import KFileError

# ---------------------------------------------------------------------------
# reference models (built directly on the core database)
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


# ---------------------------------------------------------------------------
# Abaqus .inp -- reading
# ---------------------------------------------------------------------------

_INP_CUBE = """\
*HEADING
tiny cube
** unit cube, one C3D8, base face encastre
*NODE, NSET=ALLN
1, 0., 0., 0.
2, 1., 0., 0.
3, 1., 1., 0.
4, 0., 1., 0.
5, 0., 0., 1.
6, 1., 0., 1.
7, 1., 1., 1.
8, 0., 1., 1.
*ELEMENT, TYPE=C3D8, ELSET=CUBE
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=BASE, GENERATE
1, 4, 1
*MATERIAL, NAME=STEEL
*ELASTIC
2.1e11, 0.3
*DENSITY
7850.
*SOLID SECTION, ELSET=CUBE, MATERIAL=STEEL
*BOUNDARY
BASE, ENCASTRE
"""


def test_inp_hex8_cube_reads_and_assembles(tmp_path: Path) -> None:
    p = tmp_path / "cube.inp"
    p.write_text(_INP_CUBE, encoding="utf-8")
    m = read_inp(p)

    assert m.name == "tiny cube"
    assert set(m.nodes) == {n[0] for n in _CUBE_XYZ}
    for nid, x, y, z in _CUBE_XYZ:
        np.testing.assert_allclose(m.nodes[nid].xyz, (x, y, z), atol=0.0)
    el = m.elements[1]
    assert el.type == "HEX8"
    assert el.nodes == (1, 2, 3, 4, 5, 6, 7, 8)
    mat = m.element_material(1)
    assert mat is not None
    assert (mat.E, mat.nu, mat.rho) == (STEEL["E"], STEEL["nu"], STEEL["rho"])
    assert m.element_property(1).type == "solid"
    fixed = {s.node_id for s in m.spcs}
    assert fixed == {1, 2, 3, 4}
    assert all(all(s.mask) for s in m.spcs)
    assert isinstance(m.sets["BASE"], NodeSet) and set(m.sets["BASE"]) == {1, 2, 3, 4}
    assert isinstance(m.sets["CUBE"], ElementSet) and set(m.sets["CUBE"]) == {1}
    _assert_assembles(m)


_INP_PLATE = """\
*HEADING
tiny plate
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 2., 0., 0.
4, 0., 1., 0.
5, 1., 1., 0.
6, 2., 1., 0.
*ELEMENT, TYPE=S4R, ELSET=SKIN
1, 1, 2, 5, 4
*ELEMENT, TYPE=S3, ELSET=SKIN
2, 2, 3, 6
*SHELL SECTION, ELSET=SKIN, MATERIAL=ALU
0.002, 5
*MATERIAL, NAME=ALU
*ELASTIC
7.0e10, 0.33
*DENSITY
2700.
*BOUNDARY
1, 1, 6
4, 1, 3
4, 3, 3, 0.001
"""


def test_inp_shell_sections_and_boundary_forms(tmp_path: Path) -> None:
    p = tmp_path / "plate.inp"
    p.write_text(_INP_PLATE, encoding="utf-8")
    m = read_inp(p)

    assert m.elements[1].type == "QUAD4" and m.elements[1].nodes == (1, 2, 5, 4)
    assert m.elements[2].type == "TRIA3" and m.elements[2].nodes == (2, 3, 6)
    prop = m.element_property(1)
    assert prop.type == "shell" and prop.t == pytest.approx(0.002)
    assert m.element_material(2).E == pytest.approx(7.0e10)
    # 1: all six; 4: ux-uz; 4: enforced uz = 0.001
    assert len(m.spcs) == 3
    assert m.spcs[0].mask == (True,) * 6 and m.spcs[0].value == 0.0
    assert m.spcs[1].mask == (True, True, True, False, False, False)
    assert m.spcs[2].mask == (False, False, True, False, False, False)
    assert m.spcs[2].value == pytest.approx(0.001)
    _assert_assembles(m)


_INP_BEAM = """\
*HEADING
tiny beam line
*NODE
1, 0.0, 0., 0.
2, 0.5, 0., 0.
3, 1.0, 0., 0.
*ELEMENT, TYPE=B31, ELSET=BEAMS
1, 1, 2
2, 2, 3
*BEAM GENERAL SECTION, ELSET=BEAMS, SECTION=GENERAL, DENSITY=7850.
1.0e-4, 1.0e-8, 0., 2.0e-8, 3.0e-8
0., 1., 0.
2.1e11, 8.0769e10
*BOUNDARY
1, ENCASTRE
"""


def test_inp_beam_general_section(tmp_path: Path) -> None:
    p = tmp_path / "beam.inp"
    p.write_text(_INP_BEAM, encoding="utf-8")
    m = read_inp(p)

    el = m.elements[1]
    assert el.type == "BEAM2"
    np.testing.assert_allclose(el.orientation, (0.0, 1.0, 0.0))
    prop = m.element_property(1)
    assert prop.type == "beam"
    assert prop.A == pytest.approx(1.0e-4)
    assert prop.Iy == pytest.approx(1.0e-8)  # Abaqus I11
    assert prop.Iz == pytest.approx(2.0e-8)  # Abaqus I22
    assert prop.J == pytest.approx(3.0e-8)
    mat = m.element_material(1)
    assert mat.E == pytest.approx(2.1e11)
    assert mat.nu == pytest.approx(2.1e11 / (2 * 8.0769e10) - 1.0)
    assert mat.rho == pytest.approx(7850.0)
    _assert_assembles(m)


def test_inp_beam_section_rect_and_circ(tmp_path: Path) -> None:
    deck = """\
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 2., 0., 0.
*ELEMENT, TYPE=B31, ELSET=R
1, 1, 2
*ELEMENT, TYPE=B31, ELSET=C
2, 2, 3
*MATERIAL, NAME=M
*ELASTIC
2.1e11, 0.3
*DENSITY
7850.
*BEAM SECTION, ELSET=R, MATERIAL=M, SECTION=RECT
0.02, 0.01
0., 0., 1.
*BEAM SECTION, ELSET=C, MATERIAL=M, SECTION=CIRC
0.01
0., 0., 1.
"""
    p = tmp_path / "sections.inp"
    p.write_text(deck, encoding="utf-8")
    m = read_inp(p)

    rect = m.element_property(1)
    assert rect.A == pytest.approx(0.02 * 0.01)
    assert rect.Iy == pytest.approx(0.02 * 0.01**3 / 12.0)  # I11: bending about 1-axis
    assert rect.Iz == pytest.approx(0.01 * 0.02**3 / 12.0)
    assert 0.0 < rect.J < rect.Iy + rect.Iz  # Saint-Venant J below polar moment
    circ = m.element_property(2)
    assert circ.A == pytest.approx(math.pi * 0.01**2)
    assert circ.Iy == pytest.approx(math.pi * 0.01**4 / 4.0)
    assert circ.J == pytest.approx(math.pi * 0.01**4 / 2.0)
    _assert_assembles(m)


def test_inp_truss_solid_section_area(tmp_path: Path) -> None:
    deck = """\
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
*ELEMENT, TYPE=T3D2, ELSET=RODS
1, 1, 2
*MATERIAL, NAME=M
*ELASTIC
2.1e11, 0.3
*DENSITY
7850.
*SOLID SECTION, ELSET=RODS, MATERIAL=M
3.0e-4,
"""
    p = tmp_path / "truss.inp"
    p.write_text(deck, encoding="utf-8")
    m = read_inp(p)
    assert m.elements[1].type == "BAR2"
    prop = m.element_property(1)
    assert prop.type == "bar" and prop.A == pytest.approx(3.0e-4)
    _assert_assembles(m)


def test_inp_unknown_keywords_warn_once_and_material_survives(tmp_path: Path) -> None:
    deck = """\
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
*ELEMENT, TYPE=T3D2, ELSET=RODS
1, 1, 2
*MATERIAL, NAME=M
*ELASTIC
2.1e11, 0.3
*PLASTIC
3.5e8, 0.
*DENSITY
7850.
*SOLID SECTION, ELSET=RODS, MATERIAL=M
1e-4,
*STEP
*STATIC
*END STEP
"""
    p = tmp_path / "unknown.inp"
    p.write_text(deck, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_inp(p)
    texts = [str(w.message) for w in caught]
    assert sum("*PLASTIC" in t for t in texts) == 1
    assert sum("*STEP" in t and "END STEP" not in t for t in texts) == 1
    # *PLASTIC (unknown material option) must not detach the *DENSITY
    assert m.materials[1].rho == pytest.approx(7850.0)


def test_inp_malformed_decks_raise(tmp_path: Path) -> None:
    def _read(body: str) -> FEModel:
        p = tmp_path / "bad.inp"
        p.write_text(body, encoding="utf-8")
        return read_inp(p)

    with pytest.raises(InpError, match="TYPE"):
        _read("*NODE\n1, 0., 0., 0.\n*ELEMENT\n1, 1, 1\n")
    with pytest.raises(InpError, match="real number"):
        _read("*NODE\n1, 0., abc, 0.\n")
    with pytest.raises(InpError, match="before any keyword"):
        _read("1, 0., 0., 0.\n*NODE\n")
    with pytest.raises(InpError, match="no section assignment"):
        _read("*NODE\n1, 0., 0., 0.\n2, 1., 0., 0.\n*ELEMENT, TYPE=T3D2\n1, 1, 2\n")
    with pytest.raises(InpError, match="expected 8 nodes"):
        _read("*NODE\n1, 0., 0., 0.\n*ELEMENT, TYPE=C3D8, ELSET=E\n1, 1, 1, 1, 1\n")
    assert isinstance(InpError("x"), FileFormatError)


def test_inp_write_read_roundtrip_cube_plate_beam(tmp_path: Path) -> None:
    for build in (_cube_model, _plate_model, _beam_model):
        source = build()
        path = tmp_path / f"{source.name}.inp"
        write_inp(path, source)
        loaded = read_inp(path)

        assert loaded.name == source.name
        assert set(loaded.nodes) == set(source.nodes)
        for nid in source.nodes:
            np.testing.assert_allclose(loaded.nodes[nid].xyz, source.nodes[nid].xyz)
        assert set(loaded.elements) == set(source.elements)
        for eid, el in source.elements.items():
            assert loaded.elements[eid].type == el.type
            assert loaded.elements[eid].nodes == el.nodes
        for eid in source.elements:
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
        assert len(loaded.spcs) >= len(source.spcs)
        assert {s.node_id for s in loaded.spcs} == {s.node_id for s in source.spcs}
        _assert_assembles(loaded)


# ---------------------------------------------------------------------------
# LS-DYNA .k -- reading
# ---------------------------------------------------------------------------


def _k_node_line(nid: int, x: float, y: float, z: float, tc: int = 0, rc: int = 0) -> str:
    return f"{nid:>8d}{x:>16.9e}{y:>16.9e}{z:>16.9e}{tc:>8d}{rc:>8d}"


def _k_cube_deck() -> str:
    lines = ["*KEYWORD", "*TITLE", "tiny cube", "$ nodes", "*NODE"]
    for nid, x, y, z in _CUBE_XYZ:
        tc = 7 if nid <= 4 else 0
        rc = 7 if nid <= 4 else 0
        lines.append(_k_node_line(nid, x, y, z, tc, rc))
    lines += [
        "*ELEMENT_SOLID",
        f"{1:>8d}{1:>8d}" + "".join(f"{n:>8d}" for n in (1, 2, 3, 4, 5, 6, 7, 8)),
        "*MAT_ELASTIC",
        f"{1:>10d}{7850.0:>10.1f}{2.1e11:>10.3e}{0.3:>10.2f}",
        "*SECTION_SOLID",
        f"{1:>10d}{1:>10d}",
        "*PART",
        "the cube part",
        f"{1:>10d}{1:>10d}{1:>10d}",
        "*END",
    ]
    return "\n".join(lines) + "\n"


def test_k_hex8_cube_fixed_format(tmp_path: Path) -> None:
    p = tmp_path / "cube.k"
    p.write_text(_k_cube_deck(), encoding="utf-8")
    m = read_k(p)

    assert m.name == "tiny cube"
    assert set(m.nodes) == {n[0] for n in _CUBE_XYZ}
    for nid, x, y, z in _CUBE_XYZ:
        np.testing.assert_allclose(m.nodes[nid].xyz, (x, y, z), atol=1e-12)
    el = m.elements[1]
    assert el.type == "HEX8" and el.nodes == (1, 2, 3, 4, 5, 6, 7, 8)
    mat = m.element_material(1)
    assert mat.E == pytest.approx(2.1e11)
    assert mat.nu == pytest.approx(0.3)
    assert mat.rho == pytest.approx(7850.0)
    prop = m.element_property(1)
    assert prop.type == "solid" and prop.name == "the cube part"
    fixed = {s.node_id for s in m.spcs}
    assert fixed == {1, 2, 3, 4}
    assert all(s.mask == (True,) * 6 for s in m.spcs)
    _assert_assembles(m)


_K_PLATE = """\
*KEYWORD
*TITLE
tiny plate
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
4, 0.0, 1.0, 0.0
5, 1.0, 1.0, 0.0
6, 2.0, 1.0, 0.0
*ELEMENT_SHELL
1, 1, 1, 2, 5, 4
2, 1, 2, 3, 6, 6
*SECTION_SHELL
1, 2
0.002, 0.002, 0.002, 0.002
*MAT_ELASTIC
1, 2700.0, 7.0e10, 0.33
*PART
skin
1, 1, 1
*BOUNDARY_SPC_NODE
1, 0, 1, 1, 1, 1, 1, 1
4, 0, 1, 1, 1, 0, 0, 0
*END
"""


def test_k_shell_free_format_quad_and_degenerate_tria(tmp_path: Path) -> None:
    p = tmp_path / "plate.k"
    p.write_text(_K_PLATE, encoding="utf-8")
    m = read_k(p)

    assert m.elements[1].type == "QUAD4" and m.elements[1].nodes == (1, 2, 5, 4)
    assert m.elements[2].type == "TRIA3" and m.elements[2].nodes == (2, 3, 6)
    prop = m.element_property(1)
    assert prop.type == "shell" and prop.t == pytest.approx(0.002)
    assert m.element_material(2).E == pytest.approx(7.0e10)
    assert len(m.spcs) == 2
    assert m.spcs[0].mask == (True,) * 6
    assert m.spcs[1].mask == (True, True, True, False, False, False)
    _assert_assembles(m)


_K_BEAM = """\
*KEYWORD
*TITLE
tiny beam line
*NODE
1, 0.0, 0.0, 0.0
2, 0.5, 0.0, 0.0
3, 1.0, 0.0, 0.0
4, 1.5, 0.0, 0.0
5, 0.0, 1.0, 0.0
*ELEMENT_BEAM
1, 1, 1, 2, 5
2, 1, 2, 3, 5
3, 2, 3, 4
*SECTION_BEAM
1, 2
1.0e-4, 1.0e-8, 2.0e-8, 3.0e-8
*SECTION_BEAM
2, 3
2.5e-4
*MAT_ELASTIC_TITLE
plain steel
1, 7850.0, 2.1e11, 0.3
*PART
bending run
1, 1, 1
*PART
truss tail
2, 2, 1
*NODE
6, 9.0, 9.0, 9.0
*END
"""


def test_k_beam_resultant_and_truss(tmp_path: Path) -> None:
    p = tmp_path / "beam.k"
    p.write_text(_K_BEAM, encoding="utf-8")
    m = read_k(p)

    el = m.elements[1]
    assert el.type == "BEAM2"
    np.testing.assert_allclose(el.orientation, (0.0, 1.0, 0.0))  # node 5 - node 1
    prop = m.element_property(1)
    assert prop.type == "beam"
    assert prop.A == pytest.approx(1.0e-4)
    assert prop.Iy == pytest.approx(1.0e-8)  # ISS
    assert prop.Iz == pytest.approx(2.0e-8)  # ITT
    assert prop.J == pytest.approx(3.0e-8)  # IRR
    assert m.elements[3].type == "BAR2"
    truss = m.element_property(3)
    assert truss.type == "bar" and truss.A == pytest.approx(2.5e-4)
    assert m.materials[1].name == "plain steel"
    _assert_assembles(m)


def test_k_section_beam_rect_and_tube(tmp_path: Path) -> None:
    deck = """\
*KEYWORD
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 2.0, 0.0, 0.0
*ELEMENT_BEAM
1, 1, 1, 2
2, 2, 2, 3
*SECTION_BEAM
1, 1, 1.0, 2.0, 0
0.02, 0.02, 0.01, 0.01
*SECTION_BEAM
2, 1, 1.0, 2.0, 1
0.02, 0.02, 0.016, 0.016
*MAT_ELASTIC
1, 7850.0, 2.1e11, 0.3
*PART
rect
1, 1, 1
*PART
tube
2, 2, 1
*END
"""
    p = tmp_path / "hl.k"
    p.write_text(deck, encoding="utf-8")
    m = read_k(p)

    rect = m.element_property(1)
    assert rect.A == pytest.approx(0.02 * 0.01)
    assert rect.Iy == pytest.approx(0.02 * 0.01**3 / 12.0)
    assert rect.Iz == pytest.approx(0.01 * 0.02**3 / 12.0)
    tube = m.element_property(2)
    do, di = 0.02, 0.016
    assert tube.A == pytest.approx(math.pi * (do**2 - di**2) / 4.0)
    assert tube.Iy == pytest.approx(math.pi * (do**4 - di**4) / 64.0)
    assert tube.J == pytest.approx(math.pi * (do**4 - di**4) / 32.0)
    _assert_assembles(m)


def test_k_node_constraint_codes(tmp_path: Path) -> None:
    deck = """\
*KEYWORD
*NODE
1, 0.0, 0.0, 0.0, 4, 0
2, 1.0, 0.0, 0.0, 0, 5
3, 2.0, 0.0, 0.0
*ELEMENT_BEAM
1, 1, 1, 2
2, 1, 2, 3
*SECTION_BEAM
1, 2
1.0e-4, 1.0e-8, 2.0e-8, 3.0e-8
*MAT_ELASTIC
1, 7850.0, 2.1e11, 0.3
*PART
run
1, 1, 1
*END
"""
    p = tmp_path / "tcrc.k"
    p.write_text(deck, encoding="utf-8")
    m = read_k(p)
    by_node = {s.node_id: s for s in m.spcs}
    assert by_node[1].mask == (True, True, False, False, False, False)  # tc=4: x, y
    assert by_node[2].mask == (False, False, False, False, True, True)  # rc=5: ry, rz
    assert by_node[1].sid == 0 and by_node[2].sid == 0


def test_k_solid_degeneracies_and_two_line_form(tmp_path: Path) -> None:
    deck = """\
*KEYWORD
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 1.0, 0.0
6, 0.5, 0.5, 1.5
*ELEMENT_SOLID
1, 1, 1, 2, 3, 4, 4, 4, 4, 4
2, 1
1, 2, 3, 4, 4, 4, 4, 4, 0, 0
3, 1, 1, 2, 5, 3, 4, 4, 6, 6
*SECTION_SOLID
1, 1
*MAT_ELASTIC
1, 7850.0, 2.1e11, 0.3
*PART
solids
1, 1, 1
*END
"""
    p = tmp_path / "degen.k"
    p.write_text(deck, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = read_k(p)

    assert m.elements[1].type == "TET4" and m.elements[1].nodes == (1, 2, 3, 4)
    assert m.elements[2].type == "TET4" and m.elements[2].nodes == (1, 2, 3, 4)
    assert 3 not in m.elements  # 6 unique corners: wedge, skipped
    texts = [str(w.message) for w in caught]
    assert sum("wedge/pyramid" in t for t in texts) == 1
    _assert_assembles(m)


def test_k_tet10_two_line_keeps_midsides(tmp_path: Path) -> None:
    deck = """\
*KEYWORD
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 0.5, 0.0, 0.0
6, 0.5, 0.5, 0.0
7, 0.0, 0.5, 0.0
8, 0.0, 0.0, 0.5
9, 0.5, 0.0, 0.5
10, 0.0, 0.5, 0.5
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
    assert m.elements[1].type == "TET10"
    assert m.elements[1].nodes == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert not any("TET10 -> TET4" in str(w.message) for w in caught)


def test_k_unknown_keywords_warn_once(tmp_path: Path) -> None:
    deck = _k_cube_deck().replace(
        "*END",
        "*CONTROL_TERMINATION\n0.01\n*DATABASE_BINARY_D3PLOT\n0.001\n*END",
    )
    p = tmp_path / "extra.k"
    p.write_text(deck, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_k(p)
    texts = [str(w.message) for w in caught]
    assert sum("CONTROL_TERMINATION" in t for t in texts) == 1
    assert sum("DATABASE_BINARY_D3PLOT" in t for t in texts) == 1


def test_k_malformed_decks_raise(tmp_path: Path) -> None:
    def _read(body: str) -> FEModel:
        p = tmp_path / "bad.k"
        p.write_text(body, encoding="utf-8")
        return read_k(p)

    with pytest.raises(KFileError, match="duplicate node id"):
        _read("*KEYWORD\n*NODE\n1, 0., 0., 0.\n1, 1., 0., 0.\n*END\n")
    with pytest.raises(KFileError, match="part 9 is not defined"):
        _read(
            "*KEYWORD\n*NODE\n1, 0., 0., 0.\n2, 1., 0., 0.\n"
            "*ELEMENT_BEAM\n1, 9, 1, 2\n*END\n"
        )
    with pytest.raises(KFileError, match="no LS-DYNA keywords"):
        _read("$ just a comment\n")
    with pytest.raises(KFileError, match="before any keyword"):
        _read("1, 0., 0., 0.\n*NODE\n")
    with pytest.raises(KFileError, match="not supported"):
        _read("*KEYWORD\n*NODE +\n1, 0., 0., 0.\n*END\n")
    with pytest.raises(KFileError, match="ELFORM=9"):
        _read(
            "*KEYWORD\n*NODE\n1, 0., 0., 0.\n*SECTION_BEAM\n1, 9\n1.0e-4\n*END\n"
        )
    with pytest.raises(KFileError, match="section 5 is not defined"):
        _read(
            "*KEYWORD\n*NODE\n1, 0., 0., 0.\n*MAT_ELASTIC\n1, 7850., 2.1e11, 0.3\n"
            "*PART\np\n1, 5, 1\n*END\n"
        )
    assert isinstance(KFileError("x"), FileFormatError)


# ---------------------------------------------------------------------------
# acceptance: the same three tiny models through both formats
# ---------------------------------------------------------------------------


def test_acceptance_cube_plate_beam_as_inp_and_k(tmp_path: Path) -> None:
    """R6-F2 acceptance: HEX8 cube, QUAD4 plate, BEAM2 line written as INP
    (via write_inp) and as K (text fixtures) read back and assemble."""
    for build in (_cube_model, _plate_model, _beam_model):
        source = build()
        path = tmp_path / f"{source.name}.inp"
        write_inp(path, source)
        _assert_assembles(read_inp(path))

    for deck, fname in ((_k_cube_deck(), "cube.k"), (_K_PLATE, "plate.k"), (_K_BEAM, "beam.k")):
        path = tmp_path / fname
        path.write_text(deck, encoding="utf-8")
        _assert_assembles(read_k(path))

"""Round-trip preservation tests for the three contracted model formats."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from femtools.core.errors import ModelError
from femtools.core.model import FEModel
from femtools.fea import apply_mpc
from femtools.io import bdf as bdf_module
from femtools.io import cdb as cdb_module
from femtools.io import project as project_module
from femtools.io import unv as unv_module


def _cdb_ints(*values: int) -> str:
    return "".join(f"{value:9d}" for value in values)


def _cdb_node(node_id: int, xyz: tuple[float, float, float]) -> str:
    integers = _cdb_ints(node_id, 0, 0)
    reals = "".join(f"{value:21.13E}" for value in (*xyz, 0.0, 0.0, 0.0))
    return integers + reals


def _write_cdb(
    tmp_path: Path,
    name: str,
    records: list[str],
    nodes: dict[int, tuple[float, float, float]],
) -> Path:
    node_block = [
        "NBLOCK,6,SOLID",
        "(3i9,6e21.13e3)",
        *(_cdb_node(node_id, xyz) for node_id, xyz in nodes.items()),
        _cdb_ints(-1),
    ]
    path = tmp_path / name
    path.write_text("\n".join([*records, *node_block, "FINISH", ""]), encoding="utf-8")
    return path


def _assert_basic_model_equal(
    expected: Any,
    actual: Any,
    *,
    elements: bool = True,
    name: bool = True,
) -> None:
    if name:
        assert actual.name == expected.name
    assert set(actual.nodes) == set(expected.nodes)
    for node_id in expected.nodes:
        np.testing.assert_allclose(
            actual.nodes[node_id].xyz,
            expected.nodes[node_id].xyz,
            rtol=0.0,
            atol=1.0e-12,
        )
    if elements:
        assert set(actual.elements) == set(expected.elements)
        for element_id in expected.elements:
            assert tuple(actual.elements[element_id].nodes) == tuple(
                expected.elements[element_id].nodes
            )


def _roundtrip(
    writer: Callable[[Any, Path], Any],
    reader: Callable[[Path], Any],
    model: Any,
    path: Path,
) -> Any:
    writer(model, path)
    assert path.exists()
    loaded = reader(path)
    return getattr(loaded, "model", loaded)


def _rbe2_model() -> FEModel:
    model = FEModel(name="rbe2-validation")
    model.add_node(1, (0.0, 0.0, 0.0))
    model.add_node(2, (1.0, 0.0, 0.0))
    model.add_node(3, (0.0, 1.0, 0.0))
    return model


def _rbe3_model() -> FEModel:
    model = FEModel(name="rbe3-validation")
    model.add_node(1, (0.0, 0.0, 0.0))
    model.add_node(2, (1.0, 0.0, 0.0))
    model.add_node(3, (0.0, 1.0, 0.0))
    return model


def test_add_rbe2_stores_valid_constraint_on_model() -> None:
    model = _rbe2_model()

    rbe2 = model.add_rbe2(10, independent=1, dependents=(2, 3), components=(1, 3, 6))

    assert model.rbe2 == [rbe2]
    assert rbe2.id == 10
    assert rbe2.independent == 1
    assert rbe2.dependents == (2, 3)
    assert rbe2.components == (1, 3, 6)


def test_add_rbe2_rejects_duplicate_id() -> None:
    model = _rbe2_model()
    original = model.add_rbe2(10, independent=1, dependents=(2,))

    with pytest.raises(ModelError, match="duplicate RBE2 id 10"):
        model.add_rbe2(10, independent=1, dependents=(3,))

    assert model.rbe2 == [original]


@pytest.mark.parametrize(
    ("independent", "dependents"),
    [
        (99, (2,)),
        (1, (2, 99)),
    ],
)
def test_add_rbe2_rejects_missing_nodes(
    independent: int,
    dependents: tuple[int, ...],
) -> None:
    model = _rbe2_model()

    with pytest.raises(ModelError, match=r"RBE2 10: undefined node\(s\) \[99\]"):
        model.add_rbe2(10, independent=independent, dependents=dependents)

    assert model.rbe2 == []


def test_add_rbe2_rejects_independent_node_in_dependents() -> None:
    model = _rbe2_model()

    with pytest.raises(ModelError, match="independent node cannot also be dependent"):
        model.add_rbe2(10, independent=1, dependents=(1, 2))

    assert model.rbe2 == []


@pytest.mark.parametrize("components", [(0,), (7,), (1, 2, 7)])
def test_add_rbe2_rejects_components_outside_one_through_six(
    components: tuple[int, ...],
) -> None:
    model = _rbe2_model()

    with pytest.raises(ModelError, match=r"components must be in 1\.\.6"):
        model.add_rbe2(10, independent=1, dependents=(2,), components=components)

    assert model.rbe2 == []


def test_add_rbe3_rejects_duplicate_id() -> None:
    model = _rbe3_model()
    original = model.add_rbe3(10, dependent=1, independents=(2, 3))

    with pytest.raises(ModelError, match="duplicate RBE3 id 10"):
        model.add_rbe3(10, dependent=2, independents=(1, 3))

    assert model.rbe3 == [original]


@pytest.mark.parametrize(
    ("dependent", "independents"),
    [
        (99, (2, 3)),
        (1, (2, 99)),
    ],
)
def test_add_rbe3_rejects_missing_nodes(
    dependent: int,
    independents: tuple[int, ...],
) -> None:
    model = _rbe3_model()

    with pytest.raises(ModelError, match=r"RBE3 10: undefined node\(s\) \[99\]"):
        model.add_rbe3(10, dependent=dependent, independents=independents)

    assert model.rbe3 == []


def test_add_rbe3_rejects_dependent_node_in_independents() -> None:
    model = _rbe3_model()

    with pytest.raises(ModelError, match="dependent node cannot also be independent"):
        model.add_rbe3(10, dependent=1, independents=(1, 2))

    assert model.rbe3 == []


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ((1.0,), "1 weights for 2 independents"),
        ((1.0, 0.0), "weights must be positive"),
        ((1.0, -1.0), "weights must be positive"),
    ],
)
def test_add_rbe3_rejects_bad_weights(
    weights: tuple[float, ...],
    message: str,
) -> None:
    model = _rbe3_model()

    with pytest.raises(ModelError, match=message):
        model.add_rbe3(10, dependent=1, independents=(2, 3), weights=weights)

    assert model.rbe3 == []


@pytest.mark.parametrize(
    ("component_field", "components"),
    [
        ("components", (0,)),
        ("components", (7,)),
        ("independent_components", (0,)),
        ("independent_components", (7,)),
    ],
)
def test_add_rbe3_rejects_components_outside_one_through_six(
    component_field: str,
    components: tuple[int, ...],
) -> None:
    model = _rbe3_model()
    kwargs: dict[str, Any] = {component_field: components}

    with pytest.raises(ModelError, match=rf"{component_field} must be in 1\.\.6"):
        model.add_rbe3(10, dependent=1, independents=(2, 3), **kwargs)

    assert model.rbe3 == []


def test_apply_mpc_composes_model_rbe2_and_rbe3_tables() -> None:
    model = _rbe3_model()
    model.add_rbe2(10, independent=1, dependents=(2,), components=(1, 2, 3))
    model.add_rbe3(
        11,
        dependent=3,
        independents=(1, 2),
        components=(1, 2, 3),
        weights=(1.0, 3.0),
    )

    transform = apply_mpc(model)
    matrix = transform.G.toarray()

    assert transform.n_dependent == 6
    assert transform.dependent_nodes() == [2, 3]
    assert transform.independent_nodes() == [1]
    np.testing.assert_allclose(matrix @ matrix, matrix, rtol=0.0, atol=1.0e-15)

    displacement = np.zeros(transform.n_dof)
    displacement[transform.dof_map.node_dofs(1)] = (1.0, 2.0, 3.0, 0.2, -0.1, 0.3)
    full = transform.to_full(displacement)
    node_1 = full[transform.dof_map.node_dofs(1)[:3]]
    node_2 = full[transform.dof_map.node_dofs(2)[:3]]
    node_3 = full[transform.dof_map.node_dofs(3)[:3]]
    np.testing.assert_allclose(node_3, 0.25 * node_1 + 0.75 * node_2)


def test_project_roundtrip_preserves_model(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, _ = axial_bar
    path = tmp_path / "axial.ftproj"

    loaded = _roundtrip(
        project_module.save_project,
        project_module.load_project,
        model,
        path,
    )

    _assert_basic_model_equal(model, loaded)
    assert set(loaded.materials) == set(model.materials)
    assert set(loaded.properties) == set(model.properties)
    assert len(loaded.spcs) == len(model.spcs)


def test_bdf_roundtrip_preserves_bar_connectivity(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, _ = axial_bar
    path = tmp_path / "axial.bdf"

    loaded = _roundtrip(bdf_module.write_bdf, bdf_module.read_bdf, model, path)

    _assert_basic_model_equal(model, loaded, name=False)


def test_unv_roundtrip_preserves_node_dataset(
    tmp_path: Path,
    axial_bar: tuple[object, dict[str, float]],
) -> None:
    model, _ = axial_bar
    path = tmp_path / "axial.unv"

    loaded = _roundtrip(unv_module.write_unv, unv_module.read_unv, model, path)

    # Round 1 UNV explicitly contracts node datasets; element dataset 2412 is not included.
    _assert_basic_model_equal(model, loaded, elements=False)


def test_cdb_etblock_alphanumeric_format_maps_element_type(tmp_path: Path) -> None:
    keyopts = "".join(f"{value:>9}" for value in [*(["0"] * 18), "KEYOPT"])
    path = _write_cdb(
        tmp_path,
        "etblock.cdb",
        [
            "ETBLOCK,1",
            "(2i9,19a9)",
            _cdb_ints(7, 180) + keyopts,
            _cdb_ints(-1),
            "MP,EX,3,2.1E11",
            "R,4,0.125",
            "EBLOCK,19",
            "(19i9)",
            _cdb_ints(101, 7, 4, 3, 0, 1, 2),
            _cdb_ints(-1),
        ],
        {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
    )

    model = cdb_module.read_cdb(path)

    element = model.elements[101]
    assert element.type == "BAR2"
    assert element.nodes == (1, 2)
    assert model.properties[element.property_id].attrs["ansys_type"] == 180


def test_cdb_compact_eblock_inherits_current_attributes(tmp_path: Path) -> None:
    path = _write_cdb(
        tmp_path,
        "compact.cdb",
        [
            "ET,4,SOLID185",
            "MP,EX,6,7.0E10",
            "TYPE,4",
            "MAT,6",
            "REAL,7",
            "SECNUM,9",
            "EBLOCK,19,COMPACT",
            "(19i9)",
            _cdb_ints(321, *range(11, 19)),
            _cdb_ints(-1),
        ],
        {node_id: (float(node_id), 0.0, 0.0) for node_id in range(11, 19)},
    )

    model = cdb_module.read_cdb(path)

    element = model.elements[321]
    prop = model.properties[element.property_id]
    assert element.type == "HEX8"
    assert element.nodes == tuple(range(11, 19))
    assert prop.material_id == 6
    assert prop.attrs == {"ansys_type": 185, "ansys_real": 7, "ansys_secnum": 9}


def test_cdb_rmore_starts_at_real_constant_seven(tmp_path: Path) -> None:
    path = _write_cdb(
        tmp_path,
        "rmore.cdb",
        [
            "ET,1,BEAM4",
            "MP,EX,2,2.1E11",
            "R,7,0.25,2.0E-4,3.0E-4",
            "RMORE,99.0,4.0E-4",
            "TYPE,1",
            "MAT,2",
            "REAL,7",
            "EBLOCK,19,COMPACT",
            "(19i9)",
            _cdb_ints(42, 1, 2),
            _cdb_ints(-1),
        ],
        {1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0)},
    )

    model = cdb_module.read_cdb(path)

    prop = model.properties[model.elements[42].property_id]
    assert prop.A == pytest.approx(0.25)
    assert prop.Iz == pytest.approx(2.0e-4)
    assert prop.Iy == pytest.approx(3.0e-4)
    assert prop.J == pytest.approx(4.0e-4)


def test_cdb_beam3_uses_area_and_izz_not_height(tmp_path: Path) -> None:
    path = _write_cdb(
        tmp_path,
        "beam3.cdb",
        [
            "ET,1,BEAM3",
            "MP,EX,2,2.1E11",
            "R,3,0.02,8.0E-6,0.5",
            "TYPE,1",
            "MAT,2",
            "REAL,3",
            "EBLOCK,19,COMPACT",
            "(19i9)",
            _cdb_ints(17, 1, 2),
            _cdb_ints(-1),
        ],
        {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
    )

    with pytest.warns(UserWarning, match="BEAM3 real set"):
        model = cdb_module.read_cdb(path)

    prop = model.properties[model.elements[17].property_id]
    assert prop.A == pytest.approx(0.02)
    assert prop.Iz == pytest.approx(8.0e-6)
    assert prop.Iy == pytest.approx(8.0e-6)
    assert prop.J == pytest.approx(1.6e-5)

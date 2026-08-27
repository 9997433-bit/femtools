"""femtools project files (``.ftproj``).

An ``.ftproj`` file is a zip archive combining JSON metadata with npz
array payloads (lossless binary doubles -- no text round-off):

::

    project.ftproj
    |-- manifest.json        format/version stamp + result directory
    |-- model.json           elements/materials/properties/spcs/loads/
    |                        sets/coord systems/units (scalars only)
    |-- model_arrays.npz     node ids / coordinates / cp / cd
    `-- results/<n>.npz      one archive per stored result

Results can be :class:`~femtools.core.results.ModalResult`,
:class:`StaticResult`, :class:`FRFResult` or :class:`ODSResult`, stored
under a user-chosen name.
"""

from __future__ import annotations

import datetime as _dt
import io as _io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..core.coords import CoordSys
from ..core.model import SPC, Element, FEModel, Load, Material, Node, Property
from ..core.results import FRFResult, ModalResult, ODSResult, StaticResult
from ..core.sets import ElementSet, NodeSet
from ..core.units import UnitSystem

__all__ = ["Project", "save_project", "load_project", "ProjectError"]

FORMAT_NAME = "ftproj"
FORMAT_VERSION = 1

AnyResult = ModalResult | StaticResult | FRFResult | ODSResult


class ProjectError(ValueError):
    """Raised for unreadable or incompatible project files."""


@dataclass
class Project:
    """Contents of a loaded ``.ftproj`` file."""

    model: FEModel
    results: dict[str, AnyResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# -- JSON helpers -----------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, indent=1, sort_keys=True)


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = _io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _npz_read(data: bytes) -> dict[str, np.ndarray]:
    with np.load(_io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _pairs_to_array(pairs) -> np.ndarray:
    return np.asarray([[int(a), int(b)] for a, b in pairs], dtype=np.int64)


def _array_to_pairs(arr: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(a), int(b)) for a, b in np.asarray(arr).reshape(-1, 2))


# -- model (de)serialization -------------------------------------------------------


def _model_to_json(model: FEModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "units": model.units.to_dict(),
        "elements": [
            {
                "id": el.id,
                "type": el.type,
                "nodes": list(el.nodes),
                "property_id": el.property_id,
                "orientation": None if el.orientation is None else el.orientation.tolist(),
                "dofs": None if el.dofs is None else list(el.dofs),
            }
            for el in model.elements.values()
        ],
        "materials": [
            {
                k: v
                for k, v in vars(mat).items()
                if v is not None or k in ("E", "nu", "rho")
            }
            for mat in model.materials.values()
        ],
        "properties": [
            {k: v for k, v in vars(prop).items() if v is not None or k in ("material_id",)}
            for prop in model.properties.values()
        ],
        "spcs": [
            {"node_id": s.node_id, "mask": list(s.mask), "value": s.value, "sid": s.sid}
            for s in model.spcs
        ],
        "loads": [
            {
                "sid": ld.sid,
                "node_id": ld.node_id,
                "force": None if ld.force is None else ld.force.tolist(),
                "moment": None if ld.moment is None else ld.moment.tolist(),
            }
            for ld in model.loads
        ],
        "sets": [
            {
                "name": s.name,
                "kind": "node" if isinstance(s, NodeSet) else "element",
                "ids": s.sorted_ids(),
            }
            for s in model.sets.values()
        ],
        "coord_systems": [cs.to_dict() for cs in model.coord_systems.values()],
    }


def _model_arrays(model: FEModel) -> dict[str, np.ndarray]:
    ids = model.node_ids()
    return {
        "node_ids": np.asarray(ids, dtype=np.int64),
        "node_xyz": model.xyz_array(),
        "node_cp": np.asarray([model.nodes[i].cp for i in ids], dtype=np.int64),
        "node_cd": np.asarray([model.nodes[i].cd for i in ids], dtype=np.int64),
    }


def _model_from_parts(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> FEModel:
    model = FEModel(
        name=meta.get("name", "model"),
        units=UnitSystem.from_dict(meta.get("units", {})),
    )
    for cs in meta.get("coord_systems", ()):
        model.coord_systems[int(cs["id"])] = CoordSys.from_dict(cs)
    ids = arrays.get("node_ids", np.empty(0, dtype=np.int64))
    xyz = arrays.get("node_xyz", np.empty((0, 3)))
    cp = arrays.get("node_cp", np.zeros(len(ids), dtype=np.int64))
    cd = arrays.get("node_cd", np.zeros(len(ids), dtype=np.int64))
    for i, nid in enumerate(ids):
        node = Node(id=int(nid), xyz=xyz[i], cp=int(cp[i]), cd=int(cd[i]))
        model.nodes[node.id] = node
    for m in meta.get("materials", ()):
        mat = Material(**m)
        model.materials[mat.id] = mat
    for p in meta.get("properties", ()):
        prop = Property(**p)
        model.properties[prop.id] = prop
    for e in meta.get("elements", ()):
        el = Element(
            id=e["id"],
            type=e["type"],
            nodes=tuple(e["nodes"]),
            property_id=e.get("property_id"),
            orientation=None if e.get("orientation") is None else np.asarray(e["orientation"]),
            dofs=None if e.get("dofs") is None else tuple(e["dofs"]),
        )
        model.elements[el.id] = el
    for s in meta.get("spcs", ()):
        model.spcs.append(
            SPC(node_id=s["node_id"], mask=tuple(s["mask"]), value=s.get("value", 0.0), sid=s.get("sid", 1))
        )
    for ld in meta.get("loads", ()):
        model.loads.append(
            Load(sid=ld.get("sid", 1), node_id=ld["node_id"], force=ld.get("force"), moment=ld.get("moment"))
        )
    for s in meta.get("sets", ()):
        cls = NodeSet if s.get("kind", "node") == "node" else ElementSet
        model.sets[s["name"]] = cls(s["name"], frozenset(int(i) for i in s["ids"]))
    return model


# -- result (de)serialization ---------------------------------------------------------


def _result_to_parts(res: AnyResult) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    """-> (kind, arrays, extra_json)."""
    if isinstance(res, ModalResult):
        arrays = {
            "freq_hz": res.freq_hz,
            "eigenvalues": res.eigenvalues,
            "modes": res.modes,
            "generalized_mass": res.generalized_mass,
        }
        if res.dof_index is not None:
            arrays["dof_index"] = _pairs_to_array(res.dof_index)
        if res.damping is not None:
            arrays["damping"] = res.damping
        return "modal", arrays, {}
    if isinstance(res, StaticResult):
        arrays = {"u": res.u}
        if res.dof_index is not None:
            arrays["dof_index"] = _pairs_to_array(res.dof_index)
        if res.reactions is not None:
            arrays["reactions"] = res.reactions
        return "static", arrays, {"load_case": res.load_case}
    if isinstance(res, FRFResult):
        arrays = {
            "freq_hz": res.freq_hz,
            "h_complex": res.h_complex,
            "inputs": _pairs_to_array(res.inputs),
            "outputs": _pairs_to_array(res.outputs),
        }
        return "frf", arrays, {"kind": res.kind}
    if isinstance(res, ODSResult):
        arrays = {"freq_hz": res.freq_hz, "shapes": res.shapes}
        if res.dof_index is not None:
            arrays["dof_index"] = _pairs_to_array(res.dof_index)
        return "ods", arrays, {"name": res.name}
    raise ProjectError(f"unsupported result type {type(res).__name__}")


def _result_from_parts(kind: str, arrays: dict[str, np.ndarray], extra: dict[str, Any]) -> AnyResult:
    dof_index = _array_to_pairs(arrays["dof_index"]) if "dof_index" in arrays else None
    if kind == "modal":
        return ModalResult(
            freq_hz=arrays["freq_hz"],
            eigenvalues=arrays["eigenvalues"],
            modes=arrays["modes"],
            generalized_mass=arrays["generalized_mass"],
            dof_index=dof_index,
            damping=arrays.get("damping"),
        )
    if kind == "static":
        return StaticResult(
            u=arrays["u"],
            dof_index=dof_index,
            reactions=arrays.get("reactions"),
            load_case=extra.get("load_case", 1),
        )
    if kind == "frf":
        return FRFResult(
            freq_hz=arrays["freq_hz"],
            h_complex=arrays["h_complex"],
            inputs=_array_to_pairs(arrays["inputs"]),
            outputs=_array_to_pairs(arrays["outputs"]),
            kind=extra.get("kind", "receptance"),
        )
    if kind == "ods":
        return ODSResult(
            freq_hz=arrays["freq_hz"],
            shapes=arrays["shapes"],
            dof_index=dof_index,
            name=extra.get("name", "ODS"),
        )
    raise ProjectError(f"unknown result kind {kind!r} in project file")


# -- public API ------------------------------------------------------------------------


def save_project(
    path: str | Path,
    model: FEModel,
    results: dict[str, AnyResult] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save ``model`` (+ named ``results`` and free-form ``metadata``) to ``path``.

    The conventional extension is ``.ftproj``; arrays are stored as binary
    npz so the round trip is bit-exact for all numeric data.
    """
    results = results or {}
    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "femtools_version": __version__,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "results": {},
    }
    entries: list[tuple[str, bytes]] = []
    for i, (name, res) in enumerate(results.items()):
        kind, arrays, extra = _result_to_parts(res)
        fname = f"results/{i:04d}.npz"
        manifest["results"][name] = {"kind": kind, "file": fname, "extra": extra}
        entries.append((fname, _npz_bytes(arrays)))

    with zipfile.ZipFile(Path(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _dumps(manifest))
        zf.writestr("model.json", _dumps(_model_to_json(model)))
        zf.writestr("model_arrays.npz", _npz_bytes(_model_arrays(model)))
        for fname, data in entries:
            zf.writestr(fname, data)


def load_project(path: str | Path) -> Project:
    """Load a ``.ftproj`` file written by :func:`save_project`."""
    try:
        zf = zipfile.ZipFile(Path(path), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectError(f"cannot open project file {path}: {exc}") from exc
    with zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ProjectError(f"{path} is not a femtools project (no manifest.json)")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != FORMAT_NAME:
            raise ProjectError(f"{path}: unexpected format {manifest.get('format')!r}")
        if int(manifest.get("version", 0)) > FORMAT_VERSION:
            raise ProjectError(
                f"{path}: project version {manifest.get('version')} is newer than "
                f"supported version {FORMAT_VERSION}"
            )
        meta = json.loads(zf.read("model.json")) if "model.json" in names else {}
        arrays = _npz_read(zf.read("model_arrays.npz")) if "model_arrays.npz" in names else {}
        model = _model_from_parts(meta, arrays)
        results: dict[str, AnyResult] = {}
        for name, entry in manifest.get("results", {}).items():
            data = _npz_read(zf.read(entry["file"]))
            results[name] = _result_from_parts(entry["kind"], data, entry.get("extra", {}))
    return Project(model=model, results=results, metadata=manifest.get("metadata", {}))

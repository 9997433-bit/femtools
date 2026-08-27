"""Model-file loading shared by the femtools user surfaces (CLI and GUI).

:func:`load_model_file` dispatches on the file extension and always
returns a plain :class:`~femtools.core.model.FEModel` (container types
like ``Project`` and ``UnvData`` are unwrapped), together with any
named results stored alongside the model:

========================  ==============================================
extension                 reader
========================  ==============================================
``.ftproj``               ``femtools.io.project.load_project``
``.json``                 :func:`model_from_json_dict` (schema below)
``.unv`` / ``.uff``       ``femtools.io.unv.read_unv``
``.bdf`` / ``.nas`` /     ``femtools.io.bdf.read_bdf``
``.dat``
========================  ==============================================

The sibling packages are imported lazily; a missing ``femtools.core`` /
``femtools.io`` surfaces as ``ImportError`` so each caller can map it to
its own error convention (CLI exit code 3, GUI HTTP 400).

JSON model schema
-----------------
A ``.json`` model is a single object built through the public
``FEModel.add_*`` API.  All sections except ``nodes`` are optional::

    {
      "name": "cantilever",
      "nodes":      [{"id": 1, "xyz": [0.0, 0.0, 0.0]}, ...]
                    or {"1": [0.0, 0.0, 0.0], ...},
      "materials":  [{"id": 1, "type": "isotropic",
                      "E": 210e9, "nu": 0.3, "rho": 7850}, ...],
      "properties": [{"id": 1, "type": "beam", "material_id": 1,
                      "A": 1e-4, "Iy": 1e-8, "Iz": 1e-8, "J": 2e-8}, ...],
      "elements":   [{"id": 1, "type": "BEAM2", "nodes": [1, 2],
                      "property_id": 1}, ...],
      "spcs":       [{"node": 1, "dof": [1, 2, 3, 4, 5, 6]}
                     or {"node": 1, "mask": [1, 1, 1, 1, 1, 1]}, ...]
    }

``mat`` / ``prop`` are accepted as aliases for ``material_id`` /
``property_id``; SPC DOF numbers are 1-based (1..6, ux..rz).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["LoadedModel", "load_model_file", "model_from_json_dict", "SUPPORTED_SUFFIXES"]

SUPPORTED_SUFFIXES = (".ftproj", ".json", ".unv", ".uff", ".bdf", ".nas", ".dat")


@dataclass
class LoadedModel:
    """An :class:`FEModel` plus any results that were stored with it."""

    model: Any
    results: dict[str, Any] = field(default_factory=dict)
    format: str = ""
    path: str = ""


def _spc_mask(entry: dict[str, Any]) -> tuple[bool, ...]:
    if "mask" in entry:
        mask = [bool(int(v)) for v in entry["mask"]]
        if len(mask) != 6:
            raise ValueError(f"SPC mask must have 6 entries, got {entry['mask']!r}")
        return tuple(mask)
    dofs = entry.get("dof", entry.get("dofs", "all"))
    if isinstance(dofs, str):
        if dofs.lower() == "all":
            return (True,) * 6
        raise ValueError(f"SPC dof must be 'all' or a list of 1..6, got {dofs!r}")
    if isinstance(dofs, int):
        dofs = [dofs]
    mask6 = [False] * 6
    for d in dofs:
        if not 1 <= int(d) <= 6:
            raise ValueError(f"SPC DOF numbers are 1-based (1..6), got {d!r}")
        mask6[int(d) - 1] = True
    return tuple(mask6)


def model_from_json_dict(data: dict[str, Any]) -> Any:
    """Build an :class:`FEModel` from the JSON model schema (see module docs)."""
    from femtools.core.model import FEModel

    if not isinstance(data, dict):
        raise ValueError(f"JSON model must be an object, got {type(data).__name__}")
    model = FEModel(name=str(data.get("name", "model")))

    nodes = data.get("nodes", [])
    if isinstance(nodes, dict):
        nodes = [{"id": nid, "xyz": xyz} for nid, xyz in nodes.items()]
    for n in nodes:
        model.add_node(id=int(n["id"]), xyz=n["xyz"],
                       cp=int(n.get("cp", 0)), cd=int(n.get("cd", 0)))

    for m in data.get("materials", []):
        kwargs = {k: v for k, v in m.items() if k not in ("id", "type")}
        model.add_material(id=int(m["id"]), type=str(m.get("type", "isotropic")), **kwargs)

    for p in data.get("properties", []):
        kwargs = {k: v for k, v in p.items()
                  if k not in ("id", "type", "material_id", "mat")}
        mat_id = p.get("material_id", p.get("mat"))
        model.add_property(id=int(p["id"]), type=str(p["type"]),
                           material_id=None if mat_id is None else int(mat_id), **kwargs)

    for e in data.get("elements", []):
        prop_id = e.get("property_id", e.get("prop"))
        model.add_element(
            id=int(e["id"]),
            type=str(e["type"]).upper(),
            nodes=[int(n) for n in e["nodes"]],
            property_id=None if prop_id is None else int(prop_id),
            orientation=e.get("orientation"),
        )

    for s in data.get("spcs", []):
        node_id = s.get("node_id", s.get("node"))
        if node_id is None:
            raise ValueError(f"SPC entry needs a 'node' id: {s!r}")
        model.add_spc(node_id=int(node_id), mask=_spc_mask(s),
                      value=float(s.get("value", 0.0)))
    return model


def load_model_file(path: str | Path) -> LoadedModel:
    """Load a model file by extension and unwrap it to a bare ``FEModel``.

    Raises ``ValueError`` for unsupported extensions or malformed content
    and ``ImportError`` when the required femtools module is missing.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".ftproj":
        from femtools.io.project import load_project

        project = load_project(str(p))
        return LoadedModel(model=project.model, results=dict(project.results),
                           format="ftproj", path=str(p))

    if suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        return LoadedModel(model=model_from_json_dict(data), format="json", path=str(p))

    if suffix in (".unv", ".uff"):
        from femtools.io.unv import read_unv

        data = read_unv(str(p))
        results: dict[str, Any] = {}
        if getattr(data, "modal", None) is not None:
            results["modal"] = data.modal
        if getattr(data, "frf", None) is not None:
            results["frf"] = data.frf
        return LoadedModel(model=data.model, results=results, format="unv", path=str(p))

    if suffix in (".bdf", ".nas", ".dat"):
        from femtools.io.bdf import read_bdf

        return LoadedModel(model=read_bdf(str(p)), format="bdf", path=str(p))

    raise ValueError(
        f"unsupported model file {p} "
        f"(expected one of: {', '.join(SUPPORTED_SUFFIXES)})"
    )

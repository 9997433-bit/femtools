"""Plotting for femtools models and results (matplotlib, optional plotly).

The functions here are deliberately duck-typed so they work with the
contract objects (``FEModel``, ``ModalResult``, ``FRFResult``) without a
hard import dependency on the sibling packages.  Everything runs
headless: when no display is available the Agg backend is selected
before pyplot is imported.

Backends
--------
matplotlib is the default and the only required backend.  When the
optional ``plotly`` package is installed, every plot function also
accepts ``backend="plotly"`` and returns a
:class:`plotly.graph_objects.Figure` instead of a matplotlib Figure
(``outfile=`` then writes standalone ``.html``, or a static image via
the optional ``kaleido`` package).  The process-wide default can be
switched with :func:`set_default_backend`; plotly is never imported
unless a plotly plot is actually requested.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = [
    "plot_mesh",
    "plot_mesh3d",
    "plot_stress",
    "plot_mac",
    "plot_frf",
    "plot_psd",
    "plot_mode",
    "plotly_available",
    "pyvista_available",
    "get_default_backend",
    "set_default_backend",
]

_BACKENDS = ("matplotlib", "plotly")
_default_backend = "matplotlib"


def plotly_available() -> bool:
    """True when the optional plotly package can be imported."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True


def pyvista_available() -> bool:
    """True when the optional pyvista package can be imported."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return False
    return True


def get_default_backend() -> str:
    """Name of the backend used when ``backend=`` is not given."""
    return _default_backend


def set_default_backend(backend: str) -> str:
    """Set the process-wide plotting backend; returns the previous one.

    ``"matplotlib"`` (the shipped default) or ``"plotly"`` (requires the
    optional plotly package; raises ImportError when it is missing).
    """
    global _default_backend
    name = _canonical_backend(backend)
    if name == "plotly" and not plotly_available():
        raise ImportError(
            "the plotly backend needs the optional 'plotly' package "
            "(pip install plotly); matplotlib remains the default backend"
        )
    previous = _default_backend
    _default_backend = name
    return previous


def _canonical_backend(backend: str | None) -> str:
    name = (_default_backend if backend is None else str(backend)).strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown plotting backend {backend!r}; use one of {', '.join(_BACKENDS)}")
    return name


def _use_plotly(backend: str | None, ax: Any) -> bool:
    """Resolve the backend choice for one plot call."""
    if _canonical_backend(backend) != "plotly":
        return False
    if ax is not None:
        raise ValueError("ax= is a matplotlib Axes and cannot be combined with "
                         "backend='plotly'")
    return True


def _plotly_go():
    """Import plotly.graph_objects with a clear error when it is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "this plot was requested with backend='plotly' but the optional "
            "'plotly' package is not installed (pip install plotly); "
            "matplotlib is the default backend and needs no extra install"
        ) from exc
    return go


def _finish_plotly(fig: Any, outfile: str | None) -> Any:
    """Write a plotly figure to .html (or a static image via kaleido)."""
    if outfile:
        from pathlib import Path

        if Path(outfile).suffix.lower() in (".html", ".htm"):
            fig.write_html(outfile, include_plotlyjs="cdn")
        else:
            try:
                fig.write_image(outfile)
            except Exception as exc:
                raise RuntimeError(
                    f"static image export to {outfile!r} needs the optional "
                    "'kaleido' package (pip install kaleido); alternatively "
                    "save the plotly figure to a .html file"
                ) from exc
    return fig

# edge tables per Round-1 element type (indices into the element node list)
_EDGE_TABLE: dict[str, tuple[tuple[int, int], ...]] = {
    "BAR2": ((0, 1),),
    "BEAM2": ((0, 1),),
    "TRUSS2D": ((0, 1),),
    "SPRING": ((0, 1),),
    "DAMPER": ((0, 1),),
    "TRIA3": ((0, 1), (1, 2), (2, 0)),
    "QUAD4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "TET4": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    "HEX8": (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ),
    "MASS": (),
}


def _plt():
    """Import pyplot, forcing Agg when running without a display."""
    import matplotlib

    if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, outfile: str | None):
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    return fig


def _node_coords(model: Any) -> dict[int, np.ndarray]:
    """Extract ``{node_id: xyz(3,)}`` from a model, duck-typed."""
    coords: dict[int, np.ndarray] = {}
    for nid, node in model.nodes.items():
        xyz = getattr(node, "xyz", node)
        coords[nid] = np.asarray(xyz, dtype=float).reshape(3)
    return coords


def _element_edges(model: Any) -> list[tuple[str, tuple[int, ...], int]]:
    """Yield ``(type, node_ids, element_id)`` for every element."""
    out = []
    for eid, elem in model.elements.items():
        etype = str(getattr(elem, "type", "")).upper()
        nodes = tuple(getattr(elem, "nodes", ()))
        out.append((etype, nodes, eid))
    return out


def _segments(model: Any, coords: dict[int, np.ndarray]) -> tuple[list, list]:
    """Return (line segments, point coords) for the wireframe of a model."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    points: list[np.ndarray] = []
    for etype, nodes, _eid in _element_edges(model):
        edges = _EDGE_TABLE.get(etype)
        if edges is None:  # unknown type: chain consecutive nodes, close polygons
            n = len(nodes)
            edges = tuple((i, i + 1) for i in range(n - 1))
            if n > 2:
                edges += ((n - 1, 0),)
        if not edges and nodes:  # point element (MASS)
            if nodes[0] in coords:
                points.append(coords[nodes[0]])
            continue
        for a, b in edges:
            try:
                segments.append((coords[nodes[a]], coords[nodes[b]]))
            except (KeyError, IndexError):
                continue
    return segments, points


def _is_planar(xyz: np.ndarray) -> bool:
    """True when all points lie (nearly) in the global XY plane."""
    if xyz.size == 0:
        return True
    span = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1e-30)
    return float(np.ptp(xyz[:, 2])) <= 1e-9 * span


def _draw_wireframe(ax, segments, points, *, three_d: bool, color, lw, alpha=1.0):
    for p, q in segments:
        if three_d:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, lw=lw, alpha=alpha)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha)
    if points:
        pts = np.asarray(points)
        if three_d:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, marker="s", alpha=alpha)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], color=color, marker="s", alpha=alpha)


def plot_mesh(
    model: Any,
    ax: Any = None,
    *,
    show_node_ids: bool = False,
    show_element_ids: bool = False,
    color: str = "tab:blue",
    node_color: str = "black",
    title: str | None = None,
    outfile: str | None = None,
    backend: str | None = None,
):
    """Plot the undeformed wireframe of an FE model.

    Planar (XY) models are drawn in 2D, everything else in 3D.
    Returns the matplotlib Figure (or a plotly Figure with
    ``backend="plotly"``).
    """
    if _use_plotly(backend, ax):
        return _plotly_mesh(model, show_node_ids=show_node_ids,
                            show_element_ids=show_element_ids, color=color,
                            node_color=node_color, title=title, outfile=outfile)
    plt = _plt()
    coords = _node_coords(model)
    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))
    three_d = not _is_planar(xyz)

    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d" if three_d else None)
    else:
        fig = ax.figure
        three_d = getattr(ax, "name", "") == "3d"

    segments, points = _segments(model, coords)
    _draw_wireframe(ax, segments, points, three_d=three_d, color=color, lw=1.5)

    if coords:
        if three_d:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=node_color, s=12, zorder=3)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], color=node_color, s=12, zorder=3)

    if show_node_ids:
        for nid, p in coords.items():
            pos = p if three_d else p[:2]
            ax.text(*pos, f" {nid}", fontsize=8, color=node_color)
    if show_element_ids:
        for _etype, nodes, eid in _element_edges(model):
            pts = [coords[n] for n in nodes if n in coords]
            if not pts:
                continue
            c = np.mean(pts, axis=0)
            pos = c if three_d else c[:2]
            ax.text(*pos, str(eid), fontsize=8, color=color, ha="center")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if three_d:
        ax.set_zlabel("z")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or f"{getattr(model, 'name', 'model')}: "
                          f"{len(coords)} nodes, {len(model.elements)} elements")
    return _finish(fig, outfile)


# ----------------------------------------------------------------------
# 3-D mesh rendering (optional pyvista; matplotlib fallback)
# ----------------------------------------------------------------------
# VTK cell-type ids per femtools element type (stable VTK constants,
# hard-coded so this table never needs pyvista/vtk at import time)
_VTK_CELL_TYPES: dict[str, int] = {
    "MASS": 1,      # VTK_VERTEX
    "BAR2": 3,      # VTK_LINE
    "BEAM2": 3,
    "TRUSS2D": 3,
    "SPRING": 3,
    "DAMPER": 3,
    "TRIA3": 5,     # VTK_TRIANGLE
    "QUAD4": 9,     # VTK_QUAD
    "TET4": 10,     # VTK_TETRA
    "HEX8": 12,     # VTK_HEXAHEDRON
}


def _pyvista():
    """Import pyvista with a clear error message when it is missing."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "this plot was requested with backend='pyvista' but the optional "
            "'pyvista' package is not installed (pip install pyvista); "
            "matplotlib is the default backend and needs no extra install"
        ) from exc
    return pv


def _pyvista_grid(model: Any):
    """Build a ``pyvista.UnstructuredGrid`` from an FE model (duck-typed)."""
    pv = _pyvista()
    coords = _node_coords(model)
    index = {nid: i for i, nid in enumerate(coords)}
    points = np.asarray(list(coords.values()), dtype=float).reshape(-1, 3)

    cells: list[int] = []
    celltypes: list[int] = []
    for etype, nodes, _eid in _element_edges(model):
        idx = [index[n] for n in nodes if n in index]
        if not idx:
            continue
        # unknown types degrade to a point or polyline, mirroring the
        # unknown-type chaining of the matplotlib wireframe
        vtk_type = _VTK_CELL_TYPES.get(etype, 1 if len(idx) == 1 else 4)
        cells.append(len(idx))
        cells.extend(idx)
        celltypes.append(vtk_type)

    if not celltypes:
        return pv.PolyData(points)
    return pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.asarray(celltypes, dtype=np.uint8),
        points,
    )


def plot_mesh3d(
    model: Any,
    *,
    color: str = "tab:blue",
    show_edges: bool = True,
    opacity: float = 1.0,
    title: str | None = None,
    window_size: tuple[int, int] = (1024, 768),
    show: bool = False,
    outfile: str | None = None,
    backend: str | None = None,
):
    """Render an interactive-quality 3-D view of an FE model.

    Uses the optional ``pyvista`` package when it is importable: solid
    elements become real VTK cells (shaded surfaces with edges) instead
    of a bare wireframe.  pyvista is never required — when it is not
    installed the plot silently falls back to the matplotlib 3-D
    wireframe, and matplotlib remains the default backend of
    :mod:`femtools.viz`; pyvista is not imported unless it is actually
    used.

    ``backend`` selects the renderer explicitly: ``None`` (default)
    prefers pyvista and falls back to matplotlib, ``"pyvista"``
    requires pyvista (raises ImportError when missing) and
    ``"matplotlib"`` always draws the matplotlib 3-D wireframe.

    With pyvista the return value is the ``pyvista.Plotter``
    (``outfile=`` saves an off-screen screenshot, ``show=True`` opens
    the interactive window); with matplotlib it is the
    :class:`matplotlib.figure.Figure` as for every other viz function
    (``show=`` is ignored there).
    """
    choice = "auto" if backend is None else str(backend).strip().lower()
    if choice not in ("auto", "pyvista", "matplotlib"):
        raise ValueError(f"unknown plot_mesh3d backend {backend!r}; "
                         "use 'pyvista', 'matplotlib' or None (auto)")
    if choice == "matplotlib" or (choice == "auto" and not pyvista_available()):
        return _mpl_mesh3d(model, color=color, title=title, outfile=outfile)

    pv = _pyvista()
    grid = _pyvista_grid(model)
    n_nodes = len(getattr(model, "nodes", {}))
    n_elems = len(getattr(model, "elements", {}))

    plotter = pv.Plotter(off_screen=not show, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        color=_plotly_color(color),  # hex spelling: understood by every backend
        show_edges=show_edges,
        line_width=2.0,
        point_size=8.0,
        opacity=opacity,
    )
    plotter.add_axes()
    plotter.add_text(
        title or f"{getattr(model, 'name', 'model')}: {n_nodes} nodes, {n_elems} elements",
        font_size=10,
    )
    if outfile:
        plotter.screenshot(str(outfile))
    if show:
        plotter.show()
    return plotter


def _mpl_mesh3d(model: Any, *, color: str, title: str | None, outfile: str | None):
    """matplotlib fallback for :func:`plot_mesh3d`: always a 3-D wireframe."""
    plt = _plt()
    coords = _node_coords(model)
    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    segments, points = _segments(model, coords)
    _draw_wireframe(ax, segments, points, three_d=True, color=color, lw=1.5)
    if coords:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", s=12, zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or f"{getattr(model, 'name', 'model')}: "
                          f"{len(coords)} nodes, {len(model.elements)} elements")
    return _finish(fig, outfile)


# ----------------------------------------------------------------------
# stress fringe plot (matplotlib default; optional pyvista like plot_mesh3d)
# ----------------------------------------------------------------------
# filled-face tables per element type (indices into the element node list);
# surface elements are their own face, solids expose their boundary faces
_FACE_TABLE: dict[str, tuple[tuple[int, ...], ...]] = {
    "TRIA3": ((0, 1, 2),),
    "QUAD4": ((0, 1, 2, 3),),
    "TET4": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),
    "HEX8": (
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ),
}

_LINE_TYPES = frozenset({"BAR2", "BEAM2", "TRUSS2D", "SPRING", "DAMPER"})

#: Voigt component order assumed for 6-wide stress rows (matches
#: ``femtools.fea.recover.COMPONENTS`` without importing the kernel).
_VOIGT = ("xx", "yy", "zz", "xy", "yz", "zx")
_VM_ALIASES = frozenset({"von_mises", "vonmises", "von-mises", "vm", "mises"})


def _von_mises6(rows: np.ndarray) -> np.ndarray:
    """Von Mises equivalent of ``(n, 6)`` Voigt stress rows."""
    s = np.atleast_2d(np.asarray(rows, dtype=float))
    dev = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
    shear = s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    return np.sqrt(0.5 * dev + 3.0 * shear)


def _component_scalars(rows: np.ndarray, component: Any,
                       components: tuple[str, ...]) -> tuple[np.ndarray, str]:
    """Reduce ``(n, k)`` component rows to per-element scalars ``(values, label)``."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    if isinstance(component, (int, np.integer)):
        idx = int(component)
        if not 0 <= idx < rows.shape[1]:
            raise ValueError(f"component index {idx} out of range 0..{rows.shape[1] - 1}")
        label = components[idx] if idx < len(components) else f"component {idx}"
        return rows[:, idx], f"stress {label}"
    name = str(component).strip().lower()
    if name in _VM_ALIASES:
        if rows.shape[1] != 6:
            raise ValueError(
                "von Mises needs 6 Voigt components per element, got "
                f"{rows.shape[1]}; pass component=<index> instead")
        return _von_mises6(rows), "von Mises stress"
    lookup = [str(c).lower() for c in components]
    key = name[1:] if name.startswith("s") and name[1:] in lookup else name
    if key in lookup:
        idx = lookup.index(key)
        if idx >= rows.shape[1]:
            raise ValueError(f"component {component!r} is column {idx} but the stress "
                             f"rows only have {rows.shape[1]} columns")
        return rows[:, idx], f"stress {components[idx]}"
    raise ValueError(
        f"unknown stress component {component!r}: use 'von_mises', one of "
        f"{', '.join(components)}, or a 0-based column index")


def _stress_values(stress: Any, component: Any) -> tuple[dict[Any, float], str]:
    """Duck-typed ``{element_id: scalar}`` extraction plus a colorbar label.

    Accepts a ``StressResult``-like object (``element_ids`` plus a
    ``(n, 6)`` ``stress`` array and optionally a precomputed per-element
    array attribute such as ``von_mises`` or ``max_shear``) or a plain
    mapping ``{element_id: scalar | 6 Voigt components}``.
    """
    if isinstance(stress, dict):
        if not stress:
            raise ValueError("the stress mapping is empty: nothing to color by")
        widths = {np.asarray(v, dtype=float).reshape(-1).size for v in stress.values()}
        if widths == {1}:
            name = str(component).strip().lower()
            label = "stress" if name in _VM_ALIASES else f"stress {component}"
            return ({eid: float(np.asarray(v).reshape(-1)[0])
                     for eid, v in stress.items()}, label)
        rows = np.vstack([np.asarray(v, dtype=float).reshape(1, -1)
                          for v in stress.values()])
        values, label = _component_scalars(rows, component, _VOIGT)
        return dict(zip(stress.keys(), values.tolist(), strict=True)), label

    ids = None
    for attr in ("element_ids", "eids", "elements", "ids"):
        ids = getattr(stress, attr, None)
        if ids is not None:
            break
    if ids is None:
        raise ValueError(
            f"cannot read element ids from {type(stress).__name__}; pass a "
            "StressResult or a {element_id: value} mapping")
    ids = list(ids)

    # a per-element array attribute named like the component wins: this
    # resolves 'von_mises' (and e.g. 'max_shear') on a StressResult directly
    name = str(component).strip().lower()
    attr_name = "von_mises" if name in _VM_ALIASES else name
    direct = getattr(stress, attr_name, None)
    if direct is not None:
        arr = np.asarray(direct, dtype=float).reshape(-1)
        if arr.size == len(ids):
            label = ("von Mises stress" if attr_name == "von_mises"
                     else str(component).strip())
            return dict(zip(ids, arr.tolist(), strict=True)), label

    rows = getattr(stress, "stress", None)
    if rows is None:
        raise ValueError(f"{type(stress).__name__} has no 'stress' component array")
    components = tuple(getattr(stress, "components", _VOIGT))
    values, label = _component_scalars(rows, component, components)
    if values.size != len(ids):
        raise ValueError(f"{len(ids)} element ids but {values.size} stress rows")
    return dict(zip(ids, values.tolist(), strict=True)), label


def _element_faces(etype: str, nodes: tuple[int, ...],
                   coords: dict[int, np.ndarray]) -> list[list[np.ndarray]]:
    """Filled polygons (vertex lists) of one element; [] for non-surface types."""
    faces = _FACE_TABLE.get(etype)
    if faces is None:
        # unknown types with 3+ nodes: close the node loop, like the wireframe
        if etype in _LINE_TYPES or etype == "MASS" or len(nodes) < 3:
            return []
        faces = (tuple(range(len(nodes))),)
    out = []
    for face in faces:
        try:
            out.append([coords[nodes[i]] for i in face])
        except (KeyError, IndexError):
            continue
    return out


def plot_stress(
    model: Any,
    stress: Any,
    component: Any = "von_mises",
    ax: Any = None,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_edges: bool = True,
    colorbar: bool = True,
    missing_color: str = "0.75",
    title: str | None = None,
    outfile: str | None = None,
    backend: str | None = None,
    window_size: tuple[int, int] = (1024, 768),
    show: bool = False,
):
    """Color the mesh of *model* by a recovered stress field.

    ``stress`` is a :class:`femtools.fea.recover.StressResult` (or any
    object with ``element_ids`` and a ``(n, 6)`` Voigt ``stress`` array)
    or a plain mapping ``{element_id: scalar | 6 components}``.
    ``component`` selects the fringe value: ``"von_mises"`` (default), a
    Voigt component name (``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``,
    ``"yz"``, ``"zx"``, also spelled ``"sxx"`` ...), a 0-based column
    index, or the name of a per-element array attribute of the result
    (e.g. ``"max_shear"``).

    Surface elements are drawn as filled polygons, solid elements by
    their boundary faces and line elements as colored segments; elements
    without a recovered value (e.g. ``StressResult.skipped`` entries)
    stay as a ``missing_color`` wireframe.  Planar (XY) models are drawn
    in 2D, everything else in 3D.

    matplotlib is the default and only required backend.  Pass
    ``backend="pyvista"`` to render through the same optional pyvista
    path as :func:`plot_mesh3d` (shaded VTK cells with per-cell
    scalars); that raises ImportError when pyvista is not installed and
    is never imported otherwise.  Returns the matplotlib Figure (or the
    ``pyvista.Plotter``, whose ``outfile=`` is an off-screen screenshot
    and ``show=True`` opens the interactive window).
    """
    choice = "matplotlib" if backend is None else str(backend).strip().lower()
    if choice not in ("matplotlib", "pyvista"):
        raise ValueError(f"unknown plot_stress backend {backend!r}; "
                         "use 'matplotlib', 'pyvista' or None (matplotlib)")
    values, label = _stress_values(stress, component)
    if choice == "pyvista":
        if ax is not None:
            raise ValueError("ax= is a matplotlib Axes and cannot be combined "
                             "with backend='pyvista'")
        return _pyvista_stress(model, values, label, cmap=cmap, vmin=vmin,
                               vmax=vmax, show_edges=show_edges, title=title,
                               window_size=window_size, show=show,
                               outfile=outfile)

    plt = _plt()
    from matplotlib import cm as _cm
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize

    coords = _node_coords(model)
    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))
    three_d = not _is_planar(xyz)

    polys: list[list[np.ndarray]] = []
    poly_vals: list[float] = []
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    line_vals: list[float] = []
    missing_elems: list[tuple[str, tuple[int, ...], Any]] = []
    for etype, nodes, eid in _element_edges(model):
        value = values.get(eid)
        if value is None or not np.isfinite(value):
            missing_elems.append((etype, nodes, eid))
            continue
        faces = _element_faces(etype, nodes, coords)
        if faces:
            polys.extend(faces)
            poly_vals.extend([float(value)] * len(faces))
        elif len(nodes) >= 2 and nodes[0] in coords and nodes[1] in coords:
            lines.append((coords[nodes[0]], coords[nodes[1]]))
            line_vals.append(float(value))
        else:
            missing_elems.append((etype, nodes, eid))
    if not polys and not lines:
        raise ValueError(
            "no element of the model has a stress value to color by "
            "(check that the stress result belongs to this model)")

    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d" if three_d else None)
    else:
        fig = ax.figure
        three_d = getattr(ax, "name", "") == "3d"

    all_vals = np.asarray(poly_vals + line_vals, dtype=float)
    lo = float(np.min(all_vals)) if vmin is None else float(vmin)
    hi = float(np.max(all_vals)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-12
    norm = Normalize(vmin=lo, vmax=hi)
    colormap = plt.get_cmap(cmap)

    if missing_elems:
        # uncovered elements stay visible as a neutral wireframe
        seg_miss: list[tuple[np.ndarray, np.ndarray]] = []
        pts_miss: list[np.ndarray] = []
        for etype, nodes, _eid in missing_elems:
            edges = _EDGE_TABLE.get(etype)
            if edges is None:
                n = len(nodes)
                edges = tuple((i, i + 1) for i in range(n - 1))
                if n > 2:
                    edges += ((n - 1, 0),)
            if not edges and nodes and nodes[0] in coords:
                pts_miss.append(coords[nodes[0]])
                continue
            for a, b in edges:
                try:
                    seg_miss.append((coords[nodes[a]], coords[nodes[b]]))
                except (KeyError, IndexError):
                    continue
        _draw_wireframe(ax, seg_miss, pts_miss, three_d=three_d,
                        color=missing_color, lw=1.0, alpha=0.9)

    edge_kw = {"edgecolors": "0.2", "linewidths": 0.5} if show_edges else \
              {"edgecolors": "face", "linewidths": 0.0}
    if three_d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

        if polys:
            pc = Poly3DCollection([np.asarray(p) for p in polys], **edge_kw)
            pc.set_facecolor(colormap(norm(np.asarray(poly_vals))))
            ax.add_collection3d(pc)
        if lines:
            lc = Line3DCollection([np.asarray(seg) for seg in lines], linewidths=3.0)
            lc.set_array(np.asarray(line_vals))
            lc.set_cmap(colormap)
            lc.set_norm(norm)
            ax.add_collection3d(lc)
        if len(xyz):
            ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2], had_data=True)
        ax.set_zlabel("z")
    else:
        if polys:
            pc = PolyCollection([np.asarray(p)[:, :2] for p in polys], **edge_kw)
            pc.set_facecolor(colormap(norm(np.asarray(poly_vals))))
            ax.add_collection(pc)
        if lines:
            lc = LineCollection([np.asarray(seg)[:, :2] for seg in lines],
                                linewidths=3.0)
            lc.set_array(np.asarray(line_vals))
            lc.set_cmap(colormap)
            lc.set_norm(norm)
            ax.add_collection(lc)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="datalim")

    if colorbar:
        sm = _cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    peak = float(np.max(all_vals))
    ax.set_title(title or f"{getattr(model, 'name', 'model')}: {label} "
                          f"(max {peak:.4g})")
    return _finish(fig, outfile)


def _pyvista_stress(
    model: Any,
    values: dict[Any, float],
    label: str,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    show_edges: bool,
    title: str | None,
    window_size: tuple[int, int],
    show: bool,
    outfile: str | None,
):
    """pyvista renderer for :func:`plot_stress` (same path as plot_mesh3d)."""
    pv = _pyvista()
    grid = _pyvista_grid(model)
    coords = _node_coords(model)

    # cell order matches _pyvista_grid: every element with mappable nodes
    cell_vals: list[float] = []
    for _etype, nodes, eid in _element_edges(model):
        if not any(n in coords for n in nodes):
            continue
        value = values.get(eid)
        cell_vals.append(float(value) if value is not None else float("nan"))
    scalars = np.asarray(cell_vals, dtype=float)
    if isinstance(grid, pv.PolyData) or scalars.size != grid.n_cells:
        raise ValueError(
            "no element of the model has a stress value to color by "
            "(check that the stress result belongs to this model)")
    grid.cell_data[label] = scalars

    finite = scalars[np.isfinite(scalars)]
    clim = None
    if finite.size:
        clim = (float(np.min(finite)) if vmin is None else float(vmin),
                float(np.max(finite)) if vmax is None else float(vmax))

    plotter = pv.Plotter(off_screen=not show, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        scalars=label,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        nan_color="lightgray",
        line_width=2.0,
        point_size=8.0,
        scalar_bar_args={"title": label},
    )
    plotter.add_axes()
    plotter.add_text(
        title or f"{getattr(model, 'name', 'model')}: {label}",
        font_size=10,
    )
    if outfile:
        plotter.screenshot(str(outfile))
    if show:
        plotter.show()
    return plotter


def plot_mac(
    mac: Any,
    ax: Any = None,
    *,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
    annotate: bool | None = None,
    cmap: str = "viridis",
    title: str = "MAC",
    outfile: str | None = None,
    backend: str | None = None,
):
    """Plot a MAC (or any correlation) matrix as an annotated heatmap.

    ``mac[i, j]`` is drawn at row *i* (set A), column *j* (set B).
    Values are annotated when the matrix is 15x15 or smaller (override
    with ``annotate=``). Returns the matplotlib Figure (or a plotly
    Figure with ``backend="plotly"``).
    """
    if _use_plotly(backend, ax):
        return _plotly_mac(mac, labels_a=labels_a, labels_b=labels_b,
                           annotate=annotate, cmap=cmap, title=title,
                           outfile=outfile)
    plt = _plt()
    m = np.asarray(mac, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"MAC matrix must be 2-D, got shape {m.shape}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, label="MAC")

    n_a, n_b = m.shape
    ax.set_xticks(range(n_b))
    ax.set_yticks(range(n_a))
    ax.set_xticklabels(labels_b or [str(j + 1) for j in range(n_b)])
    ax.set_yticklabels(labels_a or [str(i + 1) for i in range(n_a)])
    ax.set_xlabel("set B mode")
    ax.set_ylabel("set A mode")
    ax.set_title(title)

    if annotate is None:
        annotate = max(n_a, n_b) <= 15
    if annotate:
        for i in range(n_a):
            for j in range(n_b):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] < 0.6 else "black")
    return _finish(fig, outfile)


def _extract_frf(frf: Any, freq: Any):
    """Duck-typed extraction of (freq_hz, H) from an FRFResult or arrays."""
    if freq is None:
        for attr in ("freq_hz", "freqs_hz", "freqs", "freq", "f"):
            freq = getattr(frf, attr, None)
            if freq is not None:
                break
    data = None
    for attr in ("H", "h", "frf", "data", "values", "matrix"):
        data = getattr(frf, attr, None)
        if data is not None:
            break
    if data is None:
        data = frf  # raw ndarray
    H = np.asarray(data)
    if freq is None:
        raise ValueError("could not find a frequency vector: pass freq= explicitly")
    return np.asarray(freq, dtype=float), H


def _frf_curve(frf: Any, output: int, input_: int, freq: Any) -> tuple[Any, Any]:
    """Select one FRF curve ``(f, h)`` out of a duck-typed FRF object."""
    f, H = _extract_frf(frf, freq)
    if H.ndim == 3:
        h = H[output, input_, :]
    elif H.ndim == 2:  # (n_curves, n_freq) — take requested output row
        h = H[output, :]
    else:
        h = H
    h = np.asarray(h).reshape(-1)
    if h.shape[0] != f.shape[0]:
        raise ValueError(f"FRF length {h.shape[0]} != frequency length {f.shape[0]}")
    return f, h


def _frf_kind(kind: str) -> tuple[bool, bool]:
    want_mag = kind in ("bode", "mag", "magnitude")
    want_phase = kind in ("bode", "phase")
    if not (want_mag or want_phase):
        raise ValueError(f"kind must be 'bode', 'mag' or 'phase', got {kind!r}")
    return want_mag, want_phase


def plot_frf(
    frf: Any,
    output: int = 0,
    input: int = 0,  # noqa: A002 - matches contract terminology
    ax: Any = None,
    *,
    freq: Any = None,
    kind: str = "bode",
    db: bool = False,
    label: str | None = None,
    title: str | None = None,
    outfile: str | None = None,
    backend: str | None = None,
):
    """Plot one FRF curve (magnitude, or magnitude + phase Bode pair).

    ``frf`` may be an ``FRFResult`` (complex array ``(n_out, n_in,
    n_freq)`` plus a frequency vector) or a raw complex array combined
    with ``freq=``.  ``kind`` is ``"bode"`` (default), ``"mag"`` or
    ``"phase"``.  Set ``db=True`` for a dB magnitude axis.  Returns the
    matplotlib Figure (or a plotly Figure with ``backend="plotly"``).
    """
    if _use_plotly(backend, ax):
        return _plotly_frf(frf, output, input, freq=freq, kind=kind, db=db,
                           label=label, title=title, outfile=outfile)
    plt = _plt()
    f, h = _frf_curve(frf, output, input, freq)
    want_mag, want_phase = _frf_kind(kind)

    if ax is None:
        n_rows = 2 if (want_mag and want_phase) else 1
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(7, 5),
                                 height_ratios=[2, 1] if n_rows == 2 else None)
        axes = np.atleast_1d(axes)
    else:
        fig = ax.figure
        axes = np.atleast_1d(ax)
        if len(axes) < 2:
            want_phase = want_phase and not want_mag  # single axis: plot one quantity

    curve_label = label or f"H[{output},{input}]"
    idx = 0
    if want_mag:
        mag = np.abs(h)
        a = axes[idx]
        if db:
            with np.errstate(divide="ignore"):
                a.plot(f, 20.0 * np.log10(mag), label=curve_label)
            a.set_ylabel("|H| [dB]")
        else:
            a.semilogy(f, np.where(mag > 0, mag, np.nan), label=curve_label)
            a.set_ylabel("|H|")
        a.grid(True, which="both", alpha=0.3)
        a.legend(loc="best", fontsize=8)
        idx += 1
    if want_phase:
        a = axes[min(idx, len(axes) - 1)]
        a.plot(f, np.degrees(np.unwrap(np.angle(h))), label=curve_label)
        a.set_ylabel("phase [deg]")
        a.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frequency [Hz]")
    axes[0].set_title(title or "Frequency response function")
    return _finish(fig, outfile)


def plot_psd(
    psd: Any,
    output: int = 0,
    ax: Any = None,
    *,
    freq: Any = None,
    label: str | None = None,
    title: str | None = None,
    outfile: str | None = None,
):
    """Plot one response auto-spectrum (log-magnitude over frequency).

    ``psd`` may be a :class:`~femtools.dynamics.random.PSDResult` (real
    array ``(n_out, n_freq)`` plus its frequency vector) or a raw array
    combined with ``freq=``; ``output`` selects the response row.
    matplotlib only -- a PSD is a real, positive spectrum, so a Bode
    pair adds nothing.  Returns the matplotlib Figure.
    """
    plt = _plt()
    values = np.atleast_2d(np.asarray(getattr(psd, "psd", psd), dtype=float))
    f = getattr(psd, "freq_hz", None) if freq is None else freq
    if f is None:
        raise ValueError("plot_psd needs a frequency vector: a PSDResult, or freq=")
    f = np.asarray(f, dtype=float).reshape(-1)
    if not 0 <= int(output) < values.shape[0]:
        raise ValueError(
            f"output {output} is out of range for a block of {values.shape[0]} spectra")
    curve = values[int(output)].reshape(-1)
    if curve.shape[0] != f.shape[0]:
        raise ValueError(f"PSD length {curve.shape[0]} != frequency length {f.shape[0]}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.semilogy(f, np.where(curve > 0, curve, np.nan),
                label=label or f"S[{int(output)}]")
    ax.set_xlabel("frequency [Hz]")
    unit = str(getattr(psd, "response", "") or "")
    ax.set_ylabel({"receptance": "PSD [disp$^2$/Hz]",
                   "mobility": "PSD [vel$^2$/Hz]",
                   "accelerance": "PSD [accel$^2$/Hz]"}.get(unit, "PSD [unit$^2$/Hz]"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title or "Response power spectral density")
    return _finish(fig, outfile)


def _mode_displacements(
    phi: np.ndarray,
    coords: dict[int, np.ndarray],
    dof_map: Any,
) -> dict[int, np.ndarray]:
    """Map a mode-shape vector to per-node DOF vectors (6,): ux..uz, rx..rz.

    Supports dof_map as ``{(node_id, local_dof): global_index}`` (local
    DOF 0- or 1-based) or ``{node_id: sequence of global indices}``.
    Without a dof_map, an equal number of DOFs per node is assumed, in
    ascending node-id order, translations first, then rotations.
    """
    node_ids = sorted(coords)
    disp = {nid: np.zeros(6) for nid in node_ids}

    if dof_map:
        try:
            items = list(dof_map.items())
        except AttributeError:
            items = []
        if items and isinstance(items[0][0], tuple):
            base = min(k[1] for k, _ in items)  # 0- or 1-based local DOF
            for (nid, ldof), gidx in items:
                comp = int(ldof) - base
                if nid in disp and 0 <= comp < 6 and 0 <= int(gidx) < phi.shape[0]:
                    disp[nid][comp] = phi[int(gidx)].real
            return disp
        if items:
            for nid, gidxs in items:
                if nid not in disp:
                    continue
                if isinstance(gidxs, (int, np.integer)):
                    gidxs = [gidxs]
                for comp, gidx in enumerate(list(gidxs)[:6]):
                    if 0 <= int(gidx) < phi.shape[0]:
                        disp[nid][comp] = phi[int(gidx)].real
            return disp

    # fallback: uniform DOFs per node, ascending node id, translations first
    n_nodes = len(node_ids)
    ndof_per_node = max(1, phi.shape[0] // max(n_nodes, 1))
    for i, nid in enumerate(node_ids):
        for comp in range(min(6, ndof_per_node)):
            gidx = i * ndof_per_node + comp
            if gidx < phi.shape[0]:
                disp[nid][comp] = phi[gidx].real
    return disp


def _draw_rotations(ax, points: list, rotations: list, *, three_d: bool,
                    length: float, color: str) -> None:
    """Draw nodal rotation pseudo-vectors (right-hand rule, rx/ry/rz).

    3D axes get quiver arrows of the rotation vector.  2D (XY) axes get
    quiver arrows for the in-plane part plus signed circle markers for
    the out-of-plane component: radius ~ |theta_z|, solid = CCW (+z),
    dashed = CW (-z).
    """
    P = np.asarray(points, dtype=float)
    R = np.asarray(rotations, dtype=float)
    if P.size == 0 or R.size == 0:
        return
    rmax = float(np.max(np.linalg.norm(R, axis=1)))
    if rmax <= 0.0:
        return
    V = R * (length / rmax)
    if three_d:
        ax.quiver(P[:, 0], P[:, 1], P[:, 2], V[:, 0], V[:, 1], V[:, 2],
                  color=color, linewidth=1.2, alpha=0.9)
        # mplot3d quiver does not autoscale; include the arrow tips manually
        pts = np.vstack([P, P + V])
        ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2], had_data=True)
        return
    if np.max(np.abs(V[:, :2])) > 1e-12 * length:
        ax.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color=color,
                  angles="xy", scale_units="xy", scale=1.0, width=0.004, alpha=0.9)
    rz = R[:, 2] / rmax
    for sign, style, label in ((1.0, "-", r"$\theta_z>0$ (ccw)"),
                               (-1.0, "--", r"$\theta_z<0$ (cw)")):
        sel = sign * rz > 1e-12
        if np.any(sel):
            ax.scatter(P[sel, 0], P[sel, 1], s=250.0 * np.abs(rz[sel]),
                       facecolors="none", edgecolors=color, linestyles=style,
                       linewidths=1.3, label=label)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)


def _mode_geometry(
    model: Any, modal: Any, index: int, scale: float | None, dof_map: Any,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray],
           np.ndarray, float, float]:
    """Shared mode-shape geometry: ``(coords, disp6, deformed, xyz, bbox, scale)``."""
    coords = _node_coords(model)

    phi_all = getattr(modal, "modes", getattr(modal, "phi", modal))
    phi_all = np.asarray(phi_all)
    phi = phi_all[:, index] if phi_all.ndim == 2 else phi_all.reshape(-1)
    if dof_map is None:
        dof_map = getattr(modal, "dof_map", None)

    disp6 = _mode_displacements(phi, coords, dof_map)
    disp = {nid: v[:3] for nid, v in disp6.items()}

    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))
    bbox = float(np.linalg.norm(np.ptp(xyz, axis=0))) if len(xyz) else 1.0
    bbox = bbox if bbox > 0 else 1.0
    dmax = max((float(np.linalg.norm(d)) for d in disp.values()), default=0.0)
    if scale is None:
        scale = 0.1 * bbox / dmax if dmax > 0 else 1.0

    deformed = {nid: coords[nid] + scale * disp[nid] for nid in coords}
    return coords, disp6, deformed, xyz, bbox, float(scale)


def _mode_default_title(modal: Any, index: int) -> str:
    freq_hz = getattr(modal, "freq_hz", None)
    if freq_hz is not None and index < len(np.atleast_1d(freq_hz)):
        return f"mode {index + 1} — {float(np.atleast_1d(freq_hz)[index]):.4g} Hz"
    return f"mode {index + 1}"


def plot_mode(
    model: Any,
    modal: Any,
    index: int = 0,
    ax: Any = None,
    *,
    scale: float | None = None,
    dof_map: Any = None,
    show_undeformed: bool = True,
    show_rotations: bool = False,
    rotation_scale: float | None = None,
    color: str = "tab:red",
    title: str | None = None,
    outfile: str | None = None,
    backend: str | None = None,
):
    """Plot a deformed mode shape over the undeformed wireframe.

    ``modal`` is a ``ModalResult`` (``modes`` array ``(ndof, n_modes)``,
    optional ``freq_hz`` and ``dof_map``) or a raw mode matrix/vector.
    The displacement scale defaults to 10% of the model bounding-box
    diagonal at unit maximum displacement.

    With ``show_rotations=True`` the rotational DOF components (rx, ry,
    rz) are drawn at the deformed nodes: quiver arrows of the rotation
    pseudo-vector in 3D; in 2D, arrows for the in-plane part plus circle
    markers sized by ``|theta_z|`` (solid = CCW, dashed = CW).
    ``rotation_scale`` sets the length of the largest arrow in model
    units (default: 8% of the bounding-box diagonal).  Rotation glyphs
    are matplotlib-only.

    Returns the matplotlib Figure (or a plotly Figure with
    ``backend="plotly"``).
    """
    if _use_plotly(backend, ax):
        if show_rotations:
            raise ValueError("show_rotations glyphs are matplotlib-only; "
                             "drop backend='plotly' or show_rotations=True")
        return _plotly_mode(model, modal, index, scale=scale, dof_map=dof_map,
                            show_undeformed=show_undeformed, color=color,
                            title=title, outfile=outfile)
    plt = _plt()
    coords, disp6, deformed, xyz, bbox, scale = _mode_geometry(
        model, modal, index, scale, dof_map)
    all_pts = np.vstack([xyz, np.asarray(list(deformed.values()))]) if coords else xyz
    three_d = not _is_planar(all_pts)

    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d" if three_d else None)
    else:
        fig = ax.figure
        three_d = getattr(ax, "name", "") == "3d"

    if show_undeformed:
        seg0, pts0 = _segments(model, coords)
        _draw_wireframe(ax, seg0, pts0, three_d=three_d, color="0.7", lw=1.0, alpha=0.8)
    seg1, pts1 = _segments(model, deformed)
    _draw_wireframe(ax, seg1, pts1, three_d=three_d, color=color, lw=1.8)

    if show_rotations:
        node_ids = sorted(coords)
        arrow_len = rotation_scale if rotation_scale is not None else 0.08 * bbox
        _draw_rotations(
            ax,
            [deformed[nid] for nid in node_ids],
            [disp6[nid][3:6] for nid in node_ids],
            three_d=three_d,
            length=arrow_len,
            color=color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if three_d:
        ax.set_zlabel("z")
    else:
        ax.set_aspect("equal", adjustable="datalim")

    if title is None:
        title = _mode_default_title(modal, index)
    ax.set_title(f"{title}  (scale={scale:.3g})")
    return _finish(fig, outfile)


# ----------------------------------------------------------------------
# plotly backend (optional; imported only when a plotly plot is requested)
# ----------------------------------------------------------------------
# matplotlib color spellings used by the femtools defaults, mapped to hex
_PLOTLY_COLORS = {
    "tab:blue": "#1f77b4", "tab:orange": "#ff7f0e", "tab:green": "#2ca02c",
    "tab:red": "#d62728", "tab:purple": "#9467bd", "tab:brown": "#8c564b",
    "tab:pink": "#e377c2", "tab:gray": "#7f7f7f", "tab:grey": "#7f7f7f",
    "tab:olive": "#bcbd22", "tab:cyan": "#17becf",
}


def _plotly_color(color: str) -> str:
    """Translate matplotlib color spellings plotly does not know."""
    c = str(color)
    if c in _PLOTLY_COLORS:
        return _PLOTLY_COLORS[c]
    try:  # matplotlib grey-level strings like "0.7"
        level = float(c)
    except ValueError:
        return c
    v = int(round(255 * min(max(level, 0.0), 1.0)))
    return f"rgb({v},{v},{v})"


def _polyline_xyz(segments: list) -> tuple[list, list, list]:
    """Concatenate segments into None-gapped x/y/z polyline arrays."""
    xs: list = []
    ys: list = []
    zs: list = []
    for p, q in segments:
        xs += [float(p[0]), float(q[0]), None]
        ys += [float(p[1]), float(q[1]), None]
        zs += [float(p[2]), float(q[2]), None]
    return xs, ys, zs


def _plotly_wireframe(go, fig, segments, points, *, three_d: bool, color: str,
                      width: float, name: str, opacity: float = 1.0) -> None:
    """Add one wireframe (lines + point-element markers) to a plotly figure."""
    color = _plotly_color(color)
    xs, ys, zs = _polyline_xyz(segments)
    if xs:
        if three_d:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                                       line={"color": color, "width": width},
                                       opacity=opacity))
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                     line={"color": color, "width": width},
                                     opacity=opacity))
    if points:
        pts = np.asarray(points, dtype=float)
        marker = {"color": color, "symbol": "square", "size": 6}
        if three_d:
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                       mode="markers", marker=marker,
                                       name=f"{name} (point elements)",
                                       opacity=opacity))
        else:
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                     marker=marker,
                                     name=f"{name} (point elements)",
                                     opacity=opacity))


def _plotly_frame_layout(fig, *, three_d: bool, title: str) -> None:
    """Common axis labels / equal-aspect layout for mesh and mode plots."""
    if three_d:
        fig.update_layout(scene={"xaxis_title": "x", "yaxis_title": "y",
                                 "zaxis_title": "z", "aspectmode": "data"})
    else:
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title, showlegend=False)


def _plotly_mesh(
    model: Any,
    *,
    show_node_ids: bool = False,
    show_element_ids: bool = False,
    color: str = "tab:blue",
    node_color: str = "black",
    title: str | None = None,
    outfile: str | None = None,
):
    go = _plotly_go()
    coords = _node_coords(model)
    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))
    three_d = not _is_planar(xyz)

    fig = go.Figure()
    segments, points = _segments(model, coords)
    _plotly_wireframe(go, fig, segments, points, three_d=three_d, color=color,
                      width=3.0, name="mesh")

    if coords:
        node_ids = list(coords)
        mode = "markers+text" if show_node_ids else "markers"
        text = [str(nid) for nid in node_ids] if show_node_ids else None
        marker = {"color": _plotly_color(node_color), "size": 3 if three_d else 5}
        kwargs = {"mode": mode, "marker": marker, "name": "nodes",
                  "text": text, "textposition": "top center"}
        if three_d:
            fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], **kwargs))
        else:
            fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], **kwargs))

    if show_element_ids:
        cx, cy, cz, labels = [], [], [], []
        for _etype, nodes, eid in _element_edges(model):
            pts = [coords[n] for n in nodes if n in coords]
            if not pts:
                continue
            c = np.mean(pts, axis=0)
            cx.append(float(c[0]))
            cy.append(float(c[1]))
            cz.append(float(c[2]))
            labels.append(str(eid))
        if labels:
            kwargs = {"mode": "text", "text": labels, "name": "element ids",
                      "textfont": {"color": _plotly_color(color)}}
            if three_d:
                fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, **kwargs))
            else:
                fig.add_trace(go.Scatter(x=cx, y=cy, **kwargs))

    _plotly_frame_layout(fig, three_d=three_d,
                         title=title or f"{getattr(model, 'name', 'model')}: "
                                        f"{len(coords)} nodes, "
                                        f"{len(model.elements)} elements")
    return _finish_plotly(fig, outfile)


def _plotly_mac(
    mac: Any,
    *,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
    annotate: bool | None = None,
    cmap: str = "viridis",
    title: str = "MAC",
    outfile: str | None = None,
):
    go = _plotly_go()
    m = np.asarray(mac, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"MAC matrix must be 2-D, got shape {m.shape}")
    n_a, n_b = m.shape
    x = labels_b or [str(j + 1) for j in range(n_b)]
    y = labels_a or [str(i + 1) for i in range(n_a)]
    if annotate is None:
        annotate = max(n_a, n_b) <= 15

    heatmap = go.Heatmap(
        z=m, x=x, y=y, zmin=0.0, zmax=1.0, colorscale=cmap,
        colorbar={"title": "MAC"},
        text=[[f"{v:.2f}" for v in row] for row in m] if annotate else None,
        texttemplate="%{text}" if annotate else None,
    )
    fig = go.Figure(data=heatmap)
    fig.update_xaxes(title_text="set B mode", type="category")
    # match matplotlib's origin="upper": mode 1 of set A on the top row
    fig.update_yaxes(title_text="set A mode", type="category", autorange="reversed")
    fig.update_layout(title=title)
    return _finish_plotly(fig, outfile)


def _plotly_frf(
    frf: Any,
    output: int = 0,
    input_: int = 0,
    *,
    freq: Any = None,
    kind: str = "bode",
    db: bool = False,
    label: str | None = None,
    title: str | None = None,
    outfile: str | None = None,
):
    go = _plotly_go()
    from plotly.subplots import make_subplots

    f, h = _frf_curve(frf, output, input_, freq)
    want_mag, want_phase = _frf_kind(kind)
    curve_label = label or f"H[{output},{input_}]"

    n_rows = 2 if (want_mag and want_phase) else 1
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=[0.67, 0.33] if n_rows == 2 else None,
                        vertical_spacing=0.06)
    row = 1
    if want_mag:
        mag = np.abs(h)
        if db:
            with np.errstate(divide="ignore"):
                y_mag = 20.0 * np.log10(mag)
            fig.add_trace(go.Scatter(x=f, y=y_mag, name=curve_label,
                                     mode="lines"), row=row, col=1)
            fig.update_yaxes(title_text="|H| [dB]", row=row, col=1)
        else:
            fig.add_trace(go.Scatter(x=f, y=np.where(mag > 0, mag, np.nan),
                                     name=curve_label, mode="lines"), row=row, col=1)
            fig.update_yaxes(title_text="|H|", type="log", row=row, col=1)
        row += 1
    if want_phase:
        fig.add_trace(go.Scatter(x=f, y=np.degrees(np.unwrap(np.angle(h))),
                                 name=f"{curve_label} phase", mode="lines"),
                      row=row, col=1)
        fig.update_yaxes(title_text="phase [deg]", row=row, col=1)
    fig.update_xaxes(title_text="frequency [Hz]", row=n_rows, col=1)
    fig.update_layout(title=title or "Frequency response function")
    return _finish_plotly(fig, outfile)


def _plotly_mode(
    model: Any,
    modal: Any,
    index: int = 0,
    *,
    scale: float | None = None,
    dof_map: Any = None,
    show_undeformed: bool = True,
    color: str = "tab:red",
    title: str | None = None,
    outfile: str | None = None,
):
    go = _plotly_go()
    coords, _disp6, deformed, xyz, _bbox, scale = _mode_geometry(
        model, modal, index, scale, dof_map)
    all_pts = np.vstack([xyz, np.asarray(list(deformed.values()))]) if coords else xyz
    three_d = not _is_planar(all_pts)

    fig = go.Figure()
    if show_undeformed:
        seg0, pts0 = _segments(model, coords)
        _plotly_wireframe(go, fig, seg0, pts0, three_d=three_d, color="0.7",
                          width=2.0, name="undeformed", opacity=0.8)
    seg1, pts1 = _segments(model, deformed)
    _plotly_wireframe(go, fig, seg1, pts1, three_d=three_d, color=color,
                      width=3.5, name="deformed")

    if title is None:
        title = _mode_default_title(modal, index)
    _plotly_frame_layout(fig, three_d=three_d, title=f"{title}  (scale={scale:.3g})")
    return _finish_plotly(fig, outfile)

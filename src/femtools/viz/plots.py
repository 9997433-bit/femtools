"""Matplotlib plotting for femtools models and results.

The functions here are deliberately duck-typed so they work with the
contract objects (``FEModel``, ``ModalResult``, ``FRFResult``) without a
hard import dependency on the sibling packages.  Everything runs
headless: when no display is available the Agg backend is selected
before pyplot is imported.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = ["plot_mesh", "plot_mac", "plot_frf", "plot_mode"]

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
):
    """Plot the undeformed wireframe of an FE model.

    Planar (XY) models are drawn in 2D, everything else in 3D.
    Returns the matplotlib Figure.
    """
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
):
    """Plot a MAC (or any correlation) matrix as an annotated heatmap.

    ``mac[i, j]`` is drawn at row *i* (set A), column *j* (set B).
    Values are annotated when the matrix is 15x15 or smaller (override
    with ``annotate=``). Returns the matplotlib Figure.
    """
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
):
    """Plot one FRF curve (magnitude, or magnitude + phase Bode pair).

    ``frf`` may be an ``FRFResult`` (complex array ``(n_out, n_in,
    n_freq)`` plus a frequency vector) or a raw complex array combined
    with ``freq=``.  ``kind`` is ``"bode"`` (default), ``"mag"`` or
    ``"phase"``.  Set ``db=True`` for a dB magnitude axis.  Returns the
    matplotlib Figure.
    """
    plt = _plt()
    f, H = _extract_frf(frf, freq)
    if H.ndim == 3:
        h = H[output, input, :]
    elif H.ndim == 2:  # (n_curves, n_freq) — take requested output row
        h = H[output, :]
    else:
        h = H
    h = np.asarray(h).reshape(-1)
    if h.shape[0] != f.shape[0]:
        raise ValueError(f"FRF length {h.shape[0]} != frequency length {f.shape[0]}")

    want_mag = kind in ("bode", "mag", "magnitude")
    want_phase = kind in ("bode", "phase")
    if not (want_mag or want_phase):
        raise ValueError(f"kind must be 'bode', 'mag' or 'phase', got {kind!r}")

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


def _mode_displacements(
    phi: np.ndarray,
    coords: dict[int, np.ndarray],
    dof_map: Any,
) -> dict[int, np.ndarray]:
    """Map a mode-shape vector to per-node translation vectors (3,).

    Supports dof_map as ``{(node_id, local_dof): global_index}`` (local
    DOF 0- or 1-based) or ``{node_id: sequence of global indices}``.
    Without a dof_map, an equal number of DOFs per node is assumed, in
    ascending node-id order, translations first.
    """
    node_ids = sorted(coords)
    disp = {nid: np.zeros(3) for nid in node_ids}

    if dof_map:
        try:
            items = list(dof_map.items())
        except AttributeError:
            items = []
        if items and isinstance(items[0][0], tuple):
            base = min(k[1] for k, _ in items)  # 0- or 1-based local DOF
            for (nid, ldof), gidx in items:
                comp = int(ldof) - base
                if nid in disp and 0 <= comp < 3 and 0 <= int(gidx) < phi.shape[0]:
                    disp[nid][comp] = phi[int(gidx)].real
            return disp
        if items:
            for nid, gidxs in items:
                if nid not in disp:
                    continue
                if isinstance(gidxs, (int, np.integer)):
                    gidxs = [gidxs]
                for comp, gidx in enumerate(list(gidxs)[:3]):
                    if 0 <= int(gidx) < phi.shape[0]:
                        disp[nid][comp] = phi[int(gidx)].real
            return disp

    # fallback: uniform DOFs per node, ascending node id, translations first
    n_nodes = len(node_ids)
    ndof_per_node = max(1, phi.shape[0] // max(n_nodes, 1))
    for i, nid in enumerate(node_ids):
        for comp in range(min(3, ndof_per_node)):
            gidx = i * ndof_per_node + comp
            if gidx < phi.shape[0]:
                disp[nid][comp] = phi[gidx].real
    return disp


def plot_mode(
    model: Any,
    modal: Any,
    index: int = 0,
    ax: Any = None,
    *,
    scale: float | None = None,
    dof_map: Any = None,
    show_undeformed: bool = True,
    color: str = "tab:red",
    title: str | None = None,
    outfile: str | None = None,
):
    """Plot a deformed mode shape over the undeformed wireframe.

    ``modal`` is a ``ModalResult`` (``modes`` array ``(ndof, n_modes)``,
    optional ``freq_hz`` and ``dof_map``) or a raw mode matrix/vector.
    The displacement scale defaults to 10% of the model bounding-box
    diagonal at unit maximum displacement.  Returns the matplotlib Figure.
    """
    plt = _plt()
    coords = _node_coords(model)

    phi_all = getattr(modal, "modes", getattr(modal, "phi", modal))
    phi_all = np.asarray(phi_all)
    phi = phi_all[:, index] if phi_all.ndim == 2 else phi_all.reshape(-1)
    if dof_map is None:
        dof_map = getattr(modal, "dof_map", None)

    disp = _mode_displacements(phi, coords, dof_map)

    xyz = np.asarray(list(coords.values())) if coords else np.zeros((0, 3))
    dmax = max((float(np.linalg.norm(d)) for d in disp.values()), default=0.0)
    if scale is None:
        bbox = float(np.linalg.norm(np.ptp(xyz, axis=0))) if len(xyz) else 1.0
        scale = 0.1 * (bbox if bbox > 0 else 1.0) / dmax if dmax > 0 else 1.0

    deformed = {nid: coords[nid] + scale * disp[nid] for nid in coords}
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

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if three_d:
        ax.set_zlabel("z")
    else:
        ax.set_aspect("equal", adjustable="datalim")

    if title is None:
        freq_hz = getattr(modal, "freq_hz", None)
        if freq_hz is not None and index < len(np.atleast_1d(freq_hz)):
            title = f"mode {index + 1} — {float(np.atleast_1d(freq_hz)[index]):.4g} Hz"
        else:
            title = f"mode {index + 1}"
    ax.set_title(f"{title}  (scale={scale:.3g})")
    return _finish(fig, outfile)

"""femtools command-line interface.

Typer application exposed as the ``femtools`` console script
(``femtools.cli:app``).  Subcommands: ``solve-modes``, ``read-mesh``,
``write-mesh``, ``recover-stress``, ``plot-stress``, ``mac``,
``report-mac``, ``frf``, ``reduce``, ``estimate-frf``, ``update``,
``pretest``, ``script``, ``gui``.

Sibling packages (``femtools.core``, ``femtools.fea``, ...) are imported
lazily inside each command; if one is missing the command prints a clear
message and exits with code 3 instead of crashing with a traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

import femtools

app = typer.Typer(
    name="femtools",
    help="Solver-independent structural dynamics: FEA, correlation, updating, pretest.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

MISSING_MODULE_EXIT = 3


def _missing(feature: str, exc: ImportError) -> typer.Exit:
    err_console.print(
        f"[red]error:[/red] {feature} requires a femtools module that is not "
        f"available in this installation: [bold]{exc.name or exc}[/bold]\n"
        "Install/build the full femtools package and retry."
    )
    return typer.Exit(code=MISSING_MODULE_EXIT)


def _load_model(path: Path) -> Any:
    """Load a model file as an FEModel (dispatch on suffix).

    Accepts ``.ftproj`` / ``.json`` / ``.unv`` / ``.bdf|.nas|.dat`` /
    ``.inp`` / ``.k|.key``.  Container formats (``.ftproj`` projects,
    ``.unv`` bundles) are unwrapped to the bare model so every
    downstream solver call receives an ``FEModel``.
    """
    from femtools.script.loading import load_model_file

    try:
        return load_model_file(path).model
    except ImportError as exc:
        raise _missing(f"reading {path.suffix!r} files", exc) from exc
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _solve(model: Any, n_modes: int, shift: float) -> Any:
    try:
        from femtools.fea.eigen import solve_modes
    except ImportError as exc:
        raise _missing("modal analysis", exc) from exc
    return solve_modes(model, n_modes=n_modes, shift=shift)


def _freq_table(modal: Any, title: str) -> Table:
    import numpy as np

    table = Table(title=title)
    table.add_column("mode", justify="right")
    table.add_column("frequency [Hz]", justify="right")
    freqs = np.atleast_1d(getattr(modal, "freq_hz", modal))
    for i, f in enumerate(freqs, start=1):
        table.add_row(str(i), f"{float(f):.6g}")
    return table


def _load_modes(path: Path, n_modes: int, shift: float) -> tuple[Any, Any]:
    """Return (phi, freq_hz|None) from an .npz mode file or a model file."""
    import numpy as np

    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        for key in ("modes", "phi", "shapes"):
            if key in data:
                return data[key], data.get("freq_hz")
        err_console.print(
            f"[red]error:[/red] {path} has no 'modes'/'phi' array "
            f"(found: {sorted(data.keys())})"
        )
        raise typer.Exit(code=2)
    modal = _solve(_load_model(path), n_modes, shift)
    return modal.modes, getattr(modal, "freq_hz", None)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"femtools {femtools.__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", "-V", callback=_version_callback,
                                          is_eager=True,
                                          help="Print version and exit.")] = False,
) -> None:
    pass


# ----------------------------------------------------------------------
# solve-modes
# ----------------------------------------------------------------------
@app.command("solve-modes")
def solve_modes_cmd(
    model_file: Annotated[Path, typer.Argument(exists=True, readable=True,
                                               help="Model file (.ftproj, .json, .unv, "
                                                    ".bdf).")],
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Number of modes.")] = 10,
    shift: Annotated[float, typer.Option("--shift",
                                         help="Eigen shift (rad^2/s^2).")] = 0.0,
    output: Annotated[Path | None, typer.Option("--output", "-o",
                                                help="Save result to .npz.")] = None,
    plot: Annotated[Path | None, typer.Option("--plot",
                                              help="Save a mode-shape plot (PNG).")] = None,
    mode_index: Annotated[int, typer.Option("--mode-index", min=0,
                                            help="Mode to plot (0-based).")] = 0,
) -> None:
    """Solve the undamped eigenproblem and print natural frequencies."""
    import numpy as np

    model = _load_model(model_file)
    modal = _solve(model, n_modes, shift)
    console.print(_freq_table(modal, f"modes of {model_file.name}"))

    if output is not None:
        np.savez(
            str(output),
            freq_hz=np.asarray(modal.freq_hz),
            eigenvalues=np.asarray(getattr(modal, "eigenvalues", [])),
            modes=np.asarray(modal.modes),
            generalized_mass=np.asarray(getattr(modal, "generalized_mass", [])),
        )
        console.print(f"saved modal result to [bold]{output}[/bold]")
    if plot is not None:
        from femtools.viz import plot_mode

        n_solved = int(np.asarray(modal.modes).shape[1])
        if mode_index >= n_solved:
            err_console.print(
                f"[red]error:[/red] --mode-index {mode_index} is out of range: "
                f"only {n_solved} modes were solved (0..{n_solved - 1})")
            raise typer.Exit(code=2)
        plot_mode(model, modal, index=mode_index, outfile=str(plot))
        console.print(f"saved mode plot to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# read-mesh
# ----------------------------------------------------------------------
@app.command("read-mesh")
def read_mesh_cmd(
    mesh_file: Annotated[Path, typer.Argument(
        exists=True, readable=True,
        help="Mesh/model file: .inp (Abaqus), .k/.key (LS-DYNA), .unv/.uff, "
             ".bdf/.nas/.dat (Nastran), .ftproj, .json.")],
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save the loaded model as a .ftproj project.")] = None,
    plot: Annotated[Path | None, typer.Option(
        "--plot", help="Save a wireframe plot of the mesh (PNG).")] = None,
) -> None:
    """Load a mesh/model file by suffix and print a model summary."""
    from collections import Counter

    from femtools.script.loading import load_model_file

    try:
        loaded = load_model_file(mesh_file)
    except ImportError as exc:
        raise _missing(f"reading {mesh_file.suffix!r} files", exc) from exc
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    model = loaded.model

    table = Table(title=f"{mesh_file.name} ({loaded.format})")
    table.add_column("entity", justify="left")
    table.add_column("count", justify="right")
    table.add_row("nodes", str(len(getattr(model, "nodes", {}))))
    elements = getattr(model, "elements", {})
    table.add_row("elements", str(len(elements)))
    type_counts = Counter(str(getattr(e, "type", "?")).upper() for e in elements.values())
    for etype, n in sorted(type_counts.items()):
        table.add_row(f"  {etype}", str(n))
    table.add_row("materials", str(len(getattr(model, "materials", {}))))
    table.add_row("properties", str(len(getattr(model, "properties", {}))))
    table.add_row("SPCs", str(len(getattr(model, "spcs", []))))
    console.print(table)
    console.print(f"model name: [bold]{getattr(model, 'name', '?')}[/bold]")
    if loaded.results:
        console.print(f"stored results: {', '.join(sorted(loaded.results))}")

    if output is not None:
        try:
            from femtools.io.project import save_project
        except ImportError as exc:
            raise _missing("saving projects", exc) from exc
        save_project(model, str(output))
        console.print(f"saved model to [bold]{output}[/bold]")
    if plot is not None:
        from femtools.viz import plot_mesh

        plot_mesh(model, outfile=str(plot))
        console.print(f"saved mesh plot to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# write-mesh
# ----------------------------------------------------------------------
_WRITE_SUFFIX_HELP = ".cdb (ANSYS), .k/.key (LS-DYNA), .inp (Abaqus), " \
                     ".bdf/.nas/.dat (Nastran), .unv/.uff, .ftproj"


def _mesh_writer(suffix: str) -> Any:
    """Import the mesh writer for an output suffix (lazy, may raise ImportError)."""
    if suffix == ".cdb":
        from femtools.io.cdb import write_cdb
        return write_cdb
    if suffix in (".k", ".key"):
        from femtools.io.kfile import write_k
        return write_k
    if suffix == ".inp":
        from femtools.io.inp import write_inp
        return write_inp
    if suffix in (".bdf", ".nas", ".dat"):
        from femtools.io.bdf import write_bdf
        return write_bdf
    if suffix in (".unv", ".uff"):
        from femtools.io.unv import write_unv
        return write_unv
    err_console.print(
        f"[red]error:[/red] unsupported output suffix {suffix!r} "
        f"(expected one of: {_WRITE_SUFFIX_HELP})")
    raise typer.Exit(code=2)


@app.command("write-mesh")
def write_mesh_cmd(
    model_file: Annotated[Path, typer.Argument(
        exists=True, readable=True,
        help="Input model file (any suffix read-mesh accepts).")],
    output: Annotated[Path, typer.Argument(
        help=f"Output deck; the format is chosen by its suffix: {_WRITE_SUFFIX_HELP}.")],
) -> None:
    """Export a model as a solver deck chosen by the output file suffix.

    The writer kernels are imported lazily: an installation shipping
    without one exits with code 3 and a clear message, exactly like
    ``read-mesh`` for ``.inp``/``.k`` inputs.
    """
    model = _load_model(model_file)
    suffix = output.suffix.lower()

    if suffix == ".ftproj":
        try:
            from femtools.io.project import save_project
        except ImportError as exc:
            raise _missing("saving projects", exc) from exc
        save_project(model, str(output))
    else:
        try:
            writer = _mesh_writer(suffix)
        except ImportError as exc:
            raise _missing(f"writing {suffix!r} files", exc) from exc
        try:
            # the io writers accept (path, model) or (model, path); try the
            # canonical order first and fall back for kernels that only take
            # the other one (a str duck-typed as a model surfaces either as
            # TypeError or as AttributeError)
            try:
                writer(str(output), model)
            except (TypeError, AttributeError):
                writer(model, str(output))
        except (ValueError, OSError) as exc:
            err_console.print(f"[red]error:[/red] cannot write {output}: {exc}")
            raise typer.Exit(code=2) from exc

    console.print(
        f"wrote [bold]{getattr(model, 'name', '?')}[/bold] "
        f"({len(getattr(model, 'nodes', {}))} nodes, "
        f"{len(getattr(model, 'elements', {}))} elements) "
        f"to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# recover-stress
# ----------------------------------------------------------------------
def _model_load_mapping(model: Any) -> dict[tuple[int, int], float] | None:
    """Flatten ``model.loads`` records into ``{(node_id, dof): value}``.

    The static solver reads that mapping directly, whereas the shape of a
    stored core ``Load`` record (``force``/``moment`` vectors) is not among
    the record layouts its load builder probes for.  Returns ``None`` when
    the model carries no loads (or record types the kernel should
    interpret itself).
    """
    records = getattr(model, "loads", None)
    if not records:
        return None
    mapping: dict[tuple[int, int], float] = {}
    try:
        for record in records:
            for dof, value in record.as_dof_values():
                key = (record.node_id, dof)
                mapping[key] = mapping.get(key, 0.0) + value
    except (AttributeError, TypeError):
        return None
    return mapping or None


def _parse_point_load(spec: str, option: str = "--load") -> tuple[int, int, float]:
    """Parse a 'NODE:DOF=VALUE' spec (DOF 1-6) of ``--load`` / ``--measure``."""
    head, sep, value_str = spec.partition("=")
    try:
        node_str, dof_str = head.split(":")
        node, dof, value = int(node_str), int(dof_str), float(value_str)
    except ValueError:
        node = dof = 0
        value = 0.0
        sep = ""
    if not sep or not 1 <= dof <= 6:
        err_console.print(
            f"[red]error:[/red] bad {option} spec {spec!r} "
            "(expected NODE:DOF=VALUE with DOF 1-6, e.g. 2:3=-1000)")
        raise typer.Exit(code=2)
    return node, dof, value


def _stack_components(values: Any) -> Any:
    """Stack per-element component vectors, NaN-padding ragged lengths.

    Mixed element types can recover different component counts (a bar
    has one axial stress, a solid six); padding keeps one rectangular
    array for the table and the .npz payload.
    """
    import numpy as np

    rows = [np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1) for v in values]
    width = max((r.size for r in rows), default=0)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, : r.size] = r
    return out


def _stress_parts(result: Any) -> tuple[list[Any] | None, Any, Any]:
    """Duck-typed unpack of a stress result into ``(ids, components, von_mises)``.

    Accepts a ``StressResult``-like object (element ids plus a component
    array and optionally a von Mises vector) or a plain mapping
    ``{element_id: components}``.  Missing pieces come back as ``None``.
    """
    import numpy as np

    if isinstance(result, dict):
        return list(result.keys()), _stack_components(result.values()), None

    ids = None
    for attr in ("element_ids", "eids", "elements", "ids"):
        ids = getattr(result, attr, None)
        if ids is not None:
            break
    comps = None
    for attr in ("stress", "sigma", "values", "data"):
        comps = getattr(result, attr, None)
        if comps is not None:
            break
    vm = None
    for attr in ("von_mises", "vm", "mises"):
        vm = getattr(result, attr, None)
        if vm is not None:
            break

    if isinstance(comps, dict):
        ids = list(comps.keys())
        comps = _stack_components(comps.values())
    elif comps is not None:
        try:
            arr = np.asarray(comps, dtype=float)
        except ValueError:  # ragged per-element vectors
            arr = None
        comps = arr if arr is not None and arr.ndim == 2 else _stack_components(comps)
    if ids is None and comps is not None:
        ids = list(range(1, comps.shape[0] + 1))
    return (list(np.atleast_1d(np.asarray(ids)).tolist()) if ids is not None else None,
            comps, None if vm is None else np.asarray(vm, dtype=float).reshape(-1))


def _apply_load_specs(model: Any, load: list[str] | None) -> None:
    """Apply repeatable ``--load`` 'NODE:DOF=VALUE' specs to a model."""
    for spec in load or []:
        node, dof, value = _parse_point_load(spec)
        vec = [0.0, 0.0, 0.0]
        vec[(dof - 1) % 3] = value
        try:
            model.add_load(node_id=node,
                           force=vec if dof <= 3 else None,
                           moment=vec if dof > 3 else None)
        except ValueError as exc:
            err_console.print(f"[red]error:[/red] bad --load {spec!r}: {exc}")
            raise typer.Exit(code=2) from exc


def _recover_pipeline(model_file: Path, load: list[str] | None) -> tuple[Any, Any]:
    """Shared solve+recover pipeline of ``recover-stress`` / ``plot-stress``.

    Lazily imports the stress-recovery kernel and the static solver (an
    installation shipping without either exits with code 3), applies the
    ``--load`` specs, runs the static solve and returns ``(model,
    stress_result)``.
    """
    try:
        from femtools.fea.recover import recover_stress
    except ImportError as exc:
        raise _missing("stress recovery", exc) from exc
    try:
        from femtools.fea.static import solve_static
    except ImportError as exc:
        raise _missing("the static solver", exc) from exc

    model = _load_model(model_file)
    _apply_load_specs(model, load)
    if not getattr(model, "loads", None):
        err_console.print(
            "[yellow]warning:[/yellow] the model carries no loads and none were "
            "given with --load; the recovered field reflects enforced "
            "displacements only (zero for a plain constrained model)")

    try:
        static = solve_static(model, _model_load_mapping(model), full_result=True)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        # ValueError covers bad load specs, RuntimeError the singular
        # factorization of an under-constrained model
        err_console.print(f"[red]error:[/red] static solve failed: {exc}")
        raise typer.Exit(code=2) from exc

    import inspect

    kwargs: dict[str, Any] = {}
    try:
        if "assembly" in inspect.signature(recover_stress).parameters:
            kwargs["assembly"] = static.assembly
    except (TypeError, ValueError):
        pass
    try:
        result = recover_stress(model, static.u, **kwargs)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] stress recovery failed: {exc}")
        raise typer.Exit(code=2) from exc
    return model, result


@app.command("recover-stress")
def recover_stress_cmd(
    model_file: Annotated[Path, typer.Argument(
        exists=True, readable=True, help="Model file.")],
    load: Annotated[list[str] | None, typer.Option(
        "--load", "-l",
        help="Add a nodal load 'NODE:DOF=VALUE' (DOF 1-6: fx fy fz mx my mz); "
             "repeatable.  Loads stored in the model file are kept.")] = None,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o",
        help="Save element_ids, stress (and von_mises) to .npz.")] = None,
    max_rows: Annotated[int, typer.Option(
        "--max-rows", min=1, help="Table rows to print.")] = 20,
) -> None:
    """Solve a linear static case and recover element centroid stresses.

    Requires the stress-recovery kernel (``femtools.fea.recover``); an
    installation shipping without it exits with code 3, like every
    other lazily bound subcommand.
    """
    import numpy as np

    model, result = _recover_pipeline(model_file, load)

    ids, comps, vm = _stress_parts(result)
    if ids is None or comps is None:
        err_console.print("[red]error:[/red] the stress-recovery kernel returned "
                          f"no element stresses ({type(result).__name__})")
        raise typer.Exit(code=2)

    n_comp = int(comps.shape[1]) if comps.ndim == 2 else 1
    table = Table(title=f"element centroid stress ({model_file.name})")
    table.add_column("element", justify="right")
    comp_labels = (["sxx", "syy", "szz", "sxy", "syz", "szx"] if n_comp == 6
                   else [f"s{k + 1}" for k in range(n_comp)])
    for label in comp_labels:
        table.add_column(label, justify="right")
    if vm is not None:
        table.add_column("von Mises", justify="right")
    for row_i, eid in enumerate(ids[:max_rows]):
        row = [str(eid)] + ["-" if np.isnan(c) else f"{float(c):.5g}"
                            for c in np.atleast_1d(comps[row_i])]
        if vm is not None and row_i < vm.size:
            row.append(f"{float(vm[row_i]):.5g}")
        table.add_row(*row)
    console.print(table)
    if len(ids) > max_rows:
        console.print(f"... {len(ids) - max_rows} more elements "
                      "(raise --max-rows or save with --output)")

    if output is not None:
        payload: dict[str, Any] = {
            "element_ids": np.asarray(ids),
            "stress": np.asarray(comps, dtype=float),
        }
        if vm is not None:
            payload["von_mises"] = vm
        np.savez(str(output), **payload)
        console.print(f"saved stress result to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# plot-stress
# ----------------------------------------------------------------------
@app.command("plot-stress")
def plot_stress_cmd(
    model_file: Annotated[Path, typer.Argument(
        exists=True, readable=True, help="Model file.")],
    load: Annotated[list[str] | None, typer.Option(
        "--load", "-l",
        help="Add a nodal load 'NODE:DOF=VALUE' (DOF 1-6: fx fy fz mx my mz); "
             "repeatable.  Loads stored in the model file are kept.")] = None,
    component: Annotated[str, typer.Option(
        "--component", "-c",
        help="Fringe value: 'von_mises' or a Voigt component "
             "(xx yy zz xy yz zx, also spelled sxx ...).")] = "von_mises",
    output: Annotated[Path, typer.Option(
        "--output", "-o", help="Plot file (PNG).")] = Path("stress.png"),
    cmap: Annotated[str, typer.Option(
        "--cmap", help="matplotlib colormap name.")] = "viridis",
) -> None:
    """Solve a static case, recover stresses and save a colored mesh plot.

    Colors the mesh by von Mises stress (or a named component) with
    :func:`femtools.viz.plot_stress`.  Requires the same kernels as
    ``recover-stress`` (``femtools.fea.recover`` and the static solver);
    an installation shipping without them exits with code 3.
    """
    import numpy as np

    model, result = _recover_pipeline(model_file, load)

    from femtools.viz.plots import plot_stress

    try:
        plot_stress(model, result, component=component, cmap=cmap,
                    outfile=str(output))
    except ValueError as exc:
        # unknown component name, or a result with nothing to color by
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    n_recovered = len(getattr(result, "element_ids", []) or [])
    vm = getattr(result, "von_mises", None)
    if vm is not None and np.size(vm):
        console.print(f"recovered {n_recovered} elements, "
                      f"max von Mises {float(np.max(vm)):.6g}")
    else:
        console.print(f"recovered {n_recovered} elements")
    console.print(f"saved stress plot to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# mac
# ----------------------------------------------------------------------
@app.command("mac")
def mac_cmd(
    file_a: Annotated[Path, typer.Argument(exists=True,
                                           help="Mode set A: .npz with 'modes', or a "
                                                "model file.")],
    file_b: Annotated[Path | None, typer.Argument(
        help="Mode set B (defaults to A: self-MAC).")] = None,
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Modes to solve when input is a "
                                              "model file.")] = 10,
    shift: Annotated[float, typer.Option("--shift")] = 0.0,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save the MAC matrix (.npz or .csv).")] = None,
    plot: Annotated[Path | None, typer.Option(
        "--plot", help="Save a MAC heatmap (PNG).")] = None,
) -> None:
    """Compute the Modal Assurance Criterion matrix between two mode sets."""
    import numpy as np

    phi_a, _ = _load_modes(file_a, n_modes, shift)
    phi_b = phi_a if file_b is None else _load_modes(file_b, n_modes, shift)[0]

    try:
        from femtools.correlation.mac import mac_matrix
    except ImportError as exc:
        raise _missing("MAC computation", exc) from exc
    mac = np.asarray(mac_matrix(phi_a, phi_b))

    table = Table(title=f"MAC {file_a.name} vs {(file_b or file_a).name}")
    table.add_column("A\\B", justify="right")
    for j in range(mac.shape[1]):
        table.add_column(str(j + 1), justify="right")
    for i in range(mac.shape[0]):
        table.add_row(str(i + 1), *(f"{mac[i, j]:.3f}" for j in range(mac.shape[1])))
    console.print(table)
    if mac.size and mac.shape[0] == mac.shape[1]:
        diag = np.diag(mac)
        off_diag = float(np.max(mac - np.diag(diag))) if mac.size > 1 else 0.0
        console.print(f"diag min={float(np.min(diag)):.6f}  "
                      f"off-diag max={off_diag:.6f}")
    elif mac.size:
        console.print(f"best-match min={float(np.min(np.max(mac, axis=1))):.6f}  "
                      f"overall max={float(np.max(mac)):.6f}")

    if output is not None:
        if output.suffix.lower() == ".csv":
            np.savetxt(str(output), mac, delimiter=",", fmt="%.8g")
        else:
            np.savez(str(output), mac=mac)
        console.print(f"saved MAC to [bold]{output}[/bold]")
    if plot is not None:
        from femtools.viz import plot_mac

        plot_mac(mac, outfile=str(plot))
        console.print(f"saved MAC heatmap to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# report-mac
# ----------------------------------------------------------------------
@app.command("report-mac")
def report_mac_cmd(
    file_a: Annotated[Path, typer.Argument(exists=True,
                                           help="Mode set A: .npz with 'modes', or a "
                                                "model file.")],
    file_b: Annotated[Path | None, typer.Argument(
        help="Mode set B (defaults to A: self-MAC report).")] = None,
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Modes to solve when input is a "
                                              "model file.")] = 10,
    shift: Annotated[float, typer.Option("--shift")] = 0.0,
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Report file (.html/.htm = HTML, anything else = text).")] = Path(
            "mac-report.html"),
    fmt: Annotated[str, typer.Option(
        "--format", "-f", help="'auto' (by suffix), 'html' or 'text'.")] = "auto",
    pair: Annotated[bool, typer.Option(
        "--pair/--no-pair",
        help="Pair the two mode sets by MAC (two-file reports only).")] = True,
    mac_threshold: Annotated[float, typer.Option(
        "--mac-threshold", min=0.0, max=1.0,
        help="Reject pairs below this MAC.")] = 0.5,
    heatmap: Annotated[bool, typer.Option(
        "--heatmap/--no-heatmap",
        help="Embed a heatmap image in HTML reports.")] = True,
    title: Annotated[str | None, typer.Option(
        "--title", help="Report title.")] = None,
) -> None:
    """Write an HTML/text MAC correlation report for one or two mode sets."""
    import numpy as np

    if fmt.lower() not in ("auto", "html", "text", "txt"):
        raise typer.BadParameter(f"unknown format {fmt!r}; use 'auto', 'html' or 'text'",
                                 param_hint="'--format' / '-f'")

    phi_a, freq_a = _load_modes(file_a, n_modes, shift)
    if file_b is None:
        phi_b, freq_b = phi_a, freq_a
    else:
        phi_b, freq_b = _load_modes(file_b, n_modes, shift)

    try:
        from femtools.correlation.mac import mac_matrix
    except ImportError as exc:
        raise _missing("MAC computation", exc) from exc
    mac = np.asarray(mac_matrix(phi_a, None if file_b is None else phi_b))

    pairs = None
    if pair and file_b is not None:
        try:
            from femtools.correlation.pairing import pair_modes
        except ImportError:
            err_console.print(
                "[yellow]warning:[/yellow] mode pairing is unavailable in this "
                "installation; the report will not include a pair table"
            )
        else:
            pairs = pair_modes(phi_a, phi_b, freq_a, freq_b,
                               mac_threshold=mac_threshold, mac=mac)

    try:
        from femtools.viz.report import save_mac_report
    except ImportError as exc:
        raise _missing("MAC report generation", exc) from exc

    name_b = file_b.name if file_b is not None else f"{file_a.name} (self)"
    saved = save_mac_report(
        output, mac, fmt=fmt,
        freq_a=freq_a, freq_b=freq_b, pairs=pairs,
        title=title or f"MAC report: {file_a.name} vs {name_b}",
        name_a=file_a.name, name_b=name_b, heatmap=heatmap,
    )

    if mac.shape[0] == mac.shape[1] and mac.size:
        diag = np.diag(mac)
        off = float(np.max(mac - np.diag(diag))) if mac.size > 1 else 0.0
        console.print(f"MAC {mac.shape[0]}x{mac.shape[1]}: "
                      f"diag min={float(np.min(diag)):.4f}  off-diag max={off:.4f}")
    else:
        console.print(f"MAC {mac.shape[0]}x{mac.shape[1]}: "
                      f"max={float(np.max(mac)):.4f}" if mac.size else "MAC empty")
    if pairs is not None:
        n_paired = len(getattr(pairs, "pairs", pairs))
        console.print(f"paired {n_paired} of {min(mac.shape)} modes "
                      f"(MAC threshold {mac_threshold:g})")
    console.print(f"saved MAC report to [bold]{saved}[/bold]")


# ----------------------------------------------------------------------
# frf
# ----------------------------------------------------------------------
_DAMPING_HELP = (
    "Damping model. A bare number is a modal damping ratio applied to every mode "
    "(e.g. 0.02). Also accepted: 'modal:Z' or 'modal:Z1,Z2,...' (per-mode zeta), "
    "'rayleigh:ALPHA,BETA' or 'rayleigh:alpha=..,beta=..' (C = alpha*M + beta*K), "
    "'structural:ETA' (hysteretic loss factor), 'none', or a combined "
    "'key=value,...' list with keys zeta/alpha/beta/eta."
)


def _parse_damping_spec(spec: str) -> float | dict[str, Any] | None:
    """Parse the ``--damping`` option into a value ``as_damping`` understands.

    Returns a float (modal zeta), a dict with ``zeta``/``alpha``/``beta``/
    ``eta`` keys, or ``None`` for an undamped synthesis.
    """

    def fail(message: str) -> None:
        raise typer.BadParameter(f"{message} (from {spec!r}). {_DAMPING_HELP}",
                                 param_hint="'--damping' / '-z'")

    def floats(text: str, what: str) -> list[float]:
        try:
            values = [float(v) for v in text.split(",") if v.strip()]
        except ValueError:
            fail(f"bad {what} value in {text!r}")
        if not values:
            fail(f"missing {what} value")
        return values

    def keyvals(text: str, allowed: set[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in text.split(","):
            key, sep, value = item.partition("=")
            key = key.strip().lower()
            if not sep or key not in allowed:
                fail(f"expected {'/'.join(sorted(allowed))} assignments, got {item.strip()!r}")
            try:
                out[key] = float(value)
            except ValueError:
                fail(f"bad value for {key}: {value.strip()!r}")
        return out

    s = spec.strip()
    if not s or s.lower() in ("none", "off", "undamped"):
        return None
    try:
        return float(s)  # backward compatible: bare number = modal zeta
    except ValueError:
        pass

    kind, sep, rest = s.partition(":")
    if sep:
        kind = kind.strip().lower()
        rest = rest.strip()
        if kind in ("modal", "zeta", "viscous"):
            values = floats(rest, "zeta")
            return {"zeta": values[0] if len(values) == 1 else values}
        if kind in ("structural", "hysteretic", "eta"):
            values = floats(rest, "eta")
            return {"eta": values[0] if len(values) == 1 else values}
        if kind == "rayleigh":
            if "=" in rest:
                return keyvals(rest, {"alpha", "beta"})
            values = floats(rest, "alpha,beta")
            if len(values) != 2:
                fail(f"rayleigh needs exactly two values alpha,beta, got {len(values)}")
            return {"alpha": values[0], "beta": values[1]}
        fail(f"unknown damping type {kind!r} (use modal, rayleigh or structural)")
    if "=" in s:
        return keyvals(s, {"zeta", "alpha", "beta", "eta"})
    fail("could not parse damping spec")
    return None  # unreachable; fail() always raises


def _describe_damping(parsed: float | dict[str, Any] | None) -> str:
    if parsed is None:
        return "undamped"
    if isinstance(parsed, float):
        return f"modal zeta={parsed:g}"
    parts = []
    if "zeta" in parsed:
        parts.append(f"modal zeta={parsed['zeta']}")
    if "alpha" in parsed or "beta" in parsed:
        parts.append(f"Rayleigh alpha={parsed.get('alpha', 0.0):g} "
                     f"beta={parsed.get('beta', 0.0):g}")
    if "eta" in parsed:
        parts.append(f"structural eta={parsed['eta']}")
    return " + ".join(parts)


def _parse_dof_list(spec: str, option: str) -> list[tuple[int, int]]:
    """Parse 'node:dof[,node:dof...]' into [(node_id, dof_1based), ...]."""
    pairs = []
    for item in spec.split(","):
        try:
            node_str, dof_str = item.split(":")
            pairs.append((int(node_str), int(dof_str)))
        except ValueError:
            err_console.print(
                f"[red]error:[/red] bad {option} spec {item!r} "
                "(expected NODE:DOF, e.g. 2:1 or 2:1,5:3)"
            )
            raise typer.Exit(code=2) from None
    return pairs


def _synth_frf(model_file: Path, input_dofs: str, output_dofs: str,
               fmin: float, fmax: float, n_freq: int, n_modes: int,
               damping: str) -> tuple[Any, Any, Any, int, int]:
    """Shared modal-FRF synthesis pipeline of ``frf`` / ``dump-frf``.

    Solves the modal basis, maps the NODE:DOF selections and calls
    :func:`femtools.dynamics.frf.modal_frf` (imported lazily: an
    installation shipping without it exits with code 3).  Returns
    ``(freq_hz, frf, damping_spec, n_inputs, n_outputs)``.
    """
    import numpy as np

    damping_spec = _parse_damping_spec(damping)
    model = _load_model(model_file)
    modal = _solve(model, n_modes, 0.0)
    input_pairs = _parse_dof_list(input_dofs, "--input")
    output_pairs = _parse_dof_list(output_dofs, "--response")
    freq_hz = np.linspace(fmin, fmax, n_freq)

    # map (node, 1-based dof) to global DOF indices via the modal DOF map
    dof_map = getattr(modal, "dof_map", None)

    def _to_indices(pairs: list[tuple[int, int]], option: str) -> list:
        if dof_map is None:
            return pairs  # let modal_frf interpret the raw selection
        try:
            return [int(dof_map.index(node, dof - 1)) for node, dof in pairs]
        except KeyError as exc:
            err_console.print(f"[red]error:[/red] bad {option} DOF: {exc}")
            raise typer.Exit(code=2) from exc

    inputs = _to_indices(input_pairs, "--input")
    outputs = _to_indices(output_pairs, "--response")

    try:
        from femtools.dynamics.frf import modal_frf
    except ImportError as exc:
        raise _missing("FRF synthesis", exc) from exc
    frf = modal_frf(modal, inputs, outputs, freq_hz, damping_spec)
    return freq_hz, frf, damping_spec, len(input_pairs), len(output_pairs)


@app.command("frf")
def frf_cmd(
    model_file: Annotated[Path, typer.Argument(exists=True, help="Model file.")],
    input_dofs: Annotated[str, typer.Option(
        "--input", "-i",
        help="Excitation DOFs 'NODE:DOF[,NODE:DOF...]' (DOF 1-6).")],
    output_dofs: Annotated[str, typer.Option(
        "--response", "--output-dof", "-r",
        help="Response DOFs 'NODE:DOF[,NODE:DOF...]'.")],
    fmin: Annotated[float, typer.Option("--fmin", help="Start frequency [Hz].")] = 0.0,
    fmax: Annotated[float, typer.Option("--fmax", help="End frequency [Hz].")] = 100.0,
    n_freq: Annotated[int, typer.Option("--n-freq", min=2,
                                        help="Frequency points.")] = 500,
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Modes kept in the modal basis.")] = 20,
    damping: Annotated[str, typer.Option(
        "--damping", "-z", help=_DAMPING_HELP)] = "0.02",
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save FRF to .npz (freq_hz, H).")] = None,
    plot: Annotated[Path | None, typer.Option(
        "--plot", help="Save a Bode plot (PNG).")] = None,
) -> None:
    """Synthesize modal FRFs between input and response DOFs."""
    import numpy as np

    freq_hz, frf, damping_spec, n_in, n_out = _synth_frf(
        model_file, input_dofs, output_dofs, fmin, fmax, n_freq, n_modes, damping)

    H = None
    for attr in ("H", "h", "frf", "data", "values"):
        H = getattr(frf, attr, None)
        if H is not None:
            break
    if H is None:
        H = np.asarray(frf)
    console.print(
        f"FRF computed: {n_out} outputs x {n_in} inputs x {n_freq} "
        f"frequencies ({fmin:g}-{fmax:g} Hz), damping: {_describe_damping(damping_spec)}"
    )

    if output is not None:
        np.savez(str(output), freq_hz=freq_hz, H=np.asarray(H))
        console.print(f"saved FRF to [bold]{output}[/bold]")
    if plot is not None:
        from femtools.viz import plot_frf

        plot_frf(frf, 0, 0, freq=freq_hz, outfile=str(plot))
        console.print(f"saved FRF plot to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# dump-frf / load-frf
# ----------------------------------------------------------------------
_FRF_H_KEYS = ("H", "h", "frf")
_FRF_FREQ_KEYS = ("freq_hz", "freqs_hz", "freq", "freqs", "f")


@app.command("dump-frf")
def dump_frf_cmd(
    source_file: Annotated[Path, typer.Argument(
        exists=True, readable=True,
        help="Model file to synthesize from, or a plain .npz carrying "
             "'freq_hz' and 'H' (e.g. saved by 'frf -o' / 'estimate-frf -o').")],
    output: Annotated[Path, typer.Option(
        "--output", "-o", help="FRF archive to write (.npz).")],
    input_dofs: Annotated[str | None, typer.Option(
        "--input", "-i",
        help="Excitation DOFs 'NODE:DOF[,NODE:DOF...]' (DOF 1-6); required "
             "when synthesizing from a model file.")] = None,
    output_dofs: Annotated[str | None, typer.Option(
        "--response", "--output-dof", "-r",
        help="Response DOFs 'NODE:DOF[,NODE:DOF...]'; required when "
             "synthesizing from a model file.")] = None,
    fmin: Annotated[float, typer.Option("--fmin", help="Start frequency [Hz].")] = 0.0,
    fmax: Annotated[float, typer.Option("--fmax", help="End frequency [Hz].")] = 100.0,
    n_freq: Annotated[int, typer.Option("--n-freq", min=2,
                                        help="Frequency points.")] = 500,
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Modes kept in the modal basis.")] = 20,
    damping: Annotated[str, typer.Option(
        "--damping", "-z", help=_DAMPING_HELP)] = "0.02",
    compress: Annotated[bool, typer.Option(
        "--compress/--no-compress", help="Write a compressed archive.")] = False,
) -> None:
    """Write a reloadable FRF archive (``femtools.dynamics.frf.dump_frf``).

    Synthesizes modal FRFs like the ``frf`` command (or canonicalizes a
    plain ``.npz`` that already carries ``freq_hz`` and ``H``) and stores
    them with the format tag ``load-frf`` checks for; ``H`` and
    ``freq_hz`` round-trip bit-identical.  Requires the FRF kernel
    (``femtools.dynamics.frf``); an installation shipping without it
    exits with code 3, like every other lazily bound subcommand.
    """
    import numpy as np

    try:
        from femtools.dynamics.frf import dump_frf
    except ImportError as exc:
        raise _missing("FRF archiving", exc) from exc

    if source_file.suffix.lower() == ".npz":
        try:
            data = np.load(str(source_file))
        except (OSError, ValueError) as exc:
            err_console.print(f"[red]error:[/red] cannot read {source_file}: {exc}")
            raise typer.Exit(code=2) from exc
        block: dict[str, Any] = {}
        for name, keys in (("H", _FRF_H_KEYS), ("freq_hz", _FRF_FREQ_KEYS)):
            for key in keys:
                if key in data:
                    block[name] = data[key]
                    break
        missing = [name for name in ("H", "freq_hz") if name not in block]
        if missing:
            err_console.print(
                f"[red]error:[/red] {source_file} has no {'/'.join(missing)} array "
                f"(found: {sorted(data.keys())}); expected an .npz saved by "
                "'frf -o' or 'estimate-frf -o'")
            raise typer.Exit(code=2)
        frf: Any = block
        meta: dict[str, Any] = {"source": source_file.name}
    else:
        if input_dofs is None or output_dofs is None:
            err_console.print(
                "[red]error:[/red] synthesizing an FRF from a model file needs "
                "--input and --response DOF selections")
            raise typer.Exit(code=2)
        _, frf, damping_spec, _, _ = _synth_frf(
            source_file, input_dofs, output_dofs, fmin, fmax, n_freq, n_modes, damping)
        meta = {"source": source_file.name,
                "damping": _describe_damping(damping_spec)}

    try:
        saved = dump_frf(frf, output, compress=compress, meta=meta)
    except (TypeError, ValueError) as exc:
        err_console.print(f"[red]error:[/red] cannot archive FRF: {exc}")
        raise typer.Exit(code=2) from exc

    H = np.asarray(frf["H"] if isinstance(frf, dict) else frf.H)
    if H.ndim == 2:
        H = H[:, None, :]
    console.print(
        f"archived FRF block {H.shape[0]} outputs x {H.shape[1]} inputs x "
        f"{H.shape[2]} frequencies to [bold]{saved}[/bold]")


@app.command("load-frf")
def load_frf_cmd(
    archive: Annotated[Path, typer.Argument(
        exists=True, readable=True, help="FRF archive written by dump-frf.")],
    plot: Annotated[Path | None, typer.Option(
        "--plot", help="Save a Bode plot (PNG).")] = None,
    output_index: Annotated[int, typer.Option(
        "--plot-output", min=0, help="Output row to plot (0-based).")] = 0,
    input_index: Annotated[int, typer.Option(
        "--plot-input", min=0, help="Input column to plot (0-based).")] = 0,
) -> None:
    """Inspect an FRF archive written by ``dump-frf``.

    Loads the archive with ``femtools.dynamics.frf.load_frf``, prints a
    summary (shape, response type, frequency band, provenance) and
    optionally saves a Bode plot of one curve.  Requires the FRF kernel;
    an installation shipping without it exits with code 3.
    """
    import zipfile

    import numpy as np

    try:
        from femtools.dynamics.frf import load_frf
    except ImportError as exc:
        raise _missing("FRF archiving", exc) from exc

    try:
        frf = load_frf(archive)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    freq = np.asarray(frf.freq_hz, dtype=float).reshape(-1)
    table = Table(title=f"FRF archive {archive.name}")
    table.add_column("field", justify="left")
    table.add_column("value", justify="left")
    table.add_row("shape", f"{frf.n_out} outputs x {frf.n_in} inputs x "
                           f"{frf.n_freq} frequencies")
    if freq.size:
        table.add_row("band", f"{float(freq[0]):g} - {float(freq[-1]):g} Hz")
    table.add_row("response", str(frf.response))
    if frf.method:
        table.add_row("method", str(frf.method))
    for name in ("outputs", "inputs"):
        dofs = getattr(frf, name, None)
        if dofs is not None:
            values = np.asarray(dofs).reshape(-1).tolist()
            table.add_row(f"{name} (DOF ids)", ", ".join(str(int(v)) for v in values))
    for key, value in (frf.meta or {}).items():
        if key != "loaded_from":
            table.add_row(f"meta.{key}", str(value))
    console.print(table)

    if plot is not None:
        if output_index >= frf.n_out or input_index >= frf.n_in:
            err_console.print(
                f"[red]error:[/red] curve ({output_index},{input_index}) is out of "
                f"range for a {frf.n_out}x{frf.n_in} block")
            raise typer.Exit(code=2)
        from femtools.viz import plot_frf

        plot_frf(frf, output_index, input_index, outfile=str(plot))
        console.print(f"saved FRF plot to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# reduce
# ----------------------------------------------------------------------
_REDUCE_METHODS = ("guyan", "irs", "serep", "craig-bampton")


def _reduction_parts(result: Any) -> tuple[Any, Any, Any]:
    """Duck-typed unpack of a reduction kernel result into ``(T, Kr, Mr)``.

    Accepts ``(T, Kr)`` / ``(T, Kr, Mr)`` tuples or an object carrying the
    basis and (optionally) the reduced matrices; missing matrices are
    returned as ``None`` and recomputed by the caller via ``T``.
    """
    if isinstance(result, tuple):
        if len(result) == 2:
            return result[0], result[1], None
        if len(result) >= 3:
            return result[0], result[1], result[2]
        return result[0] if result else None, None, None
    if hasattr(result, "ndim"):  # bare basis matrix (ndarray/sparse): .T is a transpose!
        return result, None, None
    T = next((getattr(result, name) for name in ("T", "transform", "transformation", "basis")
              if getattr(result, name, None) is not None), None)
    Kr = next((getattr(result, name) for name in ("K_red", "Kr", "K_reduced", "Krr", "K")
               if getattr(result, name, None) is not None), None)
    Mr = next((getattr(result, name) for name in ("M_red", "Mr", "M_reduced", "M")
               if getattr(result, name, None) is not None), None)
    if T is None and Kr is None:
        return result, None, None  # bare transformation matrix
    return T, Kr, Mr


def _reduced_frequencies(Kr: Any, Mr: Any) -> Any:
    """Natural frequencies [Hz] of the reduced pencil ``(Kr, Mr)``.

    Solved inside the subspace the reduced mass actually spans: a SEREP
    basis built from fewer modes than masters leaves ``M_red`` singular by
    construction, so a plain ``eigh(Kr, Mr)`` fails outright.  The null
    directions carry no inertia (infinite frequency) and are dropped, so
    the returned vector can be shorter than ``Kr.shape[0]``.
    """
    import numpy as np

    Ks = 0.5 * (Kr + Kr.T)
    Ms = 0.5 * (Mr + Mr.T)
    w, V = np.linalg.eigh(Ms)  # ascending
    keep = w > max(float(w[-1]), 0.0) * Ms.shape[0] * 1.0e-14
    if not keep.any():
        raise np.linalg.LinAlgError("reduced mass matrix has no positive eigenvalues")
    B = V[:, keep] / np.sqrt(w[keep])
    KB = B.T @ Ks @ B
    lam = np.linalg.eigvalsh(0.5 * (KB + KB.T))
    return np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi)


@app.command("reduce")
def reduce_cmd(
    model_file: Annotated[Path, typer.Argument(exists=True, help="Model file.")],
    master: Annotated[str, typer.Option(
        "--master", "-m",
        help="Master (retained/boundary) DOFs 'NODE:DOF[,NODE:DOF...]' (DOF 1-6).")],
    method: Annotated[str, typer.Option(
        "--method", "-M",
        help="Reduction method: 'guyan', 'irs', 'serep' or 'craig-bampton'.")] = "guyan",
    n_modes: Annotated[int, typer.Option(
        "--n-modes", "-n", min=1,
        help="Target modes: SEREP basis size, Craig-Bampton fixed-interface "
             "modes, and the number of comparison frequencies.")] = 10,
    compare: Annotated[bool, typer.Option(
        "--compare/--no-compare",
        help="Compare reduced vs full-model frequencies.")] = True,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save T, K_red, M_red, ... to .npz.")] = None,
) -> None:
    """Reduce a model to master DOFs (Guyan/IRS/SEREP/Craig-Bampton)."""
    import numpy as np

    method_key = method.strip().lower().replace("_", "-")
    if method_key in ("cb", "craigbampton"):
        method_key = "craig-bampton"
    if method_key not in _REDUCE_METHODS:
        raise typer.BadParameter(
            f"unknown method {method!r}; use one of {', '.join(_REDUCE_METHODS)}",
            param_hint="'--method' / '-M'")

    model = _load_model(model_file)
    try:
        from femtools.fea.assemble import assemble_km
    except ImportError as exc:
        raise _missing("matrix assembly", exc) from exc
    asm = assemble_km(model)
    free = np.asarray(asm.free_dof, dtype=int)
    Kff = asm.K[free, :][:, free]
    Mff = asm.M[free, :][:, free]

    # map NODE:DOF masters to positions inside the free-DOF partition
    pairs = _parse_dof_list(master, "--master")
    pos_of = {int(g): i for i, g in enumerate(free.tolist())}
    master_free: list[int] = []
    master_global: list[int] = []
    for node, dof in pairs:
        try:
            g = int(asm.dof_map.index(node, dof - 1))
        except (KeyError, ValueError) as exc:
            err_console.print(f"[red]error:[/red] bad --master DOF {node}:{dof}: {exc}")
            raise typer.Exit(code=2) from exc
        p = pos_of.get(g)
        if p is None:
            err_console.print(
                f"[red]error:[/red] master DOF {node}:{dof} is constrained or "
                "inactive (not in the free-DOF set)")
            raise typer.Exit(code=2)
        master_free.append(p)
        master_global.append(g)
    if len(set(master_free)) != len(master_free):
        err_console.print("[red]error:[/red] duplicate DOFs in --master")
        raise typer.Exit(code=2)
    master_idx = np.asarray(master_free, dtype=int)

    modal_full = None
    if compare or method_key == "serep":
        try:
            from femtools.fea.eigen import solve_modes
        except ImportError as exc:
            raise _missing("modal analysis", exc) from exc
        n_solve = max(n_modes, 1)
        modal_full = solve_modes(model, n_modes=n_solve, assembly=asm)

    try:
        if method_key == "craig-bampton":
            try:
                from femtools.dynamics.craig_bampton import craig_bampton
            except ImportError as exc:
                raise _missing("Craig-Bampton reduction", exc) from exc
            result = craig_bampton(Kff, Mff, master_idx, n_modes=n_modes)
        else:
            try:
                import femtools.fea.reduction as reduction
            except ImportError as exc:
                raise _missing(f"{method_key} reduction", exc) from exc
            kernel = getattr(reduction, method_key, None)
            if kernel is None:
                err_console.print(
                    f"[red]error:[/red] femtools.fea.reduction does not provide "
                    f"{method_key!r} in this installation")
                raise typer.Exit(code=MISSING_MODULE_EXIT)
            if method_key == "serep":
                phi_ff = np.asarray(modal_full.modes)[free, :]
                result = kernel(phi_ff, master_idx)
            elif method_key == "irs":
                # the reduction contract works on dense arrays
                result = kernel(Kff.toarray(), Mff.toarray(), master_idx)
            else:  # guyan
                result = kernel(Kff.toarray(), master_idx)
    except (ValueError, np.linalg.LinAlgError) as exc:
        err_console.print(f"[red]error:[/red] {method_key} reduction failed: {exc}")
        raise typer.Exit(code=2) from exc

    T, Kr, Mr = _reduction_parts(result)
    if T is None:
        err_console.print("[red]error:[/red] reduction kernel returned no "
                          "transformation matrix")
        raise typer.Exit(code=2)
    T = np.asarray(T if not hasattr(T, "toarray") else T.toarray(), dtype=float)
    if Kr is None:
        Kr = T.T @ (Kff @ T)
    if Mr is None:
        Mr = T.T @ (Mff @ T)
    Kr = np.asarray(Kr if not hasattr(Kr, "toarray") else Kr.toarray(), dtype=float)
    Mr = np.asarray(Mr if not hasattr(Mr, "toarray") else Mr.toarray(), dtype=float)
    n_red = int(Kr.shape[0])

    labels = ", ".join(f"{node}:{dof}" for node, dof in pairs)
    extra = ""
    if method_key == "craig-bampton":
        extra = f" ({len(master_idx)} boundary + {n_red - len(master_idx)} modal)"
    console.print(f"{method_key} reduction: {free.size} free DOFs -> "
                  f"{n_red} generalized DOFs{extra}")
    console.print(f"masters: {labels}")

    freq_red = np.zeros(0)
    if compare:
        try:
            freq_red = np.asarray(_reduced_frequencies(Kr, Mr))
        except (ValueError, np.linalg.LinAlgError) as exc:
            err_console.print(f"[yellow]warning:[/yellow] reduced eigensolve failed "
                              f"({exc}); skipping the frequency comparison")
        else:
            if freq_red.size < n_red:
                console.print(
                    f"reduced mass has rank {freq_red.size} of {n_red}; comparing "
                    f"the {freq_red.size} finite-frequency modes")
            freq_full = np.atleast_1d(np.asarray(modal_full.freq_hz, dtype=float))
            n_cmp = int(min(n_modes, freq_red.size, freq_full.size))
            table = Table(title=f"frequency comparison ({method_key})")
            table.add_column("mode", justify="right")
            table.add_column("full [Hz]", justify="right")
            table.add_column("reduced [Hz]", justify="right")
            table.add_column("error [%]", justify="right")
            for i in range(n_cmp):
                ff, fr = float(freq_full[i]), float(freq_red[i])
                err = f"{100.0 * (fr - ff) / ff:+.4f}" if ff != 0.0 else "-"
                table.add_row(str(i + 1), f"{ff:.6g}", f"{fr:.6g}", err)
            console.print(table)

    if output is not None:
        payload: dict[str, Any] = {
            "T": T,
            "K_red": Kr,
            "M_red": Mr,
            "master_dof": np.asarray(master_global, dtype=int),
            "master_free": master_idx,
            "free_dof": free,
            "method": np.asarray(method_key),
        }
        if freq_red.size:
            payload["freq_reduced"] = freq_red
        if modal_full is not None:
            payload["freq_full"] = np.asarray(modal_full.freq_hz, dtype=float)
        np.savez(str(output), **payload)
        console.print(f"saved reduced model to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# estimate-frf
# ----------------------------------------------------------------------
_INPUT_KEYS = ("x", "u", "input", "force", "excitation")
_OUTPUT_KEYS = ("y", "a", "response", "output", "acceleration", "accel")
_FS_KEYS = ("fs", "sample_rate", "sampling_rate", "sr")
_TIME_KEYS = ("t", "time")


def _call_estimator(fn: Any, x: Any, y: Any, rate: float, **tuning: Any) -> Any:
    """Call a spectral estimator, adapting to its exact signature.

    The estimator kernels are developed independently; this keeps the CLI
    working across signature variants: an ``overlap=`` fraction is
    translated to the ``noverlap=`` sample count when the function takes
    only the latter (the scipy.signal convention), tuning keywords the
    function does not accept (e.g. ``window=``) are dropped, and the
    sampling rate is passed as the ``fs=`` keyword when the signature
    takes one (the contract-test convention) or as the third positional
    argument otherwise.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(x, y, fs=rate, **tuning)
    if ("overlap" in tuning and "overlap" not in params and "noverlap" in params
            and tuning.get("nperseg")):
        tuning = dict(tuning)
        tuning["noverlap"] = int(round(float(tuning.pop("overlap"))
                                       * int(tuning["nperseg"])))
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not has_varkw:
        tuning = {k: v for k, v in tuning.items() if k in params}
    if "fs" in params or has_varkw:
        return fn(x, y, fs=rate, **tuning)
    return fn(x, y, rate, **tuning)


def _pick_array(data: Any, explicit: str | None, candidates: tuple[str, ...],
                option: str) -> Any:
    keys = list(data.keys())
    if explicit:
        if explicit in keys:
            return data[explicit]
        err_console.print(f"[red]error:[/red] no array {explicit!r} in the data "
                          f"file (found: {sorted(keys)})")
        raise typer.Exit(code=2)
    for key in candidates:
        if key in keys:
            return data[key]
    err_console.print(
        f"[red]error:[/red] could not find a {option} time history "
        f"(tried {', '.join(candidates)}; found: {sorted(keys)}). "
        f"Select one explicitly with --{option}-key.")
    raise typer.Exit(code=2)


def _resolve_fs(data: Any, fs: float | None) -> float:
    import numpy as np

    if fs is not None:
        return float(fs)
    keys = list(data.keys())
    for key in _FS_KEYS:
        if key in keys:
            return float(np.asarray(data[key], dtype=float).reshape(-1)[0])
    for key in _TIME_KEYS:
        if key in keys:
            t = np.asarray(data[key], dtype=float).reshape(-1)
            if t.size >= 2:
                dt = float(np.median(np.diff(t)))
                if dt > 0.0:
                    return 1.0 / dt
    err_console.print(
        "[red]error:[/red] sampling rate unknown: pass --fs or store an "
        f"'fs' (or {'/'.join(_TIME_KEYS)}) array in the data file")
    raise typer.Exit(code=2)


def _frf_estimate_parts(result: Any, fs: float, nperseg: int) -> tuple[Any, Any, Any]:
    """Duck-typed unpack of an estimator result into ``(freq_hz, H, coherence)``."""
    import numpy as np

    freq = H = coh = None
    if isinstance(result, tuple):
        if len(result) >= 2:
            freq, H = result[0], result[1]
        if len(result) >= 3:
            coh = result[2]
    elif isinstance(result, np.ndarray):
        H = result
    else:
        for attr in ("freq_hz", "freqs_hz", "frequency", "frequencies", "freqs", "freq", "f"):
            freq = getattr(result, attr, None)
            if freq is not None:
                break
        for attr in ("H", "h_complex", "h", "frf", "data", "values"):
            H = getattr(result, attr, None)
            if H is not None:
                break
        for attr in ("coherence", "gamma2", "coh"):
            coh = getattr(result, attr, None)
            if coh is not None:
                break
    if H is None:
        return None, None, None
    H = np.asarray(H)
    if freq is None:
        n_f = int(H.shape[-1])
        if n_f == nperseg // 2 + 1:
            freq = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        else:
            freq = np.linspace(0.0, fs / 2.0, n_f)
        err_console.print("[yellow]warning:[/yellow] estimator returned no frequency "
                          "vector; assuming a one-sided grid up to fs/2")
    return np.asarray(freq, dtype=float).reshape(-1), H, coh


@app.command("estimate-frf")
def estimate_frf_cmd(
    data_file: Annotated[Path, typer.Argument(
        exists=True, readable=True,
        help=".npz with input/output time histories (keys like x/u/force and "
             "y/a/response; time along the last axis) plus 'fs' or 't'.")],
    fs: Annotated[float | None, typer.Option(
        "--fs", help="Sampling rate [Hz] (default: 'fs' or 't' from the file).")] = None,
    input_key: Annotated[str | None, typer.Option(
        "--input-key", "-x", help="Array key of the excitation signal.")] = None,
    output_key: Annotated[str | None, typer.Option(
        "--output-key", "-y", help="Array key of the response signal.")] = None,
    estimator: Annotated[str, typer.Option(
        "--estimator", "-e", help="FRF estimator: 'h1' or 'h2'.")] = "h1",
    nperseg: Annotated[int, typer.Option(
        "--nperseg", min=8, help="Samples per Welch segment.")] = 1024,
    overlap: Annotated[float, typer.Option(
        "--overlap", min=0.0, max=0.99, help="Segment overlap fraction.")] = 0.5,
    window: Annotated[str, typer.Option(
        "--window", help="Window function (e.g. hann, hamming, boxcar).")] = "hann",
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save freq_hz, H (and coherence) to .npz.")] = None,
    plot: Annotated[Path | None, typer.Option(
        "--plot", help="Save a Bode plot of the first FRF (PNG).")] = None,
) -> None:
    """Estimate H1/H2 FRFs (and coherence) from measured time histories."""
    import numpy as np

    est = estimator.strip().lower()
    if est not in ("h1", "h2"):
        raise typer.BadParameter(f"unknown estimator {estimator!r}; use 'h1' or 'h2'",
                                 param_hint="'--estimator' / '-e'")

    try:
        data = np.load(str(data_file))
    except (OSError, ValueError) as exc:
        err_console.print(f"[red]error:[/red] cannot read {data_file}: {exc}")
        raise typer.Exit(code=2) from exc
    x = np.asarray(_pick_array(data, input_key, _INPUT_KEYS, "input"))
    y = np.asarray(_pick_array(data, output_key, _OUTPUT_KEYS, "output"))
    rate = _resolve_fs(data, fs)

    try:
        import femtools.mpe.frf_estimation as frf_estimation
    except ImportError as exc:
        raise _missing("FRF estimation", exc) from exc
    kernel = getattr(frf_estimation, f"estimate_{est}", None)
    if kernel is None:
        err_console.print(
            f"[red]error:[/red] femtools.mpe.frf_estimation does not provide "
            f"'estimate_{est}' in this installation")
        raise typer.Exit(code=MISSING_MODULE_EXIT)

    try:
        result = _call_estimator(
            kernel, x, y, rate, nperseg=nperseg, window=window, overlap=overlap)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {est} estimation failed: {exc}")
        raise typer.Exit(code=2) from exc
    freq, H, coh = _frf_estimate_parts(result, rate, nperseg)
    if H is None:
        err_console.print("[red]error:[/red] estimator returned no FRF data")
        raise typer.Exit(code=2)

    if coh is None:
        coh_fn = getattr(frf_estimation, "coherence", None)
        if coh_fn is not None:
            try:
                coh_res = _call_estimator(
                    coh_fn, x, y, rate, nperseg=nperseg, window=window, overlap=overlap)
            except ValueError:
                coh_res = None
            if isinstance(coh_res, tuple):
                coh_res = coh_res[-1]
            for attr in ("coherence", "gamma2", "coh", "values", "C"):
                if hasattr(coh_res, attr):
                    coh_res = getattr(coh_res, attr)
                    break
            coh = coh_res
    coh_arr = None if coh is None else np.asarray(coh, dtype=float)

    n_samples = int(x.shape[-1])
    df = float(freq[1] - freq[0]) if freq.size > 1 else 0.0
    console.print(
        f"{est.upper()} estimate: {n_samples} samples at {rate:g} Hz -> "
        f"{freq.size} frequency lines ({freq[0]:g}-{freq[-1]:g} Hz, df={df:g} Hz), "
        f"nperseg={nperseg}, window={window}, overlap={overlap:g}")
    if coh_arr is not None and coh_arr.size:
        console.print(f"coherence: mean {float(np.nanmean(coh_arr)):.4f}, "
                      f"min {float(np.nanmin(coh_arr)):.4f}")

    if output is not None:
        payload = {"freq_hz": freq, "H": np.asarray(H),
                   "fs": np.asarray(rate), "estimator": np.asarray(est)}
        if coh_arr is not None:
            payload["coherence"] = coh_arr
        np.savez(str(output), **payload)
        console.print(f"saved FRF estimate to [bold]{output}[/bold]")
    if plot is not None:
        from femtools.viz import plot_frf

        plot_frf(np.asarray(H), 0, 0, freq=freq,
                 title=f"{est.upper()} FRF estimate", outfile=str(plot))
        console.print(f"saved FRF plot to [bold]{plot}[/bold]")


# ----------------------------------------------------------------------
# update
# ----------------------------------------------------------------------
@app.command("update")
def update_cmd(
    model_file: Annotated[Path, typer.Argument(exists=True, help="Model file.")],
    config_file: Annotated[Path, typer.Argument(
        exists=True, help="JSON file with update_model keyword arguments "
                          "(parameters, targets, weights, ...).")],
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save the updated model (.ftproj).")] = None,
) -> None:
    """Sensitivity-based model updating driven by a JSON configuration."""
    model = _load_model(model_file)
    with open(config_file, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        err_console.print("[red]error:[/red] update config must be a JSON object")
        raise typer.Exit(code=2)

    try:
        from femtools.updating.updater import update_model
    except ImportError as exc:
        raise _missing("model updating", exc) from exc
    try:
        result = update_model(model, **config)
    except (TypeError, ValueError) as exc:
        # bad keys/values in the user's JSON config surface here
        err_console.print(f"[red]error:[/red] invalid update configuration: {exc}")
        raise typer.Exit(code=2) from exc

    table = Table(title="updating result")
    table.add_column("field")
    table.add_column("value")
    for field in ("converged", "n_iter", "n_iterations", "message", "parameters",
                  "cost", "residual_norm"):
        value = getattr(result, field, None)
        if value is not None:
            table.add_row(field, str(value))
    if table.row_count:
        console.print(table)
    else:
        console.print(repr(result))

    if output is not None:
        updated = getattr(result, "model", model)
        try:
            from femtools.io.project import save_project
        except ImportError as exc:
            raise _missing("saving projects", exc) from exc
        save_project(updated, str(output))
        console.print(f"saved updated model to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# update-static
# ----------------------------------------------------------------------
@app.command("update-static")
def update_static_cmd(
    model_file: Annotated[Path, typer.Argument(
        exists=True, readable=True, help="Model file.")],
    measure: Annotated[list[str], typer.Option(
        "--measure", "-m",
        help="Measured static deflection 'NODE:DOF=VALUE' (DOF 1-6: "
             "ux uy uz rx ry rz); repeatable.")],
    load: Annotated[list[str] | None, typer.Option(
        "--load", "-l",
        help="Add the test load 'NODE:DOF=VALUE' (DOF 1-6: fx fy fz mx my mz); "
             "repeatable.  Loads stored in the model file are kept.")] = None,
    config_file: Annotated[Path | None, typer.Option(
        "--config", "-c", exists=True, readable=True,
        help="JSON file with extra update_from_static keyword arguments "
             "(parameters, max_iter, ...).")] = None,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save the updated model (.ftproj).")] = None,
) -> None:
    """Update the model against measured static deflections.

    Wraps ``femtools.updating.updater.update_from_static``: the measured
    deflections drive the same Gauss-Newton loop as modal updating, with
    one relative Young's modulus multiplier as the default parameter
    (override via ``--config``).  Requires the updating kernel; an
    installation shipping without it exits with code 3, like every
    other lazily bound subcommand.
    """
    try:
        from femtools.updating.updater import update_from_static
    except ImportError as exc:
        raise _missing("static model updating", exc) from exc

    model = _load_model(model_file)
    _apply_load_specs(model, load)
    if not getattr(model, "loads", None):
        err_console.print(
            "[yellow]warning:[/yellow] the model carries no loads and none were "
            "given with --load; a static update needs the test load that "
            "produced the measured deflections")

    # (node, 0-based component) keys are the kernel's self-describing form
    measured = {}
    for spec in measure:
        node, dof, value = _parse_point_load(spec, option="--measure")
        measured[(node, dof - 1)] = value

    config: dict[str, Any] = {}
    if config_file is not None:
        with open(config_file, encoding="utf-8") as fh:
            config = json.load(fh)
        if not isinstance(config, dict):
            err_console.print("[red]error:[/red] update config must be a JSON object")
            raise typer.Exit(code=2)

    try:
        result = update_from_static(model, measured, **config)
    except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        # TypeError/ValueError cover bad config keys and inconsistent
        # measurements, RuntimeError a singular static solve
        err_console.print(f"[red]error:[/red] static updating failed: {exc}")
        raise typer.Exit(code=2) from exc

    summary = getattr(result, "summary", None)
    console.print(summary() if callable(summary) else repr(result))

    if output is not None:
        updated = getattr(result, "model", None) or model
        try:
            from femtools.io.project import save_project
        except ImportError as exc:
            raise _missing("saving projects", exc) from exc
        save_project(updated, str(output))
        console.print(f"saved updated model to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# pretest
# ----------------------------------------------------------------------
def _pretest_candidates(modal: Any) -> Any | None:
    """Translational free-DOF candidate set of a modal result, if available.

    An accelerometer measures one translation of one node, so the EFI
    ranking must not consider rotational or SPC-constrained DOFs.  Returns
    ``None`` when the candidate slicing cannot run (stripped installation,
    or a modal result without a usable DOF map); the caller then falls back
    to ranking the raw mode matrix.
    """
    import numpy as np

    try:
        from femtools.pretest.candidates import translational_dofs
    except ImportError:
        return None
    try:
        cand = translational_dofs(modal)
    except (TypeError, ValueError, KeyError):
        return None
    phi = getattr(cand, "phi", None)
    if phi is None or getattr(cand, "dofs", None) is None or not np.size(phi):
        return None
    return cand


@app.command("pretest")
def pretest_cmd(
    model_file: Annotated[Path, typer.Argument(exists=True, help="Model file.")],
    n_modes: Annotated[int, typer.Option("--n-modes", "-n", min=1,
                                         help="Target modes.")] = 6,
    n_sensors: Annotated[int, typer.Option("--n-sensors", "-s", min=1,
                                           help="Sensors to keep (EFI).")] = 10,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", help="Save ranked sensors as JSON.")] = None,
) -> None:
    """Rank sensor locations with Kammer Effective Independence (EFI)."""
    import numpy as np

    model = _load_model(model_file)
    modal = _solve(model, n_modes, 0.0)
    console.print(_freq_table(modal, f"target modes of {model_file.name}"))

    try:
        from femtools.pretest.efi import effective_independence
    except ImportError as exc:
        raise _missing("pretest sensor selection", exc) from exc

    cand = _pretest_candidates(modal)
    if cand is not None:
        phi = np.asarray(cand.phi)
        candidate_ids = np.asarray(cand.dofs).reshape(-1)
        labels = list(getattr(cand, "labels", None) or [])
        label_of = {int(d): lab
                    for d, lab in zip(candidate_ids.tolist(), labels, strict=False)}
        n_dropped = int(np.size(getattr(cand, "dropped", ())))
        note = f" ({n_dropped} constrained/inactive DOFs dropped)" if n_dropped else ""
        console.print(f"candidates: {phi.shape[0]} translational free DOFs{note}")
        if n_sensors > phi.shape[0]:
            err_console.print(
                f"[yellow]warning:[/yellow] --n-sensors {n_sensors} exceeds the "
                f"{phi.shape[0]} candidate DOFs; keeping all of them"
            )
            n_sensors = int(phi.shape[0])
    else:
        phi = np.asarray(modal.modes)
        candidate_ids = None
        label_of = {}
        err_console.print(
            "[yellow]warning:[/yellow] translational candidate slicing is "
            "unavailable; ranking every row of the raw mode matrix"
        )

    kwargs = {} if candidate_ids is None else {"candidate_dofs": candidate_ids}
    try:
        try:
            ranked = effective_independence(phi, n_sensors=n_sensors, **kwargs)
        except TypeError:
            ranked = effective_independence(phi)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # duck-typed unpack: (ids, values) tuple, object with fields, or plain ids
    ids, values = None, None
    if isinstance(ranked, tuple) and len(ranked) == 2:
        ids, values = ranked
    else:
        for id_attr in ("dof_ids", "sensors", "ids", "selected"):
            if getattr(ranked, id_attr, None) is not None:
                ids = getattr(ranked, id_attr)
                break
        values = getattr(ranked, "efi", getattr(ranked, "values", None))
        if ids is None:
            ids = ranked
    ids = list(np.atleast_1d(np.asarray(ids)).tolist())[:n_sensors]
    vals = None if values is None else list(np.atleast_1d(np.asarray(values)).tolist())

    table = Table(title=f"EFI sensor ranking (top {len(ids)})")
    table.add_column("rank", justify="right")
    if label_of:
        table.add_column("sensor", justify="left")
    table.add_column("DOF id", justify="right")
    if vals is not None:
        table.add_column("EFI", justify="right")
    for rank, dof_id in enumerate(ids, start=1):
        row = [str(rank)]
        if label_of:
            row.append(label_of.get(int(dof_id), "?"))
        row.append(str(dof_id))
        if vals is not None and rank - 1 < len(vals):
            row.append(f"{float(vals[rank - 1]):.5g}")
        table.add_row(*row)
    console.print(table)

    if output is not None:
        payload: dict[str, Any] = {"dof_ids": ids}
        if label_of:
            payload["labels"] = [label_of.get(int(i), "") for i in ids]
        if vals is not None:
            payload["efi"] = vals[: len(ids)]
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"saved sensor set to [bold]{output}[/bold]")


# ----------------------------------------------------------------------
# script
# ----------------------------------------------------------------------
@app.command("script")
def script_cmd(
    script_file: Annotated[Path | None, typer.Argument(
        help="FSL script file ('-' for stdin).")] = None,
    command: Annotated[str | None, typer.Option(
        "--command", "-c", help="Run an inline FSL snippet.")] = None,
    echo: Annotated[bool, typer.Option(
        "--echo", help="Echo each executed statement.")] = False,
) -> None:
    """Run an FSL (Femtools Scripting Language) script."""
    from femtools.script.engine import ScriptEngine, ScriptError

    if command is None and script_file is None:
        err_console.print("[red]error:[/red] provide a script file or --command/-c")
        raise typer.Exit(code=2)

    sources: list[str] = []
    if script_file is not None:
        if str(script_file) == "-":
            sources.append(sys.stdin.read())
        elif script_file.exists():
            sources.append(script_file.read_text(encoding="utf-8"))
        else:
            err_console.print(f"[red]error:[/red] no such script file: {script_file}")
            raise typer.Exit(code=2)
    if command is not None:
        sources.append(command)

    engine = ScriptEngine(echo=echo)
    try:
        for source in sources:
            engine.run(source)
    except ScriptError as exc:
        err_console.print(f"[red]script error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if engine.model is not None:
        model = engine.model
        console.print(
            f"project [bold]{getattr(model, 'name', '?')}[/bold]: "
            f"{len(getattr(model, 'nodes', {}))} nodes, "
            f"{len(getattr(model, 'elements', {}))} elements, "
            f"{len(getattr(model, 'spcs', []))} SPCs"
        )
    for name, result in engine.results.items():
        freqs = getattr(result, "freq_hz", None)
        if freqs is not None:
            console.print(_freq_table(result, f"result {name!r}"))
        else:
            console.print(f"result [bold]{name}[/bold]: {type(result).__name__}")


# ----------------------------------------------------------------------
# gui
# ----------------------------------------------------------------------
@app.command("gui")
def gui_cmd(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="TCP port.")] = 8765,
    backend: Annotated[str, typer.Option(
        "--backend", help="'auto', 'stdlib' or 'fastapi'.")] = "auto",
    model_file: Annotated[Path | None, typer.Option(
        "--model", "-m", help="Model to preload.")] = None,
    open_browser: Annotated[bool, typer.Option(
        "--open-browser/--no-open-browser",
        help="Open a web browser on start.")] = False,
) -> None:
    """Launch the femtools web GUI (headless-friendly local server)."""
    from femtools.gui import run_gui

    model = _load_model(model_file) if model_file is not None else None
    run_gui(host=host, port=port, backend=backend, model=model, open_browser=open_browser)


def main() -> None:  # convenience for `python -m femtools.cli`
    app()


if __name__ == "__main__":
    main()

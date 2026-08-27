"""femtools command-line interface.

Typer application exposed as the ``femtools`` console script
(``femtools.cli:app``).  Subcommands: ``solve-modes``, ``mac``, ``frf``,
``update``, ``pretest``, ``script``, ``gui``.

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
    """Load a model file (.ftproj / .json / .unv / .bdf|.nas|.dat) as an FEModel.

    Container formats (``.ftproj`` projects, ``.unv`` bundles) are
    unwrapped to the bare model so every downstream solver call receives
    an ``FEModel``.
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

        plot_mode(model, modal, index=mode_index, outfile=str(plot))
        console.print(f"saved mode plot to [bold]{plot}[/bold]")


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
    off_diag = float(np.max(mac - np.diag(np.diag(mac)))) if mac.size > 1 else 0.0
    console.print(f"diag min={float(np.min(np.diag(mac))):.6f}  "
                  f"off-diag max={off_diag:.6f}")

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

    H = None
    for attr in ("H", "h", "frf", "data", "values"):
        H = getattr(frf, attr, None)
        if H is not None:
            break
    if H is None:
        H = np.asarray(frf)
    console.print(
        f"FRF computed: {len(output_pairs)} outputs x {len(input_pairs)} inputs x {n_freq} "
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
    result = update_model(model, **config)

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
# pretest
# ----------------------------------------------------------------------
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
    try:
        ranked = effective_independence(np.asarray(modal.modes), n_sensors=n_sensors)
    except TypeError:
        ranked = effective_independence(np.asarray(modal.modes))

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
    table.add_column("DOF id", justify="right")
    if vals is not None:
        table.add_column("EFI", justify="right")
    for rank, dof_id in enumerate(ids, start=1):
        row = [str(rank), str(dof_id)]
        if vals is not None and rank - 1 < len(vals):
            row.append(f"{float(vals[rank - 1]):.5g}")
        table.add_row(*row)
    console.print(table)

    if output is not None:
        payload = {"dof_ids": ids}
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

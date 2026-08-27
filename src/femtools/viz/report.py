"""Self-contained correlation (MAC) reports in HTML or plain text.

The generators are pure string builders: importing this module needs only
numpy.  :func:`mac_report_html` optionally embeds a heatmap image rendered
through :func:`femtools.viz.plots.plot_mac`; when matplotlib is not
installed (or fails headless) the image is skipped and the report is still
produced, so a stripped installation degrades gracefully.

Inputs are duck-typed to match the rest of the package: ``pairs`` accepts a
:class:`femtools.correlation.pairing.PairingResult`, a list of ``ModePair``
objects, or a plain sequence of ``(index_a, index_b)`` tuples.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["mac_report_html", "mac_report_text", "save_mac_report"]

#: MAC verdict thresholds used by both report flavors.
GOOD_MAC = 0.9
WARN_MAC = 0.7

#: Above this many modes per axis the HTML value table is replaced by the
#: heatmap alone (a 100x100 annotated table is unreadable and huge).
MAX_TABLE_MODES = 50


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
def _as_mac(mac: Any) -> np.ndarray:
    m = np.asarray(mac, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"MAC matrix must be 2-D, got shape {m.shape}")
    return m


def _as_freq(freq: Any, n: int) -> np.ndarray:
    """Coerce an optional frequency vector to length ``n`` (NaN-padded)."""
    out = np.full(n, np.nan)
    if freq is None:
        return out
    arr = np.asarray(freq, dtype=float).reshape(-1)
    k = min(n, arr.size)
    out[:k] = arr[:k]
    return out


def _pair_rows(
    pairs: Any, mac: np.ndarray, freq_a: np.ndarray, freq_b: np.ndarray
) -> list[tuple[int, int, float, float, float]] | None:
    """Normalize ``pairs`` into ``(ia, ib, mac, f_a, f_b)`` rows."""
    if pairs is None:
        return None
    items = getattr(pairs, "pairs", pairs)  # PairingResult -> list[ModePair]
    rows: list[tuple[int, int, float, float, float]] = []
    for item in items:
        if hasattr(item, "index_a"):
            ia, ib = int(item.index_a), int(item.index_b)
            fa = float(getattr(item, "freq_a", np.nan))
            fb = float(getattr(item, "freq_b", np.nan))
            m = float(getattr(item, "mac", np.nan))
        else:
            ia, ib = (int(v) for v in tuple(item)[:2])
            fa = fb = m = float("nan")
        if not (0 <= ia < mac.shape[0] and 0 <= ib < mac.shape[1]):
            continue
        if np.isnan(m):
            m = float(mac[ia, ib])
        if np.isnan(fa):
            fa = float(freq_a[ia])
        if np.isnan(fb):
            fb = float(freq_b[ib])
        rows.append((ia, ib, m, fa, fb))
    return rows


def _summary(
    mac: np.ndarray, rows: list[tuple[int, int, float, float, float]] | None
) -> list[tuple[str, str]]:
    """Key/value summary statistics shared by the HTML and text reports."""
    n_a, n_b = mac.shape
    out = [("modes", f"{n_a} (A) x {n_b} (B)")]
    if n_a == n_b and n_a > 0:
        diag = np.diag(mac)
        out.append(("diagonal MAC min", f"{float(np.min(diag)):.4f}"))
        if mac.size > 1:
            off = mac - np.diag(diag)
            out.append(("off-diagonal MAC max", f"{float(np.max(off)):.4f}"))
    elif mac.size:
        out.append(("MAC max", f"{float(np.max(mac)):.4f}"))
    if rows is not None:
        out.append(("paired modes", str(len(rows))))
        if rows:
            macs = [r[2] for r in rows]
            out.append(("worst pair MAC", f"{min(macs):.4f}"))
            errs = [
                abs(100.0 * (fa - fb) / fb)
                for _, _, _, fa, fb in rows
                if np.isfinite(fa) and np.isfinite(fb) and fb != 0.0
            ]
            if errs:
                out.append(("max |freq error|", f"{max(errs):.3f} %"))
    return out


def _verdict(value: float) -> str:
    if value >= GOOD_MAC:
        return "good"
    if value >= WARN_MAC:
        return "weak"
    return "poor"


def _fmt_hz(value: float) -> str:
    return f"{value:.6g}" if np.isfinite(value) else "-"


def _fmt_err(fa: float, fb: float) -> str:
    if np.isfinite(fa) and np.isfinite(fb) and fb != 0.0:
        return f"{100.0 * (fa - fb) / fb:+.3f}"
    return "-"


def _heatmap_base64(
    mac: np.ndarray,
    labels_a: list[str] | None,
    labels_b: list[str] | None,
    title: str,
) -> str | None:
    """PNG heatmap as base64, or ``None`` when matplotlib is unavailable."""
    try:
        from femtools.viz.plots import plot_mac
    except ImportError:
        return None
    try:
        # pin matplotlib: the embed needs fig.savefig PNG bytes even if the
        # process-wide default backend was switched to plotly
        fig = plot_mac(mac, labels_a=labels_a, labels_b=labels_b, title=title,
                       backend="matplotlib")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # decoration only: a broken backend must not kill the report
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _default_labels(labels: Any, n: int) -> list[str]:
    if labels is None:
        return [str(i + 1) for i in range(n)]
    out = [str(lab) for lab in list(labels)[:n]]
    out += [str(i + 1) for i in range(len(out), n)]
    return out


# ----------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------
_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 2em auto; max-width: 70em; color: #1a2330; padding: 0 1em; }
h1 { font-size: 1.5em; border-bottom: 2px solid #2c6e91; padding-bottom: .3em; }
h2 { font-size: 1.15em; margin-top: 1.6em; color: #2c6e91; }
table { border-collapse: collapse; margin: .8em 0; font-size: .85em; }
th, td { border: 1px solid #c9d4dd; padding: .28em .55em; text-align: right; }
th { background: #eef3f7; }
td.rowhead { background: #eef3f7; font-weight: 600; }
td.diag { outline: 2px solid #1a2330; outline-offset: -2px; }
.summary td { text-align: left; }
.meta { color: #66707c; font-size: .8em; }
.verdict-good { color: #1a7d36; font-weight: 600; }
.verdict-weak { color: #a67c00; font-weight: 600; }
.verdict-poor { color: #b02a2a; font-weight: 600; }
img.heatmap { max-width: 100%; height: auto; border: 1px solid #c9d4dd; }
.note { color: #66707c; font-style: italic; }
"""


def _mac_cell(value: float, diagonal: bool) -> str:
    v = float(min(max(value, 0.0), 1.0))
    fg = "#ffffff" if v >= 0.7 else "#1a2330"
    style = f"background-color: rgba(26, 125, 84, {v:.3f}); color: {fg};"
    cls = ' class="diag"' if diagonal else ""
    return f'<td{cls} style="{style}">{v:.3f}</td>'


def mac_report_html(
    mac: Any,
    *,
    freq_a: Any = None,
    freq_b: Any = None,
    labels_a: Any = None,
    labels_b: Any = None,
    pairs: Any = None,
    title: str = "MAC correlation report",
    name_a: str = "set A",
    name_b: str = "set B",
    heatmap: bool = True,
) -> str:
    """Build a self-contained HTML MAC report and return it as a string.

    Parameters
    ----------
    mac:
        MAC matrix ``(n_a, n_b)`` with values in ``[0, 1]``.
    freq_a, freq_b:
        Optional natural frequencies [Hz] of the two mode sets.
    labels_a, labels_b:
        Optional mode labels (default ``1..n``).
    pairs:
        Optional mode pairing (``PairingResult``, ``ModePair`` list, or
        ``(index_a, index_b)`` tuples) rendered as a pair table with
        frequency errors and MAC verdicts.
    heatmap:
        Embed a base64 PNG heatmap (skipped silently without matplotlib).
    """
    m = _as_mac(mac)
    n_a, n_b = m.shape
    fa = _as_freq(freq_a, n_a)
    fb = _as_freq(freq_b, n_b)
    la = _default_labels(labels_a, n_a)
    lb = _default_labels(labels_b, n_b)
    rows = _pair_rows(pairs, m, fa, fb)

    try:
        from femtools import __version__ as _version
    except ImportError:  # pragma: no cover - femtools is our own parent package
        _version = "unknown"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    esc = html.escape
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{esc(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{esc(title)}</h1>",
        f'<p class="meta">generated {stamp} &middot; femtools {esc(_version)} &middot; '
        f"A = {esc(name_a)}, B = {esc(name_b)}</p>",
    ]

    parts.append("<h2>Summary</h2><table class='summary'>")
    for key, value in _summary(m, rows):
        parts.append(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>")
    parts.append("</table>")

    if heatmap:
        png = _heatmap_base64(m, la, lb, title="MAC")
        if png is not None:
            parts.append(
                "<h2>Heatmap</h2>"
                f'<img class="heatmap" alt="MAC heatmap" '
                f'src="data:image/png;base64,{png}">'
            )
        else:
            parts.append('<p class="note">heatmap skipped: matplotlib unavailable</p>')

    if max(n_a, n_b) <= MAX_TABLE_MODES:
        parts.append("<h2>MAC matrix</h2><table><tr><th>A \\ B</th>")
        parts.extend(f"<th>{esc(lab)}</th>" for lab in lb)
        parts.append("</tr>")
        for i in range(n_a):
            parts.append(f'<tr><td class="rowhead">{esc(la[i])}</td>')
            parts.extend(_mac_cell(m[i, j], n_a == n_b and i == j) for j in range(n_b))
            parts.append("</tr>")
        parts.append("</table>")
    else:
        parts.append(
            f'<p class="note">MAC value table omitted '
            f"({n_a}x{n_b} exceeds {MAX_TABLE_MODES} modes)</p>"
        )

    if np.isfinite(fa).any() or np.isfinite(fb).any():
        parts.append(
            "<h2>Natural frequencies</h2><table>"
            f"<tr><th>mode</th><th>{esc(name_a)} [Hz]</th><th>{esc(name_b)} [Hz]</th></tr>"
        )
        for i in range(max(n_a, n_b)):
            va = _fmt_hz(fa[i]) if i < n_a else "-"
            vb = _fmt_hz(fb[i]) if i < n_b else "-"
            parts.append(f"<tr><td>{i + 1}</td><td>{va}</td><td>{vb}</td></tr>")
        parts.append("</table>")

    if rows is not None:
        parts.append(
            "<h2>Mode pairs</h2><table><tr><th>#</th><th>A</th><th>B</th>"
            "<th>f_A [Hz]</th><th>f_B [Hz]</th><th>&Delta;f [%]</th>"
            "<th>MAC</th><th>verdict</th></tr>"
        )
        for k, (ia, ib, mv, pfa, pfb) in enumerate(rows, start=1):
            verdict = _verdict(mv)
            parts.append(
                f"<tr><td>{k}</td><td>{esc(la[ia])}</td><td>{esc(lb[ib])}</td>"
                f"<td>{_fmt_hz(pfa)}</td><td>{_fmt_hz(pfb)}</td>"
                f"<td>{_fmt_err(pfa, pfb)}</td><td>{mv:.4f}</td>"
                f'<td class="verdict-{verdict}">{verdict}</td></tr>'
            )
        if not rows:
            parts.append('<tr><td colspan="8">no pairs above the MAC threshold</td></tr>')
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# plain-text report
# ----------------------------------------------------------------------
def mac_report_text(
    mac: Any,
    *,
    freq_a: Any = None,
    freq_b: Any = None,
    labels_a: Any = None,
    labels_b: Any = None,
    pairs: Any = None,
    title: str = "MAC correlation report",
    name_a: str = "set A",
    name_b: str = "set B",
) -> str:
    """Build the plain-text flavor of :func:`mac_report_html`."""
    m = _as_mac(mac)
    n_a, n_b = m.shape
    fa = _as_freq(freq_a, n_a)
    fb = _as_freq(freq_b, n_b)
    la = _default_labels(labels_a, n_a)
    lb = _default_labels(labels_b, n_b)
    rows = _pair_rows(pairs, m, fa, fb)

    try:
        from femtools import __version__ as _version
    except ImportError:  # pragma: no cover
        _version = "unknown"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        title,
        "=" * len(title),
        f"generated {stamp} | femtools {_version} | A = {name_a}, B = {name_b}",
        "",
    ]
    for key, value in _summary(m, rows):
        lines.append(f"  {key:<22} {value}")
    lines.append("")

    width = max(7, max((len(s) for s in la + lb), default=1) + 1)
    lines.append("MAC matrix (rows = A, columns = B)")
    lines.append(" " * width + "".join(f"{lab:>{width}}" for lab in lb))
    for i in range(n_a):
        lines.append(f"{la[i]:>{width}}" + "".join(f"{m[i, j]:>{width}.3f}" for j in range(n_b)))
    lines.append("")

    if np.isfinite(fa).any() or np.isfinite(fb).any():
        lines.append(f"{'mode':>5} {'f_A [Hz]':>12} {'f_B [Hz]':>12}")
        for i in range(max(n_a, n_b)):
            va = _fmt_hz(fa[i]) if i < n_a else "-"
            vb = _fmt_hz(fb[i]) if i < n_b else "-"
            lines.append(f"{i + 1:>5} {va:>12} {vb:>12}")
        lines.append("")

    if rows is not None:
        lines.append(
            f"{'#':>3} {'A':>5} {'B':>5} {'f_A [Hz]':>12} {'f_B [Hz]':>12} "
            f"{'df [%]':>9} {'MAC':>7}  verdict"
        )
        for k, (ia, ib, mv, pfa, pfb) in enumerate(rows, start=1):
            lines.append(
                f"{k:>3} {la[ia]:>5} {lb[ib]:>5} {_fmt_hz(pfa):>12} {_fmt_hz(pfb):>12} "
                f"{_fmt_err(pfa, pfb):>9} {mv:>7.4f}  {_verdict(mv)}"
            )
        if not rows:
            lines.append("  (no pairs above the MAC threshold)")
        lines.append("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# file writer
# ----------------------------------------------------------------------
def save_mac_report(path: str | Path, mac: Any, *, fmt: str = "auto", **kwargs: Any) -> Path:
    """Write a MAC report to *path* and return the path.

    ``fmt`` is ``"html"``, ``"text"`` or ``"auto"`` (default), which picks
    HTML for ``.html``/``.htm`` suffixes and plain text otherwise.  All
    other keyword arguments are forwarded to the generator; ``heatmap=``
    only applies to HTML output.
    """
    out = Path(path)
    choice = fmt.lower()
    if choice == "auto":
        choice = "html" if out.suffix.lower() in (".html", ".htm") else "text"
    if choice == "html":
        content = mac_report_html(mac, **kwargs)
    elif choice in ("text", "txt"):
        kwargs.pop("heatmap", None)
        content = mac_report_text(mac, **kwargs)
    else:
        raise ValueError(f"unknown report format {fmt!r}; use 'auto', 'html' or 'text'")
    out.write_text(content, encoding="utf-8")
    return out

"""Self-contained ``report.html`` (05-metrics.md section 6).

The report is forwarded around and opened without any environment, so it must be
a single file with no external references: all charts are inline SVG and all
images are base64-embedded. No matplotlib, no CDN, no JS.

Charts: position-error CDF, recall curve (A@d vs d), terrain breakdown, and a
gallery of the worst-50 failures with their correspondences.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .summary import Summary

RECALL_D_M = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100]


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=np.float64))
    if xs.size == 0:
        return np.array([]), np.array([])
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def _svg_line_chart(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    width: int = 640,
    height: int = 320,
    xlabel: str = "",
    ylabel: str = "",
    xlog: bool = False,
    xmax: float | None = None,
) -> str:
    if xs.size == 0:
        return "<p>(no data)</p>"
    left, right, top, bottom = 60, 20, 20, 50
    pw, ph = width - left - right, height - top - bottom
    x0 = float(np.min(xs))
    x1 = float(np.max(xs)) if xmax is None else xmax
    if xlog and x0 <= 0:
        x0 = max(float(np.min(xs[xs > 0])), 1e-9) if np.any(xs > 0) else 1.0
    y0, y1 = 0.0, 1.05

    def px(v):
        if xlog:
            lx0, lx1 = np.log10(x0), np.log10(max(x1, x0 * 1.0001))
            return left + (np.log10(max(v, x0)) - lx0) / (lx1 - lx0) * pw
        return left + (v - x0) / (x1 - x0) * pw

    def py(v):
        return top + (1.0 - (v - y0) / (y1 - y0)) * ph

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    # axis ticks
    ticks = ""
    for t in range(6):
        v = y0 + (y1 - y0) * t / 5
        ticks += f'<line x1="{left}" y1="{py(v):.1f}" x2="{width - right}" y2="{py(v):.1f}" '
        ticks += f'stroke="#eee"/><text x="{left - 6}" y="{py(v) + 4:.1f}" text-anchor="end" '
        ticks += f'font-size="11">{v:.1f}</text>'
    xlabel_svg = (
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12">{html.escape(xlabel)}</text>'
    )
    ylabel_svg = (
        f'<text x="14" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {(top + bottom) / 2:.1f})">{html.escape(ylabel)}</text>'
    )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#ccc"/>'
        f"{ticks}{xlabel_svg}{ylabel_svg}"
        f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
        "</svg>"
    )


def _svg_bar_chart(
    labels: list[str], values: list[float], *, width: int = 640, height: int = 300, ylabel: str = ""
) -> str:
    if not labels:
        return "<p>(no data)</p>"
    left, right, top, bottom = 70, 20, 20, 50
    pw, ph = width - left - right, height - top - bottom
    y1 = max(max(values), 1e-9) * 1.15
    bw = pw / len(labels) * 0.7
    bars = ""
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + pw * (i + 0.5) / len(labels)
        h = ph * (val / y1)
        bars += (
            f'<rect x="{x - bw / 2:.1f}" y="{top + ph - h:.1f}" width="{bw:.1f}" '
            f'height="{h:.1f}" fill="#1f77b4"/>'
        )
        bars += (
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" '
            f'font-size="11">{html.escape(lab)}</text>'
        )
        bars += (
            f'<text x="{x:.1f}" y="{top + ph - h - 4:.1f}" text-anchor="middle" '
            f'font-size="11">{val:.2f}</text>'
        )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f"{bars}"
        f'<text x="14" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {(top + bottom) / 2:.1f})">{html.escape(ylabel)}</text>'
        "</svg>"
    )


def _table(rows: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(a)}</td><td style='text-align:right'>{html.escape(b)}</td>"
        f"<td style='text-align:right'>{html.escape(c)}</td></tr>"
        for a, b, c in rows
    )
    return (
        "<table style='border-collapse:collapse'>"
        "<thead><tr><th style='text-align:left'>metric</th><th>with bias</th>"
        "<th>without bias</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _image_data_uri(path: Path, max_side: int = 220) -> str:
    try:
        from PIL import Image
    except ImportError:
        return ""
    try:
        img = Image.open(path)
        img.thumbnail((max_side, max_side))
        img = img.convert("RGB")
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "—"
        return f"{x:.{nd}f}"
    return str(x)


def render_report(
    summary: Summary, records, out_path: Path, failures: Iterable[dict] | None = None
) -> Path:
    """Write a self-contained HTML report and return its path."""
    rec = records.fixes
    accepted = rec["accepted"]
    with_bias = summary.fix_level
    without_bias = summary.fix_level_without_bias

    east_err = rec["est_east"] - rec["gt_east"]
    north_err = rec["est_north"] - rec["gt_north"]
    pos_err = np.hypot(east_err, north_err)[accepted]

    # recall curve
    recall_x = RECALL_D_M
    recall_y = [float(np.mean(pos_err <= d)) if pos_err.size else float("nan") for d in recall_x]
    recall_y = [0.0 if not np.isfinite(v) else v for v in recall_y]

    cdf_x, cdf_y = _cdf(pos_err)

    # terrain breakdown
    tlabels = list(summary.by_terrain.keys())
    tacc = [summary.by_terrain[k]["acceptance_rate"] for k in tlabels]
    ta20 = [summary.by_terrain[k]["A@20"] for k in tlabels]

    metrics_to_show = [
        "n_attempts",
        "n_accepted",
        "acceptance_rate",
        "A@5",
        "A@10",
        "A@20",
        "A@50",
        "RE_med_deg",
        "RE_p95_deg",
        "IFR",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    table_rows = [(m, _fmt(with_bias.get(m)), _fmt(without_bias.get(m))) for m in metrics_to_show]

    golden_html = ""
    if summary.golden_delta is not None:
        g = summary.golden_delta
        g_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{_fmt(v, 5)}</td></tr>"
            for k, v in g.items()
        )
        golden_html = (
            "<h3>Golden delta</h3><table style='border-collapse:collapse'>"
            f"<tbody>{g_rows}</tbody></table>"
        )

    gallery = ""
    if failures is not None:
        cards = []
        for f in list(failures)[:50]:
            patch = _image_data_uri(Path(f["patch"])) if f.get("patch") else ""
            mapimg = _image_data_uri(Path(f["map"])) if f.get("map") else ""
            imgs = ""
            if patch:
                imgs += f'<img src="{patch}" style="max-height:160px;margin:2px"/>'
            if mapimg:
                imgs += f'<img src="{mapimg}" style="max-height:160px;margin:2px"/>'
            cards.append(
                "<div style='display:inline-block;margin:6px;border:1px solid #ccc;padding:6px'>"
                f"<div style='font-size:12px'>err {_fmt(f.get('error_m'), 1)} m "
                f"| {html.escape(str(f.get('note', '')))}</div>{imgs}</div>"
            )
        gallery = "<h3>Worst 50 failures</h3>" + "".join(cards)

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>geoloc metrics — {html.escape(summary.run_id)}</title>
<style>
body{{font-family:sans-serif;max-width:1000px;margin:24px auto;color:#222}}
h1,h2,h3{{border-bottom:1px solid #ddd;padding-bottom:4px}}
table td,table th{{padding:4px 10px;border-bottom:1px solid #eee}}
</style></head><body>
<h1>geoloc metrics report</h1>
<p>run <b>{html.escape(summary.run_id)}</b> — level {html.escape(summary.level)},
target {html.escape(summary.target)} — sha {html.escape(summary.git_sha)} —
bias ({_fmt(summary.bias[0])}, {_fmt(summary.bias[1])}) m</p>
<h2>Fix-level metrics</h2>
{_table(table_rows)}
<h2>Position error CDF (accepted fixes)</h2>
{_svg_line_chart(cdf_x, cdf_y, xlabel="position error (m)", ylabel="CDF")}
<h2>Recall curve A@d</h2>
{
        _svg_line_chart(
            np.array(recall_x, dtype=float),
            np.array(recall_y),
            xlabel="d (m)",
            ylabel="A@d",
            xlog=True,
            xmax=100,
        )
    }
<h2>Terrain breakdown — acceptance rate</h2>
{_svg_bar_chart(tlabels, tacc, ylabel="acceptance_rate")}
<h2>Terrain breakdown — A@20</h2>
{_svg_bar_chart(tlabels, ta20, ylabel="A@20")}
<h2>Trajectory</h2>
<pre>{html.escape(str(summary.trajectory))}</pre>
{golden_html}
{gallery}
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path

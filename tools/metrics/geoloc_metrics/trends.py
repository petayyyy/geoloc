"""Trend accumulation: append each run's summary to a shared store and render a
static dashboard (T12 deliverable 7).

The shared store is a single CSV so it works with no pandas/pyarrow; if pandas
is present the same table is also maintained as a parquet (the "common parquet"
named in the task card). The dashboard is a self-contained HTML page with a
per-run bar chart of the key metrics.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

from .summary import KEY_METRICS

KEY_FIELDS = [p.split(".")[-1] for p in KEY_METRICS.values()]


def _summary_row(summary: dict) -> dict:
    row = {
        "run_id": summary.get("run_id", ""),
        "level": summary.get("level", ""),
        "target": summary.get("target", ""),
        "git_sha": summary.get("git_sha", ""),
    }
    fix = summary.get("fix_level", {})
    for f in KEY_FIELDS:
        row[f] = fix.get(f)
    return row


def append_summary(summary: dict, store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    csv_path = store_dir / "trends.csv"
    row = _summary_row(summary)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        df.to_parquet(store_dir / "trends.parquet", index=False)
    except Exception:
        pass

    return csv_path


def _bar(labels: list[str], values: list[float], title: str) -> str:
    if not labels:
        return ""
    w, h = 640, 220
    left, right, top, bottom = 70, 20, 20, 40
    pw, ph = w - left - right, h - top - bottom
    ymax = max(max(values), 1e-9) * 1.2
    bars = ""
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + pw * (i + 0.5) / len(labels)
        bh = ph * (val / ymax) if val is not None and val == val else 0
        bars += (
            f'<rect x="{x - pw / len(labels) * 0.35:.1f}" y="{top + ph - bh:.1f}" '
            f'width="{pw / len(labels) * 0.7:.1f}" height="{bh:.1f}" fill="#1f77b4"/>'
        )
        bars += (
            f'<text x="{x:.1f}" y="{h - 10}" text-anchor="middle" '
            f'font-size="10">{html.escape(lab[:12])}</text>'
        )
    return (
        f'<h3>{html.escape(title)}</h3><svg width="{w}" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg">{bars}</svg>'
    )


def render_dashboard(store_dir: Path, out: Path | None = None) -> Path:
    csv_path = store_dir / "trends.csv"
    out = out or (store_dir / "dashboard.html")
    if not csv_path.exists():
        out.write_text("<html><body><p>no runs yet</p></body></html>", encoding="utf-8")
        return out
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    run_ids = [r["run_id"] for r in rows]
    charts = ""
    for field in KEY_FIELDS:
        vals = []
        for r in rows:
            v = r.get(field)
            vals.append(float(v) if v not in (None, "", "None") else float("nan"))
        charts += _bar(run_ids, vals, field)
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>geoloc trends</title>"
        "<style>body{font-family:sans-serif;margin:24px;color:#222}</style></head><body>"
        f"<h1>geoloc metric trends</h1>{charts}</body></html>"
    )
    out.write_text(page, encoding="utf-8")
    return out

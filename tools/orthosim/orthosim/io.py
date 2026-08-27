"""Run outputs (02-level-a-orthosim.md §6): config, results, summary, failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(_to_uint8(arr)).save(path)


def write_config(out_dir: Path, cfg: dict) -> None:
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def write_results(out_dir: Path, records: list[dict]) -> None:
    keys = list(records[0].keys())
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: _csv_value(r[k]) for k in keys})

    try:
        import pyarrow as pa  # noqa: F401  # optional
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        return
    import pandas as pd

    pd.DataFrame(records).to_parquet(out_dir / "results.parquet", index=False)


def write_summary(
    out_dir: Path, records: list[dict], cfg: dict, matcher_error: list[float] | None
) -> None:
    n = len(records)
    summary = {
        "run_id": cfg.get("run_id", ""),
        "set": cfg.get("set"),
        "seed": cfg.get("seed"),
        "n_pairs": n,
        "cross_provider_guard": "PASS",
        "providers": cfg.get("providers"),
        "by_terrain": _terrain_counts(records),
        "matcher": None,
    }
    if matcher_error is not None:
        err = np.asarray(matcher_error)
        summary["matcher"] = {
            "n": int(len(err)),
            "error_median_m": float(np.median(err)),
            "error_p95_m": float(np.percentile(err, 95)),
            "A@20": float(np.mean(err <= 20.0)),
            "IFR@50": float(np.mean(err > 50.0)),
        }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)


def write_failures(out_dir: Path, pairs: list, n: int) -> None:
    """Visual gallery of the ``n`` worst (or first, when no matcher) pairs.

    ``pairs`` are :class:`~orthosim.pairs.RenderedPair` instances, already sorted
    worst-first by match error when a matcher is available.
    """
    fdir = out_dir / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    for pair in pairs[:n]:
        q = _to_uint8(pair.query)
        w = _to_uint8(pair.map_window)
        h = max(q.shape[0], w.shape[0])
        canvas = np.zeros((h, q.shape[1] + w.shape[1], 3), dtype=np.uint8)
        canvas[: q.shape[0], : q.shape[1]] = q
        canvas[: w.shape[0], q.shape[1] : q.shape[1] + w.shape[1]] = w
        save_png(fdir / f"pair_{pair.spec.id:05d}.png", canvas)


def _csv_value(v):
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, sort_keys=True)
    return v


def _terrain_counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in records:
        cls = r.get("terrain_class", "background")
        counts[cls] = counts.get(cls, 0) + 1
    return counts

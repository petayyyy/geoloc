"""``summary.json`` -- the machine-readable report for CI (05-metrics.md section 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from .bias import DEFAULT_BIAS
from .schema import Records
from .terrain import by_terrain

# Key metrics whose degradation CI blocks on (05-metrics.md section 7). Map a
# human name to the summary field it lives in.
KEY_METRICS = {
    "A@20": "fix_level.A@20",
    "RE_p95": "fix_level.RE_p95_deg",
    "IFR": "fix_level.IFR",
    "acceptance_rate": "fix_level.acceptance_rate",
    "latency_p95": "fix_level.latency_p95_ms",
}


@dataclass
class Summary:
    """All the numbers that go into ``summary.json``."""

    run_id: str
    level: str
    target: str
    git_sha: str = ""
    seed: int | None = None
    bias: tuple[float, float] = DEFAULT_BIAS
    fix_level: dict = field(default_factory=dict)
    fix_level_without_bias: dict = field(default_factory=dict)
    trajectory: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    by_terrain: dict = field(default_factory=dict)
    by_terrain_without_bias: dict = field(default_factory=dict)
    terrain_counts: dict = field(default_factory=dict)
    golden_delta: dict | None = None

    @classmethod
    def from_records(
        cls,
        records: Records,
        *,
        level: str = "B",
        target: str = "x86",
        git_sha: str = "",
        seed: int | None = None,
        bias: tuple[float, float] = DEFAULT_BIAS,
    ) -> Summary:
        rec = records.fixes
        fix = M.fix_level_table(rec, bias=bias, bias_mode="with")
        fix_nobias = M.fix_level_table(rec, bias=bias, bias_mode="without")
        traj = (
            M.trajectory_level_table(records.trajectory) if records.trajectory is not None else {}
        )
        return cls(
            run_id=records.run_id or "unnamed",
            level=level,
            target=target,
            git_sha=git_sha,
            seed=seed,
            bias=bias,
            fix_level=fix,
            fix_level_without_bias=fix_nobias,
            trajectory=traj,
            quality=M.quality_distribution(rec),
            by_terrain=by_terrain(rec, M.fix_level_table, bias=bias, bias_mode="with"),
            by_terrain_without_bias=by_terrain(
                rec, M.fix_level_table, bias=bias, bias_mode="without"
            ),
            terrain_counts=_terrain_counts(rec),
        )

    def to_dict(self) -> dict:
        out = {
            "run_id": self.run_id,
            "level": self.level,
            "target": self.target,
            "git_sha": self.git_sha,
            "seed": self.seed,
            "bias_east": self.bias[0],
            "bias_north": self.bias[1],
            "fix_level": self.fix_level,
            "fix_level_without_bias": self.fix_level_without_bias,
            "trajectory": self.trajectory,
            "quality": self.quality,
            "by_terrain": self.by_terrain,
            "by_terrain_without_bias": self.by_terrain_without_bias,
            "terrain_counts": self.terrain_counts,
        }
        if self.golden_delta is not None:
            out["golden_delta"] = self.golden_delta
        return out

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(self.to_dict()), f, indent=2, sort_keys=True)


def _terrain_counts(rec: np.ndarray) -> dict:
    from .terrain import terrain_counts

    return terrain_counts(rec)


def _sanitize(obj):
    """Replace NaN/Inf floats with ``None`` so ``summary.json`` is valid JSON.

    ``A@d`` / ``IFR`` are undefined when there are no accepted fixes; ``None``
    is the machine-readable way to say "not computed" (``NaN`` is not valid JSON).
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

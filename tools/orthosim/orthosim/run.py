"""Run orchestration: build a scene, generate pairs, write the run outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import io
from .camera import normalize_angle
from .geopack import GeoPack
from .pairs import PairSpec, RenderedPair, generate_pair
from .scene import Scene, classify
from .scene import build_scene as build_scene_from_geopack
from .sets import ADVERSARIAL_KINDS, preset
from .synthetic import make_synthetic_scene

KIND_CLASS = {
    "periodic": "farmland",
    "water": "water",
    "snow": "background",
    "forest": "forest",
    "stale_map": "urban",
    "symmetric": "roads",
}


@dataclass
class RunResult:
    out_dir: Path
    records: list[dict]
    pairs: list[RenderedPair]
    matcher_error: list[float] | None


def build_scene(cfg: dict, seed: int, kind: str | None = None) -> Scene:
    scene_cfg = cfg.get("scene", {})
    source = scene_cfg.get("source", "synthetic")
    if source == "geopack":
        gdir = scene_cfg.get("geopack_dir")
        if not gdir:
            raise ValueError("scene.source=geopack requires scene.geopack_dir")
        terrain = scene_cfg.get("terrain", "dem")
        return build_scene_from_geopack(GeoPack(Path(gdir)), terrain_mode=terrain)
    return make_synthetic_scene(kind or scene_cfg.get("kind", "periodic"), seed)


def _region(cfg: dict, scene) -> tuple[float, float, float, float]:
    r = cfg.get("region", {})
    if "east_min" in r:
        return (r["east_min"], r["east_max"], r["north_min"], r["north_max"])
    if isinstance(scene, Scene):
        # geopack scene: sample inside the ortho_a raster, with a small margin.
        f = scene.ortho_a
        margin = cfg.get("region_margin_m", 20.0)
        h, w = f.data.shape[:2]
        east_min = f.east_min + margin
        east_max = f.east_min + w * f.gsd - margin
        north_max = f.north_max - margin
        north_min = f.north_max - h * f.gsd + margin
        return (east_min, east_max, north_min, north_max)
    half = cfg.get("region_half_m", 150.0)
    return (-half, half, -half, half)


def _classify_scene(scene, east: float, north: float, cfg: dict) -> str:
    if isinstance(scene, Scene) and scene.semantic is not None:
        window_m = cfg.get("patch", {}).get("gsd_m", 0.5) * 64
        return classify(scene.semantic, east, north, window_m)
    return "background"


def _specs(
    rng: np.random.Generator,
    providers: dict,
    preset_cfg: dict,
    n_pairs: int,
    region: tuple[float, float, float, float],
    scene,
    kind: str | None,
    cfg: dict,
) -> list[PairSpec]:
    e_min, e_max, n_min, n_max = region
    pos_err = preset_cfg.get("prior_pos_err_m", 15.0)
    yaw_std = preset_cfg.get("prior_yaw_std_rad", 0.05)
    augment = preset_cfg.get("augment", {})
    specs = []
    for i in range(n_pairs):
        gt_east = rng.uniform(e_min, e_max)
        gt_north = rng.uniform(n_min, n_max)
        gt_yaw = rng.uniform(-np.pi, np.pi)
        th = rng.uniform(-np.pi, np.pi)
        d = rng.uniform(0.0, pos_err)
        prior_east = gt_east + d * np.cos(th)
        prior_north = gt_north + d * np.sin(th)
        prior_yaw = normalize_angle(gt_yaw + rng.uniform(-yaw_std, yaw_std))
        if kind:
            terrain_class = KIND_CLASS.get(kind, "background")
        else:
            terrain_class = _classify_scene(scene, gt_east, gt_north, cfg)
        specs.append(
            PairSpec(
                id=i,
                gt_east=float(gt_east),
                gt_north=float(gt_north),
                gt_yaw=float(gt_yaw),
                prior_east=float(prior_east),
                prior_north=float(prior_north),
                prior_yaw=float(prior_yaw),
                query_provider=providers["query"],
                map_provider=providers["map"],
                augment=dict(augment),
                terrain_class=terrain_class,
            )
        )
    return specs


def run(cfg: dict, matcher=None) -> RunResult:
    seed = int(cfg.get("seed", 42))
    set_name = cfg.get("set", "smoke")
    rng = np.random.default_rng(seed)
    preset_cfg = preset(set_name)
    n_pairs = int(cfg.get("n_pairs") or preset_cfg.get("n_pairs", 20))
    providers = cfg.get("providers", {"query": "b", "map": "a"})
    patch = cfg.get("patch", {})
    gsd = float(patch.get("gsd_m", 0.5))
    patch_size = int(patch.get("patch_size", 128))
    window_size = int(patch.get("window_size", 256))
    aa = int(patch.get("aa", 2))
    query_mode = cfg.get("query_mode", "direct")

    records: list[dict] = []
    pairs: list[RenderedPair] = []
    matcher_error: list[float] | None = [] if matcher is not None else None

    emit = lambda specs, scene, kind: _emit(  # noqa: E731
        specs,
        scene,
        cfg,
        gsd,
        patch_size,
        window_size,
        aa,
        query_mode,
        rng,
        matcher,
        kind,
        records,
        pairs,
        matcher_error,
    )

    if set_name == "adversarial":
        kinds = cfg.get("kinds") or ADVERSARIAL_KINDS
        per = int(cfg.get("n_per_kind") or preset_cfg.get("n_per_kind", 200))
        for kind in kinds:
            scene = build_scene(cfg, seed, kind)
            region = _region(cfg, scene)
            specs = _specs(rng, providers, preset_cfg, per, region, scene, kind, cfg)
            emit(specs, scene, kind)
    else:
        scene = build_scene(cfg, seed)
        scene_cfg = cfg.get("scene", {})
        kind = None
        if scene_cfg.get("source", "synthetic") == "synthetic":
            kind = scene_cfg.get("kind")
        region = _region(cfg, scene)
        specs = _specs(rng, providers, preset_cfg, n_pairs, region, scene, kind, cfg)
        emit(specs, scene, kind)

    out_dir = Path(cfg.get("out_dir", f"runs/{cfg.get('run_id', 'run')}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_config(out_dir, cfg)
    io.write_results(out_dir, records)
    io.write_summary(out_dir, records, cfg, matcher_error)
    if matcher_error is not None:
        order = np.argsort(-np.asarray(matcher_error))
        pairs = [pairs[i] for i in order]
    io.write_failures(out_dir, pairs, int(cfg.get("failures_n", 50)))
    return RunResult(out_dir=out_dir, records=records, pairs=pairs, matcher_error=matcher_error)


def _emit(
    specs,
    scene,
    cfg,
    gsd,
    patch_size,
    window_size,
    aa,
    query_mode,
    rng,
    matcher,
    kind,
    records,
    pairs,
    matcher_error,
):
    for spec in specs:
        pair = generate_pair(
            scene, spec, rng, gsd, patch_size, window_size, aa=aa, query_mode=query_mode
        )
        rec = {
            "id": spec.id,
            "set": cfg.get("set"),
            "kind": kind,
            "terrain_class": spec.terrain_class,
            "gt_east": spec.gt_east,
            "gt_north": spec.gt_north,
            "gt_yaw": spec.gt_yaw,
            "prior_east": spec.prior_east,
            "prior_north": spec.prior_north,
            "prior_yaw": spec.prior_yaw,
            "delta_east": spec.delta[0],
            "delta_north": spec.delta[1],
            "delta_yaw": spec.delta[2],
            "provider_query": spec.query_provider,
            "provider_map": spec.map_provider,
            "augment": spec.augment,
        }
        if matcher is not None:
            est = matcher(pair.query, pair.map_window, gsd)
            err = float(np.hypot(est[0] - spec.delta[0], est[1] - spec.delta[1]))
            rec["match_east"] = est[0]
            rec["match_north"] = est[1]
            rec["match_yaw"] = est[2]
            rec["error_m"] = err
            matcher_error.append(err)
        records.append(rec)
        pairs.append(pair)

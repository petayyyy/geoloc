"""Command-line interface: orthosim run --config <yaml>."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from .run import run


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def _cmd_run(args) -> int:
    cfg = _load_config(Path(args.config))
    cfg.setdefault("run_id", f"{cfg.get('set', 'smoke')}_{cfg.get('seed', 42)}")
    if cfg.get("out_dir") is None:
        cfg["out_dir"] = f"runs/{cfg['run_id']}"
    t0 = time.time()
    result = run(cfg)
    print(
        f"orthosim: set={cfg['set']} pairs={len(result.records)} "
        f"out={result.out_dir} ({time.time() - t0:.1f}s)"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="orthosim", description="T10/T11 OrthoSim pair generator")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("run", help="generate a set")
    p.add_argument("--config", required=True, help="run config YAML")
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

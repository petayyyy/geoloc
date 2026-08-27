"""Command line interface: ``geoloc-metrics report | compare | trends``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .golden import compare, load_golden
from .report import render_report
from .schema import load_records
from .summary import Summary
from .trends import append_summary, render_dashboard


def _cmd_report(args) -> int:
    records = load_records(Path(args.records))
    summary = Summary.from_records(
        records,
        level=args.level,
        target=args.target,
        git_sha=args.git_sha,
        seed=args.seed,
        bias=(args.bias_east, args.bias_north),
    )
    if args.golden:
        summary.golden_delta = compare(summary.to_dict(), load_golden(Path(args.golden)))
    out = Path(args.out_dir)
    summary.write(out / "summary.json")
    render_report(summary, records, out / "report.html")
    print(f"summary.json + report.html written to {out}")
    return 0


def _cmd_compare(args) -> int:
    current = load_golden(Path(args.current))
    golden = load_golden(Path(args.golden))
    delta = compare(current, golden)
    import json

    print(json.dumps(delta, indent=2))
    return 0 if delta["verdict"] == "PASS" else 1


def _cmd_trends(args) -> int:
    import json

    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)
    store = Path(args.store_dir)
    append_summary(summary, store)
    dashboard = render_dashboard(store)
    print(f"appended to {store / 'trends.csv'}; dashboard at {dashboard}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-metrics", description="T12 metrics harness")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("report", help="records -> summary.json + report.html")
    p.add_argument("records", help="records base path (fixes/trajectory CSV)")
    p.add_argument("--out-dir", default="runs/latest")
    p.add_argument("--level", default="B")
    p.add_argument("--target", default="x86")
    p.add_argument("--git-sha", default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--bias-east", type=float, default=0.0)
    p.add_argument("--bias-north", type=float, default=0.0)
    p.add_argument("--golden", help="golden summary.json to compare against")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("compare", help="compare two summary.json (golden gate)")
    p.add_argument("current")
    p.add_argument("golden")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("trends", help="append a summary and refresh the dashboard")
    p.add_argument("summary")
    p.add_argument("--store-dir", default="runs/trends")
    p.set_defaults(func=_cmd_trends)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

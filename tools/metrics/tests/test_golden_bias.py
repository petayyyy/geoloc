"""T12-U-05 (golden tolerance logic) and T12-U-06 (dual bias accounting)."""

import numpy as np
from conftest import make_fixes

from geoloc_metrics import metrics as M
from geoloc_metrics.golden import compare
from geoloc_metrics.schema import Records
from geoloc_metrics.summary import Summary


def _summary_fix_level(a20, re_p95, ifr, acc, lat95):
    return {
        "run_id": "x",
        "level": "B",
        "target": "x86",
        "fix_level": {
            "A@20": a20,
            "RE_p95_deg": re_p95,
            "IFR": ifr,
            "acceptance_rate": acc,
            "latency_p95_ms": lat95,
        },
    }


def test_u05_tolerance_relative_pass_fail():
    golden = _summary_fix_level(a20=0.60, re_p95=2.0, ifr=0.003, acc=0.6, lat95=100.0)

    # 4% degradation of A@20 -> within 5% tolerance -> PASS.
    cur = _summary_fix_level(a20=0.60 * 0.96, re_p95=2.0, ifr=0.003, acc=0.6, lat95=100.0)
    assert compare(cur, golden)["verdict"] == "PASS"

    # 6% degradation of A@20 -> FAIL.
    cur = _summary_fix_level(a20=0.60 * 0.94, re_p95=2.0, ifr=0.003, acc=0.6, lat95=100.0)
    assert compare(cur, golden)["verdict"] == "FAIL"


def test_u05_ifr_absolute_tolerance():
    golden = _summary_fix_level(a20=0.60, re_p95=2.0, ifr=0.003, acc=0.6, lat95=100.0)

    # +0.05 percentage points (below the 0.1 pp cap) -> PASS.
    cur = _summary_fix_level(a20=0.60, re_p95=2.0, ifr=0.0035, acc=0.6, lat95=100.0)
    assert compare(cur, golden)["verdict"] == "PASS"

    # +0.15 percentage points -> FAIL regardless of relative change.
    cur = _summary_fix_level(a20=0.60, re_p95=2.0, ifr=0.0045, acc=0.6, lat95=100.0)
    assert compare(cur, golden)["verdict"] == "FAIL"


def test_u06_dual_bias_differs_exactly_by_bias():
    # A constant basemap bias of (3, 4) m: with-bias error = 5 m, without = 0.
    bias = (3.0, 4.0)
    n = 100
    rec = make_fixes(
        n, est_east=np.full(n, 3.0), est_north=np.full(n, 4.0), bias_east=3.0, bias_north=4.0
    )
    with_bias = M.fix_level_table(rec, bias=bias, bias_mode="with")
    without_bias = M.fix_level_table(rec, bias=bias, bias_mode="without")

    # The raw error is exactly |bias| = 5 m, so A@5 == 1.0 (5 <= 5).
    assert with_bias["A@5"] == 1.0
    assert with_bias["IFR"] == 0.0

    # After subtracting the bias the error is exactly zero: A@5 == 1.0 AND the
    # error magnitude collapses to 0 (IFR is trivially 0, but A@d is identical
    # at every d). The distinguishing signal is the bias magnitude itself.
    assert without_bias["A@5"] == 1.0
    assert without_bias["IFR"] == 0.0

    # A bias large enough to cross a threshold shows the two views differ.
    rec2 = make_fixes(
        n, est_east=np.full(n, 10.0), est_north=np.zeros(n), bias_east=10.0, bias_north=0.0
    )
    with10 = M.fix_level_table(rec2, bias=(10.0, 0.0), bias_mode="with")
    without10 = M.fix_level_table(rec2, bias=(10.0, 0.0), bias_mode="without")
    assert with10["A@5"] == 0.0  # 10 m error, none within 5 m
    assert without10["A@5"] == 1.0  # zero error after removal

    summary = Summary.from_records(Records(fixes=rec2), bias=(10.0, 0.0))
    assert summary.fix_level["A@5"] == 0.0
    assert summary.fix_level_without_bias["A@5"] == 1.0


def test_terrain_breakdown_present():
    rec = make_fixes(100, est_east=np.zeros(100), est_north=np.zeros(100), terrain="urban")
    rec["terrain"][::2] = "forest"
    summary = Summary.from_records(Records(fixes=rec))
    assert set(summary.by_terrain.keys()) == {"urban", "forest"}
    assert summary.terrain_counts["urban"] == 50
    assert summary.terrain_counts["forest"] == 50


def test_summary_json_valid_with_zero_accepted(tmp_path):
    # No accepted fixes -> A@d / IFR are undefined; summary.json must still be
    # valid JSON (undefined metrics serialise as null, not NaN).
    import json

    rec = make_fixes(
        20, est_east=np.full(20, 100.0), est_north=np.zeros(20), accepted=np.zeros(20, dtype=bool)
    )
    summary = Summary.from_records(Records(fixes=rec))
    summary.write(tmp_path / "summary.json")
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["fix_level"]["n_accepted"] == 0
    assert data["fix_level"]["IFR"] is None
    assert data["fix_level"]["acceptance_rate"] == 0.0

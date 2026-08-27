"""Golden-run comparison with degradation tolerances (05-metrics.md section 7).

The tolerances are asymmetric on purpose:

* any key metric that degrades more than **5% relative** to the golden run
  fails the CI check;
* ``IFR`` additionally gets a tightened **absolute** tolerance: +0.1 percentage
  point (0.001) fails regardless of the relative change -- an accepted-false-fix
  rate is a safety number, not an accuracy number.

Golden runs may only be updated by an explicit commit; auto-updating is
forbidden (the harness never writes a golden, it only compares).
"""

from __future__ import annotations

import json
from pathlib import Path

from .summary import KEY_METRICS

REL_TOL = 0.05
IFR_ABS_TOL = 0.001

# Which key metrics are "higher is better" (the rest are "lower is better").
HIGHER_BETTER = {"A@20", "acceptance_rate"}


def _get_nested(table: dict, dotted_path: str):
    node = table
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def compare(
    current: dict, golden: dict, *, rel_tol: float = REL_TOL, ifr_abs_tol: float = IFR_ABS_TOL
) -> dict:
    """Return a ``golden_delta`` dict: per-key-metric delta and a verdict.

    ``delta`` is ``current - golden`` (signed), matching the section 6 example.
    The verdict is ``PASS`` unless any key metric degrades past its tolerance.
    """
    deltas: dict = {}
    verdict = "PASS"
    for name, path in KEY_METRICS.items():
        cur = _get_nested(current, path)
        gold = _get_nested(golden, path)
        if cur is None or gold is None:
            deltas[name] = None
            continue
        delta = cur - gold
        deltas[name] = delta

        if name == "IFR":
            if delta > ifr_abs_tol:
                verdict = "FAIL"
        elif gold is not None:
            if name in HIGHER_BETTER:
                if cur < gold * (1.0 - rel_tol):
                    verdict = "FAIL"
            else:
                if cur > gold * (1.0 + rel_tol):
                    verdict = "FAIL"

    deltas["verdict"] = verdict
    return deltas


def load_golden(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

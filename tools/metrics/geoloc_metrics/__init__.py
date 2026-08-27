"""geoloc_metrics -- the single implementation of every metric in
``docs/plan/testing/05-metrics.md`` (task T12).

One metric, one implementation, used by every test level. The harness consumes a
common, level-independent record format (``schema.py``) so that a result run
through level A and a result run through level B produce comparable numbers --
the argument about "whose number to believe" has no resolution otherwise.
"""

__version__ = "0.1.0"

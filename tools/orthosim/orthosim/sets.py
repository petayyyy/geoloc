"""Named OrthoSim sets (02-level-a-orthosim.md §5) and deterministic spec lists.

The counts below are the plan's targets; ``smoke`` defaults to a small number so
a single commit check stays under a minute, and the full sizes are selected via
``n_pairs`` in the run config.
"""

from __future__ import annotations

ADVERSARIAL_KINDS = ["periodic", "water", "snow", "forest", "stale_map", "symmetric"]

PRESETS: dict[str, dict] = {
    "smoke": {
        "n_pairs": 20,
        "prior_pos_err_m": 15.0,
        "prior_yaw_std_rad": 0.05,
        "augment": {},
    },
    "regression": {
        "n_pairs": 3000,
        "prior_pos_err_m": 15.0,
        "prior_yaw_std_rad": 0.05,
        "augment": {
            "gamma": 1.05,
            "white_balance": [1.03, 0.99, 1.01],
            "contrast": 1.05,
            "gaussian_noise": 3.0,
        },
    },
    "adversarial": {
        "n_per_kind": 200,
        "prior_pos_err_m": 15.0,
        "prior_yaw_std_rad": 0.05,
        "augment": {},
    },
    "sweep_altitude": {
        "n_pairs": 1500,
        "prior_pos_err_m": 15.0,
        "prior_yaw_std_rad": 0.05,
        "augment": {},
    },
    "sweep_domain": {
        "n_pairs": 2500,
        "prior_pos_err_m": 15.0,
        "prior_yaw_std_rad": 0.05,
        "augment": {},
    },
    "sweep_degradation": {
        "n_pairs": 2000,
        "prior_pos_err_m": 30.0,
        "prior_yaw_std_rad": 0.10,
        "augment": {"gaussian_noise": 5.0},
    },
}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"unknown set {name!r}; available: {sorted(PRESETS)}")
    return dict(PRESETS[name])

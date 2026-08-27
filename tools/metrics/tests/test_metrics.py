"""T12-U-01 .. T12-U-04 and T12-I-01: the metric functions themselves."""

import numpy as np
import pytest
from conftest import make_fixes, make_trajectory

from geoloc_metrics import metrics as M
from geoloc_metrics.schema import Records, covariance_3x3
from geoloc_metrics.summary import Summary


def test_u01_a_at_d_exact():
    # T12-U-01: a known error distribution -> exact A@d.
    err = np.arange(1, 101, dtype=np.float64)  # errors 1..100 m
    # Build fixes whose accepted errors are exactly err.
    rec = make_fixes(100, est_east=err, est_north=np.zeros(100))
    acc_err = M.position_error(rec["est_east"] - rec["gt_east"], rec["est_north"] - rec["gt_north"])
    assert M.a_at_d(acc_err, 5.0) == pytest.approx(5 / 100)
    assert M.a_at_d(acc_err, 20.0) == pytest.approx(20 / 100)
    assert M.a_at_d(acc_err, 50.0) == pytest.approx(50 / 100)


def test_u02_ifr():
    # T12-U-02: N known-false accepted fixes -> IFR = N / accepted.
    n_accepted = 1000
    err = np.full(n_accepted, 5.0)  # 970 good
    err[:30] = 80.0  # 30 false (error > 50 m)
    assert M.ifr(err) == pytest.approx(30 / n_accepted)


def test_u03_ate_rmse():
    # T12-U-03: a trajectory with a known constant offset -> ATE_RMSE = offset.
    n = 1000
    gt_e = np.linspace(0, 500, n)
    gt_n = np.zeros(n)
    traj = make_trajectory(
        n, gt_east=gt_e, gt_north=gt_n, est_east=gt_e + 3.0, est_north=gt_n + 4.0
    )
    assert M.ate_rmse(
        M.position_error(traj["est_east"] - traj["gt_east"], traj["est_north"] - traj["gt_north"])
    ) == pytest.approx(5.0)


def test_u04_nees():
    # T12-U-04: perfectly calibrated covariance -> NEES ~ DoF; understated -> inflated.
    rng = np.random.default_rng(0)
    n = 20000
    sigma = 2.0
    cov_true = np.tile(
        np.array([[sigma**2, 0.0, 0.0], [0.0, sigma**2, 0.0], [0.0, 0.0, 0.01]]), (n, 1, 1)
    )
    err_e = rng.normal(0, sigma, n)
    err_n = rng.normal(0, sigma, n)
    err_y = rng.normal(0, 0.1, n)
    nees_good = M.nees_mean(err_e, err_n, err_y, cov_true, dof=2)
    assert nees_good == pytest.approx(2.0, rel=0.05)

    cov_small = cov_true / 4.0  # understated covariance
    nees_bad = M.nees_mean(err_e, err_n, err_y, cov_small, dof=2)
    assert nees_bad > nees_good
    assert nees_bad == pytest.approx(8.0, rel=0.1)


def test_lateral_p95():
    # Heading due east (yaw=0); a pure-north error is fully lateral.
    n = 1000
    gt_yaw = np.zeros(n)
    east_err = np.zeros(n)
    north_err = np.full(n, 30.0)
    assert M.lateral_p95(east_err, north_err, gt_yaw) == pytest.approx(30.0)


def test_i01_level_uniformity():
    # T12-I-01: the same set fed as level A and level B gives identical metrics.
    rng = np.random.default_rng(1)
    n = 500
    err_e = rng.normal(0, 8, n)
    err_n = rng.normal(0, 8, n)
    a = make_fixes(n, est_east=err_e, est_north=err_n, level="A")
    b = make_fixes(n, est_east=err_e, est_north=err_n, level="B")
    sa = Summary.from_records(Records(fixes=a), level="A")
    sb = Summary.from_records(Records(fixes=b), level="B")
    for k in ("A@20", "IFR", "acceptance_rate", "RE_med_deg", "RE_p95_deg"):
        assert sa.fix_level[k] == pytest.approx(sb.fix_level[k])


def test_acceptance_rate_conditioned():
    # A@d and IFR are over ACCEPTED fixes only; acceptance_rate is separate.
    rec = make_fixes(
        100,
        est_east=np.full(100, 80.0),  # every fix 80 m off (would be "false")
        est_north=np.zeros(100),
        accepted=np.array([True] * 10 + [False] * 90),
    )
    accepted = rec["accepted"]
    pos_err = M.position_error(rec["est_east"] - rec["gt_east"], rec["est_north"] - rec["gt_north"])
    acc_err = pos_err[accepted]
    assert M.a_at_d(acc_err, 20.0) == 0.0  # none of the accepted are within 20 m
    assert M.ifr(acc_err) == 1.0  # all accepted are > 50 m
    assert M.acceptance_rate(int(np.sum(accepted)), len(rec)) == pytest.approx(0.1)


def test_covariance_roundtrip():
    rec = make_fixes(3, est_east=np.zeros(3), est_north=np.zeros(3))
    rec["cov_ee"] = [1.0, 2.0, 3.0]
    rec["cov_nn"] = [4.0, 5.0, 6.0]
    rec["cov_yy"] = [7.0, 8.0, 9.0]
    rec["cov_en"] = [0.1, 0.2, 0.3]
    cov = covariance_3x3(rec)
    assert cov.shape == (3, 3, 3)
    assert cov[0, 0, 0] == 1.0
    assert cov[0, 1, 1] == 4.0
    assert cov[0, 2, 2] == 7.0
    assert cov[0, 0, 1] == cov[0, 1, 0] == 0.1


def test_records_roundtrip(tmp_path):
    from geoloc_metrics.schema import load_records, save_records

    rec = make_fixes(10, est_east=np.arange(10.0), est_north=np.zeros(10))
    traj = make_trajectory(10, gt_east=np.arange(10.0), gt_north=np.zeros(10))
    records = Records(fixes=rec, trajectory=traj)
    save_records(tmp_path / "records", records)
    assert (tmp_path / "records.fixes.csv").exists()
    assert (tmp_path / "records.trajectory.csv").exists()
    loaded = load_records(tmp_path / "records")
    assert len(loaded.fixes) == 10
    assert len(loaded.trajectory) == 10
    assert np.allclose(loaded.fixes["est_east"], rec["est_east"])

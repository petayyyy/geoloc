// Covariance validation and helpers for the (east, north, yaw) state.
//
// WHY THIS EXISTS AS A SEPARATE HEADER:
//
// The chain is
//   matching quality -> covariance (T21) -> eph in GPS_INPUT -> EKF2's trust.
// Understate it and the autopilot believes a bad fix. Overstate it and it
// ignores a good one. ADR-006 makes integrity the top priority, and an honest
// covariance is half of that.
//
// Two mistakes account for most failures of checkpoint KT-3:
//
//  1. Forgetting the additive systematic term. A pure Sigma ~ 1/n_inliers
//     model yields centimetre sigma at high inlier counts -- impossible on a
//     satellite basemap with 2-5 m georeferencing bias. However many inliers
//     you have, they cannot reduce the MAP's own error. See addSystematic().
//
//  2. Isotropic covariance. Heading is constrained by inlier DISTRIBUTION in a
//     completely different way than position. An isotropic model
//     simultaneously overstates one and understates the other.

#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace geoloc {

using Cov3 = Eigen::Matrix3d;  // row/col order: (east, north, yaw)

/// Symmetric to within tolerance.
inline bool isSymmetric(const Cov3& C, double eps = 1e-9) noexcept {
  return (C - C.transpose()).cwiseAbs().maxCoeff() <= eps;
}

/// Positive definite (all eigenvalues > eps). Uses LDLT rather than a full
/// eigen-decomposition: cheap enough for a debug-build assertion in the loop.
inline bool isPositiveDefinite(const Cov3& C, double eps = 1e-12) noexcept {
  if (!isSymmetric(C, 1e-9)) return false;
  Eigen::SelfAdjointEigenSolver<Cov3> es(C, Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) return false;
  return es.eigenvalues().minCoeff() > eps;
}

inline bool isValidCovariance(const Cov3& C, double eps = 1e-12) noexcept {
  return C.allFinite() && isPositiveDefinite(C, eps);
}

/// Force exact symmetry. Cheap repair for accumulated float asymmetry;
/// it does NOT fix a non-positive-definite matrix.
inline Cov3 symmetrize(const Cov3& C) noexcept { return 0.5 * (C + C.transpose()); }

/// Add the basemap georeferencing systematic (T09) to the position block.
///
/// THIS TERM DOES NOT SHRINK WITH INLIER COUNT. That is the entire point:
/// the matcher cannot measure away the map's own error.
inline Cov3 addSystematic(const Cov3& C, double sigma_bias_m) noexcept {
  Cov3 out = C;
  const double v = sigma_bias_m * sigma_bias_m;
  out(0, 0) += v;
  out(1, 1) += v;
  return out;
}

/// Enforce a floor on the diagonal, so no stage can claim implausibly small
/// uncertainty. Applied after all quality-based scaling.
inline Cov3 applyFloor(const Cov3& C, double min_sigma_pos_m,
                       double min_sigma_yaw_rad) noexcept {
  Cov3 out = C;
  out(0, 0) = std::max(out(0, 0), min_sigma_pos_m * min_sigma_pos_m);
  out(1, 1) = std::max(out(1, 1), min_sigma_pos_m * min_sigma_pos_m);
  out(2, 2) = std::max(out(2, 2), min_sigma_yaw_rad * min_sigma_yaw_rad);
  return out;
}

/// Horizontal position sigma -- exactly the value that becomes `eph` in
/// GPS_INPUT. Never smooth it: a sharp rise entering COASTING is the truth
/// EKF2 must learn immediately.
inline double ephFromCovariance(const Cov3& C) noexcept {
  return std::sqrt(C(0, 0) + C(1, 1));
}

/// Prior window radius for the matcher (ADR-005):
///   R = k * sqrt(sigma_e^2 + sigma_n^2) + margin,  k = 3 covers 99.7%
inline double priorWindowRadius(const Cov3& C, double margin_m, double r_min,
                                double r_max, double k = 3.0) noexcept {
  const double r = k * ephFromCovariance(C) + margin_m;
  return r < r_min ? r_min : (r > r_max ? r_max : r);
}

/// Squared Mahalanobis distance of a residual -- the chi-squared gate statistic.
/// With 3 DoF the 99% threshold is 11.34.
inline double mahalanobisSq(const Eigen::Vector3d& residual, const Cov3& C) {
  return residual.transpose() * C.ldlt().solve(residual);
}

/// Normalised Estimation Error Squared. The ONLY way to know whether the
/// covariance is honest: mean NEES over a Monte-Carlo set must land inside the
/// 95% chi-squared interval for 3 DoF. Calibrate T21 against this, NOT against
/// A@20 -- and never by tweaking constants in geoloc_mavlink.
inline double nees(const Eigen::Vector3d& error, const Cov3& C) {
  return mahalanobisSq(error, C);
}

}  // namespace geoloc

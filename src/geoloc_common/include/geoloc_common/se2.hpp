// SE(2) transform: the estimation model of this project.
//
// ADR-002: we estimate 3 DoF (dx, dy, dyaw) and do NOT estimate scale.
// Scale is directly observable -- AGL from the lidar, surface height from the
// DSM. On a near-planar nadir scene the full homography is over-parameterised
// and unstable at low inlier counts; OrthoLoC shows PnP degrading below 20%
// covisibility and exposes an f-t_z ambiguity. Adding a degenerate DoF drags
// the others with it.

#pragma once

#include <Eigen/Dense>

#include "geoloc_common/angles.hpp"

namespace geoloc {

/// Rigid transform in the plane. Stored as (translation, yaw) rather than a
/// matrix so that yaw normalisation is enforced at construction and after
/// every composition.
class SE2 {
 public:
  SE2() : t_(Eigen::Vector2d::Zero()), yaw_(0.0) {}
  SE2(double x, double y, double yaw) : t_(x, y), yaw_(normalizeAngle(yaw)) {}
  SE2(const Eigen::Vector2d& t, double yaw) : t_(t), yaw_(normalizeAngle(yaw)) {}

  const Eigen::Vector2d& translation() const noexcept { return t_; }
  double x() const noexcept { return t_.x(); }
  double y() const noexcept { return t_.y(); }
  double yaw() const noexcept { return yaw_; }

  Eigen::Matrix2d rotation() const noexcept {
    const double c = std::cos(yaw_), s = std::sin(yaw_);
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    return R;
  }

  Eigen::Matrix3d matrix() const noexcept {
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    M.topLeftCorner<2, 2>() = rotation();
    M.topRightCorner<2, 1>() = t_;
    return M;
  }

  /// Composition: this * rhs.
  SE2 operator*(const SE2& rhs) const noexcept {
    return SE2(t_ + rotation() * rhs.t_, yaw_ + rhs.yaw_);
  }

  /// Apply to a point.
  Eigen::Vector2d operator*(const Eigen::Vector2d& p) const noexcept {
    return rotation() * p + t_;
  }

  SE2 inverse() const noexcept {
    const Eigen::Matrix2d Rt = rotation().transpose();
    return SE2(-(Rt * t_), -yaw_);
  }

  /// Relative transform from this to other: this^-1 * other.
  SE2 between(const SE2& other) const noexcept { return inverse() * other; }

  /// Log map to the tangent space (x, y, yaw). Used for residuals and for
  /// building factor-graph error terms.
  Eigen::Vector3d log() const noexcept { return {t_.x(), t_.y(), yaw_}; }

  static SE2 identity() noexcept { return SE2(); }

  bool isApprox(const SE2& o, double eps = 1e-9) const noexcept {
    return (t_ - o.t_).norm() < eps && std::abs(angleDiff(yaw_, o.yaw_)) < eps;
  }

 private:
  Eigen::Vector2d t_;
  double yaw_;  // always normalised to [-pi, pi]
};

}  // namespace geoloc

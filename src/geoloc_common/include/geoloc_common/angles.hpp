// Angle utilities.
//
// Angle normalisation is the source of half the bugs in SE(2) systems.
// Everything here is tested with property tests, not by eye.
//
// Convention (plan/03-interfaces.md section 5): radians internally, degrees
// only in diagnostics; ENU yaw measured counter-clockwise from East;
// normalise to [-pi, pi] ALWAYS.

#pragma once

#include <cmath>

namespace geoloc {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2.0 * kPi;

/// Normalise an angle to [-pi, pi].
///
/// Uses remainder() rather than a fmod-and-fix chain: remainder() is exact for
/// the rounding step and does not accumulate error for large inputs, which
/// matters because yaw accumulates over a 10 km route.
inline double normalizeAngle(double a) noexcept {
  const double r = std::remainder(a, kTwoPi);
  // remainder() returns a value in [-pi, pi]; guard the boundary so that
  // exactly -pi maps to +pi consistently, giving a single representation.
  return (r == -kPi) ? kPi : r;
}

/// Smallest signed difference a - b, normalised to [-pi, pi].
inline double angleDiff(double a, double b) noexcept {
  return normalizeAngle(a - b);
}

/// Circular mean of a set of angles. Returns 0 for an empty range.
///
/// The naive arithmetic mean is wrong near the wrap point: mean(179 deg,
/// -179 deg) is 180 deg, not 0. We need this for heading convergence
/// diagnostics, where fixes cluster near an arbitrary heading.
template <typename It>
inline double circularMean(It begin, It end) noexcept {
  double sx = 0.0, sy = 0.0;
  std::size_t n = 0;
  for (It it = begin; it != end; ++it, ++n) {
    sx += std::cos(*it);
    sy += std::sin(*it);
  }
  if (n == 0 || (sx == 0.0 && sy == 0.0)) return 0.0;
  return std::atan2(sy, sx);
}

inline constexpr double rad2deg(double r) noexcept { return r * 180.0 / kPi; }
inline constexpr double deg2rad(double d) noexcept { return d * kPi / 180.0; }

}  // namespace geoloc

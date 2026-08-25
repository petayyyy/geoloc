// WGS84 <-> local ENU conversion.
//
// The global frame of this project is a local ENU tangent plane anchored at a
// mission reference point (lat0, lon0, h0). The reference point is a MISSION
// PARAMETER, never a constant in code.
//
// Accuracy requirement (T01-U-01): round-trip error < 1e-6 m at ranges up to
// 20 km from the anchor -- comfortably covering the 5-10 km route.

#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace geoloc {

// WGS84 ellipsoid
inline constexpr double kWgs84A = 6378137.0;             // semi-major axis, m
inline constexpr double kWgs84F = 1.0 / 298.257223563;   // flattening
inline constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);  // first eccentricity squared

/// Geodetic coordinates. Angles in RADIANS (project convention); alt is
/// ellipsoidal height, not orthometric.
///
/// Beware: Copernicus DEM ships orthometric heights over EGM2008, while GNSS
/// and lidar work in ellipsoidal heights. The difference is tens of metres at
/// mid-latitudes. T06 converts; do not mix them here.
struct Geodetic {
  double lat{0.0};
  double lon{0.0};
  double alt{0.0};
};

/// Local ENU tangent plane anchored at a geodetic reference point.
class LocalEnu {
 public:
  LocalEnu() = default;

  explicit LocalEnu(const Geodetic& origin) { setOrigin(origin); }

  void setOrigin(const Geodetic& origin) {
    origin_ = origin;
    ecef_origin_ = geodeticToEcef(origin);
    const double sLat = std::sin(origin.lat), cLat = std::cos(origin.lat);
    const double sLon = std::sin(origin.lon), cLon = std::cos(origin.lon);
    // Rows: East, North, Up expressed in ECEF.
    R_ecef_to_enu_ << -sLon,          cLon,         0.0,
                      -sLat * cLon,  -sLat * sLon,  cLat,
                       cLat * cLon,   cLat * sLon,  sLat;
  }

  const Geodetic& origin() const noexcept { return origin_; }

  Eigen::Vector3d toEnu(const Geodetic& g) const {
    return R_ecef_to_enu_ * (geodeticToEcef(g) - ecef_origin_);
  }

  Geodetic fromEnu(const Eigen::Vector3d& enu) const {
    return ecefToGeodetic(ecef_origin_ + R_ecef_to_enu_.transpose() * enu);
  }

  static Eigen::Vector3d geodeticToEcef(const Geodetic& g) {
    const double sLat = std::sin(g.lat), cLat = std::cos(g.lat);
    const double sLon = std::sin(g.lon), cLon = std::cos(g.lon);
    const double N = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sLat * sLat);
    return {(N + g.alt) * cLat * cLon,
            (N + g.alt) * cLat * sLon,
            (N * (1.0 - kWgs84E2) + g.alt) * sLat};
  }

  /// ECEF -> geodetic via Bowring's method with one Newton refinement.
  ///
  /// Bowring alone is good to well under a millimetre for terrestrial
  /// altitudes; the refinement pushes the round-trip below the 1e-6 m
  /// requirement with margin.
  static Geodetic ecefToGeodetic(const Eigen::Vector3d& p) {
    const double x = p.x(), y = p.y(), z = p.z();
    const double lon = std::atan2(y, x);
    const double r = std::hypot(x, y);

    // Degenerate case: on the polar axis.
    if (r < 1e-12) {
      const double sign = (z >= 0.0) ? 1.0 : -1.0;
      const double b = kWgs84A * (1.0 - kWgs84F);
      return {sign * kPiHalf(), lon, std::abs(z) - b};
    }

    const double b = kWgs84A * (1.0 - kWgs84F);
    const double ep2 = (kWgs84A * kWgs84A - b * b) / (b * b);
    const double th = std::atan2(kWgs84A * z, b * r);
    const double s3 = std::sin(th) * std::sin(th) * std::sin(th);
    const double c3 = std::cos(th) * std::cos(th) * std::cos(th);
    double lat = std::atan2(z + ep2 * b * s3, r - kWgs84E2 * kWgs84A * c3);

    // One Newton step on the standard latitude equation.
    for (int i = 0; i < 2; ++i) {
      const double sLat = std::sin(lat);
      const double N = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sLat * sLat);
      const double h = r / std::cos(lat) - N;
      lat = std::atan2(z, r * (1.0 - kWgs84E2 * N / (N + h)));
    }

    const double sLat = std::sin(lat);
    const double N = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sLat * sLat);
    const double alt = r / std::cos(lat) - N;
    return {lat, lon, alt};
  }

 private:
  static constexpr double kPiHalf() { return 1.57079632679489661923; }

  Geodetic origin_{};
  Eigen::Vector3d ecef_origin_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_ecef_to_enu_{Eigen::Matrix3d::Identity()};
};

}  // namespace geoloc

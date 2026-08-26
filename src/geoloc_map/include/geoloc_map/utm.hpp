// Copyright 2026 geoloc team.
// UTM (Universal Transverse Mercator) <-> WGS84 conversion.
//
// The geopack mosaic is georeferenced in a projected CRS (a UTM zone, e.g.
// EPSG:32637), while the onboard frame is a local ENU tangent plane anchored at
// the mission origin (geoloc_common/geodetic.hpp). To serve a window around an
// ENU position this node must walk:
//
//     ENU --LocalEnu--> WGS84 --transverse Mercator--> UTM (east, north)
//
// and back. This header implements the transverse Mercator on the WGS84
// ellipsoid with Karney's Krueger n-series ("Transverse Mercator with an
// accuracy of a few nanometers"), which round-trips to well below the 1e-6 m
// georeferencing requirement (T07-U-01). The forward conformal latitude uses a
// closed form; the inverse solves it with the same fixed-point iteration as
// GeographicLib's Math::tauf.
//
// Only the UTM case of EPSG:326xx / EPSG:327xx is supported; anything else is
// a hard error, not a silent wrong coordinate.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "geoloc_common/angles.hpp"

namespace geoloc {

// WGS84 ellipsoid.
inline constexpr double kUtmSemiMajor = 6378137.0;
inline constexpr double kUtmF = 1.0 / 298.257223563;
inline constexpr double kUtmE2 = kUtmF * (2.0 - kUtmF);
inline constexpr double kUtmE = 0.08181919084262149;  // sqrt(kUtmE2)
inline constexpr double kUtmK0 = 0.9996;
inline constexpr double kUtmFalseEasting = 500000.0;

// Third flattening and the rectifying radius for the n-series.
inline constexpr double kUtmN = kUtmF / (2.0 - kUtmF);  // 0.0016792203863837
inline constexpr double kUtmRectifying = 6367449.145823415;

// Krueger series coefficients (Karney 2011), forward (alpha) and inverse (beta).
inline constexpr double kUtmAlpha[8] = {
    0.0008377318206244698,  7.608527773572486e-07,  1.1976455032424919e-09, 2.4291706803970904e-12,
    5.7118183704280194e-15, 1.4799979313796632e-17, 4.1076241093707195e-20, 1.2107850389225785e-22,
};
inline constexpr double kUtmBeta[8] = {
    0.0008377321640579488, 5.9058701522203654e-08, 1.673482665343825e-10,  2.1647981104906419e-13,
    3.78793096862602e-16,  7.236769021815623e-19,  1.4934798247781068e-21, 3.2595225458381582e-24,
};

struct UtmZone {
  int zone{0};       // 1..60
  bool north{true};  // true = N (EPSG:326xx), false = S (EPSG:327xx)
};

/// Parse "EPSG:32637" / "EPSG:32704" into a UTM zone. Throws on anything else.
inline UtmZone utmZoneFromEpsg(const std::string& crs) {
  const std::string prefix = "EPSG:";
  if (crs.size() < prefix.size() + 5 || crs.compare(0, prefix.size(), prefix) != 0) {
    throw std::invalid_argument("unsupported CRS (expected UTM EPSG:326xx/327xx): " + crs);
  }
  const std::string code = crs.substr(prefix.size());
  bool north;
  if (code.compare(0, 3, "326") == 0) {
    north = true;
  } else if (code.compare(0, 3, "327") == 0) {
    north = false;
  } else {
    throw std::invalid_argument("unsupported CRS (expected UTM EPSG:326xx/327xx): " + crs);
  }
  const int zone = std::stoi(code.substr(3));
  if (zone < 1 || zone > 60) {
    throw std::invalid_argument("UTM zone out of range: " + crs);
  }
  return {zone, north};
}

inline double utmCentralMeridian(int zone) {
  return deg2rad(zone * 6 - 183);
}

inline double utmFalseNorthing(bool north) {
  return north ? 0.0 : 10000000.0;
}

namespace detail {

/// tan(conformal latitude) from tan(geodetic latitude), closed form.
inline double utmTaupf(double tau) {
  const double tau1 = std::hypot(1.0, tau);
  const double sig = std::sinh(kUtmE * std::atanh(kUtmE * tau / tau1));
  return std::hypot(1.0, sig) * tau - sig * tau1;
}

/// tan(geodetic latitude) from tan(conformal latitude), fixed-point iteration
/// (mirrors GeographicLib's Math::tauf; converges in ~2 steps for |lat| < 85).
inline double utmTauf(double taup) {
  const double e2m = 1.0 - kUtmE2;
  double tau = taup / e2m;
  const double tol = std::sqrt(std::numeric_limits<double>::epsilon()) / 10.0;
  const double stol = tol * std::max(1.0, std::abs(taup));
  for (int i = 0; i < 6; ++i) {
    const double taupa = utmTaupf(tau);
    const double dtau = (taup - taupa) * (1.0 + e2m * tau * tau) /
                        (e2m * std::hypot(1.0, tau) * std::hypot(1.0, taupa));
    tau += dtau;
    if (std::abs(dtau) < stol) break;
  }
  return tau;
}

}  // namespace detail

/// WGS84 (radians) -> UTM easting/northing (metres).
inline void wgs84ToUtm(double lat, double lon, const UtmZone& zone, double& easting,
                       double& northing) {
  const double dlon = lon - utmCentralMeridian(zone.zone);
  const double taup = detail::utmTaupf(std::tan(lat));
  const double cdl = std::cos(dlon);
  const double sdl = std::sin(dlon);
  const double xi1 = std::atan2(taup, cdl);
  const double eta1 = std::asinh(sdl / std::hypot(taup, cdl));

  double xi = xi1;
  double eta = eta1;
  for (int i = 0; i < 8; ++i) {
    const double k = 2.0 * (i + 1);
    xi += kUtmAlpha[i] * std::sin(k * xi1) * std::cosh(k * eta1);
    eta += kUtmAlpha[i] * std::cos(k * xi1) * std::sinh(k * eta1);
  }

  easting = kUtmFalseEasting + kUtmK0 * kUtmRectifying * eta;
  northing = utmFalseNorthing(zone.north) + kUtmK0 * kUtmRectifying * xi;
}

/// UTM easting/northing (metres) -> WGS84 (radians).
inline void utmToWgs84(double easting, double northing, const UtmZone& zone, double& lat,
                       double& lon) {
  const double x = (easting - kUtmFalseEasting) / (kUtmK0 * kUtmRectifying);
  const double y = (northing - utmFalseNorthing(zone.north)) / (kUtmK0 * kUtmRectifying);

  double xi1 = y;
  double eta1 = x;
  for (int i = 0; i < 8; ++i) {
    const double k = 2.0 * (i + 1);
    xi1 -= kUtmBeta[i] * std::sin(k * y) * std::cosh(k * x);
    eta1 -= kUtmBeta[i] * std::cos(k * y) * std::sinh(k * x);
  }

  const double taup = std::sin(xi1) / std::hypot(std::sinh(eta1), std::cos(xi1));
  lat = std::atan(detail::utmTauf(taup));
  lon = utmCentralMeridian(zone.zone) + std::atan2(std::sinh(eta1), std::cos(xi1));
}

}  // namespace geoloc

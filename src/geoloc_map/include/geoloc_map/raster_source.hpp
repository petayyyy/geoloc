// Copyright 2026 geoloc team.
// RasterSource: the abstraction the window service reads from.
//
// The window service logic (pyramid selection, georeferencing, cache, bounds
// handling) is written against this interface so it is unit-testable against a
// synthetic in-memory raster (no COG / libtiff). The production implementation
// is geoloc_map::Geopack, which maps the request ENU coordinates through the
// mission UTM CRS into COG pixels.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "geoloc_map/pyramid.hpp"

namespace geoloc {

/// A served window: grayscale image, validity mask, and ENU georeferencing.
struct WindowResult {
  bool success{false};
  std::string message;
  uint32_t width{0};
  uint32_t height{0};
  double origin_east{0.0};  // ENU of the window's top-left corner
  double origin_north{0.0};
  double gsd{0.0};                // served (pyramid) GSD
  std::vector<uint8_t> image;     // mono8, width*height
  std::vector<uint8_t> validity;  // mono8, width*height
  bool descriptors_requested{false};
  bool cache_hit{false};
};

/// A raster layer the service can pull windows from. Coordinates are ENU on the
/// public interface; implementations map them into their own storage frame
/// (the Geopack maps ENU -> UTM -> pixel).
class RasterSource {
 public:
  virtual ~RasterSource() = default;

  virtual std::vector<PyramidLevel> pyramid() const = 0;

  /// ENU (east, north) -> continuous pixel (col, row) at the given level.
  virtual Eigen::Vector2d enuToPixel(const Eigen::Vector2d& enu,
                                     const PyramidLevel& level) const = 0;

  /// Continuous pixel (col, row) at the given level -> ENU (east, north).
  virtual Eigen::Vector2d pixelToEnu(double col, double row, const PyramidLevel& level) const = 0;

  /// Read the pixel-aligned rectangle [x0, x0+w) x [y0, y0+h) at the given
  /// level into `gray` and `validity` (both resized to w*h). Returns false on a
  /// decode failure without throwing.
  virtual bool readWindow(const PyramidLevel& level, int x0, int y0, int w, int h,
                          std::vector<uint8_t>& gray, std::vector<uint8_t>& validity) = 0;
};

}  // namespace geoloc

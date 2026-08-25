// Geo-referenced raster: pixel <-> ENU mapping.
//
// Used identically by the basemap (geoloc_map), the local DSM (geoloc_ortho)
// and the true-ortho patch. Having ONE implementation is deliberate: a
// half-pixel convention mismatch between two of them produces a constant
// offset that looks exactly like a matcher bias and is very hard to attribute.
//
// CONVENTION: pixel (0,0) is the TOP-LEFT corner of the raster; pixel CENTRES
// are at (col + 0.5, row + 0.5). North is up, so row increases southward.

#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

namespace geoloc {

class GeoRaster {
 public:
  GeoRaster() = default;

  GeoRaster(double origin_east, double origin_north, double gsd, int width, int height)
      : origin_east_(origin_east),
        origin_north_(origin_north),
        gsd_(gsd),
        width_(width),
        height_(height) {}

  double originEast() const noexcept { return origin_east_; }
  double originNorth() const noexcept { return origin_north_; }
  double gsd() const noexcept { return gsd_; }
  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }

  /// Continuous pixel coordinates (col, row) -> ENU (east, north).
  Eigen::Vector2d pixelToEnu(double col, double row) const noexcept {
    return {origin_east_ + col * gsd_, origin_north_ - row * gsd_};
  }

  /// ENU -> continuous pixel coordinates (col, row).
  Eigen::Vector2d enuToPixel(double east, double north) const noexcept {
    return {(east - origin_east_) / gsd_, (origin_north_ - north) / gsd_};
  }

  Eigen::Vector2d enuToPixel(const Eigen::Vector2d& enu) const noexcept {
    return enuToPixel(enu.x(), enu.y());
  }

  /// Centre of pixel (col, row) in ENU.
  Eigen::Vector2d pixelCenterToEnu(int col, int row) const noexcept {
    return pixelToEnu(col + 0.5, row + 0.5);
  }

  /// True if the continuous pixel coordinate falls inside the raster.
  bool contains(double col, double row) const noexcept {
    return col >= 0.0 && row >= 0.0 && col <= static_cast<double>(width_) &&
           row <= static_cast<double>(height_);
  }

  bool containsEnu(const Eigen::Vector2d& enu) const noexcept {
    const Eigen::Vector2d p = enuToPixel(enu);
    return contains(p.x(), p.y());
  }

  /// ENU bounds as (east_min, north_min, east_max, north_max).
  Eigen::Vector4d boundsEnu() const noexcept {
    return {origin_east_,
            origin_north_ - height_ * gsd_,
            origin_east_ + width_ * gsd_,
            origin_north_};
  }

  /// Sub-window with the same GSD, clamped to this raster. Returned raster
  /// carries the correct origin so geo-referencing survives cropping -- the
  /// place where offsets usually creep in.
  GeoRaster subWindow(int col0, int row0, int w, int h) const noexcept {
    col0 = clampInt(col0, 0, width_);
    row0 = clampInt(row0, 0, height_);
    w = clampInt(w, 0, width_ - col0);
    h = clampInt(h, 0, height_ - row0);
    const Eigen::Vector2d o = pixelToEnu(col0, row0);
    return GeoRaster(o.x(), o.y(), gsd_, w, h);
  }

  /// Window of the given radius (metres) around an ENU centre, clamped.
  /// This is the prior-window request shape from ADR-005.
  GeoRaster windowAround(const Eigen::Vector2d& center_enu, double radius_m) const noexcept {
    const Eigen::Vector2d c = enuToPixel(center_enu);
    const double rpx = radius_m / gsd_;
    const int col0 = static_cast<int>(std::floor(c.x() - rpx));
    const int row0 = static_cast<int>(std::floor(c.y() - rpx));
    const int side = static_cast<int>(std::ceil(2.0 * rpx));
    return subWindow(col0, row0, side, side);
  }

  /// Resampled view at a different GSD, same geographic extent. Used when the
  /// matcher asks for a coarser pyramid level during cold start / LOST.
  GeoRaster atGsd(double new_gsd) const noexcept {
    const double sx = gsd_ / new_gsd;
    return GeoRaster(origin_east_, origin_north_, new_gsd,
                     static_cast<int>(std::round(width_ * sx)),
                     static_cast<int>(std::round(height_ * sx)));
  }

 private:
  static int clampInt(int v, int lo, int hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  double origin_east_{0.0};
  double origin_north_{0.0};
  double gsd_{1.0};
  int width_{0};
  int height_{0};
};

}  // namespace geoloc

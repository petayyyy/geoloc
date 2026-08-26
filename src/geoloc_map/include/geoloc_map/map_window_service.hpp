// Copyright 2026 geoloc team.
// Window service: the T07 core logic.
//
// Given a prior position (ENU), a radius and a requested GSD, this produces a
// basemap window with georeferencing and an optional descriptor set. It owns
// the three T07 decisions:
//
//   * pyramid level selection (coarse level for cold start / LOST),
//   * descriptor cache invalidation (centre shift >30%, GSD change, model
//     version change),
//   * bounds handling (a request outside the mission package is a normal
//     situation, not an exception -- clean refusal, no crash).
//
// It is written against the RasterSource interface and has no ROS or COG
// dependency, so the T07-U-* unit tests exercise it against a synthetic raster.

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "geoloc_map/descriptor_cache.hpp"
#include "geoloc_map/pyramid.hpp"
#include "geoloc_map/raster_source.hpp"

namespace geoloc {

/// A served window plus optional descriptors, in the MapWindow.srv layout.
struct MapWindowResponse {
  bool success{false};
  std::string message;
  uint32_t width{0};
  uint32_t height{0};
  double origin_east{0.0};  // ENU of the window's top-left corner
  double origin_north{0.0};
  double gsd{0.0};  // served (pyramid) GSD, not the requested one
  std::vector<uint8_t> image;
  std::vector<uint8_t> validity;
  DescriptorSet descriptors;
  bool cache_hit{false};
};

class MapWindowService {
 public:
  struct Config {
    double shift_fraction{0.30};
    bool invalidate_on_model_change{true};
  };

  MapWindowService(const Config& cfg, std::shared_ptr<DescriptorEngine> engine)
      : cache_(cfg.shift_fraction, cfg.invalidate_on_model_change), engine_(std::move(engine)) {}

  const Config& config() const noexcept { return config_; }
  const DescriptorCache& cache() const noexcept { return cache_; }

  MapWindowResponse serve(RasterSource& src, const Eigen::Vector2d& center, double radius_m,
                          double requested_gsd, bool with_descriptors,
                          const std::string& model_version) {
    MapWindowResponse resp;
    const std::vector<PyramidLevel> levels = src.pyramid();
    if (levels.empty() || levels.front().gsd <= 0.0) {
      resp.message = "raster source has no base level";
      return resp;
    }

    // 1. Pyramid level selection by requested GSD (coarse for cold start / LOST).
    const PyramidLevel level = selectPyramidLevel(requested_gsd, levels);
    resp.gsd = level.gsd;

    // 2. Bounds: the request centre must lie inside the mission package. A
    //    request outside is a NORMAL situation (aircraft left the route).
    const PyramidLevel& base = levels.front();
    const Eigen::Vector2d cbase = src.enuToPixel(center, base);
    const bool inside = cbase.x() >= 0.0 && cbase.x() <= static_cast<double>(base.width) &&
                        cbase.y() >= 0.0 && cbase.y() <= static_cast<double>(base.height);
    if (!inside) {
      resp.success = false;
      resp.message = "request centre outside mission package";
      return resp;
    }

    // 3. Window rectangle in level pixels, clamped to the raster edge.
    const Eigen::Vector2d cp = src.enuToPixel(center, level);
    const double rpx = radius_m / level.gsd;
    const int col0 = static_cast<int>(std::floor(cp.x() - rpx));
    const int row0 = static_cast<int>(std::floor(cp.y() - rpx));
    const int side = static_cast<int>(std::ceil(2.0 * rpx));
    const int W = level.width, H = level.height;
    const int x0 = std::max(0, std::min(col0, W));
    const int y0 = std::max(0, std::min(row0, H));
    const int x1 = std::max(0, std::min(col0 + side, W));
    const int y1 = std::max(0, std::min(row0 + side, H));
    const int w = x1 - x0, h = y1 - y0;
    if (w <= 0 || h <= 0) {
      resp.success = false;
      resp.message = "window outside mission package";
      return resp;
    }

    // 4. Read pixels + validity mask.
    if (!src.readWindow(level, x0, y0, w, h, resp.image, resp.validity)) {
      resp.success = false;
      resp.message = "tile decode failed";
      return resp;
    }

    // 5. Georeferencing of the served window (top-left corner, ENU).
    const Eigen::Vector2d origin = src.pixelToEnu(x0, y0, level);
    resp.origin_east = origin.x();
    resp.origin_north = origin.y();
    resp.width = static_cast<uint32_t>(w);
    resp.height = static_cast<uint32_t>(h);

    // 6. Descriptors: reuse the cache unless the window shifted >30%, the GSD
    //    changed, or the model version changed.
    if (with_descriptors) {
      if (cache_.valid(center, radius_m, level.gsd, model_version)) {
        resp.descriptors = cache_.descriptors();
        resp.cache_hit = true;
      } else {
        DescriptorSet ds = engine_->compute(resp.image, w, h, resp.validity);
        cache_.incrementRecomputeCount();
        cache_.store(center, radius_m, level.gsd, model_version, ds);
        resp.descriptors = std::move(ds);
        resp.cache_hit = false;
      }
    }

    resp.success = true;
    return resp;
  }

 private:
  Config config_;
  DescriptorCache cache_;
  std::shared_ptr<DescriptorEngine> engine_;
};

}  // namespace geoloc

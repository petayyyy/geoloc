// Copyright 2026 geoloc team.
// T07 unit tests (level 0): georeferencing, pyramid selection, cache
// invalidation, bounds handling.
//
// Deliberately dependency-free (no libtiff / yaml-cpp / ROS) like
// geoloc_common's tests: everything under test here is header-only logic. The
// COG-backed integration path is covered by test_geopack.cpp.
//
// Test IDs (task card T07):
//   T07-U-01  georeferencing (ENU <-> UTM <-> pixel round-trip, < 1e-6 m)
//   T07-U-02  pyramid level selection (2 m request -> 4x level, not base)
//   T07-U-03  cache invalidation (29% shift -> hit, 31% -> recompute)
//   T07-U-04  bounds (outside the package -> clean refusal, no crash)

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "geoloc_common/angles.hpp"
#include "geoloc_common/geodetic.hpp"
#include "geoloc_map/descriptor_cache.hpp"
#include "geoloc_map/map_window_service.hpp"
#include "geoloc_map/pyramid.hpp"
#include "geoloc_map/raster_source.hpp"
#include "geoloc_map/utm.hpp"

using namespace geoloc;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* name, const char* detail = "") {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL: %s %s\n", name, detail);
  }
}

void checkNear(double a, double b, double tol, const char* name) {
  ++g_checks;
  if (!(std::abs(a - b) <= tol)) {
    ++g_failures;
    std::printf("  FAIL: %s (got %.17g, want %.17g, tol %.3g)\n", name, a, b, tol);
  }
}

// ---------------------------------------------------------------------------
// T07-U-01: georeferencing. Reference UTM values were computed with pyproj
// 3.6.1 (EPSG:4326 -> EPSG:32637) for the Maykop corridor anchor.
// ---------------------------------------------------------------------------
void testGeoreferencing() {
  std::printf("T07-U-01  georeferencing (ENU <-> UTM <-> pixel)\n");

  const Geodetic origin{deg2rad(44.8285), deg2rad(39.922), 80.0};
  const LocalEnu enu(origin);
  const UtmZone zone = utmZoneFromEpsg("EPSG:32637");

  // Validate the transverse Mercator against pyproj control points.
  double e, n;
  wgs84ToUtm(origin.lat, origin.lon, zone, e, n);
  checkNear(e, 572884.1806432105, 1e-6, "origin -> UTM easting");
  checkNear(n, 4964312.650495112, 1e-6, "origin -> UTM northing");

  double lat, lon;
  utmToWgs84(e, n, zone, lat, lon);
  checkNear(lat, origin.lat, 1e-11, "UTM -> lat (inverse)");
  checkNear(lon, origin.lon, 1e-11, "UTM -> lon (inverse)");

  // Pixel <-> ENU through the real ortho_a grid (corner convention, matching
  // GeoRaster): origin (572559.6, 4965038.4), gsd 0.3. This is the served
  // window's georeferencing round trip (UTM <-> WGS84 <-> ENU) and must be
  // exact to <1e-6 m -- the T07-U-01 criterion.
  const double oe = 572559.6, on = 4965038.4, gsd = 0.3;
  double worst = 0.0;
  for (int i = 0; i < 20000; ++i) {
    const double col = (i % 2164) + 0.25;
    const double row = (i % 4838) + 0.25;
    const double pe = oe + col * gsd;
    const double pn = on - row * gsd;
    utmToWgs84(pe, pn, zone, lat, lon);
    const Eigen::Vector3d env = enu.toEnu(Geodetic{lat, lon, origin.alt});
    // and back
    const Geodetic g2 = enu.fromEnu(env);
    wgs84ToUtm(g2.lat, g2.lon, zone, e, n);
    const double col2 = (e - oe) / gsd;
    const double row2 = (on - n) / gsd;
    worst = std::max(worst, std::hypot(col2 - col, row2 - row) * gsd);
  }
  std::printf("  worst pixel<->ENU georeferencing error: %.3e m\n", worst);
  check(worst < 1e-6, "pixel<->ENU georeferencing < 1e-6 m");

  // A known GCP: ortho_a tile, east 572515.817765 / north 4965077.461323.
  // Pixel (corner convention) must equal the GCP centre minus the 0.5 offset.
  const double gcp_col_center = -146.440784;
  const double gcp_row_center = -130.704410;
  const double col = (572515.817765 - oe) / gsd;
  const double row = (on - 4965077.461323) / gsd;
  checkNear(col - 0.5, gcp_col_center, 1e-5, "GCP easting -> pixel col");
  checkNear(row - 0.5, gcp_row_center, 1e-5, "GCP northing -> pixel row");
}

// ---------------------------------------------------------------------------
// T07-U-02: pyramid level selection.
// ---------------------------------------------------------------------------
void testPyramid() {
  std::printf("T07-U-02  pyramid level selection\n");
  const auto levels = buildPyramid(0.5, {2, 4, 8});

  check(levels.size() == 4, "4 levels (base + 3 overviews)");
  const auto l2 = selectPyramidLevel(2.0, levels);
  check(l2.factor == 4 && std::abs(l2.gsd - 2.0) < 1e-12, "2 m request -> 4x level (not base)");

  const auto l15 = selectPyramidLevel(1.5, levels);
  check(l15.factor == 2, "1.5 m request -> 2x level");

  const auto lbase = selectPyramidLevel(0.5, levels);
  check(lbase.factor == 1, "0.5 m request -> base");

  const auto lfine = selectPyramidLevel(0.1, levels);
  check(lfine.factor == 1, "request finer than base clamps to base");

  const auto lcoarse = selectPyramidLevel(10.0, levels);
  check(lcoarse.factor == 8, "request coarser than pyramid clamps to 8x");
}

// ---------------------------------------------------------------------------
// T07-U-03: cache invalidation. Exercised through the service so the shift
// fraction, GSD and model-version rules are all covered end to end.
// ---------------------------------------------------------------------------
class SyntheticSource : public RasterSource {
 public:
  SyntheticSource(double origin_east, double origin_north, double gsd, int width, int height)
      : oe_(origin_east), on_(origin_north), gsd_(gsd), width_(width), height_(height) {
    for (int f : {1, 2, 4, 8}) {
      PyramidLevel lv;
      lv.factor = f;
      lv.gsd = gsd * f;
      lv.width = (width + f - 1) / f;
      lv.height = (height + f - 1) / f;
      levels_.push_back(lv);
    }
  }

  std::vector<PyramidLevel> pyramid() const override { return levels_; }
  Eigen::Vector2d enuToPixel(const Eigen::Vector2d& enu, const PyramidLevel& level) const override {
    return {(enu.x() - oe_) / level.gsd, (on_ - enu.y()) / level.gsd};
  }
  Eigen::Vector2d pixelToEnu(double col, double row, const PyramidLevel& level) const override {
    return {oe_ + col * level.gsd, on_ - row * level.gsd};
  }
  bool readWindow(const PyramidLevel&, int, int, int w, int h, std::vector<uint8_t>& gray,
                  std::vector<uint8_t>& validity) override {
    gray.assign(static_cast<size_t>(w) * h, 128);
    validity.assign(static_cast<size_t>(w) * h, 255);
    return true;
  }

 private:
  double oe_, on_, gsd_;
  int width_, height_;
  std::vector<PyramidLevel> levels_;
};

void testCache() {
  std::printf("T07-U-03  descriptor cache invalidation\n");
  SyntheticSource src(0.0, 1000.0, 0.5, 2000, 2000);
  MapWindowService::Config cfg;
  cfg.shift_fraction = 0.30;
  cfg.invalidate_on_model_change = true;
  MapWindowService svc(cfg, std::make_shared<GridDescriptorEngine>(16));

  const double radius = 100.0;  // window size = 200 m
  const Eigen::Vector2d c0(500.0, 500.0);

  // First request computes descriptors.
  auto r1 = svc.serve(src, c0, radius, 0.5, true, "xfeat_v1");
  check(r1.success && !r1.cache_hit, "first request computes descriptors");

  // 29% shift (58 m) -> still a hit.
  const Eigen::Vector2d c29(500.0 + 58.0, 500.0);
  auto r2 = svc.serve(src, c29, radius, 0.5, true, "xfeat_v1");
  check(r2.success && r2.cache_hit, "29%% shift -> cache hit");

  // 31% shift (62 m) -> recompute.
  const Eigen::Vector2d c31(500.0 + 62.0, 500.0);
  auto r3 = svc.serve(src, c31, radius, 0.5, true, "xfeat_v1");
  check(r3.success && !r3.cache_hit, "31%% shift -> recompute");

  // GSD change -> recompute even without a shift.
  auto r4 = svc.serve(src, c31, radius, 1.0, true, "xfeat_v1");
  check(!r4.cache_hit, "requested GSD change -> recompute");

  // Model version change -> recompute.
  auto r5 = svc.serve(src, c31, radius, 0.5, true, "xfeat_v2");
  check(!r5.cache_hit, "model version change -> recompute");

  check(svc.cache().recomputeCount() == 4, "exactly 4 recomputes across the sequence");
}

// ---------------------------------------------------------------------------
// T07-U-04: bounds handling.
// ---------------------------------------------------------------------------
void testBounds() {
  std::printf("T07-U-04  request beyond the package bounds\n");
  SyntheticSource src(0.0, 1000.0, 0.5, 2000, 2000);  // ENU east [0,1000], north [0,1000]
  MapWindowService::Config cfg;
  MapWindowService svc(cfg, std::make_shared<GridDescriptorEngine>(16));

  // Inside: success, window clamped to the raster.
  const auto inside = svc.serve(src, Eigen::Vector2d(500.0, 500.0), 300.0, 0.5, false, "");
  check(inside.success, "centre inside -> success");
  check(inside.width > 0 && inside.height > 0, "inside window is non-empty");
  check(inside.origin_east <= 500.0 && inside.origin_north >= 500.0,
        "window origin is the top-left corner (north-up)");

  // At the very edge: clamped, not empty, no crash.
  const auto edge = svc.serve(src, Eigen::Vector2d(5.0, 5.0), 300.0, 0.5, false, "");
  check(edge.success, "centre near the edge -> success (clamped)");
  check(edge.width > 0 && edge.height > 0, "edge window is clamped, not empty");

  // Far outside: clean refusal, no crash.
  const auto outside = svc.serve(src, Eigen::Vector2d(5000.0, 5000.0), 300.0, 0.5, false, "");
  check(!outside.success, "centre far outside -> refusal");
  check(!outside.message.empty(), "refusal carries a reason");
}

}  // namespace

int main() {
  std::printf("geoloc_map unit tests (T07-U-01 .. T07-U-04)\n");
  std::printf("=============================================\n");
  testGeoreferencing();
  testPyramid();
  testCache();
  testBounds();
  std::printf("=============================================\n");
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) std::printf("ALL PASS\n");
  return g_failures == 0 ? 0 : 1;
}

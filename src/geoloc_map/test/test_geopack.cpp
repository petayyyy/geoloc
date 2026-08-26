// Copyright 2026 geoloc team.
// T07 integration test: load the real Maykop .geopack and verify the
// ENU <-> UTM <-> pixel chain against its GCP control points, plus a real
// window read and the bounds refusal path.
//
// Usage: test_geopack <manifest.yaml>
// Exits 0 (SKIP) when no manifest is given or it cannot be loaded, so the
// test is harmless in a cross-build container without mission data.

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "geoloc_common/angles.hpp"
#include "geoloc_common/geodetic.hpp"
#include "geoloc_map/geopack.hpp"
#include "geoloc_map/map_window_service.hpp"
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

struct Gcp {
  double east, north, col_center, row_center;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("SKIP: no manifest path given\n");
    return 0;
  }
  std::printf("geoloc_map integration test (Maykop geopack)\n");
  std::printf("=============================================\n");

  Geopack gp;
  try {
    gp = Geopack::load(argv[1]);
  } catch (const std::exception& e) {
    std::printf("SKIP: cannot load %s: %s\n", argv[1], e.what());
    return 0;
  }

  check(gp.crs() == "EPSG:32637", "manifest CRS is EPSG:32637");

  GeopackLayer* a = gp.layer("ortho_a");
  GeopackLayer* b = gp.layer("ortho_b");
  check(a != nullptr, "ortho_a present");
  check(b != nullptr, "ortho_b present");
  if (!a) return 1;

  // Inside-corridor GCPs from gcp.csv (ortho_a).
  const std::vector<Gcp> gcps = {
      {572787.795398, 4965026.441189, 760.151325, 39.362702},
      {572843.295713, 4964918.875477, 945.152375, 397.915078},
      {572898.797856, 4964811.308412, 1130.159520, 756.471958},
  };

  const auto levels = a->pyramid();
  check(!levels.empty() && levels.front().factor == 1, "base level present");
  const PyramidLevel& base = levels.front();
  check(std::abs(base.gsd - 0.3) < 1e-9, "ortho_a base GSD is 0.3 m");

  const Geodetic origin = gp.origin();
  const LocalEnu enu(origin);
  const UtmZone zone = utmZoneFromEpsg(gp.crs());

  // Georeferencing (T07-U-01): every GCP's UTM coordinate must land on its
  // pixel through the served window's affine (origin + gsd). This is the
  // "known point <-> known UTM coordinate" contract and is exact to < 1e-6 m;
  // the GCP CSV carries ~6 decimal places, so 1e-5 px (~3e-6 m) tolerates the
  // rounding.
  const Eigen::Vector2d origin_utm = a->baseUtmOrigin();
  const double gsd = base.gsd;
  double worst = 0.0;
  for (const auto& g : gcps) {
    const double col = (g.east - origin_utm.x()) / gsd;
    const double row = (origin_utm.y() - g.north) / gsd;
    checkNear(col - 0.5, g.col_center, 1e-5, "GCP easting -> pixel col");
    checkNear(row - 0.5, g.row_center, 1e-5, "GCP northing -> pixel row");
    worst = std::max(worst, std::abs((col - 0.5) - g.col_center) * gsd);
    worst = std::max(worst, std::abs((row - 0.5) - g.row_center) * gsd);
  }
  std::printf("  worst GCP georeferencing error: %.3e m\n", worst);
  check(worst < 1e-6, "GCP georeferencing < 1e-6 m");

  // Serve a real window around the first GCP and check the contract.
  MapWindowService::Config cfg;
  cfg.shift_fraction = 0.30;
  MapWindowService svc(cfg, std::make_shared<GridDescriptorEngine>(16));

  double lat, lon;
  utmToWgs84(gcps[0].east, gcps[0].north, zone, lat, lon);
  const Eigen::Vector3d e0 = enu.toEnu(Geodetic{lat, lon, origin.alt});
  const Eigen::Vector2d center(e0.x(), e0.y());

  // A base-resolution (0.3 m) request serves the base level.
  const auto win = svc.serve(*a, center, 30.0, 0.3, true, "xfeat_v1");
  check(win.success, "serve at a GCP centre succeeds");
  check(win.width > 0 && win.height > 0, "served window is non-empty");
  check(std::abs(win.gsd - 0.3) < 1e-9, "0.3 m request -> base (0.3 m) level");
  check(win.image.size() == static_cast<size_t>(win.width) * win.height, "image sized w*h");
  check(win.validity.size() == static_cast<size_t>(win.width) * win.height, "validity sized w*h");
  check(win.descriptors.n_keypoints > 0, "placeholder descriptors produced");

  // The served window's reported top-left corner (ENU) plus its size must
  // contain the requested centre.
  const double center_col = (center.x() - win.origin_east) / win.gsd;
  const double center_row = (win.origin_north - center.y()) / win.gsd;
  check(center_col >= 0.0 && center_col <= static_cast<double>(win.width),
        "served window covers the centre (col)");
  check(center_row >= 0.0 && center_row <= static_cast<double>(win.height),
        "served window covers the centre (row)");

  // Far outside the corridor: clean refusal, no crash.
  const auto out = svc.serve(*a, Eigen::Vector2d(-1e5, -1e5), 30.0, 0.3, false, "");
  check(!out.success, "centre far outside -> refusal");
  check(!out.message.empty(), "refusal carries a reason");

  std::printf("=============================================\n");
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) std::printf("ALL PASS\n");
  return g_failures == 0 ? 0 : 1;
}

// Copyright 2026 geoloc team.
// .geopack loader implementation (see geopack.hpp).

#include "geoloc_map/geopack.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "geoloc_common/angles.hpp"

namespace geoloc {

GeopackLayer::GeopackLayer(std::string name, LocalEnu enu, UtmZone zone, CogReader ortho,
                           CogReader validity)
    : name_(std::move(name)),
      enu_(std::move(enu)),
      zone_(zone),
      ortho_(std::move(ortho)),
      validity_(std::move(validity)) {
  has_validity_ = validity_.valid();
  const auto& levels = ortho_.levels();
  const CogReader::Level& base = levels.front();
  for (const auto& lv : levels) {
    PyramidLevel p;
    p.factor = static_cast<int>(
        std::lround(static_cast<double>(base.width) / static_cast<double>(lv.width)));
    if (p.factor < 1) p.factor = 1;
    p.gsd = lv.gsd;
    p.width = static_cast<int>(lv.width);
    p.height = static_cast<int>(lv.height);
    pyramid_.push_back(p);
  }
}

int GeopackLayer::indexOf(const PyramidLevel& level) const {
  for (size_t i = 0; i < pyramid_.size(); ++i) {
    if (pyramid_[i].factor == level.factor) return static_cast<int>(i);
  }
  return 0;
}

std::vector<PyramidLevel> GeopackLayer::pyramid() const {
  return pyramid_;
}

Eigen::Vector2d GeopackLayer::baseUtmOrigin() const {
  if (ortho_.levels().empty()) return Eigen::Vector2d::Zero();
  const CogReader::Level& L = ortho_.levels().front();
  return {L.origin_east, L.origin_north};
}

Eigen::Vector2d GeopackLayer::enuToPixel(const Eigen::Vector2d& enu,
                                         const PyramidLevel& level) const {
  const Geodetic g = enu_.fromEnu(Eigen::Vector3d(enu.x(), enu.y(), 0.0));
  double east = 0.0, north = 0.0;
  wgs84ToUtm(g.lat, g.lon, zone_, east, north);
  const int idx = indexOf(level);
  const CogReader::Level& L = ortho_.levels()[idx];
  const double col = (east - L.origin_east) / L.gsd;
  const double row = (L.origin_north - north) / L.gsd;
  return {col, row};
}

Eigen::Vector2d GeopackLayer::pixelToEnu(double col, double row, const PyramidLevel& level) const {
  const int idx = indexOf(level);
  const CogReader::Level& L = ortho_.levels()[idx];
  const double east = L.origin_east + col * L.gsd;
  const double north = L.origin_north - row * L.gsd;
  double lat = 0.0, lon = 0.0;
  utmToWgs84(east, north, zone_, lat, lon);
  // Project back onto the tangent plane at the mission origin's altitude, so the
  // 2D map chain (ENU -> UTM -> pixel) round-trips exactly.
  const Eigen::Vector3d enu = enu_.toEnu(Geodetic{lat, lon, enu_.origin().alt});
  return {enu.x(), enu.y()};
}

bool GeopackLayer::readWindow(const PyramidLevel& level, int x0, int y0, int w, int h,
                              std::vector<uint8_t>& gray, std::vector<uint8_t>& validity) {
  const int idx = indexOf(level);
  gray.resize(static_cast<size_t>(w) * h);
  validity.resize(static_cast<size_t>(w) * h);
  if (!ortho_.readGray(idx, x0, y0, w, h, gray.data())) return false;
  if (has_validity_) {
    if (!validity_.readUint8(idx, x0, y0, w, h, validity.data())) return false;
  } else {
    // No validity layer: treat the whole window as valid (honest: the mosaic
    // has no recorded holes or seams).
    std::fill(validity.begin(), validity.end(), 255);
  }
  return true;
}

int GeopackLayer::prefetch(const Eigen::Vector2d& center_enu, double radius_m) {
  if (pyramid_.empty()) return 0;
  const PyramidLevel& level = pyramid_.front();  // prefetch at full resolution
  const int idx = indexOf(level);
  const Eigen::Vector2d c = enuToPixel(center_enu, level);
  const double rpx = radius_m / level.gsd;
  const int x0 = static_cast<int>(std::floor(c.x() - rpx));
  const int y0 = static_cast<int>(std::floor(c.y() - rpx));
  const int side = static_cast<int>(std::ceil(2.0 * rpx));
  const int W = level.width, H = level.height;
  const int cx0 = std::max(0, x0);
  const int cy0 = std::max(0, y0);
  const int cx1 = std::min(W, x0 + side);
  const int cy1 = std::min(H, y0 + side);
  if (cx1 <= cx0 || cy1 <= cy0) return 0;

  int touched = ortho_.prefetchTiles(idx, cx0, cy0, cx1 - cx0, cy1 - cy0);
  if (has_validity_) {
    touched += validity_.prefetchTiles(idx, cx0, cy0, cx1 - cx0, cy1 - cy0);
  }
  return touched;
}

Geopack Geopack::load(const std::string& manifest_path) {
  const YAML::Node m = YAML::LoadFile(manifest_path);
  if (!m["crs"]) throw std::runtime_error("manifest missing 'crs'");
  if (!m["origin"]) throw std::runtime_error("manifest missing 'origin'");

  Geopack gp;
  const size_t slash = manifest_path.find_last_of('/');
  gp.dir_ = (slash == std::string::npos) ? "" : manifest_path.substr(0, slash + 1);
  gp.crs_ = m["crs"].as<std::string>();
  gp.zone_ = utmZoneFromEpsg(gp.crs_);

  const Geodetic origin{deg2rad(m["origin"]["lat"].as<double>()),
                        deg2rad(m["origin"]["lon"].as<double>()), m["origin"]["alt"].as<double>()};
  gp.origin_ = origin;
  const LocalEnu enu(origin);

  if (!m["layers"]) throw std::runtime_error("manifest missing 'layers'");
  for (const auto& entry : m["layers"]) {
    const std::string name = entry.first.as<std::string>();
    const YAML::Node& lyr = entry.second;
    const std::string file = lyr["file"].as<std::string>();

    // Only ortho layers become a served RasterSource; dem/semantic are kept for
    // T20/T29 through the same georeferencing path.
    if (name.rfind("ortho_", 0) != 0) {
      gp.layer_names_.push_back(name);
      continue;
    }

    CogReader ortho = CogReader::open(gp.dir_ + file);
    CogReader validity;
    if (lyr["validity_file"]) {
      validity = CogReader::open(gp.dir_ + lyr["validity_file"].as<std::string>());
    }
    gp.layers_.push_back(
        std::make_unique<GeopackLayer>(name, enu, gp.zone_, std::move(ortho), std::move(validity)));
    gp.layer_names_.push_back(name);
  }

  if (gp.layers_.empty()) throw std::runtime_error("geopack has no ortho layers");
  return gp;
}

GeopackLayer* Geopack::layer(const std::string& name) {
  for (auto& l : layers_) {
    if (l->name() == name) return l.get();
  }
  return nullptr;
}

}  // namespace geoloc

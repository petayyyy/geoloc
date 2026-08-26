// Copyright 2026 geoloc team.
// .geopack loader: manifest + COG layers, mapping ENU <-> UTM <-> pixel.
//
// A .geopack is a directory (not an archive) with manifest.yaml and COG layers
// (see docs/plan/03-interfaces.md section 4). The layers are georeferenced in a
// projected CRS (a UTM zone), while the onboard frame is the local ENU tangent
// plane anchored at the manifest `origin`. This header owns that chain:
//
//     ENU --LocalEnu--> WGS84 --transverse Mercator--> UTM --> COG pixel
//
// and exposes it through the RasterSource interface the window service reads.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "geoloc_common/geodetic.hpp"
#include "geoloc_map/cog.hpp"
#include "geoloc_map/raster_source.hpp"
#include "geoloc_map/utm.hpp"

namespace geoloc {

/// One ortho layer (primary/secondary provider) with its validity mask.
class GeopackLayer : public RasterSource {
 public:
  GeopackLayer(std::string name, LocalEnu enu, UtmZone zone, CogReader ortho, CogReader validity);

  const std::string& name() const noexcept { return name_; }
  bool hasValidity() const noexcept { return has_validity_; }

  /// Top-left corner of the base level in the geopack CRS (e.g. UTM easting,
  /// northing). Exposed so the served window's pixel <-> CRS affine can be
  /// verified against control points directly.
  Eigen::Vector2d baseUtmOrigin() const;

  std::vector<PyramidLevel> pyramid() const override;
  Eigen::Vector2d enuToPixel(const Eigen::Vector2d& enu, const PyramidLevel& level) const override;
  Eigen::Vector2d pixelToEnu(double col, double row, const PyramidLevel& level) const override;
  bool readWindow(const PyramidLevel& level, int x0, int y0, int w, int h,
                  std::vector<uint8_t>& gray, std::vector<uint8_t>& validity) override;

  /// Warm the tile cache ahead of the aircraft (prefetch). Reads and decodes the
  /// tiles overlapping the ENU rectangle so the OS page cache is hot when the
  /// matcher asks for them. Returns the number of tiles touched.
  int prefetch(const Eigen::Vector2d& center_enu, double radius_m);

 private:
  int indexOf(const PyramidLevel& level) const;

  std::string name_;
  LocalEnu enu_;
  UtmZone zone_;
  CogReader ortho_;
  CogReader validity_;
  bool has_validity_{false};
  std::vector<PyramidLevel> pyramid_;
};

/// A loaded mission package.
class Geopack {
 public:
  static Geopack load(const std::string& manifest_path);

  const Geodetic& origin() const noexcept { return origin_; }
  const std::string& crs() const noexcept { return crs_; }

  GeopackLayer* layer(const std::string& name);
  const std::vector<std::string>& layerNames() const noexcept { return layer_names_; }

 private:
  std::string dir_;
  std::string crs_;
  Geodetic origin_;
  UtmZone zone_;
  std::vector<std::string> layer_names_;
  std::vector<std::unique_ptr<GeopackLayer>> layers_;
};

}  // namespace geoloc

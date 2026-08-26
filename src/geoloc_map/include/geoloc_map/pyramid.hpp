// Copyright 2026 geoloc team.
// Pyramid level selection for the COG overview pyramid.
//
// The basemap mosaic is a tiled COG with an overview pyramid (factors 2, 4, 8
// in the mapprep build). The matcher asks for a window at a requested GSD; the
// node serves it from the finest pyramid level whose GSD is still <= the
// request, so a coarse cold-start / LOST search does not read the full
// resolution (T07-U-02: a 2 m request on a 0.5 m base must land on the 4x
// level, not the base level).

#pragma once

#include <cstdint>
#include <vector>

namespace geoloc {

/// A resolved pyramid level: factor relative to base, its GSD in m/px, and the
/// raster dimensions at that level (filled by the raster source; 0 when the
/// caller only cares about the factor/gsd geometry).
struct PyramidLevel {
  int factor{1};  // 1 == base, 2 == first overview, ...
  double gsd{0.0};
  int width{0};
  int height{0};
};

/// Build the pyramid level list from a base GSD and overview factors.
/// Overview factors are expected ascending (e.g. {2, 4, 8}); the base level
/// (factor 1) is always first.
inline std::vector<PyramidLevel> buildPyramid(double base_gsd,
                                              const std::vector<int>& overview_factors) {
  std::vector<PyramidLevel> levels;
  levels.push_back({1, base_gsd});
  for (int f : overview_factors) {
    if (f <= 1) continue;  // a 1x "overview" is not an overview
    levels.push_back({f, base_gsd * static_cast<double>(f)});
  }
  return levels;
}

/// Select the pyramid level for a requested GSD.
///
/// Rule: the coarsest level whose GSD is <= the requested GSD, never finer than
/// the base. A request coarser than the coarsest overview clamps to it. A
/// request finer than the base level clamps to the base.
inline PyramidLevel selectPyramidLevel(double requested_gsd,
                                       const std::vector<PyramidLevel>& levels) {
  if (levels.empty()) return {1, 0.0};
  // levels are ordered finest (base) first, coarsest last. Walk coarsest ->
  // finest and return the first level whose GSD is fine enough (<= requested),
  // so a coarse request never touches the base level.
  for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
    if (it->gsd <= requested_gsd) return *it;
  }
  // Request finer than the base level: clamp to the base.
  return levels.front();
}

}  // namespace geoloc

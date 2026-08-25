// geoloc_map -- SKELETON (task T07)
//
// RESPONSIBLE FOR: Storing and serving the basemap window, DEM and semantics;
// caching map-side descriptors.
// NOT RESPONSIBLE FOR: Matching.
//
// Prior window from the fusion covariance replaces a retrieval network
// (ADR-005). Descriptors are cached ON BOARD, not precomputed offline.
//
// Publishes:
//   (none)
//   ~/map_window                                       [service]
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T07.

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "geoloc_common/angles.hpp"
#include "geoloc_common/covariance.hpp"
#include "geoloc_common/geodetic.hpp"
#include "geoloc_common/raster.hpp"
#include "geoloc_common/se2.hpp"

using namespace std::chrono_literals;

namespace geoloc {

class GeolocMapNode : public rclcpp::Node {
 public:
  GeolocMapNode() : rclcpp::Node("geoloc_map") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 1);

    // TODO(T07): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_map skeleton up (task T07)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocMapNode>());
  rclcpp::shutdown();
  return 0;
}

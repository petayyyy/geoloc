// geoloc_ortho -- SKELETON (task T14, T15)
//
// RESPONSIBLE FOR: Geometry: cloud -> deskew -> local DSM -> true-ortho patch
// + per-pixel confidence map.
// NOT RESPONSIBLE FOR: Anything involving features or the map.
//
// Avia FOV is NARROWER than the camera frame -- ~30% of patch width has no
// lidar coverage and is filled from Copernicus DEM. The confidence map must be
// honest about that.
//
// Publishes:
//   /geoloc/ortho_patch                                @ 2 Hz
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T14, T15.

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

class GeolocOrthoNode : public rclcpp::Node {
 public:
  GeolocOrthoNode() : rclcpp::Node("geoloc_ortho") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 2);

    // TODO(T14, T15): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_ortho skeleton up (task T14, T15)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocOrthoNode>());
  rclcpp::shutdown();
  return 0;
}

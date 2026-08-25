// geoloc_fusion -- SKELETON (task T23, T24)
//
// RESPONSIBLE FOR: Factor graph, global pose, prior window, mode machine, tf
// map_enu -> odom.
// NOT RESPONSIBLE FOR: The autopilot interface.
//
// A fix corrects map_enu -> odom, NOT the aircraft pose directly. Odometry
// stays smooth and a correction jump never becomes a velocity jump for the
// control loop.
//
// Publishes:
//   /geoloc/global_pose                                @ 10 Hz
//   /geoloc/status                                     @ 1 Hz
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T23, T24.

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

class GeolocFusionNode : public rclcpp::Node {
 public:
  GeolocFusionNode() : rclcpp::Node("geoloc_fusion") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 10);

    // TODO(T23, T24): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_fusion skeleton up (task T23, T24)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocFusionNode>());
  rclcpp::shutdown();
  return 0;
}

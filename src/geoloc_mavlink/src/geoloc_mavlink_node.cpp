// geoloc_mavlink -- SKELETON (task T25-T27)
//
// RESPONSIBLE FOR: Translating the estimate into MAVLink GPS_INPUT; failsafe
// escalation levels.
// NOT RESPONSIBLE FOR: Judging quality. And NEVER commanding a flight mode
// change -- T27-U-02 audits for it.
//
// GPS_INPUT, not VISION_POSITION_ESTIMATE (ADR-003). eph comes straight from
// the covariance and is NEVER smoothed.
//
// Publishes:
//   /mavros/gps_input/gps_input                        @ 5 Hz
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T25-T27.

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

class GeolocMavlinkNode : public rclcpp::Node {
 public:
  GeolocMavlinkNode() : rclcpp::Node("geoloc_mavlink") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 5);

    // TODO(T25-T27): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_mavlink skeleton up (task T25-T27)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocMavlinkNode>());
  rclcpp::shutdown();
  return 0;
}

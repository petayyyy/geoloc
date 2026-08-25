// geoloc_integrity -- SKELETON (task T21, T22)
//
// RESPONSIBLE FOR: The accept/reject decision and the covariance estimate.
// NOT RESPONSIBLE FOR: Estimating SE(2).
//
// THE main safety interlock. Target IFR < 0.5%. The worst case is a SERIES of
// mutually consistent false fixes -- the consistency check cannot catch those;
// only peak_ratio and the chi2 gate can. Isaac scenario S-05 exists for
// exactly this and is blocking.
//
// Publishes:
//   /geoloc/fix                                        @ 2 Hz
//   /geoloc/fix_rejected                               (event-driven)
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T21, T22.

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

class GeolocIntegrityNode : public rclcpp::Node {
 public:
  GeolocIntegrityNode() : rclcpp::Node("geoloc_integrity") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 2);

    // TODO(T21, T22): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_integrity skeleton up (task T21, T22)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocIntegrityNode>());
  rclcpp::shutdown();
  return 0;
}

// geoloc_matcher -- SKELETON (task T16-T20)
//
// RESPONSIBLE FOR: Features, correspondences, SE(2) estimation, channel
// arbitration (primary -> fallback).
// NOT RESPONSIBLE FOR: Deciding whether a fix is valid. NEVER add a 'this one
// is obviously bad' shortcut here.
//
// Estimates 3 DoF only (ADR-002). peak_ratio is computed over SPATIALLY
// SEPARATED hypotheses -- on periodic structures it is the only signal
// distinguishing a correct match from one shifted by exactly one period.
//
// Publishes:
//   /geoloc/fix_raw                                    @ 2 Hz
//
// This is a day-one skeleton: it brings the topic up with placeholder content
// so the whole graph launches and every work stream has something to plug
// into. Implementation lands with task T16-T20.

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

class GeolocMatcherNode : public rclcpp::Node {
 public:
  GeolocMatcherNode() : rclcpp::Node("geoloc_matcher") {
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", 2);

    // TODO(T16-T20): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "geoloc_matcher skeleton up (task T16-T20)");
  }
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocMatcherNode>());
  rclcpp::shutdown();
  return 0;
}

// geoloc_matcher -- features, correspondences, SE(2) estimation, channel
// arbitration (T16-T20).
//
// RESPONSIBLE FOR: Features, correspondences, SE(2) estimation, channel
// arbitration (primary -> fallback).
// NOT RESPONSIBLE FOR: Deciding whether a fix is valid. NEVER add a 'this one
// is obviously bad' shortcut here. That decision lives only in geoloc_integrity.
//
// This node currently runs the T19 fallback channel: log-polar phase
// correlation on gradient orientation (CHANNEL_PHASE_CORR). The primary XFeat
// channel and the arbitration strategy (primary -> fallback after N failed
// frames) land with T16/T17. The channel always publishes a result with
// quality metrics and never judges it.
//
// Publishes:
//   /geoloc/fix_raw        geoloc_msgs/SE2Fix   (channel = 1)  @ up to 2 Hz
//
// Topic/service names follow 03-interfaces.md; final remapping and the exact
// prior-window derivation (from the fusion covariance) are T17's integration.

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "geoloc_msgs/msg/ortho_patch.hpp"
#include "geoloc_msgs/msg/se2_fix.hpp"
#include "geoloc_msgs/srv/map_window.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "geoloc_common/angles.hpp"
#include "geoloc_matcher/phase_corr.hpp"

using namespace std::chrono_literals;

namespace geoloc {

namespace {

GrayImage mono8ToGray(const sensor_msgs::msg::Image& img) {
  GrayImage g(img.width, img.height);
  for (size_t i = 0; i < img.data.size(); ++i) {
    g.data[i] = static_cast<double>(img.data[i]) / 255.0;
  }
  return g;
}

}  // namespace

class GeolocMatcherNode : public rclcpp::Node {
 public:
  GeolocMatcherNode() : rclcpp::Node("geoloc_matcher") {
    // T19 phase-correlation knobs (P0 rule 6: thresholds are CONFIGURATION).
    declare_parameter<double>("grad_thresh_rel", 0.0);
    declare_parameter<double>("coarse_max_deg", 18.0);
    declare_parameter<double>("coarse_step_deg", 6.0);
    declare_parameter<int>("nrho", 64);
    declare_parameter<int>("ntheta", 512);
    declare_parameter<double>("scale_check_tolerance", 0.10);
    declare_parameter<double>("prior_window_radius_m", 60.0);  // T17: from fusion covariance
    declare_parameter<double>("position_sigma_m", 8.0);
    declare_parameter<double>("yaw_sigma_deg", 1.5);
    declare_parameter<double>("basemap_bias_sigma_m", 3.0);

    matcher_config_.grad_thresh_rel = get_parameter("grad_thresh_rel").as_double();
    matcher_config_.coarse_max_deg = get_parameter("coarse_max_deg").as_double();
    matcher_config_.coarse_step_deg = get_parameter("coarse_step_deg").as_double();
    matcher_config_.nrho = get_parameter("nrho").as_int();
    matcher_config_.ntheta = get_parameter("ntheta").as_int();
    matcher_config_.scale_check_tolerance = get_parameter("scale_check_tolerance").as_double();
    matcher_ = std::make_unique<PhaseCorrMatcher>(matcher_config_);

    covariance_config_.position_sigma_m = get_parameter("position_sigma_m").as_double();
    covariance_config_.yaw_sigma_deg = get_parameter("yaw_sigma_deg").as_double();
    covariance_config_.basemap_bias_sigma_m = get_parameter("basemap_bias_sigma_m").as_double();

    patch_sub_ = create_subscription<geoloc_msgs::msg::OrthoPatch>(
        "/geoloc/ortho_patch", rclcpp::SensorDataQoS(),
        [this](const geoloc_msgs::msg::OrthoPatch::SharedPtr p) { onPatch(p); });

    fix_pub_ = create_publisher<geoloc_msgs::msg::SE2Fix>("/geoloc/fix_raw", 10);

    map_client_ = create_client<geoloc_msgs::srv::MapWindow>("/geoloc_map/map_window");

    RCLCPP_INFO(get_logger(), "geoloc_matcher up (T19 phase-correlation channel)");
  }

 private:
  void onPatch(const geoloc_msgs::msg::OrthoPatch::SharedPtr patch) {
    if (!map_client_->wait_for_service(50ms)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "map_window service unavailable");
      return;
    }

    auto req = std::make_shared<geoloc_msgs::srv::MapWindow::Request>();
    // Prior centre is the patch centre; the phase-correlation channel searches
    // the prior window for the true location. T17 replaces this fixed radius
    // with R = 3*sqrt(sigma_e^2 + sigma_n^2) + margin from the fusion prior.
    req->center_east = patch->origin_east + patch->image.width * patch->gsd / 2.0;
    req->center_north = patch->origin_north - patch->image.height * patch->gsd / 2.0;
    req->radius_m = get_parameter("prior_window_radius_m").as_double();
    req->gsd = patch->gsd;
    req->with_descriptors = false;
    req->model_version = "";

    auto future = map_client_->async_send_request(req);
    const auto rc = rclcpp::spin_until_future_complete(shared_from_this(), future, 200ms);
    if (rc != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "map_window request timed out");
      return;
    }
    const auto resp = future.get();
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "map_window refused: %s", resp->message.c_str());
      return;
    }
    matchAndPublish(patch, *resp);
  }

  void matchAndPublish(const geoloc_msgs::msg::OrthoPatch::SharedPtr& patch,
                       const geoloc_msgs::srv::MapWindow::Response& window) {
    const auto t0 = get_clock()->now();

    GrayImage query = mono8ToGray(patch->image);
    GrayImage map = mono8ToGray(window.image);
    GrayImage confidence = mono8ToGray(patch->confidence);

    const PhaseCorrResult r = matcher_->match(query, map, &confidence);

    geoloc_msgs::msg::SE2Fix fix;
    fix.header.stamp = patch->header.stamp;  // sensor timestamp, never receive time
    fix.header.frame_id = "map_enu";
    fix.channel = geoloc_msgs::msg::SE2Fix::CHANNEL_PHASE_CORR;

    PhaseCorrFix f = phaseCorrToFix(r, patch->gsd, covariance_config_);

    // The shift from the matcher is in window pixel coordinates; fold in the
    // offset between the window origin and the patch's reported origin so the
    // delta is a correction in map_enu.
    const double win_origin_east = window.origin_east;
    const double win_origin_north = window.origin_north;
    fix.delta_east = (win_origin_east + r.shift_east_px * window.gsd) - patch->origin_east;
    fix.delta_north = (win_origin_north + r.shift_north_px * window.gsd) - patch->origin_north;
    fix.delta_yaw = f.delta_yaw;

    for (int i = 0; i < 9; ++i) fix.covariance[i] = f.covariance(i / 3, i % 3);

    fix.n_correspondences = f.n_correspondences;
    fix.n_inliers = f.n_inliers;
    fix.inlier_ratio = f.inlier_ratio;
    fix.covisibility = f.covisibility;
    fix.peak_ratio = f.peak_ratio;
    fix.residual_rms_px = f.residual_rms_px;
    fix.spatial_spread = f.spatial_spread;
    fix.mean_confidence = f.mean_confidence;
    fix.scale_check = f.scale_check;

    const auto t1 = get_clock()->now();
    fix.processing_time_ms = static_cast<float>((t1 - t0).seconds() * 1e3);

    fix_pub_->publish(fix);
  }

  PhaseCorrConfig matcher_config_;
  PhaseCorrCovarianceConfig covariance_config_;
  std::unique_ptr<PhaseCorrMatcher> matcher_;

  rclcpp::Subscription<geoloc_msgs::msg::OrthoPatch>::SharedPtr patch_sub_;
  rclcpp::Publisher<geoloc_msgs::msg::SE2Fix>::SharedPtr fix_pub_;
  rclcpp::Client<geoloc_msgs::srv::MapWindow>::SharedPtr map_client_;
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::GeolocMatcherNode>());
  rclcpp::shutdown();
  return 0;
}

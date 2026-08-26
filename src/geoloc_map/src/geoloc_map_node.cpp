// Copyright 2026 geoloc team.
// geoloc_map -- the onboard basemap window service (task T07).
//
// RESPONSIBLE FOR: Storing and serving the basemap window, DEM and semantics;
// caching map-side descriptors.
// NOT RESPONSIBLE FOR: Matching.
//
// The prior window from the fusion covariance replaces a retrieval network
// (ADR-005). Descriptors are cached ON BOARD, not precomputed offline. The
// cache invalidates on a >30% window shift, a requested-GSD change, or a
// matcher model-version change -- the last one is easy to forget and produces a
// silent incompatibility after a model update.
//
// Services:
//   ~/map_window        geoloc_msgs/srv/MapWindow

#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "geoloc_msgs/srv/map_window.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "geoloc_map/geopack.hpp"
#include "geoloc_map/map_window_service.hpp"

#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

namespace geoloc {

namespace {

/// Thresholds the mission YAML may supply as defaults (P0 rule 6: thresholds
/// are CONFIGURATION, never constants in code).
struct MapDefaults {
  std::string geopack_path;
  std::string primary_layer{"ortho_a"};
  double shift_fraction{0.30};
  bool invalidate_on_model_change{true};
  bool prefetch_enabled{true};
  double prefetch_lookahead_s{5.0};
  double prefetch_rate_hz{1.0};
  int descriptor_grid_stride{16};
  int max_memory_mb{200};
};

MapDefaults defaultsFromYaml(const std::string& path, const MapDefaults& fallback) {
  MapDefaults d = fallback;
  const YAML::Node m = YAML::LoadFile(path);
  if (m["mission"] && m["mission"]["geopack_path"]) {
    d.geopack_path = m["mission"]["geopack_path"].as<std::string>();
  }
  if (m["map"]) {
    const YAML::Node map = m["map"];
    if (map["primary_layer"]) d.primary_layer = map["primary_layer"].as<std::string>();
    if (map["descriptor_cache"] && map["descriptor_cache"]["invalidate_on_shift_fraction"]) {
      d.shift_fraction = map["descriptor_cache"]["invalidate_on_shift_fraction"].as<double>();
    }
    if (map["descriptor_cache"] && map["descriptor_cache"]["invalidate_on_model_change"]) {
      d.invalidate_on_model_change =
          map["descriptor_cache"]["invalidate_on_model_change"].as<bool>();
    }
    if (map["prefetch_enabled"]) d.prefetch_enabled = map["prefetch_enabled"].as<bool>();
    if (map["prefetch_lookahead_s"]) {
      d.prefetch_lookahead_s = map["prefetch_lookahead_s"].as<double>();
    }
    if (map["max_memory_mb"]) d.max_memory_mb = map["max_memory_mb"].as<int>();
  }
  return d;
}

sensor_msgs::msg::Image makeMono8(const std::vector<uint8_t>& pixels, uint32_t width,
                                  uint32_t height) {
  sensor_msgs::msg::Image img;
  img.height = height;
  img.width = width;
  img.encoding = "mono8";
  img.is_bigendian = false;
  img.step = width;
  img.data = pixels;  // copy: the service owns the reusable buffer
  return img;
}

}  // namespace

class GeolocMapNode : public rclcpp::Node {
 public:
  GeolocMapNode() : rclcpp::Node("geoloc_map") {
    declare_parameter<std::string>("mission_config", "");

    MapDefaults defaults;
    const std::string mission_config = get_parameter("mission_config").as_string();
    if (!mission_config.empty()) {
      defaults = defaultsFromYaml(mission_config, defaults);
    }

    // Declare every knob with the YAML value as default so launch-file
    // overrides still win.
    declare_parameter<std::string>("geopack_path", defaults.geopack_path);
    declare_parameter<std::string>("primary_layer", defaults.primary_layer);
    declare_parameter<double>("invalidate_on_shift_fraction", defaults.shift_fraction);
    declare_parameter<bool>("invalidate_on_model_change", defaults.invalidate_on_model_change);
    declare_parameter<bool>("prefetch_enabled", defaults.prefetch_enabled);
    declare_parameter<double>("prefetch_lookahead_s", defaults.prefetch_lookahead_s);
    declare_parameter<double>("prefetch_rate_hz", defaults.prefetch_rate_hz);
    declare_parameter<int>("descriptor_grid_stride", defaults.descriptor_grid_stride);
    declare_parameter<int>("max_memory_mb", defaults.max_memory_mb);

    const std::string geopack_path = get_parameter("geopack_path").as_string();
    if (geopack_path.empty()) {
      RCLCPP_FATAL(get_logger(), "no geopack configured; set 'mission_config' or 'geopack_path'");
      throw std::runtime_error("no geopack path");
    }
    geopack_ = std::make_unique<Geopack>(Geopack::load(geopack_path));
    RCLCPP_INFO(get_logger(), "geopack loaded: crs=%s origin=(%.6f, %.6f)", geopack_->crs().c_str(),
                rad2deg(geopack_->origin().lat), rad2deg(geopack_->origin().lon));

    const std::string layer_name = get_parameter("primary_layer").as_string();
    layer_ = geopack_->layer(layer_name);
    if (!layer_) {
      RCLCPP_FATAL(get_logger(), "primary layer '%s' not in geopack", layer_name.c_str());
      throw std::runtime_error("missing primary layer");
    }

    MapWindowService::Config cfg;
    cfg.shift_fraction = get_parameter("invalidate_on_shift_fraction").as_double();
    cfg.invalidate_on_model_change = get_parameter("invalidate_on_model_change").as_bool();
    service_ = std::make_unique<MapWindowService>(
        cfg,
        std::make_shared<GridDescriptorEngine>(get_parameter("descriptor_grid_stride").as_int()));

    srv_ = create_service<geoloc_msgs::srv::MapWindow>(
        "~/map_window", [this](const std::shared_ptr<geoloc_msgs::srv::MapWindow::Request> req,
                               std::shared_ptr<geoloc_msgs::srv::MapWindow::Response> resp) {
          handleRequest(*req, *resp);
        });

    if (get_parameter("prefetch_enabled").as_bool()) {
      const double rate = std::max(get_parameter("prefetch_rate_hz").as_double(), 0.1);
      prefetch_timer_ =
          create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { prefetchTick(); });
    }

    RCLCPP_INFO(get_logger(), "geoloc_map serving layer '%s'", layer_name.c_str());
  }

 private:
  void handleRequest(const geoloc_msgs::srv::MapWindow::Request& req,
                     geoloc_msgs::srv::MapWindow::Response& resp) {
    const Eigen::Vector2d center(req.center_east, req.center_north);

    // Track the request centre for the prefetch velocity estimate.
    const auto now = get_clock()->now();
    if (have_prev_request_) {
      const double dt = (now - prev_request_time_).seconds();
      if (dt > 1e-3) {
        velocity_ = (center - prev_request_center_) / dt;
      }
    }
    prev_request_center_ = center;
    prev_request_time_ = now;
    have_prev_request_ = true;

    const MapWindowResponse win = service_->serve(*layer_, center, req.radius_m, req.gsd,
                                                  req.with_descriptors, req.model_version);

    resp.success = win.success;
    resp.message = win.message;
    if (!win.success) {
      RCLCPP_WARN(get_logger(), "MapWindow refused: %s", win.message.c_str());
      return;
    }

    resp.origin_east = win.origin_east;
    resp.origin_north = win.origin_north;
    resp.gsd = win.gsd;
    resp.image = makeMono8(win.image, win.width, win.height);
    resp.validity = makeMono8(win.validity, win.width, win.height);

    if (req.with_descriptors) {
      resp.n_keypoints = win.descriptors.n_keypoints;
      resp.keypoints_xy = win.descriptors.keypoints_xy;
      resp.descriptors = win.descriptors.descriptors;
    }
  }

  void prefetchTick() {
    if (!have_prev_request_ || !layer_) return;
    const double lookahead = get_parameter("prefetch_lookahead_s").as_double();
    const double speed = velocity_.norm();
    if (speed < 1e-3) return;  // stationary: nothing ahead to prefetch
    const Eigen::Vector2d ahead = prev_request_center_ + velocity_ * lookahead;
    layer_->prefetch(ahead, 2.0 * lookahead * speed);
  }

  std::unique_ptr<Geopack> geopack_;
  GeopackLayer* layer_{nullptr};
  std::unique_ptr<MapWindowService> service_;
  rclcpp::Service<geoloc_msgs::srv::MapWindow>::SharedPtr srv_;
  rclcpp::TimerBase::SharedPtr prefetch_timer_;

  bool have_prev_request_{false};
  rclcpp::Time prev_request_time_;
  Eigen::Vector2d prev_request_center_{Eigen::Vector2d::Zero()};
  Eigen::Vector2d velocity_{Eigen::Vector2d::Zero()};
};

}  // namespace geoloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<geoloc::GeolocMapNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("geoloc_map"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

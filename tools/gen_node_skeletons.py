#!/usr/bin/env python3
"""Generate skeleton ROS 2 packages for the geoloc pipeline nodes.

Rationale (task card T01): the graph must come up on day one, empty. Every node
publishes its contract topics with placeholder content and a launch file brings
up the whole graph. This gives six parallel work streams something to plug into
and makes integration continuous rather than an event late in the project.

Each node's responsibility boundary is copied into its header comment from
plan/02-architecture.md section 3, because the most important boundary in this
system is easy to erode:

    geoloc_matcher ALWAYS produces a result with quality metrics and NEVER
    decides whether it is valid. That decision lives only in geoloc_integrity.

Run from the repo root:  python3 tools/gen_node_skeletons.py
"""

import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

NODES = [
    {
        "pkg": "geoloc_map",
        "task": "T07",
        "responsible": "Storing and serving the basemap window, DEM and semantics; "
                       "caching map-side descriptors.",
        "not_responsible": "Matching.",
        "publishes": [],
        "services": [("geoloc_msgs/srv/MapWindow", "~/map_window")],
        "note": "Prior window from the fusion covariance replaces a retrieval "
                "network (ADR-005). Descriptors are cached ON BOARD, not "
                "precomputed offline.",
    },
    {
        "pkg": "geoloc_ortho",
        "task": "T14, T15",
        "responsible": "Geometry: cloud -> deskew -> local DSM -> true-ortho patch "
                       "+ per-pixel confidence map.",
        "not_responsible": "Anything involving features or the map.",
        "publishes": [("geoloc_msgs/msg/OrthoPatch", "/geoloc/ortho_patch", 2.0)],
        "services": [],
        "note": "Avia FOV is NARROWER than the camera frame -- ~30% of patch "
                "width has no lidar coverage and is filled from Copernicus DEM. "
                "The confidence map must be honest about that.",
    },
    {
        "pkg": "geoloc_matcher",
        "task": "T16-T20",
        "responsible": "Features, correspondences, SE(2) estimation, channel "
                       "arbitration (primary -> fallback).",
        "not_responsible": "Deciding whether a fix is valid. NEVER add a "
                           "'this one is obviously bad' shortcut here.",
        "publishes": [("geoloc_msgs/msg/SE2Fix", "/geoloc/fix_raw", 2.0)],
        "services": [],
        "note": "Estimates 3 DoF only (ADR-002). peak_ratio is computed over "
                "SPATIALLY SEPARATED hypotheses -- on periodic structures it is "
                "the only signal distinguishing a correct match from one shifted "
                "by exactly one period.",
    },
    {
        "pkg": "geoloc_integrity",
        "task": "T21, T22",
        "responsible": "The accept/reject decision and the covariance estimate.",
        "not_responsible": "Estimating SE(2).",
        "publishes": [
            ("geoloc_msgs/msg/SE2Fix", "/geoloc/fix", 2.0),
            ("geoloc_msgs/msg/SE2Fix", "/geoloc/fix_rejected", 0.0),
        ],
        "services": [],
        "note": "THE main safety interlock. Target IFR < 0.5%. The worst case is "
                "a SERIES of mutually consistent false fixes -- the consistency "
                "check cannot catch those; only peak_ratio and the chi2 gate can. "
                "Isaac scenario S-05 exists for exactly this and is blocking.",
    },
    {
        "pkg": "geoloc_fusion",
        "task": "T23, T24",
        "responsible": "Factor graph, global pose, prior window, mode machine, "
                       "tf map_enu -> odom.",
        "not_responsible": "The autopilot interface.",
        "publishes": [
            ("geometry_msgs/msg/PoseWithCovarianceStamped", "/geoloc/global_pose", 10.0),
            ("geoloc_msgs/msg/GeolocStatus", "/geoloc/status", 1.0),
        ],
        "services": [],
        "note": "A fix corrects map_enu -> odom, NOT the aircraft pose directly. "
                "Odometry stays smooth and a correction jump never becomes a "
                "velocity jump for the control loop.",
    },
    {
        "pkg": "geoloc_mavlink",
        "task": "T25-T27",
        "responsible": "Translating the estimate into MAVLink GPS_INPUT; "
                       "failsafe escalation levels.",
        "not_responsible": "Judging quality. And NEVER commanding a flight mode "
                           "change -- T27-U-02 audits for it.",
        "publishes": [("mavros_msgs/msg/GPSINPUT", "/mavros/gps_input/gps_input", 5.0)],
        "services": [],
        "note": "GPS_INPUT, not VISION_POSITION_ESTIMATE (ADR-003). eph comes "
                "straight from the covariance and is NEVER smoothed.",
    },
]

PACKAGE_XML = """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{pkg}</name>
  <version>0.1.0</version>
  <description>{desc}</description>
  <maintainer email="dev@example.invalid">geoloc team</maintainer>
  <license>Proprietary</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>geoloc_msgs</depend>
  <depend>geoloc_common</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

CMAKE = """cmake_minimum_required(VERSION 3.16)
project({pkg})

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geoloc_msgs REQUIRED)
find_package(geoloc_common REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable({pkg}_node src/{pkg}_node.cpp)
ament_target_dependencies({pkg}_node
  rclcpp geoloc_msgs geoloc_common std_msgs sensor_msgs geometry_msgs tf2_ros)

install(TARGETS {pkg}_node DESTINATION lib/${{PROJECT_NAME}})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
"""


def comment_block(text, width=76):
    """Wrap prose into `// ` comment lines at a fixed width."""
    return "\n".join("// " + line for line in textwrap.wrap(text, width))


def node_source(n):
    pubs = "\n".join(
        f"//   {t:<50} @ {r:g} Hz" if r else f"//   {t:<50} (event-driven)"
        for _, t, r in n["publishes"]
    ) or "//   (none)"
    srvs = "\n".join(f"//   {t:<50} [service]" for _, t in n["services"])

    header = "\n".join(filter(None, [
        f'// {n["pkg"]} -- SKELETON (task {n["task"]})',
        "//",
        comment_block("RESPONSIBLE FOR: " + n["responsible"]),
        comment_block("NOT RESPONSIBLE FOR: " + n["not_responsible"]),
        "//",
        comment_block(n["note"]),
        "//",
        "// Publishes:",
        pubs,
        srvs,
        "//",
        comment_block(
            "This is a day-one skeleton: it brings the topic up with placeholder "
            "content so the whole graph launches and every work stream has "
            f'something to plug into. Implementation lands with task {n["task"]}.'
        ),
    ]))

    body = f'''
#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "geoloc_common/angles.hpp"
#include "geoloc_common/covariance.hpp"
#include "geoloc_common/geodetic.hpp"
#include "geoloc_common/raster.hpp"
#include "geoloc_common/se2.hpp"

using namespace std::chrono_literals;

namespace geoloc {{

class {n["cls"]} : public rclcpp::Node {{
 public:
  {n["cls"]}() : rclcpp::Node("{n["pkg"]}") {{
    declare_parameter<std::string>("mission_config", "");
    declare_parameter<double>("rate_hz", {n["rate"]:g});

    // TODO({n["task"]}): declare the real parameters from
    // configs/mission_template.yaml. Thresholds are CONFIGURATION,
    // never constants in code.

    RCLCPP_INFO(get_logger(), "{n["pkg"]} skeleton up (task {n["task"]})");
  }}
}};

}}  // namespace geoloc

int main(int argc, char** argv) {{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<geoloc::{n["cls"]}>());
  rclcpp::shutdown();
  return 0;
}}
'''
    return header + "\n" + body


def main():
    for n in NODES:
        n["cls"] = "".join(p.capitalize() for p in n["pkg"].split("_")) + "Node"
        n["rate"] = n["publishes"][0][2] if n["publishes"] and n["publishes"][0][2] else 1.0
        n["desc"] = f'{n["responsible"]} Skeleton; implementation in task {n["task"]}.'

        pkg_dir = SRC / n["pkg"]
        (pkg_dir / "src").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "package.xml").write_text(PACKAGE_XML.format(**n), encoding="utf-8")
        (pkg_dir / "CMakeLists.txt").write_text(CMAKE.format(**n), encoding="utf-8")
        (pkg_dir / "src" / f'{n["pkg"]}_node.cpp').write_text(node_source(n), encoding="utf-8")
        print(f'generated {n["pkg"]}')


if __name__ == "__main__":
    main()

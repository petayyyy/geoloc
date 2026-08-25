"""Bring up the full geoloc graph.

Task card T01 requires this to work on day one, with every node still empty.
An empty graph that launches gives six parallel work streams something to plug
into, and makes integration continuous rather than an event late in the project.

Verify with:
    ros2 launch geoloc_bringup all.launch.py
    ros2 topic list        # must show every topic from plan/03-interfaces.md

Note this launches only OUR nodes. FAST-LIVO2, the Livox driver, the camera
driver and MAVROS come up separately -- FAST-LIVO2 already runs on the board and
we do not want to own its lifecycle.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Node order matches the data flow in plan/02-architecture.md section 2:
#   ortho -> matcher -> integrity -> fusion -> mavlink,  with map serving matcher.
GEOLOC_NODES = [
    ("geoloc_map", "T07"),
    ("geoloc_ortho", "T14, T15"),
    ("geoloc_matcher", "T16-T20"),
    ("geoloc_integrity", "T21, T22"),
    ("geoloc_fusion", "T23, T24"),
    ("geoloc_mavlink", "T25-T27"),
]


def generate_launch_description():
    mission_config = LaunchConfiguration("mission_config")
    log_level = LaunchConfiguration("log_level")
    enable_mavlink = LaunchConfiguration("enable_mavlink")

    args = [
        DeclareLaunchArgument(
            "mission_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("geoloc_bringup"), "config", "mission_template.yaml"]
            ),
            description="Mission YAML: geopack path, ENU anchor, all thresholds.",
        ),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument(
            "enable_mavlink",
            default_value="true",
            description=(
                "Set false for offline replay and OrthoSim runs, where nothing "
                "should reach an autopilot."
            ),
        ),
    ]

    nodes = []
    for pkg, _task in GEOLOC_NODES:
        # geoloc_mavlink is the only node that talks to the aircraft, so it is
        # the only one that is conditionally disabled.
        condition = IfCondition(enable_mavlink) if pkg == "geoloc_mavlink" else None
        nodes.append(
            Node(
                package=pkg,
                executable=f"{pkg}_node",
                name=pkg,
                output="screen",
                parameters=[{"mission_config": mission_config}],
                arguments=["--ros-args", "--log-level", log_level],
                condition=condition,
            )
        )

    return LaunchDescription(args + nodes)

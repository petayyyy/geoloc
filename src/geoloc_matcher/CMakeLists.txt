cmake_minimum_required(VERSION 3.16)
project(geoloc_matcher)

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

add_executable(geoloc_matcher_node src/geoloc_matcher_node.cpp)
ament_target_dependencies(geoloc_matcher_node
  rclcpp geoloc_msgs geoloc_common std_msgs sensor_msgs geometry_msgs tf2_ros)

install(TARGETS geoloc_matcher_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

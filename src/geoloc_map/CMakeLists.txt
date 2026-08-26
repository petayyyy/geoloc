cmake_minimum_required(VERSION 3.16)
project(geoloc_map)

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
find_package(TIFF REQUIRED)
find_package(yaml-cpp REQUIRED)

# Core library: COG reader + geopack (shared by the node and the integration
# test). Header-only parts (utm, pyramid, descriptor cache, window service)
# are installed for the dependency-free unit test.
add_library(geoloc_map_core SHARED
  src/cog.cpp
  src/geopack.cpp
)
target_include_directories(geoloc_map_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(geoloc_map_core geoloc_common)
target_include_directories(geoloc_map_core PUBLIC ${TIFF_INCLUDE_DIRS})
target_link_libraries(geoloc_map_core ${TIFF_LIBRARIES} yaml-cpp)

add_executable(geoloc_map_node src/geoloc_map_node.cpp)
target_link_libraries(geoloc_map_node geoloc_map_core)
ament_target_dependencies(geoloc_map_node
  rclcpp geoloc_msgs geoloc_common std_msgs sensor_msgs geometry_msgs tf2_ros)

install(TARGETS geoloc_map_node geoloc_map_core
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # T07-U-01..04: pure unit tests, dependency-free (header-only logic + Eigen).
  add_executable(test_map_core test/test_map_core.cpp)
  target_include_directories(test_map_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_target_dependencies(test_map_core geoloc_common)
  ament_add_test(map_core COMMAND $<TARGET_FILE:test_map_core> TIMEOUT 120)

  # Integration test against the real Maykop geopack. Skips (exit 0) when the
  # mission data is absent, e.g. in a bare cross-build container.
  add_executable(test_geopack test/test_geopack.cpp)
  target_link_libraries(test_geopack geoloc_map_core)
  ament_target_dependencies(test_geopack geoloc_common)
  set(_GEOPACK_MANIFEST
    ${CMAKE_CURRENT_SOURCE_DIR}/../../data/missions/maykop-corridor-2026-08.geopack/manifest.yaml)
  ament_add_test(geopack_integration
    COMMAND $<TARGET_FILE:test_geopack> ${_GEOPACK_MANIFEST} TIMEOUT 120)
endif()

ament_package()

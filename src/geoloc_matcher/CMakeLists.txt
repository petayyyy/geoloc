cmake_minimum_required(VERSION 3.16)
project(geoloc_matcher)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geoloc_msgs REQUIRED)
find_package(geoloc_common REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Header-only matcher core (T16-T20): FFT + phase correlation + SE2Fix mapping.
# Dependency-free on purpose (no FFTW), so the level-0 test builds in a bare
# cross-build container.
add_library(geoloc_matcher_core INTERFACE)
target_include_directories(geoloc_matcher_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(geoloc_matcher_core INTERFACE geoloc_common Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS geoloc_matcher_core EXPORT ${PROJECT_NAME}Targets)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(geoloc_common Eigen3)
ament_export_include_directories(include)

add_executable(geoloc_matcher_node src/geoloc_matcher_node.cpp)
target_link_libraries(geoloc_matcher_node geoloc_matcher_core)
ament_target_dependencies(geoloc_matcher_node
  rclcpp geoloc_msgs geoloc_common std_msgs sensor_msgs geometry_msgs tf2_ros)

install(TARGETS geoloc_matcher_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # T19-U-01..05 + peak_ratio / SE2Fix mapping. Dependency-free.
  add_executable(test_phase_corr test/test_phase_corr.cpp)
  target_link_libraries(test_phase_corr geoloc_matcher_core)
  ament_add_test(phase_corr COMMAND $<TARGET_FILE:test_phase_corr> TIMEOUT 120
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)
endif()

ament_package()

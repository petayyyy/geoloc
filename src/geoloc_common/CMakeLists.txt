cmake_minimum_required(VERSION 3.16)
project(geoloc_common)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Property tests T01-U-01 .. T01-U-05. Dependency-free on purpose, so the
  # same binary runs in a bare cross-build container.
  add_executable(test_geoloc_common test/test_geoloc_common.cpp)
  target_link_libraries(test_geoloc_common ${PROJECT_NAME})
  ament_add_test(geoloc_common_properties
    COMMAND $<TARGET_FILE:test_geoloc_common>
    TIMEOUT 120
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
  )
endif()

ament_package()

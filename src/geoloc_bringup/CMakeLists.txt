cmake_minimum_required(VERSION 3.16)
project(geoloc_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../configs/
        DESTINATION share/${PROJECT_NAME}/config)

ament_package()

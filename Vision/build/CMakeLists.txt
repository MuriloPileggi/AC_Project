cmake_minimum_required(VERSION 3.1)
project(GeneralVision)
project(RoadVision)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(GeneralVision ../src/main_general.cpp)
add_executable(RoadVision ../src/main_road.cpp)
target_link_libraries(GeneralVision ${OpenCV_LIBS})
target_link_libraries(RoadVision ${OpenCV_LIBS})


# CPack configuration - Package executable for Raspberry Pi testing
#set(CPACK_GENERATOR "DEB")
#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "AC_VISION")
#set(CPACK_PACKAGE_VERSION "1.0.0")
#include(CPack)
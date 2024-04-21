cmake_minimum_required(VERSION 3.1)
project(Vision)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Vision ../src/main.cpp)
target_link_libraries(Vision ${OpenCV_LIBS})


# CPack configuration - Package executable for Raspberry Pi testing
#set(CPACK_GENERATOR "DEB")
#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "AC_VISION")
#set(CPACK_PACKAGE_VERSION "1.0.0")
#include(CPack)
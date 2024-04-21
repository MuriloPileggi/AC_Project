cmake_minimum_required(VERSION 3.1)
project(Serial)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Serial ../../Serial/src/main_serial.cpp)
target_link_libraries(Serial ${OpenCV_LIBS})


# CPack configuration - Package executable for Raspberry Pi testing
#set(CPACK_GENERATOR "DEB")
#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "AC_VISION")
#set(CPACK_PACKAGE_VERSION "1.0.0")
#include(CPack)
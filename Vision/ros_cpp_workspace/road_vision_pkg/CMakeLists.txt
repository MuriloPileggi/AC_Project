cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_PREFIX_PATH "/opt/ros/iron")

#project(general_vision)
project(road_vision_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)

#add_executable(general_vision ../src/main_general.cpp)
add_executable(road_vision_node ../../src/main_road_vision_node.cpp)

#target_link_libraries(general_vision ${OpenCV_LIBS})
target_link_libraries(road_vision_node ${OpenCV_LIBS})

#ament_target_dependencies(general_vision rclcpp std_msgs)
ament_target_dependencies(road_vision_node rclcpp std_msgs std_srvs)

install(TARGETS
  #general_vision
  road_vision_node
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()

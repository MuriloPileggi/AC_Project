from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='road_vision_pkg',
            executable='road_vision_node',  # This should match the executable name in your CMakeLists.txt
            name='road_vision_node',
            output='screen'
        )
    ])
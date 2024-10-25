from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='frontal_vision_pkg',
            executable='frontal_vision_node',  # This should match the executable name in your CMakeLists.txt
            name='frontal_vision_node',
            output='screen'
        )
    ])
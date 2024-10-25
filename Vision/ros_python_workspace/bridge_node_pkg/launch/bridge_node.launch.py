from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bridge_node_pkg',
            executable='bridge_node',  # This should match the name in your setup.py entry_points
            name='bridge_node',
            output='screen'
        )
    ])
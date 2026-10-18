from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cave_explorer',
            executable='demo_controller',
            name='controller_node',
            output='screen',
            prefix='xterm -e',
        ),
    ])
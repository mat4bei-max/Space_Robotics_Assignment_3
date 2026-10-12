from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Test arguments
    publish_period = DeclareLaunchArgument('publish_period', default_value='2.0')

    # Image publisher and detection subscriber; launch YOLO separately
    test_node = Node(
        package='cave_explorer', executable='yolo_test', output='screen',
        parameters=[{
            'publish_period': ParameterValue(LaunchConfiguration('publish_period'), value_type=float),
            'use_sim_time': False,
        }],
    )
    return LaunchDescription([publish_period, test_node])

"""Run geometry analysis alongside the existing SLAM/navigation launches."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='cave_explorer', executable='cave_geometry', output='screen',
             parameters=[PathJoinSubstitution([FindPackageShare('cave_explorer'),
                                              'config', 'geometry_params.yaml']),
                         {'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])

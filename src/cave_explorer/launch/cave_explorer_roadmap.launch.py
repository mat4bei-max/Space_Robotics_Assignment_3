"""Online roadmap alongside exploration; selected and baseline share map/pose updates."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('mode', default_value='distance_guided',
                              choices=['baseline', 'uniform', 'distance_guided']),
        DeclareLaunchArgument('metrics_file', default_value=''),
        DeclareLaunchArgument('record_directory', default_value=''),
        DeclareLaunchArgument('evaluation_pairs_file', default_value=''),
        DeclareLaunchArgument('compare_uniform', default_value='true'),
        Node(package='cave_explorer', executable='online_roadmap', output='screen',
             parameters=[PathJoinSubstitution([FindPackageShare('cave_explorer'),
                                              'config', 'roadmap_params.yaml']),
                         {'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'mode': LaunchConfiguration('mode'),
                          'metrics_file': ParameterValue(LaunchConfiguration('metrics_file'), value_type=str),
                          'record_directory': ParameterValue(LaunchConfiguration('record_directory'), value_type=str),
                          'evaluation_pairs_file': ParameterValue(LaunchConfiguration('evaluation_pairs_file'), value_type=str),
                          'compare_uniform': ParameterValue(LaunchConfiguration('compare_uniform'), value_type=bool)}]),
    ])

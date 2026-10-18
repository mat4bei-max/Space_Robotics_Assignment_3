"""Advanced 1 perception with optional autonomous exploration."""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package = Path(get_package_share_directory('cave_explorer'))
    config_file = package / 'config/detection_config.yaml'
    config = yaml.safe_load(config_file.read_text())
    detector = config['/yolo/yolo_node']['ros__parameters']
    default_model = package / 'artifact_detection' / detector['model']
    return LaunchDescription([
        DeclareLaunchArgument('enable_planning', default_value='false',
                              choices=['true', 'false'],
                              description='Enable exploration; requires the navigation launch.'),
        DeclareLaunchArgument('effect', default_value='none',
                              choices=['none', 'darkness', 'motion_blur']),
        DeclareLaunchArgument('severity', default_value='mild',
                              choices=['mild', 'moderate', 'severe']),
        DeclareLaunchArgument('model', default_value=str(default_model)),
        DeclareLaunchArgument('threshold', default_value=str(detector['threshold'])),
        DeclareLaunchArgument('device', default_value=detector['device']),
        Node(package='cave_explorer', executable='cave_explorer',
             name='cave_explorer_node', output='screen', parameters=[str(config_file), {
                 'use_sim_time': True,
                 'enable_planning': ParameterValue(LaunchConfiguration('enable_planning'), value_type=bool),
                 'use_depth_detection': False, 'activate_rgb_capture': False,
                 'activate_depth_capture': False, 'activate_processed_depth_capture': False,
                 'degradation_effect': LaunchConfiguration('effect'),
                 'degradation_severity': LaunchConfiguration('severity'),
             }]),
        Node(package='yolo_ros', executable='yolo_node', name='yolo_node',
             namespace='yolo', output='screen', parameters=[str(config_file), {
                 'use_sim_time': True, 'model': LaunchConfiguration('model'),
                 'device': LaunchConfiguration('device'),
                 'threshold': ParameterValue(LaunchConfiguration('threshold'), value_type=float),
             }], remappings=[('image_raw', '/artifact_detection/image')]),
    ])

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_detector(context):
    # Paths to configuration and detection assets
    package_path = Path(get_package_share_directory('cave_explorer'))
    config_file = package_path / 'config' / 'detection_config.yaml'
    with config_file.open() as stream:
        config = yaml.safe_load(stream)

    # Select the weights configured for the requested mode
    mode = LaunchConfiguration('detection_mode').perform(context)
    models = config['/yolo/yolo_node']['ros__parameters']['detection_models']
    if mode not in models:
        raise ValueError(f'No detection model configured for mode: {mode}')
    model_path = package_path / 'artifact_detection' / models[mode]
    if not model_path.is_file():
        raise FileNotFoundError(f'Artifact model is not installed: {model_path}')

    # YOLO detection node; all other parameters come from the existing config
    yolo_node = Node(
        package='yolo_ros',
        executable='yolo_node',
        name='yolo_node',
        namespace='yolo',
        output='screen',
        parameters=[
            str(config_file),
            {
                'model': str(model_path),
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool,
                ),
            },
        ],
        remappings=[('image_raw', '/artifact_detection/image')],
    )
    return [LogInfo(msg=f'YOLO mode: {mode}; weights: {model_path}'), yolo_node]


def generate_launch_description():
    # Launch arguments
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'detection_mode',
            default_value='rgb',
            choices=['rgb', 'depth', 'degraded'],
            description='Select the weights from detection_models in detection_config.yaml',
        ),
        OpaqueFunction(function=launch_detector),
    ])

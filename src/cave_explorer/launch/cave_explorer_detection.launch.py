from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Paths to configuration and detection assets
    package_path = Path(get_package_share_directory('cave_explorer'))
    config_file = package_path / 'config' / 'detection_config.yaml'

    with config_file.open() as stream:
        config = yaml.safe_load(stream)

    relative_model = config['/yolo/yolo_node']['ros__parameters']['model']
    model_path = package_path / 'artifact_detection' / relative_model

    if not model_path.is_file():
        raise FileNotFoundError(f'Artifact model is not installed: {model_path}')

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true'
    )

    # YOLO detection node
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
                    LaunchConfiguration('use_sim_time'),
                    value_type=bool,
                ),
            },
        ],
        remappings=[
            ('image_raw', '/artifact_detection/image'),
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        yolo_node,
    ])
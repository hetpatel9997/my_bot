import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    params = os.path.join(pkg, 'config', 'coverage_params.yaml')
    return LaunchDescription([
        Node(
            package='my_bot',
            executable='coverage_planner_node.py',
            name='coverage_planner',
            output='screen',
            parameters=[params],
        )
    ])

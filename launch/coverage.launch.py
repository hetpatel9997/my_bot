import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    params = os.path.join(pkg, 'config', 'coverage_params.yaml')
    # use_sim_time:=true in Gazebo, so goal stamps and TF lookups use /clock like Nav2 does.
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation (Gazebo) clock if true'),
        Node(
            package='my_bot',
            executable='coverage_planner_node.py',
            name='coverage_planner',
            output='screen',
            parameters=[params, {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
        )
    ])

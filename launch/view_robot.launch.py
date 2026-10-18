"""Shows the WeedBot model in RViz with sliders for the five arm joints and the caster.
   ros2 launch my_bot view_robot.launch.py            (real-robot geometry)
   ros2 launch my_bot view_robot.launch.py sim_mode:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    xacro_file = os.path.join(pkg, 'description', 'robot.urdf.xacro')
    sim_mode = LaunchConfiguration('sim_mode')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' use_ros2_control:=false sim_mode:=', sim_mode]), value_type=str)
    rviz_cfg = os.path.join(pkg, 'config', 'view_robot.rviz')
    return LaunchDescription([
        DeclareLaunchArgument('sim_mode', default_value='false'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_cfg] if os.path.exists(rviz_cfg) else []),
    ])

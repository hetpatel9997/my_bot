# Velocity multiplexer: Nav2 (/cmd_vel), keyboard (/cmd_vel_keyboard), joystick (/cmd_vel_joy)
# and the /e_stop lock -> /cmd_vel_mux -> collision_monitor -> /diff_cont/cmd_vel_unstamped.
# Priorities in config/twist_mux.yaml.
# Included by launch_robot.launch.py and launch_sim.launch.py (use_sim_time:=true).
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory('my_bot'), 'config', 'twist_mux.yaml')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[params, {'use_sim_time': use_sim_time}],
        # -> collision_monitor (launch/safety.launch.py) -> /diff_cont/cmd_vel_unstamped
        remappings=[('cmd_vel_out', '/cmd_vel_mux')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation (Gazebo) clock if true'),
        twist_mux,
    ])

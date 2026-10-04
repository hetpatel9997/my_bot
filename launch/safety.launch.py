# Safety layer (sim and real robot), included by launch_robot.launch.py and launch_sim.launch.py:
#   collision_monitor: /cmd_vel_mux (twist_mux output) -> /diff_cont/cmd_vel_unstamped, stopping
#                      or slowing the robot when /scan shows something in its zones
#   scan_self_filter:  /scan -> /scan_filtered without returns from the robot itself
#   safety_state_node: /safety_state = clear | slow | stop (spray only when clear)
# Zones and settings: config/safety.yaml. If the collision monitor is not active, NO command
# reaches diff_cont (fail safe).
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('my_bot'), 'config', 'safety.yaml')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='nav2_collision_monitor', executable='collision_monitor',
             name='collision_monitor', output='screen',
             parameters=[cfg, {'use_sim_time': use_sim_time}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_safety', output='screen',
             parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                          'bond_timeout': 10.0, 'node_names': ['collision_monitor']}]),
        Node(package='my_bot', executable='scan_self_filter.py', name='scan_self_filter',
             output='screen', parameters=[cfg, {'use_sim_time': use_sim_time}]),
        Node(package='my_bot', executable='safety_state_node.py', name='safety_state',
             output='screen', parameters=[cfg, {'use_sim_time': use_sim_time}]),
    ])

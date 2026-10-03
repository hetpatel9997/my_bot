# LD06 LiDAR -> /scan in laser_frame. Uses ldlidar_stl_ros2 from ~/robot_ws (not the retired ~/ros2_ws).
# The base_link -> laser_frame transform comes from the URDF (lidar.xacro) via robot_state_publisher,
# so unlike the vendor ld06.launch.py this does NOT start a static_transform_publisher.
#
#   ros2 launch my_bot lidar.launch.py                         # default port /dev/ldlidar
#   ros2 launch my_bot lidar.launch.py port_name:=/dev/ttyUSB1  # override the port
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('my_bot'), 'config', 'ldlidar.yaml')
    port_name = LaunchConfiguration('port_name')

    ldlidar = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ldlidar_node',
        output='screen',
        parameters=[params, {'port_name': port_name}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('port_name', default_value='/dev/ldlidar',
                              description='LiDAR serial port'),
        ldlidar,
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    ekf_params = os.path.join(pkg, 'config', 'ekf.yaml')

    imu = Node(
        package='my_bot',
        executable='mpu6050_node.py',
        name='mpu6050_node',
        output='screen',
        parameters=[{
            'i2c_bus': 1,
            'i2c_address': 0x68,
            'frame_id': 'imu_link',
            'rate_hz': 50.0,
            'calibration_samples': 300,
            'axis_signs': [1.0, 1.0, 1.0],
        }],
    )

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params],
        remappings=[('odometry/filtered', 'odom')],
    )

    return LaunchDescription([imu, ekf])

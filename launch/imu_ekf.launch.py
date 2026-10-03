import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    ekf_params = os.path.join(pkg, 'config', 'ekf.yaml')
    # Real robot: defaults (MPU6050 driver on I2C + EKF, wall clock).
    # Gazebo (launch_sim.launch.py): use_mpu6050:=false (Gazebo's IMU plugin publishes
    # /imu/data_raw instead, see description/imu_sim.xacro) and use_sim_time:=true.
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    use_mpu6050 = LaunchConfiguration('use_mpu6050')

    imu = Node(
        package='my_bot',
        executable='mpu6050_node.py',
        name='mpu6050_node',
        output='screen',
        condition=IfCondition(use_mpu6050),
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
        parameters=[ekf_params, {'use_sim_time': use_sim_time}],
        remappings=[('odometry/filtered', 'odom')],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument('use_mpu6050', default_value='true',
                              description='Start the MPU6050 I2C driver (false in Gazebo)'),
        imu,
        ekf,
    ])

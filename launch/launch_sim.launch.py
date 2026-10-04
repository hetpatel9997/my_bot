import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

def generate_launch_description():

    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!
    package_name='my_bot'

    # 1. Include the robot_state_publisher launch file
    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','rsp.launch.py'
                )]), launch_arguments={'use_sim_time': 'true',
                                  'wheel_mu': LaunchConfiguration('wheel_mu'),
                                  'sim_camera': LaunchConfiguration('sim_camera')}.items()
    )

    # 2. Include the Gazebo Classic launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                launch_arguments={'world': LaunchConfiguration('world'),
                                  'gui': LaunchConfiguration('gui')}.items()
             )

    # 3. Run the spawner node from the gazebo_ros package
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'my_bot'],
                        output='screen')


    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"]
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"]
    )


    # Same odometry pipeline as the real robot: robot_localization EKF fuses /diff_cont/odom with
    # /imu/data_raw (here from Gazebo's IMU plugin, description/imu_sim.xacro) and publishes /odom
    # plus odom->base_link (diff_cont's own odom TF is off, see config/gaz_ros2_ctl_use_sim.yaml).
    imu_ekf = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','imu_ekf.launch.py'
                )]), launch_arguments={'use_sim_time': 'true', 'use_mpu6050': 'false'}.items()
    )

    # Velocity multiplexer: Nav2 /cmd_vel (low), keyboard /cmd_vel_keyboard (higher), joystick,
    # and the /e_stop lock -> /diff_cont/cmd_vel_unstamped (config/twist_mux.yaml).
    twist_mux = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','twist_mux.launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Safety layer: collision_monitor between twist_mux and diff_cont + /safety_state
    safety = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','safety.launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Launch them all!
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='',
                              description='Gazebo world file (empty = Gazebo default empty world)'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show the Gazebo window (false = headless, much lighter)'),
        DeclareLaunchArgument('wheel_mu', default_value='1.0',
                              description='Wheel friction; e.g. 0.03 to test slip'),
        DeclareLaunchArgument('sim_camera', default_value='false',
                              description='Simulate the camera (costly; needed from Phase 1)'),
        rsp,
        gazebo,
        spawn_entity,
        diff_drive_spawner,
        joint_broad_spawner,
        imu_ekf,
        twist_mux,
        safety
    ])
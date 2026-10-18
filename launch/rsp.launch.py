import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import xacro


def _strip_comments(node):
    for child in list(node.childNodes):
        if child.nodeType == child.COMMENT_NODE:
            node.removeChild(child)
        else:
            _strip_comments(child)


def _robot_state_publisher(context):
    # Process the URDF file with the launch arguments
    pkg_path = os.path.join(get_package_share_directory('my_bot'))
    xacro_file = os.path.join(pkg_path,'description','robot.urdf.xacro')
    arg = lambda name: LaunchConfiguration(name).perform(context)
    doc = xacro.process_file(xacro_file, mappings={'use_ros2_control': arg('use_ros2_control'),
                                                   'sim_mode': arg('use_sim_time'),
                                                   'wheel_mu': arg('wheel_mu'),
                                                   'sim_camera': arg('sim_camera')})
    # Comments removed: Gazebo's ros2_control plugin hands robot_description to its controller
    # manager as a "--param robot_description:=..." argument, which is parsed as YAML, so any
    # "word: " in a URDF comment stopped the wheel controllers from loading. ROS never needs them.
    _strip_comments(doc)

    # Create a robot_state_publisher node
    params = {'robot_description': ParameterValue(doc.toxml(), value_type=str),
              'use_sim_time': LaunchConfiguration('use_sim_time')}
    return [Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )]


def generate_launch_description():

    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),

        DeclareLaunchArgument(
            'use_ros2_control',
            default_value='true',
            description='Use ros2_control if true'),

        DeclareLaunchArgument(
            'wheel_mu',
            default_value='1.0',
            description='Gazebo wheel friction (sim only, ignored on the real robot)'),

        DeclareLaunchArgument(
            'sim_camera',
            default_value='false',
            description='Simulate the camera in Gazebo (sim only)'),

        OpaqueFunction(function=_robot_state_publisher)
    ])

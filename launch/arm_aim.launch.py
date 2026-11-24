"""arm_aim.launch.py - MoveIt 2 for the spray arm (and, in later steps, the aim + spray nodes).

  ros2 launch my_bot arm_aim.launch.py                  # with the Gazebo sim (use_sim_time true)
  ros2 launch my_bot arm_aim.launch.py rviz:=true       # + RViz with MoveIt's MotionPlanning panel

What it starts, and why:
  - move_group: MoveIt's planner. It plans collision-free arm moves and sends them to
    arm_controller (/arm_controller/follow_joint_trajectory). Other nodes use it through
      /move_action            (moveit_msgs/action/MoveGroup: plan + move to a goal)
      /check_state_validity   (moveit_msgs/srv/GetStateValidity: "is this pose collision-free?")
      /apply_planning_scene   (add obstacles, e.g. the ground box)
    It needs the robot model: the URDF is built from the same xacro as robot_state_publisher
    (sim_mode follows use_sim_time, so in Gazebo it is the sim variant), plus MoveIt's own files
    in config/moveit/ (SRDF = arm group, park pose, collision pairs; IK; speed limits; controller).
    Planning uses OMPL with moveit_configs_utils' default settings (there is no
    config/ompl_planning.yaml on purpose).
  - RViz (only with rviz:=true): config/arm_aim.rviz, fixed frame base_link, robot model and the
    MotionPlanning panel for group "arm". It gets the same MoveIt parameters as move_group.
Needs the sim (launch_sim.launch.py) or, later, the robot with an arm driver already running:
move_group reads the arm angles from /joint_states and TF from robot_state_publisher.
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils import MoveItConfigsBuilder


def _is_true(text):
    return text.strip().lower() in ('true', '1', 'yes', 'on')


def _arm_aim(context):
    # The launch argument as a plain string ('true' / 'false'): xacro needs a string mapping.
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    # The builder prints "Cannot infer URDF/SRDF ... using config/weedbot.urdf" when it starts
    # (it looks for a MoveIt Setup Assistant file this package does not have). Harmless: both
    # files are given explicitly below.
    moveit_config = (
        MoveItConfigsBuilder('weedbot', package_name='my_bot')
        .robot_description(file_path='description/robot.urdf.xacro',
                           mappings={'use_ros2_control': 'true',
                                     'sim_mode': use_sim_time,
                                     'wheel_mu': '1.0',
                                     'sim_camera': 'false'})
        .robot_description_semantic(file_path='config/moveit/weedbot.srdf')
        .robot_description_kinematics(file_path='config/moveit/kinematics.yaml')
        .joint_limits(file_path='config/moveit/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    params = moveit_config.to_dict()
    # The URDF and SRDF are XML texts with comments such as "USE it: ...". launch_ros reads every
    # plain string parameter as YAML, and "word: " is YAML for a key, so the launch would stop
    # with "Unable to parse the value of parameter robot_description as yaml". Marking them as
    # strings passes them through unchanged.
    for key in ('robot_description', 'robot_description_semantic'):
        params[key] = ParameterValue(params[key], value_type=str)
    params['use_sim_time'] = _is_true(use_sim_time)

    nodes = [Node(package='moveit_ros_move_group', executable='move_group',
                  output='screen', parameters=[params])]
    if _is_true(LaunchConfiguration('rviz').perform(context)):
        rviz_config = os.path.join(get_package_share_directory('my_bot'), 'config', 'arm_aim.rviz')
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2_arm_aim',
                          output='log', arguments=['-d', rviz_config], parameters=[params]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true = Gazebo sim (sim clock, sim URDF variant)'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='true = also open RViz with the MotionPlanning panel'),
        OpaqueFunction(function=_arm_aim),
    ])

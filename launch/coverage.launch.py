# Coverage planner. Defaults come from config/coverage_params.yaml; these launch arguments
# override them for one run (leave empty to keep the YAML value):
#   ros2 launch my_bot coverage.launch.py use_sim_time:=true pattern:=dense lane_angle:=90 perimeter_laps:=2
#   pattern: dense (0.25 m) | medium (0.35 m) | wide (0.50 m) | custom (uses lane_spacing)
# At runtime: ros2 param set /coverage_planner pattern wide   (then /coverage/plan or /coverage/start)
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_node(context):
    pkg = get_package_share_directory('my_bot')
    params = [os.path.join(pkg, 'config', 'coverage_params.yaml'),
              {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context) == 'true'}]
    overrides = {}
    for name in ('pattern', 'lane_spacing', 'lane_angle', 'perimeter_laps'):
        val = LaunchConfiguration(name).perform(context).strip()
        if not val:
            continue
        if name == 'perimeter_laps':
            overrides[name] = int(val)
        elif name == 'lane_spacing':
            overrides[name] = float(val)
        elif name == 'lane_angle' and val.lower() != 'auto':
            overrides[name] = float(val)
        else:
            overrides[name] = val
    if overrides:
        params.append(overrides)
    return [Node(package='my_bot', executable='coverage_planner_node.py', name='coverage_planner',
                 output='screen', parameters=params)]


def generate_launch_description():
    args = [DeclareLaunchArgument('use_sim_time', default_value='false',
                                  description='Use simulation (Gazebo) clock if true')]
    for name, desc in (('pattern', 'dense | medium | wide | custom (empty = YAML)'),
                       ('lane_spacing', 'm, used with pattern:=custom (empty = YAML)'),
                       ('lane_angle', "degrees or 'auto' (empty = YAML)"),
                       ('perimeter_laps', '0, 1 or 2 (empty = YAML)')):
        args.append(DeclareLaunchArgument(name, default_value='', description=desc))
    return LaunchDescription(args + [OpaqueFunction(function=make_node)])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('my_bot')
    return LaunchDescription([
        DeclareLaunchArgument('camera_info_url', default_value='',
                              description='camera_calibration YAML for the phone; empty = use ARKit intrinsics'),
        DeclareLaunchArgument('cert_dir', default_value=os.path.expanduser('~/phone_bridge_certs')),
        DeclareLaunchArgument('fallback_identity_anchor', default_value='false',
                              description='sim/testing only: allow /phone/anchor_here without a robot pose'),
        Node(
            package='my_bot',
            executable='phone_bridge_node.py',
            name='phone_bridge',
            output='screen',
            parameters=[{
                'ws_port': 8444,                 # native app
                'https_port': 8443,              # Safari page (needs make_cert.sh, otherwise skipped)
                'http_port': 8080,
                'cert_dir': LaunchConfiguration('cert_dir'),
                'www_dir': os.path.join(pkg, 'www'),
                'frame_id': 'phone_link',
                'optical_frame': 'phone_optical',
                'gps_frame_id': 'phone_gps',
                'ar_frame': 'ar_origin',
                'base_frame': 'base_link',
                'map_frame': 'map',
                'publish_optical_tf': False,     # True if phone_optical is NOT in the URDF (phone_mount.xacro adds it)
                'gyro_sign': 1.0,
                'accel_sign': 1.0,
                'camera_info_url': LaunchConfiguration('camera_info_url'),
                'twist_window_s': 0.15,
                'twist_sigma_v': 0.03,
                'twist_sigma_w': 0.02,
                'auto_anchor_on_handheld': True,
                'fallback_identity_anchor': ParameterValue(LaunchConfiguration('fallback_identity_anchor'), value_type=bool),
                'estop_on_thermal_critical': True,
                'estop_on_disconnect': False,
                'image_hz': 10.0,
                'depth_hz': 5.0,
                'pose_hz': 30.0,
                'jpeg_quality': 0.6,
                'image_width': 640,
            }],
        ),
    ])

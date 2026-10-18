#!/usr/bin/env python3
"""
arm_joint_state_stub.py  -  placeholder /joint_states for joints that have no driver yet.

The arm (arm_base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll) has no servo driver yet,
and nothing measures the real caster's swivel/wheel angles. robot_state_publisher only publishes
TF for a moving joint once it gets a value for it, so without this node there is no TF for
spray_tip (and, on the real robot, for the caster links). This node publishes those joints at
fixed positions (default 0 = arm straight up) on /joint_states. The wheel joints come from
ros2_control's joint_broad as before; robot_state_publisher merges both. Names that are not in the
URDF (the caster joints in Gazebo, where the caster is a fixed ball) are ignored.

The real servo driver will replace this node: publish the same arm joint names, remove them here.
Parameters (config/arm_joint_state_stub.yaml): joints, positions (same length, rad), rate_hz.
Started by launch_robot.launch.py and launch_sim.launch.py.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateStub(Node):
    def __init__(self):
        super().__init__('arm_joint_state_stub')
        self.declare_parameter('joints', ['arm_base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll'])
        self.declare_parameter('positions', [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('rate_hz', 5.0)
        self.joints = list(self.get_parameter('joints').value)
        self.positions = [float(v) for v in self.get_parameter('positions').value]
        if len(self.positions) != len(self.joints):
            self.get_logger().warn(f'{len(self.positions)} positions for {len(self.joints)} joints: using 0 for all')
            self.positions = [0.0] * len(self.joints)
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.get_logger().info(f'Publishing fixed joint states for {self.joints} (no driver yet)')

    def _tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()   # sim time in Gazebo (use_sim_time)
        msg.name = self.joints
        msg.position = self.positions
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():          # Ctrl-C already shut the context down
            rclpy.shutdown()


if __name__ == '__main__':
    main()

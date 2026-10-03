#!/usr/bin/env python3
"""
odom_relay.py  -  republishes nav_msgs/Odometry from one topic to another, unchanged.

Used in simulation only (launch_sim.launch.py): Gazebo has no IMU/EKF, so nothing publishes
/odom, which Nav2 reads (odom_topic in nav2_params.yaml). This copies /diff_cont/odom to /odom.
On the real robot the robot_localization EKF publishes /odom instead; do not run this there.

Topics (set with launch remappings):  odom_in -> odom_out
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomRelay(Node):
    def __init__(self):
        super().__init__('odom_relay')
        self.pub = self.create_publisher(Odometry, 'odom_out', 10)
        self.create_subscription(Odometry, 'odom_in', self.pub.publish, 10)
        self.get_logger().info('Relaying %s -> %s' % (
            self.resolve_topic_name('odom_in'), self.resolve_topic_name('odom_out')))


def main():
    rclpy.init()
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

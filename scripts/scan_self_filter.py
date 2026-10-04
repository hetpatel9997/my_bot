#!/usr/bin/env python3
"""
scan_self_filter.py  -  removes LiDAR returns that hit the robot itself.

Subscribes /scan, sets every reading whose point lies inside the robot outline plus a margin
(in base_link) to +inf ("no return"), and publishes /scan_filtered. Everything else (costmaps,
AMCL, SLAM, collision monitor, /safety_state) uses /scan_filtered, so the robot's own chassis,
post or wheels never look like an obstacle (in the sim they made the collision monitor stop
the robot permanently). If this node dies, /scan_filtered stops and the collision monitor
stops the robot (source_timeout): fail safe.

Parameters: box [x_min, x_max, y_min, y_max] in base_link (config/safety.yaml).
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


class SelfFilter(Node):
    def __init__(self):
        super().__init__('scan_self_filter')
        self.declare_parameter('box', [-0.13, 0.23, -0.255, 0.255])
        self.declare_parameter('base_frame', 'base_link')
        self.box = [float(v) for v in self.get_parameter('box').value]
        self.base = self.get_parameter('base_frame').value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose = None                      # laser pose in base_link (static), looked up once
        self.pub = self.create_publisher(LaserScan, 'scan_filtered', qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self._cb, qos_profile_sensor_data)
        self.removed = 0
        self.get_logger().info(f'Removing returns inside base_link box {self.box} -> /scan_filtered')

    def _cb(self, msg):
        if self.pose is None:
            try:
                t = self.tf_buffer.lookup_transform(self.base, msg.header.frame_id, Time())
            except Exception:
                return                         # no output yet -> collision monitor keeps the robot stopped
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose = (t.transform.translation.x, t.transform.translation.y, yaw)
            a = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            self.cos, self.sin = np.cos(a + yaw), np.sin(a + yaw)
        lx, ly, _ = self.pose
        r = np.array(msg.ranges, dtype=np.float32)
        if len(r) != len(self.cos):            # scan layout changed: recompute next time
            self.pose = None
            return
        x, y = lx + r * self.cos, ly + r * self.sin
        x0, x1, y0, y1 = self.box
        own = np.isfinite(r) & (x > x0) & (x < x1) & (y > y0) & (y < y1)
        r[own] = np.inf
        if own.any() and self.removed == 0:
            self.get_logger().info(f'Filtering {int(own.sum())} self-hit returns per scan')
        self.removed += int(own.sum())
        msg.ranges = r.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

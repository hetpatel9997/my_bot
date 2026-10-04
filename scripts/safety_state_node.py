#!/usr/bin/env python3
"""
safety_state_node.py  -  publishes /safety_state (std_msgs/String): "clear", "slow" or "stop".

Zones from config/safety.yaml, evaluated on /scan_filtered in base_link:
  stop   3+ laser points in stop_zone (outline + 0.04 m, safety_state section), OR /e_stop is
         true, OR no fresh scan / no TF (fail safe)
  slow   3+ laser points in the collision monitor's slowdown polygon (corridor ahead), OR an
         UNKNOWN obstacle within unknown_radius (1.5 m): 3+ scan points farther than
         unknown_tolerance from anything on /map (a person, a moved object), or no map/TF yet
  clear  otherwise

Speed governor (in the command chain, so it limits EVERY source):
  /cmd_vel_mux (twist_mux) -> this node -> /cmd_vel_capped -> collision_monitor -> diff_cont
  While an unknown obstacle is within unknown_radius, linear speed is capped at unknown_speed
  (0.08 m/s; angular scaled by the same factor so the turn radius is kept), and Nav2 gets the
  same limit on /speed_limit. While /e_stop is true it sends zero commands (immediate braking).
  If this node dies, nothing reaches the wheels (fail safe).
A state is held for hold_time after its zone clears so it does not flicker.

RULE (CLAUDE.md): the spray controller may only open the spray valve while this topic is "clear".
The topic is latched (transient local), published at rate_hz and on every change.
"""
import math
import os
import time

import numpy as np
import yaml
from scipy import ndimage

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener


def inside(poly, x, y):
    """Vectorised point-in-polygon (even-odd rule). poly: (N, 2); x, y: arrays."""
    res = np.zeros(len(x), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        res ^= cross
        j = i
    return res


class SafetyState(Node):
    def __init__(self):
        super().__init__('safety_state')
        self.declare_parameter('safety_config_file', os.path.join(
            get_package_share_directory('my_bot'), 'config', 'safety.yaml'))
        self.declare_parameter('stop_zone', [0.24, 0.265, 0.24, -0.265, -0.14, -0.265, -0.14, 0.265])
        for name, default in (('scan_topic', '/scan'), ('e_stop_topic', '/e_stop'),
                              ('state_topic', '/safety_state'), ('rate_hz', 10.0),
                              ('scan_timeout', 0.5), ('hold_time', 0.5),
                              ('map_topic', '/map'), ('map_frame', 'map'),
                              ('unknown_radius', 1.5), ('unknown_tolerance', 0.25),
                              ('unknown_min_points', 3), ('unknown_speed', 0.08),
                              ('unknown_hold', 1.0), ('cmd_in', '/cmd_vel_mux'),
                              ('cmd_out', '/cmd_vel_capped'), ('speed_limit_topic', '/speed_limit')):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        cm = yaml.safe_load(open(g('safety_config_file')))['collision_monitor']['ros__parameters']
        self.zones = {'stop': (np.array(g('stop_zone'), dtype=float).reshape(-1, 2), 2)}
        for name in cm['polygons']:
            z = cm[name]
            if z['action_type'] == 'slowdown':
                self.zones['slowdown'] = (np.array(z['points'], dtype=float).reshape(-1, 2),
                                          int(z.get('max_points', 3)))
        self.base = cm.get('base_frame_id', 'base_link')
        self.scan_timeout, self.hold = g('scan_timeout'), g('hold_time')
        self.scan = None
        self.scan_wall = 0.0
        self.e_stop = False
        self.state, self.last_trigger = 'stop', {'stop': 0.0, 'slow': 0.0}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, g('scan_topic'), self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Bool, g('e_stop_topic'), self._e_stop_cb, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, g('state_topic'), latched)
        # unknown obstacles + speed governor
        self.map_dist = None
        self.unknown_near, self.last_unknown, self.cap_active = True, 0.0, None
        self.create_subscription(OccupancyGrid, g('map_topic'), self._map_cb, latched)
        self.cmd_pub = self.create_publisher(Twist, g('cmd_out'), 10)
        self.limit_pub = self.create_publisher(SpeedLimit, g('speed_limit_topic'), 10)
        self.create_subscription(Twist, g('cmd_in'), self._cmd_cb, 10)
        self.create_timer(1.0 / g('rate_hz'), self._tick)
        self.get_logger().info(f'safety_state: zones {list(self.zones)} from '
                               f'{os.path.basename(g("safety_config_file"))}; publishing {g("state_topic")}')

    def _e_stop_cb(self, msg):
        self.e_stop = msg.data
        if msg.data:                       # brake at once, not at the next tick
            self.cmd_pub.publish(Twist())

    def _scan_cb(self, msg):
        self.scan, self.scan_wall = msg, time.time()

    def _map_cb(self, m):
        occ = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width) >= 65
        self.map_info = m.info
        # metres from each cell to the nearest occupied map cell
        self.map_dist = ndimage.distance_transform_edt(~occ) * m.info.resolution

    def _unknown_near(self):
        """True if 3+ scan points within unknown_radius are not explained by the map."""
        g = lambda n: self.get_parameter(n).value
        if self.map_dist is None:
            return True                                   # no map yet: be careful
        s = self.scan
        try:
            t = self.tf_buffer.lookup_transform(g('map_frame'), s.header.frame_id, Time())
        except Exception:
            return True                                   # not localised: be careful
        r = np.array(s.ranges, dtype=float)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r >= s.range_min) & (r <= g('unknown_radius'))
        if not ok.any():
            return False
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x = t.transform.translation.x + r[ok] * np.cos(a[ok] + yaw)
        y = t.transform.translation.y + r[ok] * np.sin(a[ok] + yaw)
        i = self.map_info
        c = np.floor((x - i.origin.position.x) / i.resolution).astype(int)
        rr = np.floor((y - i.origin.position.y) / i.resolution).astype(int)
        inside = (c >= 0) & (c < i.width) & (rr >= 0) & (rr < i.height)
        far = np.ones(len(x), dtype=bool)                 # outside the map = unknown
        far[inside] = self.map_dist[rr[inside], c[inside]] > g('unknown_tolerance')
        return int(far.sum()) >= g('unknown_min_points')

    def _cmd_cb(self, msg):
        """Speed governor: cap every command while an unknown obstacle is near."""
        if self.e_stop:
            return
        cap = self.get_parameter('unknown_speed').value
        if self.cap_active is not False and abs(msg.linear.x) > cap:   # None (starting) counts as capped
            f = cap / abs(msg.linear.x)
            msg.linear.x *= f
            msg.angular.z *= f
        self.cmd_pub.publish(msg)

    def _points_in_base(self):
        s = self.scan
        try:
            t = self.tf_buffer.lookup_transform(self.base, s.header.frame_id, Time())
        except Exception:
            return None
        r = np.array(s.ranges, dtype=float)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
        r, a = r[ok], a[ok]
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x = t.transform.translation.x + r * np.cos(a + yaw)
        y = t.transform.translation.y + r * np.sin(a + yaw)
        return x, y

    def _tick(self):
        now = time.time()
        reason = None
        if self.e_stop:
            raw, reason = 'stop', 'e_stop'
        elif self.scan is None or now - self.scan_wall > self.scan_timeout:
            raw, reason = 'stop', 'no fresh /scan'
        else:
            pts = self._points_in_base()
            if pts is None:
                raw, reason = 'stop', f'no TF {self.base} <- {self.scan.header.frame_id}'
            else:
                raw = 'clear'
                for action, state in (('stop', 'stop'), ('slowdown', 'slow')):
                    if action in self.zones:
                        poly, max_pts = self.zones[action]
                        if inside(poly, *pts).sum() > max_pts:
                            raw = state
                            break
                if self._unknown_near():
                    self.last_unknown = now
                    if raw == 'clear':
                        raw, reason = 'slow', 'unknown obstacle within range'
        # speed cap while an unknown obstacle is near (held unknown_hold s), or without fresh data
        stale = self.scan is None or now - self.scan_wall > self.scan_timeout
        capped = stale or now - self.last_unknown < self.get_parameter('unknown_hold').value
        if capped != self.cap_active:
            lim = SpeedLimit()
            lim.header.stamp = self.get_clock().now().to_msg()
            lim.percentage = False
            lim.speed_limit = float(self.get_parameter('unknown_speed').value) if capped else 0.0
            self.limit_pub.publish(lim)
            self.get_logger().info(f'speed cap {"ON (unknown obstacle near)" if capped else "off"}')
            self.cap_active = capped
        if raw in self.last_trigger:
            self.last_trigger[raw] = now
        # hold: stay stop/slow for hold_time after the zone cleared
        if now - self.last_trigger['stop'] < self.hold:
            state = 'stop'
        elif now - self.last_trigger['slow'] < self.hold:
            state = 'slow'
        else:
            state = 'clear'
        if state != self.state:
            self.get_logger().info(f'safety state: {self.state} -> {state}'
                                   + (f' ({reason})' if reason else ''))
        self.state = state
        self.pub.publish(String(data=state))
        if self.e_stop:
            # twist_mux publishes nothing while locked; send explicit zeros so the wheels brake
            # now instead of after diff_cont's command timeout
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = SafetyState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

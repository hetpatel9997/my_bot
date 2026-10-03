#!/usr/bin/env python3
"""
sim_odom_check.py  -  SIMULATION ONLY drive test: compares odometry with Gazebo's true pose.

Drives the simulated robot 1 m forward (0.2 m/s), stops, then turns 360 deg in place (0.5 rad/s),
(both measured on Gazebo's true pose, so the robot really travels 1 m and really turns 360 deg)
and reports, for each phase, the true motion (/ground_truth/odom, Gazebo p3d plugin at 50 Hz,
description/sim_ground_truth.xacro) against
  /odom            (robot_localization EKF: wheels + IMU)    and
  /diff_cont/odom  (wheels only).
Heading is accumulated (unwrapped), so a full 360 deg turn is measured as such.

Run (Gazebo running via launch_sim.launch.py, after `ros_local`):
  ros2 run my_bot sim_odom_check.py
With slippery wheels (start the sim with wheel_mu:=0.1) the wheels-only heading should drift
while the EKF heading stays close to Gazebo, because the EKF trusts the gyro yaw rate.

REFUSES to run unless /clock exists (i.e. simulation). Never use on the real robot.
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Track:
    """Keeps position and an unwrapped (accumulated) heading for one pose source."""
    def __init__(self):
        self.x = self.y = self.yaw = None
        self.acc = 0.0

    def update(self, x, y, yaw):
        if self.yaw is not None:
            self.acc += wrap(yaw - self.yaw)
        self.x, self.y, self.yaw = x, y, yaw

    def snap(self):
        return (self.x, self.y, self.acc)


class Check(Node):
    def __init__(self):
        super().__init__('sim_odom_check', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.tracks = {'gazebo': Track(), 'ekf /odom': Track(), 'wheels /diff_cont/odom': Track()}
        for topic, name in (('/ground_truth/odom', 'gazebo'), ('/odom', 'ekf /odom'),
                            ('/diff_cont/odom', 'wheels /diff_cont/odom')):
            self.create_subscription(Odometry, topic, lambda m, n=name: self._odom(n, m), 50)
        # All three sources arrive as ROS topics at 10-50 Hz, so headings unwrap reliably (an
        # earlier version polled `gz model -p`, whose multi-second stalls lost whole turns).

    def _odom(self, name, m):
        p = m.pose.pose
        self.tracks[name].update(p.position.x, p.position.y, yaw_of(p.orientation))

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def run_for(self, sec, twist):
        # Timed in SIMULATION time (/clock), so the commanded distance/angle is right even when
        # Gazebo runs slower than real time.
        while self.sim_now() == 0.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        end = self.sim_now() + sec
        next_pub = 0.0
        while self.sim_now() < end:
            if time.time() >= next_pub:                    # steady 20 Hz command stream
                self.pub.publish(twist)
                next_pub = time.time() + 0.05
            rclpy.spin_once(self, timeout_sec=0.01)

    def run_until(self, done, twist, timeout):
        """Publish twist until done(gazebo_track) is true (TRUE pose), or timeout s of sim time."""
        end = self.sim_now() + timeout
        next_pub = 0.0
        while self.sim_now() < end:
            if done(self.tracks['gazebo']):
                return True
            if time.time() >= next_pub:
                self.pub.publish(twist)
                next_pub = time.time() + 0.05
            rclpy.spin_once(self, timeout_sec=0.01)
        print('  (timeout before reaching the target)')
        return False

    def snaps(self):
        return {k: t.snap() for k, t in self.tracks.items()}


def report(title, a, b):
    print(f'\n{title}')
    g = (b['gazebo'][0] - a['gazebo'][0], b['gazebo'][1] - a['gazebo'][1], b['gazebo'][2] - a['gazebo'][2])
    gd = math.hypot(g[0], g[1])
    print(f'  {"source":24s} {"distance m":>10s} {"heading deg":>12s} {"dist err":>9s} {"head err deg":>13s}')
    for k in b:
        d = (b[k][0] - a[k][0], b[k][1] - a[k][1], b[k][2] - a[k][2])
        dd = math.hypot(d[0], d[1])
        print(f'  {k:24s} {dd:10.3f} {math.degrees(d[2]):12.1f} {dd - gd:+9.3f} {math.degrees(d[2] - g[2]):+13.1f}')


def main():
    rclpy.init()
    n = Check()
    names = [t for t, _ in n.get_topic_names_and_types()]
    if '/clock' not in names:
        print('No /clock topic: this is not a simulation. Refusing to drive.'); sys.exit(1)
    still = Twist()
    n.run_for(3.0, still)                                  # collect initial poses
    for k in [k for k, t in n.tracks.items() if t.x is None]:
        if k == 'gazebo' or k.startswith('wheels'):
            print(f'No data from {k}; is the sim running?'); sys.exit(1)
        print(f'WARNING: no data from {k} (EKF not running?); it is left out of the report.')
        del n.tracks[k]
    fwd = Twist(); fwd.linear.x = 0.2
    turn = Twist(); turn.angular.z = 0.5
    # Targets are on Gazebo's TRUE pose, so the robot really moves 1 m and really turns 360 deg.
    s0 = n.snaps()
    x0, y0 = s0['gazebo'][0], s0['gazebo'][1]
    n.run_until(lambda g: math.hypot(g.x - x0, g.y - y0) >= 1.0, fwd, 30.0)
    n.run_for(2.0, still); s1 = n.snaps()
    report('FORWARD 1 m at 0.2 m/s (stopped on true distance)', s0, s1)
    a0 = s1['gazebo'][2]
    n.run_until(lambda g: g.acc - a0 >= 2 * math.pi, turn, 40.0)
    n.run_for(2.0, still); s2 = n.snaps()
    report('TURN 360 deg at 0.5 rad/s (stopped on true heading)', s1, s2)
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()

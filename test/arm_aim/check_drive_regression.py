#!/usr/bin/env python3
"""
check_drive_regression.py  -  SIMULATION ONLY: driving still works after the arm changes.

Why this test exists: making the arm joints movable in Gazebo changes the robot model (revolute
joints, arm links without gravity, a third controller). This checks that the robot still drives
exactly as it did in the 2026-10-19 sim regression (CLAUDE.md "Sim regression v21", test 1).

What it does:
  - publishes /cmd_vel_keyboard linear x 0.10 m/s at 10 Hz for 3 s (sim time), i.e. the same way
    the keyboard teleop drives: through the whole safety chain
    (twist_mux -> safety_state -> collision_monitor -> diff_cont), never straight to diff_cont;
  - there is no /map, so safety_state treats everything as "unknown" and caps every command at
    its `unknown_speed`. The baseline was measured with that cap at 0.08 m/s. On 2026-10-31 the
    owner raised `unknown_speed` to 0.50 in config/safety.yaml (no cap any more), and then the same
    command drives ~0.31 m. So, to compare with the baseline, this test sets
    /safety_state unknown_speed to 0.08 while it drives and puts the old value back afterwards
    (a lower cap only makes the test robot slower; nothing is changed on disk);
  - waits 2 s for the robot to stop, then checks
      * Gazebo's true motion (/ground_truth/odom) along the robot's starting heading is
        0.243 m +- 0.02 (2026-10-19 baseline), and
      * the robot's own estimate (/odom, EKF of wheels + IMU) agrees with the truth within 0.02 m.

Run (sim started by test/arm_aim/start_sim.sh, same terminal settings):
  source test/arm_aim/sim_env.sh && python3 test/arm_aim/check_drive_regression.py
Last line: "DRIVE REGRESSION PASS" (exit 0) or "DRIVE REGRESSION FAIL: <why>" (exit 1).
It drives the robot 0.24 m forward. It refuses to run without a simulation clock.
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

CMD_SPEED = 0.10        # m/s commanded (safety_state caps it to 0.08 without a map)
BASELINE_CAP = 0.08     # m/s, safety_state unknown_speed when the baseline was measured
CMD_HZ = 10.0
DRIVE_S = 3.0           # s sim time
SETTLE_S = 2.0          # s sim time after the last command
BASELINE = 0.243        # m, 2026-10-19 (truth, /odom and /diff_cont/odom all 0.243)
TOL = 0.02              # m


class Fail(Exception):
    pass


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DriveCheck(Node):
    def __init__(self):
        super().__init__('check_drive_regression',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(Twist, '/cmd_vel_keyboard', 10)
        self.latest = {'/ground_truth/odom': None, '/odom': None}
        for topic in self.latest:
            self.create_subscription(Odometry, topic,
                                     lambda m, t=topic: self.latest.__setitem__(t, m), 20)

    def _call(self, srv_type, name, request):
        client = self.create_client(srv_type, name)
        try:
            if not client.wait_for_service(timeout_sec=10.0):
                raise Fail(f'no service {name} (is safety_state running?)')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is None:
                raise Fail(f'no answer from {name}')
            return future.result()
        finally:
            self.destroy_client(client)

    def get_cap(self):
        """safety_state's speed cap for unknown obstacles (m/s)."""
        req = GetParameters.Request(names=['unknown_speed'])
        return self._call(GetParameters, '/safety_state/get_parameters', req).values[0].double_value

    def set_cap(self, value):
        req = SetParameters.Request(parameters=[ParameterMsg(
            name='unknown_speed',
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))])
        res = self._call(SetParameters, '/safety_state/set_parameters', req).results[0]
        if not res.successful:
            raise Fail(f'safety_state refused unknown_speed {value}: {res.reason}')

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, sim_s, every=None, wall_s=60.0):
        """Spin for sim_s of sim time, calling every() at CMD_HZ (sim time) if given."""
        t0, t_wall, next_tick = self.sim_now(), time.monotonic(), self.sim_now()
        while self.sim_now() - t0 < sim_s:
            if time.monotonic() - t_wall > wall_s:
                raise Fail(f'sim time too slow ({self.sim_now() - t0:.1f} s sim in {wall_s:.0f} s wall)')
            if every is not None and self.sim_now() >= next_tick:
                every()
                next_tick += 1.0 / CMD_HZ
            rclpy.spin_once(self, timeout_sec=0.01)

    def pose(self, topic):
        m = self.latest[topic]
        p = m.pose.pose
        return p.position.x, p.position.y, yaw_of(p.orientation)

    def run(self):
        t_wall = time.monotonic()
        # With use_sim_time the node clock stays at 0 until the first /clock message arrives.
        while not (self.sim_now() > 0.0 and all(self.latest.values())):
            if time.monotonic() - t_wall > 30.0:
                missing = [t for t, m in self.latest.items() if m is None]
                raise Fail('no /clock: is the simulation running?' if self.sim_now() <= 0.0
                           else f'no messages on {missing} after 30 s')
            rclpy.spin_once(self, timeout_sec=0.05)
        self.spin_for(0.5)                      # fresh poses before the start
        start = {t: self.pose(t) for t in self.latest}

        cmd = Twist()
        cmd.linear.x = CMD_SPEED
        self.spin_for(DRIVE_S, every=lambda: self.pub.publish(cmd))
        # No zero command on purpose (same as letting go of the key): twist_mux drops the input
        # after 0.5 s and diff_cont stops the wheels 0.25 s after its last command.
        self.spin_for(SETTLE_S)
        end = {t: self.pose(t) for t in self.latest}

        moved = {}
        for t in self.latest:
            x0, y0, yaw0 = start[t]
            x1, y1, _ = end[t]
            # distance along the starting heading: + = forward (toward the arm)
            moved[t] = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
        truth, odom = moved['/ground_truth/odom'], moved['/odom']
        print(f'  truth (/ground_truth/odom) {truth:+.4f} m, /odom {odom:+.4f} m, '
              f'baseline {BASELINE} +- {TOL}, /odom - truth {odom - truth:+.4f} m')
        if abs(truth - BASELINE) > TOL:
            raise Fail(f'robot moved {truth:+.4f} m, expected {BASELINE} +- {TOL}')
        if abs(odom - truth) > TOL:
            raise Fail(f'/odom says {odom:+.4f} m but Gazebo says {truth:+.4f} m '
                       f'(more than {TOL} m apart)')


def main():
    rclpy.init()
    node = DriveCheck()
    ok = False
    old_cap = None
    try:
        old_cap = node.get_cap()
        if old_cap != BASELINE_CAP:
            node.set_cap(BASELINE_CAP)
            print(f'  /safety_state unknown_speed {old_cap} -> {BASELINE_CAP} for this test '
                  f'(the baseline value), restored afterwards')
        node.run()
        ok = True
    except Fail as e:
        print(f'DRIVE REGRESSION FAIL: {e}')
    except (KeyboardInterrupt, ExternalShutdownException):
        print('DRIVE REGRESSION FAIL: interrupted')
    finally:
        if old_cap is not None and old_cap != BASELINE_CAP and rclpy.ok():
            try:
                node.set_cap(old_cap)
            except Fail as e:
                ok = False
                print(f'DRIVE REGRESSION FAIL: could not restore unknown_speed {old_cap}: {e}')
        if ok:
            print('DRIVE REGRESSION PASS')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

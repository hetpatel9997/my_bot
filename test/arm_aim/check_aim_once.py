#!/usr/bin/env python3
"""
check_aim_once.py  -  SIMULATION ONLY: one full aim -> spray -> park cycle of weed_aim in Gazebo.

What it does, in this order:
  1. Refuses to run without a simulation clock (/clock): it moves the arm and sprays.
  2. start_map_fixture (test/arm_aim/fixtures.py): empty /map + map -> odom, so that
     /safety_state can turn `clear` in the empty test world; waits up to 30 s for `clear`.
  3. Waits until MoveIt (/check_state_validity, /move_action) and weed_aim (subscribed to
     /aim_target) are up.
  4. Publishes ONE target on /aim_target: (0.34, 0, 0) in base_link = a weed on the ground 34 cm
     in front of the drive axle.
  5. Within 40 s exactly this must come back on /aim/result: reachable true, aim_error_m <= 0.01
     (the nozzle's centre line passes within 1 cm of the weed), sprayed true, spray_reason
     'sprayed 1.00 s', state 'ok'.
  6. Afterwards /joint_states shows every arm joint within 1 deg of the park pose (SRDF).
It also prints what it saw: the aim joints, tilt, standoff, the durations weed_aim reports, and
the relay on-time measured here from /spray/relay (wall clock, as the spray controller times it).

Run (sim from test/arm_aim/start_sim.sh and arm_aim.launch.py running, same terminal settings):
  source test/arm_aim/sim_env.sh && python3 test/arm_aim/check_aim_once.py
Last line: "AIM ONCE PASS" (exit 0) or "AIM ONCE FAIL: <why>" (exit 1).

Cleanup afterwards (pass or fail): stop arm_aim.launch.py, then
  ~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42     (must print "Clean")
  for p in $(pgrep -f move_group); do grep -qzx ROS_DOMAIN_ID=42 /proc/$p/environ && kill $p; done
  ros2 daemon stop                                             (with sim_env.sh sourced)
"""
import json
import math
import os
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import PointStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetStateValidity
from rclpy.action import ActionClient
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
sys.path.insert(0, HERE)
from aim_solver import JOINT_NAMES  # noqa: E402
from arm_config import load_park  # noqa: E402
from fixtures import start_map_fixture  # noqa: E402

SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
TARGET = (0.34, 0.0, 0.0)           # base_link, on the ground
CLEAR_WAIT_S = 30.0                 # s for /safety_state to turn clear
READY_WAIT_S = 120.0                # s for move_group + weed_aim after the launch
RESULT_WAIT_S = 40.0                # s from the target to its /aim/result
MAX_AIM_ERROR = 0.01                # m
PARK_TOL = math.radians(1.0)        # rad
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


class Fail(Exception):
    pass


class AimOnce(Node):
    def __init__(self):
        super().__init__('check_aim_once')
        self.clock_seen = False
        self.safety = None
        self.results = []           # (wall time, dict)
        self.relay_edges = []       # (wall time, on)
        self.joints, self.joints_time = {}, None
        # Gazebo publishes /clock best effort: a reliable reader would never hear it
        self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(String, '/safety_state', self._on_safety, RELIABLE)
        self.create_subscription(String, '/aim/result', self._on_result, RELIABLE)
        self.create_subscription(Bool, '/spray/relay', self._on_relay, LATCHED)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.target_pub = self.create_publisher(PointStamped, '/aim_target', RELIABLE)
        self.validity = self.create_client(GetStateValidity, '/check_state_validity')
        self.move_action = ActionClient(self, MoveGroup, '/move_action')

    def _on_clock(self, msg):
        self.clock_seen = True

    def _on_safety(self, msg):
        self.safety = msg.data

    def _on_result(self, msg):
        self.results.append((time.monotonic(), json.loads(msg.data)))

    def _on_relay(self, msg):
        on = bool(msg.data)
        if not self.relay_edges or self.relay_edges[-1][1] != on:
            self.relay_edges.append((time.monotonic(), on))

    def _on_joints(self, msg):
        seen = False
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self.joints[name] = pos
                seen = True
        if seen:
            self.joints_time = time.monotonic()

    def spin_until(self, done, wall_s):
        end = time.monotonic() + wall_s
        while rclpy.ok() and time.monotonic() < end:
            if done():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(done())

    def spin_for(self, wall_s):
        self.spin_until(lambda: False, wall_s)


def run(node):
    park = load_park(SRDF)
    if not node.spin_until(lambda: node.clock_seen, 30.0):
        raise Fail('no /clock: is the simulation running? (refusing to run without it)')
    start_map_fixture(node)
    t0 = time.monotonic()
    if not node.spin_until(lambda: node.safety == 'clear', CLEAR_WAIT_S):
        raise Fail(f'/safety_state not clear after {CLEAR_WAIT_S:.0f} s (last: {node.safety})')
    print(f'  /safety_state clear after {time.monotonic() - t0:.1f} s')

    t0 = time.monotonic()
    if not node.spin_until(lambda: (node.validity.service_is_ready()
                                    and node.move_action.server_is_ready()
                                    and node.target_pub.get_subscription_count() > 0),
                           READY_WAIT_S):
        raise Fail(f'MoveIt / weed_aim not up after {READY_WAIT_S:.0f} s (is arm_aim.launch.py '
                   'running? see $ARM_TEST_LOG/arm_aim.log)')
    print(f'  MoveIt and weed_aim up after {time.monotonic() - t0:.1f} s')

    msg = PointStamped()
    msg.header.frame_id = 'base_link'
    msg.point.x, msg.point.y, msg.point.z = TARGET
    node.results.clear()
    node.relay_edges = [e for e in node.relay_edges[-1:] if not e[1]]   # keep a latched "off"
    t_send = time.monotonic()
    node.target_pub.publish(msg)
    print(f'  target {TARGET} in base_link sent')
    if not node.spin_until(lambda: node.results, RESULT_WAIT_S):
        raise Fail(f'no /aim/result within {RESULT_WAIT_S:.0f} s (see $ARM_TEST_LOG/arm_aim.log)')
    t_result, res = node.results[0]
    node.spin_for(1.0)          # a second result for the same target would be a bug
    print(f'  /aim/result after {t_result - t_send:.1f} s: {json.dumps(res)}')
    if len(node.results) != 1:
        raise Fail(f'{len(node.results)} results for one target')

    on_times = [(t, t2 - t) for (t, a), (t2, b) in zip(node.relay_edges, node.relay_edges[1:])
                if a and not b]
    if on_times:
        print(f'  relay on {on_times[0][0] - t_send:.2f} s after the target, for '
              f'{on_times[0][1]:.3f} s (measured here from /spray/relay)')
    print(f'  aim joints {res.get("joints_deg")} deg, tilt {res.get("tilt_deg")} deg, '
          f'standoff {res.get("standoff_m")} m, durations {res.get("durations")}')

    checks = [
        (res.get('reachable') is True, f'reachable is {res.get("reachable")!r}'),
        (isinstance(res.get('aim_error_m'), (int, float))
         and res['aim_error_m'] <= MAX_AIM_ERROR,
         f'aim_error_m {res.get("aim_error_m")!r} (must be <= {MAX_AIM_ERROR})'),
        (res.get('sprayed') is True, f'sprayed is {res.get("sprayed")!r}'),
        (res.get('spray_reason') == 'sprayed 1.00 s',
         f'spray_reason {res.get("spray_reason")!r} (must be \'sprayed 1.00 s\')'),
        (res.get('state') == 'ok', f'state {res.get("state")!r}, reason {res.get("reason")!r}'),
        (len(on_times) == 1, f'{len(on_times)} relay on-periods seen (expected 1)'),
    ]
    bad = [why for ok, why in checks if not ok]
    if bad:
        raise Fail('; '.join(bad))
    print(f'  aim error {res["aim_error_m"] * 1000:.1f} mm, spray reason '
          f'{res["spray_reason"]!r}, state ok')

    # The arm must be back in park: a FRESH joint state after the result.
    t_after = time.monotonic()
    if not node.spin_until(lambda: node.joints_time is not None and node.joints_time > t_after
                           and all(j in node.joints for j in JOINT_NAMES), 5.0):
        raise Fail('no /joint_states with all arm joints after the result')
    off = {j: math.degrees(node.joints[j] - p) for j, p in zip(JOINT_NAMES, park)}
    if not all(abs(d) <= math.degrees(PARK_TOL) for d in off.values()):
        raise Fail(f'arm not back in park (deg off): {({j: round(d, 2) for j, d in off.items()})}')
    print(f'  back in park: max {max(abs(d) for d in off.values()):.3f} deg off')


def main():
    rclpy.init()
    node = AimOnce()
    ok = False
    try:
        run(node)
        ok = True
        print('AIM ONCE PASS')
    except Fail as e:
        print(f'AIM ONCE FAIL: {e}')
    except (KeyboardInterrupt, ExternalShutdownException):
        print('AIM ONCE FAIL: interrupted')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

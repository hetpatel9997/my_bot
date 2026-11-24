#!/usr/bin/env python3
"""
sim_aim_check.py  -  SIMULATION ONLY: the whole arm aim -> spray -> park behaviour in Gazebo,
8 cases (spec 2026-11-23 section 7 C; run by test/arm_aim/run_sim_aim_test.sh).

Needs: the headless sim (test/arm_aim/start_sim.sh) and arm_aim.launch.py running, same terminal
settings (source test/arm_aim/sim_env.sh). It refuses to run without a simulation clock (/clock):
it moves the arm, sprays (the sim relay is only a ROS message), drives the robot and drops a
"person" into Gazebo.

Set-up: start_map_fixture (test/arm_aim/fixtures.py: empty latched /map + static map -> odom),
so /safety_state can turn `clear` in the empty test world; wait for `clear`, MoveIt and weed_aim.
Recorded the whole time, by arrival time in this process (wall clock): /spray/relay (every
change), /joint_states (arm joints), the raw /scan converted to base_link (returns outside the
scan_self_filter box), /safety_state and /aim/result.

Cases (each prints `PASS <n> ...` with what it measured, or `FAIL <n> ...: <why>`):
  1. 9 targets x 0.28 / 0.34 / 0.40, y -0.10 / 0 / 0.10, z 0 (base_link). Each: reachable,
     aim_error_m <= 0.01, sprayed, the relay went on only while the arm was within 1 deg of
     joints_deg (arrived first), one on-period of 1.0 +- 0.15 s, the arm back within 1 deg of
     park; in the aim pose no raw /scan return outside the box (-0.53, 0.18, -0.16, 0.16) and
     /safety_state stays `clear`. (Returns while the arm travels to / from the aim pose are
     counted and printed too, but they are not a pass criterion: the solver's LiDAR keep-clear
     rule is for the aim pose, spec section 1 / 7 C "in every aim pose".)
  2. (0.80, 0, 0) -> `too far`; (-0.30, 0, 0) -> `behind the arm`; (0.05, 0, 0) ->
     `too close / under the arm` or `blocked by the robot body`. All `refused`, every joint
     within 0.5 deg of park the whole time, relay never on.
  3. Keepout: a temporary zones file (zone `test_bed`, square 0.1 m around the target, margin 0)
     set as /spray_controller zones_file; target (0.34, 0.10, 0) -> aims, `sprayed` false,
     spray_reason starts with `inside keepout zone`, relay never on, arm parks; zones_file back
     to ''. The square is centred on where the target is in `map` right now (the robot has
     driven since the start, so map (0.34, 0.10) is no longer that target).
  4. Robot moving: /cmd_vel_keyboard x 0.05 m/s at 10 Hz from the target until its result ->
     spray_reason `robot moving` or `aborted: robot moving`, relay never on, arm parks; then a
     zero command and 2 s to stop.
  5. E-stop: spray_seconds 2.0; target (0.34, 0, 0); at the relay's rising edge /e_stop true ->
     relay off within 0.1 s (from publishing /e_stop to receiving /spray/relay false, here),
     spray_reason `aborted: e-stop`, reason `e-stop`, state `error`, the arm stays where it is;
     /aim/park refused (`e-stop`) while the e-stop is on; /e_stop false -> /aim/park succeeds,
     arm within 1 deg of park; spray_seconds back to 1.0.
  6. Person: `sim_person.py spawn --ahead 1.0 --left 0.3`; wait until /safety_state is not
     `clear`; target (0.34, 0, 0) -> spray_reason starts with `safety_state`, relay never on, arm
     parks; `sim_person.py remove`, wait for `clear`. Why 0.3 m to the side and not straight
     ahead: straight ahead at 1.0 m the parked arm hides a person completely from the LD06 (the
     blind wedge in CLAUDE.md "Robot model": 0 returns at 1.0 m, first returns at 0.96 m), so
     /safety_state would stay `clear`.
  7. Target (0.34, 0, 0) in frame `nowhere` -> reason `no transform nowhere -> base_link`,
     `refused`, every joint within 0.5 deg of park, relay never on.
  8. Target A (0.34, 0, 0), 0.5 s later target B (0.28, 0.10, 0): exactly one result (A's, `ok`)
     and, for 15 s after it, no second result, every joint within 0.5 deg of park and the relay
     off (so B was dropped, not queued: a queued B would move the arm at once and report within
     one cycle, longest measured 11.5 s); then B again -> `ok`, and the arm left park only after
     B again was sent. (The brief's quiet window was 5 s; a whole cycle is longer than that.)
Before each case: /safety_state `clear` and the arm parked (if a failed case left it elsewhere
or in ERROR, /aim/park is called and the line says so). Parameters this test changes
(zones_file, spray_seconds), the e-stop, the drive command and the person are always undone.

Run (sim + arm_aim.launch.py running):
  source test/arm_aim/sim_env.sh && python3 test/arm_aim/sim_aim_check.py
Last line: "ARM AIM SIM TEST: PASS (8/8)" (exit 0) or "ARM AIM SIM TEST: FAIL (<passed>/8)"
(exit 1). Cleanup afterwards: see run_sim_aim_test.sh.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, Twist
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetStateValidity
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
sys.path.insert(0, HERE)
from aim_solver import JOINT_NAMES  # noqa: E402
from arm_config import load_park  # noqa: E402
from fixtures import start_map_fixture  # noqa: E402

SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
N_CASES = 8
TARGETS_9 = [(x, y, 0.0) for x in (0.28, 0.34, 0.40) for y in (-0.10, 0.0, 0.10)]
TARGET = (0.34, 0.0, 0.0)
SCAN_BOX = (-0.53, 0.18, -0.16, 0.16)   # base_link xmin, xmax, ymin, ymax: scan_self_filter box
MAX_AIM_ERROR = 0.01                    # m
ON_TIME, ON_TOL = 1.0, 0.15             # s, relay on-time with spray_seconds 1.0
OFF_LATENCY = 0.1                       # s, e-stop -> relay off
AIM_TOL = math.radians(1.0)             # arrived at the aim pose
PARK_TOL = math.radians(1.0)            # back in park
STILL_TOL = math.radians(0.5)           # refused target: the arm never left park
CLEAR_WAIT_S = 30.0
READY_WAIT_S = 120.0
RESULT_WAIT_S = 60.0                    # target -> its /aim/result
QUIET_S = 15.0                          # case 8: > one whole cycle (longest measured 11.5 s)
PERSON_WAIT_S = 20.0
KEEPOUT_HALF = 0.05                     # m, half the side of the case 3 square
DRIVE_SPEED, DRIVE_HZ = 0.05, 10.0      # case 4
PERSON_ARGS = ['--ahead', '1.0', '--left', '0.3']
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


class Fail(Exception):
    pass


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def deg(rad):
    return math.degrees(rad)


def fmt_xyz(xyz):
    return '(' + ', '.join(f'{v:.2f}' for v in xyz[:2]) + ')'


class SimAimCheck(Node):
    def __init__(self):
        super().__init__('sim_aim_check')
        self.clock_seen = False
        self.results = []           # (t, dict)
        self.relay = []             # (t, on) every change of /spray/relay
        self.joints = []            # (t, np.array of the 5 arm joints, JOINT_NAMES order)
        self.scans = []             # (t, returns outside SCAN_BOX, all returns, max x outside)
        self.safety = []            # (t, state) every /safety_state message
        self.laser_tf = None        # (R, t) base_link <- the scan's frame
        self.on_relay_rise = None   # case 5: called in the relay callback on a rising edge
        self.park_pose = None       # SRDF park, 5 rad (set_up)
        self._joint_buf = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Gazebo publishes /clock best effort: a reliable reader would never hear it
        self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(String, '/safety_state', self._on_safety, RELIABLE)
        self.create_subscription(String, '/aim/result', self._on_result, RELIABLE)
        self.create_subscription(Bool, '/spray/relay', self._on_relay, LATCHED)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.target_pub = self.create_publisher(PointStamped, '/aim_target', RELIABLE)
        self.e_stop_pub = self.create_publisher(Bool, '/e_stop', LATCHED)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_keyboard', 10)
        self.park_cli = self.create_client(Trigger, '/aim/park')
        self.spray_params = self.create_client(SetParameters, '/spray_controller/set_parameters')
        self.validity = self.create_client(GetStateValidity, '/check_state_validity')
        self.move_action = ActionClient(self, MoveGroup, '/move_action')

    # ------------------------------------------------------------------ recorders

    def _on_clock(self, msg):
        self.clock_seen = True

    def _on_safety(self, msg):
        self.safety.append((time.monotonic(), msg.data))

    def _on_result(self, msg):
        self.results.append((time.monotonic(), json.loads(msg.data)))

    def _on_relay(self, msg):
        now, on = time.monotonic(), bool(msg.data)
        rising = on and (not self.relay or not self.relay[-1][1])
        if not self.relay or self.relay[-1][1] != on:
            self.relay.append((now, on))
        if rising and self.on_relay_rise is not None:
            self.on_relay_rise(now)

    def _on_joints(self, msg):
        seen = False
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self._joint_buf[name] = pos
                seen = True
        if seen and all(j in self._joint_buf for j in JOINT_NAMES):
            self.joints.append((time.monotonic(),
                                np.array([self._joint_buf[j] for j in JOINT_NAMES])))

    def _on_scan(self, msg):
        now = time.monotonic()
        if self.laser_tf is None:
            try:
                tf = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, Time())
            except TransformException:
                return
            tr = tf.transform
            self.laser_tf = (quat_matrix(tr.rotation),
                             np.array([tr.translation.x, tr.translation.y, tr.translation.z]))
        r = np.array(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        rot, trans = self.laser_tf
        pts = (rot @ np.vstack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]),
                                np.zeros(int(ok.sum()))])).T + trans
        xmin, xmax, ymin, ymax = SCAN_BOX
        out = ~((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))
        max_x = float(pts[out, 0].max()) if out.any() else None
        self.scans.append((now, int(out.sum()), int(ok.sum()), max_x))

    # ------------------------------------------------------------------ helpers

    def spin_until(self, done, wall_s, every=None, every_s=None):
        """Spin until done() (True) or wall_s (False); call every() each every_s if given."""
        end = time.monotonic() + wall_s
        next_tick = time.monotonic()
        while rclpy.ok() and time.monotonic() < end:
            if done():
                return True
            if every is not None and time.monotonic() >= next_tick:
                every()
                next_tick += every_s
            rclpy.spin_once(self, timeout_sec=0.01)
        return bool(done())

    def spin_for(self, wall_s):
        self.spin_until(lambda: False, wall_s)

    def safety_now(self):
        return self.safety[-1][1] if self.safety else None

    def wait_clear(self, wall_s=CLEAR_WAIT_S):
        if not self.spin_until(lambda: self.safety_now() == 'clear', wall_s):
            raise Fail(f'/safety_state not clear after {wall_s:.0f} s '
                       f'(last: {self.safety_now()})')

    def fresh_joints(self, after):
        """The first arm joint sample received after `after` (waits up to 5 s)."""
        if not self.spin_until(lambda: self.joints and self.joints[-1][0] > after, 5.0):
            raise Fail('no /joint_states with all arm joints')
        return self.joints[-1][1]

    def off_park_deg(self, q):
        return max(abs(deg(v - p)) for v, p in zip(q, self.park_pose))

    def send_target(self, xyz, frame='base_link'):
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in xyz)
        t = time.monotonic()
        self.target_pub.publish(msg)
        return t

    def wait_result(self, since, wall_s=RESULT_WAIT_S, every=None, every_s=None):
        """The first /aim/result received after `since`: (t, dict)."""
        def got():
            return any(t > since for t, _ in self.results)
        if not self.spin_until(got, wall_s, every, every_s):
            raise Fail(f'no /aim/result within {wall_s:.0f} s (see $ARM_TEST_LOG/arm_aim.log)')
        return next((t, r) for t, r in self.results if t > since)

    def results_between(self, t0, t1):
        return [r for t, r in self.results if t0 < t <= t1]

    def on_periods(self, t0, t1):
        """[(t_on, t_off or None)] for every rising edge of the relay between t0 and t1."""
        out = []
        for i, (t, on) in enumerate(self.relay):
            if on and t0 <= t <= t1:
                t_off = next((t2 for t2, on2 in self.relay[i + 1:] if not on2), None)
                out.append((t, t_off))
        return out

    def max_dev_deg(self, t0, t1, ref):
        """(largest joint deviation from `ref` in deg, number of samples) between t0 and t1."""
        devs = [max(abs(deg(v - r)) for v, r in zip(q, ref))
                for t, q in self.joints if t0 <= t <= t1]
        if not devs:
            raise Fail('no /joint_states samples in the time window')
        return max(devs), len(devs)

    def call(self, client, request, wall_s=60.0):
        if not client.wait_for_service(timeout_sec=10.0):
            raise Fail(f'service {client.srv_name} not available')
        future = client.call_async(request)
        if not self.spin_until(future.done, wall_s):
            client.remove_pending_request(future)
            raise Fail(f'no answer from {client.srv_name} within {wall_s:.0f} s')
        return future.result()

    def park(self):
        res = self.call(self.park_cli, Trigger.Request())
        return bool(res.success), res.message

    def set_spray_param(self, name, value):
        if isinstance(value, str):
            pv = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
        else:   # spray_seconds is a DOUBLE parameter: 2 (int) would be refused, 2.0 is not
            pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        req = SetParameters.Request(parameters=[ParameterMsg(name=name, value=pv)])
        res = self.call(self.spray_params, req).results[0]
        if not res.successful:
            raise Fail(f'spray_controller refused {name} = {value!r}: {res.reason}')

    def run_person(self, *args):
        """`ros2 run my_bot sim_person.py <args>`, spinning meanwhile (recorders stay live)."""
        proc = subprocess.Popen(['ros2', 'run', 'my_bot', 'sim_person.py', *args],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.spin_until(lambda: proc.poll() is not None, 90.0)
        if proc.poll() is None:
            proc.kill()
            raise Fail(f'sim_person.py {" ".join(args)} did not finish in 90 s')
        out = proc.stdout.read().strip()
        if proc.returncode != 0:
            raise Fail(f'sim_person.py {" ".join(args)} failed: {out}')
        return out

    def map_point(self, xyz_base):
        """A base_link point in `map`, from the latest TF (map -> odom is the fixture's)."""
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except TransformException as e:
            raise Fail(f'no TF map <- base_link: {e}')
        tr = tf.transform
        return quat_matrix(tr.rotation) @ np.asarray(xyz_base) + np.array(
            [tr.translation.x, tr.translation.y, tr.translation.z])

    # ------------------------------------------------------------------ shared result checks

    def check_result(self, res, xyz, frame='base_link'):
        """The result is for this target (we did not pick up another one)."""
        got = (res.get('target') or {})
        if got.get('frame') != frame:
            raise Fail(f'result for frame {got.get("frame")!r}, expected {frame!r}: {res}')
        if frame == 'base_link':
            b = got.get('base_link_xyz')
            if not b or max(abs(u - v) for u, v in zip(b, xyz)) > 0.001:
                raise Fail(f'result for target {b}, expected {list(xyz)}')

    def expect_parked_after(self, t_result, what='arm'):
        q = self.fresh_joints(t_result)
        off = self.off_park_deg(q)
        if off > deg(PARK_TOL):
            raise Fail(f'{what} not back in park after the result: {off:.2f} deg off')
        return off

    def expect_no_relay(self, t0, t1):
        periods = self.on_periods(t0, t1)
        if periods:
            raise Fail(f'relay went on {len(periods)} time(s)')

    def aimed_and_refused_spray(self, xyz, reason_ok, label, every=None, every_s=None):
        """Shared by cases 3, 4, 6: aims, the spray is refused, parks, relay never on."""
        t_send = self.send_target(xyz)
        t_res, res = self.wait_result(t_send, every=every, every_s=every_s)
        self.check_result(res, xyz)
        if res.get('state') != 'ok' or res.get('reachable') is not True:
            raise Fail(f'state {res.get("state")!r}, reachable {res.get("reachable")!r}, '
                       f'reason {res.get("reason")!r} (expected an aim with a refused spray)')
        if res.get('sprayed') is not False or not reason_ok(res.get('spray_reason', '')):
            raise Fail(f'sprayed {res.get("sprayed")!r}, spray_reason '
                       f'{res.get("spray_reason")!r} (expected {label})')
        self.spin_for(0.5)
        self.expect_no_relay(t_send, time.monotonic())
        off = self.expect_parked_after(t_res)
        return res, off


# ---------------------------------------------------------------------- the cases

def one_target(node, xyz):
    """One of the 9 reachable targets: every check of case 1. Returns the measured numbers."""
    t_send = node.send_target(xyz)
    t_res, res = node.wait_result(t_send)
    node.spin_for(0.5)      # the relay's last message, a duplicate result
    node.check_result(res, xyz)
    if len(node.results_between(t_send, time.monotonic())) != 1:
        raise Fail('more than one result for one target')
    bad = []
    if res.get('state') != 'ok':
        bad.append(f'state {res.get("state")!r} reason {res.get("reason")!r}')
    if res.get('reachable') is not True:
        bad.append(f'reachable {res.get("reachable")!r}')
    err = res.get('aim_error_m')
    if not isinstance(err, (int, float)) or err > MAX_AIM_ERROR:
        bad.append(f'aim_error_m {err!r} (max {MAX_AIM_ERROR})')
    if res.get('sprayed') is not True:
        bad.append(f'sprayed {res.get("sprayed")!r} ({res.get("spray_reason")!r})')
    if bad:
        raise Fail('; '.join(bad))
    aim = np.radians(res['joints_deg'])

    periods = node.on_periods(t_send, t_res)
    if len(periods) != 1 or periods[0][1] is None:
        raise Fail(f'{len(periods)} complete relay on-periods (expected 1)')
    t_on, t_off = periods[0]
    on_time = t_off - t_on
    if abs(on_time - ON_TIME) > ON_TOL:
        raise Fail(f'relay on {on_time:.3f} s (expected {ON_TIME} +- {ON_TOL})')

    # The relay may go on only after the arm arrived: the newest joint sample before the rising
    # edge must be within 1 deg of the aim pose. "arrived" = the start of that unbroken run of
    # samples within 1 deg.
    before = [(t, q) for t, q in node.joints if t_send <= t <= t_on]
    if not before:
        raise Fail('no joint states between the target and the relay')

    def at_aim(q):
        return max(abs(v - a) for v, a in zip(q, aim)) <= AIM_TOL
    if not at_aim(before[-1][1]):
        off = max(abs(deg(v - a)) for v, a in zip(before[-1][1], aim))
        raise Fail(f'relay went on with the arm {off:.2f} deg from the aim pose')
    k = len(before) - 1
    while k > 0 and at_aim(before[k - 1][1]):
        k -= 1
    t_arrive = before[k][0]
    # The aim pose lasts until the first sample after the relay went off that is more than
    # 1 deg away again (park started).
    t_leave = next((t for t, q in node.joints if t > t_off and not at_aim(q)), t_res)

    # LiDAR keep-clear rule and /safety_state, in the aim pose
    in_pose = [s for s in node.scans if t_arrive <= s[0] <= t_leave]
    if not in_pose:
        raise Fail('no raw /scan while the arm was in the aim pose')
    outside = sum(s[1] for s in in_pose)
    if outside:
        max_x = max(s[3] for s in in_pose if s[3] is not None)
        raise Fail(f'{outside} raw /scan returns outside the box in the aim pose '
                   f'(farthest x {max_x:.3f})')
    states = [s for t, s in node.safety if t_arrive <= t <= t_leave]
    if not states:
        raise Fail('no /safety_state while the arm was in the aim pose')
    if any(s != 'clear' for s in states):
        raise Fail(f'/safety_state in the aim pose: {sorted(set(states))}')
    # Information only: returns while the arm travels (to the aim pose and back to park)
    moving = [s for s in node.scans
              if t_send <= s[0] <= t_res and not t_arrive <= s[0] <= t_leave]
    transit_x = [s[3] for s in moving if s[3] is not None]
    transit_states = sorted({s for t, s in node.safety
                             if t_send <= t <= t_res and not t_arrive <= t <= t_leave})
    off_park = node.expect_parked_after(t_res)
    print(f'  aim error {err * 1000:.1f} mm, joints {res["joints_deg"]} deg, tilt '
          f'{res.get("tilt_deg")} deg, standoff {res.get("standoff_m")} m; relay on '
          f'{on_time:.3f} s, {t_on - t_arrive:.2f} s after arriving; in the aim pose '
          f'{len(in_pose)} scans with 0 returns outside the box, /safety_state clear '
          f'({len(states)} msgs); back in park {off_park:.2f} deg off; durations '
          f'{res.get("durations")}; while moving: {sum(s[1] for s in moving)} returns outside '
          f'the box' + (f' (farthest x {max(transit_x):.3f})' if transit_x else '')
          + f', /safety_state {transit_states}', flush=True)
    return err, on_time, off_park, sum(s[1] for s in moving)


def case_nine_targets(node):
    """All 9 targets run even if one fails, so a failure does not hide the others."""
    done, failed = [], []
    for i, xyz in enumerate(TARGETS_9, 1):
        label = f'target {i}/9 {fmt_xyz(xyz)}'
        print(f'  {label}:', end='', flush=True)
        try:
            done.append(one_target(node, xyz))
        except Fail as e:
            print(f' FAIL {e}', flush=True)
            failed.append(f'{label}: {e}')
            before_case(node)       # parked and clear again before the next target
    if failed:
        raise Fail(f'{len(failed)} of 9 targets failed: ' + ' | '.join(failed))
    errors, on_times, parks, transit = zip(*done)
    return (f'aim error {min(errors) * 1000:.1f}-{max(errors) * 1000:.1f} mm, relay on '
            f'{min(on_times):.3f}-{max(on_times):.3f} s, park within {max(parks):.2f} deg, '
            f'0 raw scan returns outside the box in the aim poses (while moving: '
            f'{sum(transit)})')


def case_impossible(node):
    parts = []
    for xyz, ok_reasons in (((0.80, 0.0, 0.0), ('too far',)),
                            ((-0.30, 0.0, 0.0), ('behind the arm',)),
                            ((0.05, 0.0, 0.0), ('too close / under the arm',
                                                'blocked by the robot body'))):
        t_send = node.send_target(xyz)
        t_res, res = node.wait_result(t_send)
        node.spin_for(1.0)      # the arm must not move after the result either
        node.check_result(res, xyz)
        if res.get('state') != 'refused' or res.get('reason') not in ok_reasons:
            raise Fail(f'{fmt_xyz(xyz)}: state {res.get("state")!r} reason '
                       f'{res.get("reason")!r} (expected refused, {" or ".join(ok_reasons)})')
        dev, n = node.max_dev_deg(t_send, time.monotonic(), node.park_pose)
        if dev > deg(STILL_TOL):
            raise Fail(f'{fmt_xyz(xyz)}: a joint moved {dev:.2f} deg from park')
        node.expect_no_relay(t_send, time.monotonic())
        parts.append(f'{fmt_xyz(xyz)} {res["reason"]!r} (joints max {dev:.2f} deg off park, '
                     f'{n} samples)')
    return '; '.join(parts) + '; relay never on'


def case_keepout(node):
    xyz = (0.34, 0.10, 0.0)
    cx, cy, _ = node.map_point(xyz)
    h = KEEPOUT_HALF
    log_dir = os.environ.get('ARM_TEST_LOG') or tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix='keepout_test_', suffix='.yaml', dir=log_dir)
    with os.fdopen(fd, 'w') as f:
        f.write('# temporary test zone (sim_aim_check.py case 3)\nframe_id: map\nzones:\n'
                f'  - name: test_bed\n    polygon: [[{cx - h:.4f}, {cy - h:.4f}], '
                f'[{cx + h:.4f}, {cy - h:.4f}], [{cx + h:.4f}, {cy + h:.4f}], '
                f'[{cx - h:.4f}, {cy + h:.4f}]]\n    margin: 0.0\n')
    try:
        node.set_spray_param('zones_file', path)
        res, off = node.aimed_and_refused_spray(
            xyz, lambda r: r.startswith('inside keepout zone'), "'inside keepout zone ...'")
    finally:
        try:
            node.set_spray_param('zones_file', '')
        finally:
            os.unlink(path)
    return (f'zone test_bed = 0.1 m square around map ({cx:.3f}, {cy:.3f}); aim error '
            f'{res["aim_error_m"] * 1000:.1f} mm, sprayed false, {res["spray_reason"]!r}, relay '
            f'never on, parked ({off:.2f} deg off); zones_file back to \'\'')


def case_robot_moving(node):
    cmd = Twist()
    cmd.linear.x = DRIVE_SPEED
    try:
        res, off = node.aimed_and_refused_spray(
            TARGET, lambda r: r in ('robot moving', 'aborted: robot moving'),
            "'robot moving' or 'aborted: robot moving'",
            every=lambda: node.cmd_pub.publish(cmd), every_s=1.0 / DRIVE_HZ)
    finally:
        node.cmd_pub.publish(Twist())
        node.spin_for(2.0)
    return (f'/cmd_vel_keyboard {DRIVE_SPEED} m/s during the cycle -> sprayed false, '
            f'{res["spray_reason"]!r}, relay never on, parked ({off:.2f} deg off); stopped')


def case_e_stop(node):
    hit = {}

    def e_stop_on_rise(t_rise):
        if 't_e_stop' not in hit:
            node.e_stop_pub.publish(Bool(data=True))
            hit['t_rise'], hit['t_e_stop'] = t_rise, time.monotonic()

    node.set_spray_param('spray_seconds', 2.0)
    try:
        node.on_relay_rise = e_stop_on_rise
        t_send = node.send_target(TARGET)
        t_res, res = node.wait_result(t_send)
        node.on_relay_rise = None
        node.check_result(res, TARGET)
        if 't_e_stop' not in hit:
            raise Fail(f'the relay never went on (result {res})')
        t_off = next((t for t, on in node.relay if not on and t >= hit['t_rise']), None)
        if t_off is None:
            raise Fail('no /spray/relay false after the e-stop')
        latency = t_off - hit['t_e_stop']
        if latency > OFF_LATENCY:
            raise Fail(f'relay off {latency * 1000:.1f} ms after /e_stop (max '
                       f'{OFF_LATENCY * 1000:.0f} ms)')
        if (res.get('spray_reason') != 'aborted: e-stop' or res.get('state') != 'error'
                or res.get('reason') != 'e-stop'):
            raise Fail(f'spray_reason {res.get("spray_reason")!r}, state {res.get("state")!r}, '
                       f'reason {res.get("reason")!r} (expected aborted: e-stop, error, e-stop)')
        aim = np.radians(res['joints_deg'])
        node.spin_for(1.0)
        ok, msg = node.park()
        if ok or msg != 'e-stop':
            raise Fail(f'/aim/park during the e-stop answered ({ok}, {msg!r}), expected '
                       f'(False, \'e-stop\')')
        # no motion of its own after the e-stop: still at the aim pose
        drift, n = node.max_dev_deg(hit['t_e_stop'] + 0.2, time.monotonic(), aim)
        if drift > deg(AIM_TOL):
            raise Fail(f'the arm moved {drift:.2f} deg after the e-stop')
        node.e_stop_pub.publish(Bool(data=False))
        node.spin_for(0.5)
        t_park = time.monotonic()
        ok, msg = node.park()
        if not ok:
            raise Fail(f'/aim/park after the release answered ({ok}, {msg!r})')
        off = node.expect_parked_after(time.monotonic())
    finally:
        node.on_relay_rise = None
        node.e_stop_pub.publish(Bool(data=False))
        node.set_spray_param('spray_seconds', 1.0)
    return (f'relay off {latency * 1000:.1f} ms after /e_stop (by message receipt), '
            f'{res["spray_reason"]!r}, reason {res["reason"]!r}, state error; arm held the aim '
            f'pose ({drift:.2f} deg, {n} samples); /aim/park refused \'e-stop\'; after release '
            f'/aim/park -> {msg!r} in {time.monotonic() - t_park:.1f} s, {off:.2f} deg off '
            f'park; spray_seconds back to 1.0')


def case_person(node):
    out = node.run_person('spawn', *PERSON_ARGS)
    try:
        t0 = time.monotonic()
        if not node.spin_until(lambda: node.safety_now() not in (None, 'clear'), PERSON_WAIT_S):
            raise Fail(f'/safety_state still {node.safety_now()!r} {PERSON_WAIT_S:.0f} s after '
                       f'the person was spawned ({out})')
        state = node.safety_now()
        t_seen = time.monotonic() - t0
        res, off = node.aimed_and_refused_spray(
            TARGET, lambda r: r.startswith('safety_state'), "'safety_state ...'")
    finally:
        node.run_person('remove')
    node.wait_clear()
    return (f'person {" ".join(PERSON_ARGS)} -> /safety_state {state!r} after {t_seen:.1f} s; '
            f'sprayed false, {res["spray_reason"]!r}, relay never on, parked ({off:.2f} deg '
            f'off); removed, clear again')


def case_no_transform(node):
    t_send = node.send_target(TARGET, frame='nowhere')
    t_res, res = node.wait_result(t_send)
    node.spin_for(1.0)
    node.check_result(res, TARGET, frame='nowhere')
    want = 'no transform nowhere -> base_link'
    if res.get('reason') != want or res.get('state') != 'refused':
        raise Fail(f'state {res.get("state")!r} reason {res.get("reason")!r} (expected '
                   f'refused, {want!r})')
    dev, n = node.max_dev_deg(t_send, time.monotonic(), node.park_pose)
    if dev > deg(STILL_TOL):
        raise Fail(f'a joint moved {dev:.2f} deg from park')
    node.expect_no_relay(t_send, time.monotonic())
    return f'{res["reason"]!r}, refused, joints max {dev:.2f} deg off park ({n} samples)'


def case_busy(node):
    a, b = TARGET, (0.28, 0.10, 0.0)
    t_a = node.send_target(a)
    node.spin_for(0.5)
    node.send_target(b)
    t_res, res = node.wait_result(t_a)
    node.check_result(res, a)
    if res.get('state') != 'ok':
        raise Fail(f'A: state {res.get("state")!r} reason {res.get("reason")!r}')
    # B must have been DROPPED, not queued: a queued B would start moving the arm right after A's
    # result and report within one cycle, i.e. inside this window.
    node.spin_for(QUIET_S)
    t_quiet = time.monotonic()
    got = node.results_between(t_a, t_quiet)
    if len(got) != 1:
        raise Fail(f'{len(got)} results for A then B (expected only A\'s): {got}')
    dev, n = node.max_dev_deg(t_res, t_quiet, node.park_pose)
    if dev > deg(STILL_TOL):
        raise Fail(f'the arm left park after A\'s result ({dev:.2f} deg): B was queued?')
    node.expect_no_relay(t_res, t_quiet)
    t_b = node.send_target(b)
    t_res_b, res_b = node.wait_result(t_b)
    node.check_result(res_b, b)
    if res_b.get('state') != 'ok':
        raise Fail(f'B again: state {res_b.get("state")!r} reason {res_b.get("reason")!r}')
    # this B's own cycle moved the arm: it left park only after B again was sent
    t_left = next((t for t, q in node.joints
                   if t > t_res and node.off_park_deg(q) > deg(STILL_TOL)), None)
    if t_left is None or t_left <= t_b:
        raise Fail(f'B again: the arm left park at {t_left} s, not after B was sent ({t_b} s)')
    node.expect_parked_after(t_res_b)
    return (f'A then B 0.5 s later -> 1 result (A, ok, aim error '
            f'{res["aim_error_m"] * 1000:.1f} mm); in the next {QUIET_S:.0f} s no result, joints '
            f'max {dev:.2f} deg off park ({n} samples), relay off; B again -> ok (aim error '
            f'{res_b["aim_error_m"] * 1000:.1f} mm, {res_b["spray_reason"]!r}), arm left park '
            f'{t_left - t_b:.2f} s after B was sent')


CASES = [
    (1, 'nine reachable targets', case_nine_targets),
    (2, 'impossible targets refused', case_impossible),
    (3, 'keepout zone', case_keepout),
    (4, 'robot moving', case_robot_moving),
    (5, 'e-stop during the spray', case_e_stop),
    (6, 'person near', case_person),
    (7, 'unknown frame', case_no_transform),
    (8, 'busy: second target dropped', case_busy),
]


# ---------------------------------------------------------------------- set-up and main

def set_up(node):
    node.park_pose = load_park(SRDF)
    if not node.spin_until(lambda: node.clock_seen, 30.0):
        raise Fail('no /clock: is the simulation running? (refusing to run without it)')
    start_map_fixture(node)
    t0 = time.monotonic()
    node.wait_clear()
    print(f'  /safety_state clear after {time.monotonic() - t0:.1f} s (map fixture)')
    t0 = time.monotonic()
    if not node.spin_until(lambda: (node.validity.service_is_ready()
                                    and node.move_action.server_is_ready()
                                    and node.park_cli.service_is_ready()
                                    and node.spray_params.service_is_ready()
                                    and node.target_pub.get_subscription_count() > 0
                                    and node.laser_tf is not None and node.joints),
                           READY_WAIT_S):
        raise Fail(f'MoveIt / weed_aim / spray_controller / scan / joint states not up after '
                   f'{READY_WAIT_S:.0f} s (is arm_aim.launch.py running? $ARM_TEST_LOG)')
    print(f'  MoveIt, weed_aim, spray_controller up after {time.monotonic() - t0:.1f} s; '
          f'laser at x {node.laser_tf[1][0]:.4f} z {node.laser_tf[1][2]:.4f} in base_link',
          flush=True)


def before_case(node):
    """/safety_state clear and the arm parked; recover with /aim/park if a failed case did not."""
    node.wait_clear()
    q = node.fresh_joints(time.monotonic())
    last_state = node.results[-1][1].get('state') if node.results else None
    if node.off_park_deg(q) > deg(PARK_TOL) or last_state == 'error':
        ok, msg = node.park()
        print(f'  (recovery before this case: /aim/park -> {ok}, {msg!r})')
        node.wait_clear()


def main():
    rclpy.init()
    node = SimAimCheck()
    passed = 0
    try:
        set_up(node)
        for num, title, fn in CASES:
            try:
                before_case(node)
                summary = fn(node)
                passed += 1
                print(f'PASS {num} {title}: {summary}', flush=True)
            except Fail as e:
                print(f'FAIL {num} {title}: {e}', flush=True)
            except (KeyboardInterrupt, ExternalShutdownException):
                raise
            except Exception as e:      # a bug in one case must not hide the others
                print(f'FAIL {num} {title}: {type(e).__name__}: {e}', flush=True)
    except Fail as e:
        print(f'  set-up failed: {e}')
    except (KeyboardInterrupt, ExternalShutdownException):
        print('  interrupted')
    finally:
        try:
            node.e_stop_pub.publish(Bool(data=False))
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()
    verdict = 'PASS' if passed == N_CASES else 'FAIL'
    print(f'ARM AIM SIM TEST: {verdict} ({passed}/{N_CASES})', flush=True)
    sys.exit(0 if passed == N_CASES else 1)


if __name__ == '__main__':
    main()

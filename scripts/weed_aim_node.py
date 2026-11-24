#!/usr/bin/env python3
"""
weed_aim_node.py  -  aims the spray arm at ONE target point, sprays it, parks the arm, reports.

Give it a point (a weed on the ground) and it runs one CYCLE:
   1. TARGET    /aim_target or /clicked_point (geometry_msgs/PointStamped, any frame TF knows;
                RViz's "Publish Point" tool sends /clicked_point)
   2. WHERE     TF: target -> base_link (z = 0 is the ground) -> arm_base
   3. SOLVE     aim_solver.solve(): ranked joint-angle candidates that put the nozzle's centre
                line through the point, from 8-25 cm away, without the arm in the LiDAR plane
   4. VALIDATE  MoveIt /check_state_validity, candidate by candidate: the first one that does not
                hit the robot (or the ground box this node adds to MoveIt's world) wins
   5. MOVE      MoveIt /move_action: plan + execute that joint goal
   6. ARRIVED?  /joint_states: every arm joint within arrive_tolerance_deg for arrive_hold s AND
                the arm STILL (rule R18): every joint moved at most arrive_still_speed x
                arrive_hold over the last arrive_hold s, judged like the spray controller judges
                "arm still" but stricter (0.01 vs its 0.02 rad/s). Without it the spray was asked
                for while the arm still crept at 0.1-0.2 rad/s inside the last degree, and the
                spray controller refused "arm moving" (Task 6: 3 of 15 sprays).
   7. MEASURE   aim error = distance from the target to the nozzle's centre line, taken from the
                spray_tip TF (what the arm really does, not the solver's own numbers)
   8. SPRAY     /spray/trigger, only if parameter `spray` is true. The spray controller makes ALL
                the safety decisions (armed, arm + robot still, keepout, e-stop, /safety_state);
                this node only asks and reports the answer.
   9. PARK      MoveIt back to the SRDF `park` pose (arrival checked the same way)
  10. REPORT    /aim/result

One cycle at a time: a target that arrives while a cycle runs (or while the node is in ERROR) is
logged and dropped - no result, nothing queued (a detector may publish many).

Outputs
  /aim/result   std_msgs/String, JSON, one per cycle:
                  target {frame, base_link_xyz}, reachable, reason, joints_deg, tilt_deg,
                  standoff_m, aim_error_m, sprayed, spray_reason,
                  durations {solve, move, spray, park} (wall s; solve = solver + MoveIt
                  validity checks, move = plan + execute + arrival check), state
                state  ok       the arm aimed (aim error measured), the spray step ran (`sprayed`
                                and `spray_reason` say how it went) and the arm is parked again
                       refused  the target was not aimed at (`reason` says why); the arm never
                                moved, or it moved and is parked again
                       error    e-stop or PARK FAILED: the arm may be anywhere. Targets are dropped
                                until /aim/park succeeds.
                `reachable` = a collision-free aim pose was found (MoveIt was asked to move).
  /aim/markers  visualization_msgs/MarkerArray (base_link): target sphere, nozzle centre line,
                spray cone only while /spray/relay is true
  /aim/park     std_srvs/Trigger: park the arm now (refused with "busy" during a cycle and with
                "e-stop" while /e_stop is true)

E-stop (/e_stop true) at ANY point of a cycle (also before the arm moved, rule R17): the node
stops the arm at once, never moves it again by itself (no retry, no park), reports `e-stop` with
state `error`, and stays in ERROR until the e-stop is released and /aim/park succeeds. During a
spray the spray controller switches the relay off by itself; its answer ("aborted: e-stop")
becomes `spray_reason`.
A SHORT e-stop counts too: the change to true is REMEMBERED (`_e_stop_seen`) until the next
cycle or /aim/park starts. Without that, a true -> false pulse while the node only waits (the 2 s
pause before a CONTROL_FAILED retry, a goal being accepted, the spray) would be over when the
cycle looks next, and the arm would move again although the e-stop was pressed.
How the arm is stopped: the node publishes "stop" on /trajectory_execution_event (MoveIt's own stop
signal, the same one RViz's Stop button sends: move_group cancels the arm_controller goal and the
arm holds where it is) AND cancels the MoveGroup goal. The cancel alone does NOT stop the arm in
MoveIt 2.5.9 (measured 2026-11-24: move_group handles a MoveGroup cancel only after the move has
finished; the arm drove on to the aim pose). "stop" did: the arm stood still within ~0.15 s. A
"stop" that comes while MoveIt is still planning finds nothing to stop, so the node repeats it
every 50 ms until MoveIt reports the goal finished.

How the node runs (why one thread):
  Everything runs in ONE thread (rclpy SingleThreadedExecutor) and the cycle is written as an
  `async` function: every `await` hands control back to the executor, which then handles
  /e_stop, /joint_states, TF and the MoveIt answers before the cycle continues. So the cycle reads
  top to bottom like a recipe, yet an e-stop is handled within one executor round (its callback
  stops the arm right away), and nothing needs locks. No callback ever blocks: there are no sleeps
  and no blocking service calls.
  Why not a MultiThreadedExecutor like spray_controller: this node needs use_sim_time (MoveIt and
  TF use the sim clock), and rclpy adds its own /clock subscription with the default QoS event
  handler. In rclpy 3.3.19 (Humble) a MultiThreadedExecutor busy-loops on such subscriptions and
  freezes the process for 0.1-0.35 s at a time (found in the spray controller work). The single
  threaded executor does not: measured on this VM, 50 Hz topic, default handler vs none: same CPU,
  worst 100 Hz timer gap 16 ms vs 12 ms (multi-threaded: 80 % CPU, 133 ms gaps). So plain
  subscriptions and tf2_ros.TransformListener are fine here.

Times: the node's clock (sim time in Gazebo, wall time on the robot) for all timeouts; the
reported durations are wall seconds from a steady clock (Gazebo's /clock ticks only every 0.1 s,
too coarse for them). A steady 20 Hz timer makes sure the executor wakes up even when no topic
arrives (and repeats the "stop" after an e-stop).

Run (normally via launch/arm_aim.launch.py, which passes the installed safety.yaml and SRDF):
  ros2 run my_bot weed_aim_node.py --ros-args --params-file config/arm_aim.yaml \
      -p use_sim_time:=true
"""
import json
import math
import os
import signal
import sys
import time
from typing import Sequence

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import Point, PointStamped, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, ColorRGBA, String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aim_solver import JOINT_NAMES, AimLimits, ArmGeometry, ray_point_distance, solve
from arm_config import load_keep_box, load_park

GROUP = 'arm'               # MoveIt planning group (config/moveit/weedbot.srdf)
GOAL_TOL = 0.001            # rad, MoveIt joint goal tolerance
TICK_S = 0.05               # steady wake-up timer: waits and timeouts are checked this often
MOVEIT_WAIT_S = 10.0        # a target that arrives while MoveIt is still starting waits this long
# move_group offers its services a moment before its OWN action client has found arm_controller;
# a goal sent in that gap fails with CONTROL_FAILED (seen 2026-11-23, goal 0.23 s after start).
# So the first goal waits this long after MoveIt appeared, and a CONTROL_FAILED is retried once.
MOVEIT_SETTLE_S = 3.0
RETRY_PAUSE_S = 2.0         # pause before that one retry
ANSWER_S = 5.0              # longest wait for a MoveIt service answer / goal acceptance
SPRAY_ANSWER_S = 10.0       # longest wait for /spray/trigger (the spray itself is capped at 3 s)
GROUND_RETRY_S = 5.0        # ask again if /apply_planning_scene did not answer the ground box
STOP_TOPIC = '/trajectory_execution_event'  # MoveIt stops the running arm move on "stop"
STOP_REPEAT_S = 10.0        # after an e-stop, repeat "stop" until the goal ends, at most this long
# Older /joint_states do not count as "arrived"; a longer gap in them restarts the "still" window
# (the same 0.5 s as spray_controller's joint_states_timeout).
JOINTS_MAX_AGE_S = 0.5
GROUND_SIZE = (4.0, 4.0, 0.1)   # m, the ground box MoveIt keeps the arm out of
CONE_HALF_ANGLE = math.radians(12.0)    # spray cone drawn in RViz (looks only)

# Parameters: name -> (default, description, fixed at start?). The fixed ones are read-only, so a
# `ros2 param set` on them fails loudly; all others are read again at the start of every cycle.
PARAMS = {
    'base_frame': ('base_link', 'robot frame, z = 0 is the ground', True),
    'arm_base_frame': ('arm_base', 'arm base frame (the aim solver works in it)', True),
    'laser_frame': ('laser_frame', 'LiDAR frame: its height = the scan plane the arm keeps out of',
                    True),
    'nozzle_frame': ('spray_tip', 'nozzle frame, +z = spray direction (aim error, markers)', True),
    'safety_yaml': ('', 'safety.yaml with the scan_self_filter box; "" = the installed one', True),
    'srdf_path': ('', 'MoveIt SRDF with the park pose; "" = the installed one', True),
    'keep_clear_shrink': (0.02, 'the arm may cross the scan plane only this far inside the '
                                'self-filter box (m)', True),
    'scan_band': (0.03, 'LiDAR keep-clear band: +- this around the scan plane height (m)', False),
    'tilt_max_deg': (50.0, 'largest nozzle tilt from straight down (deg)', False),
    'standoff_min': (0.08, 'nearest nozzle tip -> target distance (m)', False),
    'standoff_max': (0.25, 'farthest nozzle tip -> target distance (m)', False),
    'tip_clearance': (0.03, 'nozzle tip at least this far above the target (m)', False),
    'spray': (True, 'ask /spray/trigger after aiming (false = aim, measure, park only)', False),
    'velocity_scaling': (0.5, 'MoveIt velocity scaling (0..1 of config/moveit/joint_limits)',
                         False),
    'acceleration_scaling': (0.5, 'MoveIt acceleration scaling (0..1)', False),
    'arrive_tolerance_deg': (1.0, 'arrived = every joint this close to the goal (deg) ...', False),
    'arrive_hold': (0.3, '... for this long (s), and the arm still over this long ...', False),
    'arrive_still_speed': (0.01, '... still = every joint slower than this (rad/s); keep it below '
                                 'spray_controller arm_still_speed', False),
    'arrive_timeout': (15.0, '... within this long after MoveIt finished (s)', False),
    'planning_time': (5.0, 'MoveIt planning time limit (s)', False),
    'ground_top': (0.02, 'top of the ground box in base_link (m): the arm stays above it', True),
}


# ---------------------------------------------------------------------- helpers (no ROS graph)

def reached(current: dict, goal: Sequence[float], tol_rad: float) -> bool:
    """True if every arm joint in `current` (name -> rad) is within tol_rad of `goal`
    (JOINT_NAMES order). A missing joint is NOT arrived: never assume what we cannot see."""
    for name, target in zip(JOINT_NAMES, goal):
        value = current.get(name)
        # `not (a <= b)` so that a NaN position counts as not arrived
        if value is None or not (abs(value - target) <= tol_rad):
            return False
    return True


def joint_goal_constraints(goal: Sequence[float], tol: float = GOAL_TOL) -> Constraints:
    """MoveIt goal "be at these five joint angles" (JOINT_NAMES order, +-tol rad)."""
    c = Constraints()
    for name, position in zip(JOINT_NAMES, goal):
        c.joint_constraints.append(JointConstraint(
            joint_name=name, position=float(position),
            tolerance_above=tol, tolerance_below=tol, weight=1.0))
    return c


def clamp_to_limits(goal: Sequence[float], lower: Sequence[float],
                    upper: Sequence[float]) -> tuple:
    """Pull every joint value inside [lower, upper] (rule R13). Why: the SRDF park value
    1.5707963268 is 5e-12 rad above the URDF limit pi/2, and solver values can be 1e-12 past a
    limit; a goal outside the limits, by however little, can be refused by MoveIt or the
    controller, so we never send one."""
    return tuple(float(min(max(v, lo), hi)) for v, lo, hi in zip(goal, lower, upper))


def add_joint_sample(hist: list, t: float, positions: Sequence[float], keep: float,
                     max_gap: float) -> None:
    """Add one arm sample (time s, positions in JOINT_NAMES order) to `hist`, for settled().

    A gap longer than max_gap empties `hist` first: we do not know what the arm did in it, so
    the "still" window starts again. Only what a window of `keep` s needs is kept: the newest
    sample at or before t - keep, and everything after it.
    """
    if hist and t - hist[-1][0] > max_gap:
        hist.clear()
    hist.append((t, tuple(float(v) for v in positions)))
    while len(hist) >= 2 and hist[1][0] <= t - keep:
        hist.pop(0)


def settled(hist: Sequence, now: float, window: float, max_speed: float) -> bool:
    """True if the arm was still for the last `window` s: every joint's position spread over
    the samples covering [now - window, now] is at most max_speed x window. Same way of
    measuring as spray_controller's "arm still" check (positions, not the velocity field: a
    servo driver may not report velocity). Not enough history to cover the window, or a NaN
    position, = not still."""
    start = now - window
    first = None
    for i, (t, _) in enumerate(hist):
        if t <= start:
            first = i
    if first is None:
        return False
    pos = np.array([p for _, p in hist[first:]], dtype=float)
    spread = pos.max(axis=0) - pos.min(axis=0)
    # `<=` is False for NaN, so a NaN position counts as moving
    return bool(np.all(spread <= max_speed * window))


def error_name(val) -> str:
    """MoveItErrorCodes number -> its name (e.g. -1 -> PLANNING_FAILED), for readable logs."""
    for name in dir(MoveItErrorCodes):
        if name.isupper() and getattr(MoveItErrorCodes, name) == val:
            return name
    return str(val)


def _num(value, digits):
    """A float rounded for the JSON report, or None if it is missing or not finite."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _tf_point(tf, xyz) -> np.ndarray:
    """Apply a TransformStamped to a point (x, y, z)."""
    p = PointStamped()
    p.point.x, p.point.y, p.point.z = (float(v) for v in xyz)
    q = do_transform_point(p, tf).point
    return np.array([q.x, q.y, q.z])


class _OneRound:
    """`await _OneRound()` hands control back to the executor for one round: it runs whatever
    callbacks are ready (e-stop, joint states, MoveIt answers ...) and then resumes us."""

    def __await__(self):
        yield


class _Refused(Exception):
    """The target is not aimed at; the message is the result's `reason`."""


class _EStop(Exception):
    """/e_stop was (or still is) true during a cycle or a park."""


class _ParkFailed(Exception):
    """Parking did not work: the arm may be anywhere."""


# ---------------------------------------------------------------------- the node

class WeedAim(Node):
    def __init__(self, **node_kwargs):
        super().__init__('weed_aim', **node_kwargs)
        for name, (default, doc, fixed) in PARAMS.items():
            self.declare_parameter(name, default, ParameterDescriptor(description=doc,
                                                                      read_only=fixed))
        p = self._read_params()
        self._base, self._arm_base = p['base_frame'], p['arm_base_frame']
        self._laser, self._nozzle = p['laser_frame'], p['nozzle_frame']
        self._ground_top = p['ground_top']
        share = None
        if not p['safety_yaml'] or not p['srdf_path']:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('my_bot')
        safety_yaml = p['safety_yaml'] or os.path.join(share, 'config', 'safety.yaml')
        srdf = p['srdf_path'] or os.path.join(share, 'config', 'moveit', 'weedbot.srdf')
        # Both raise (= the node refuses to start) if the file is missing or incomplete: aiming
        # without the LiDAR keep-clear box or without a park pose must never happen.
        self._keep_box = load_keep_box(safety_yaml, p['keep_clear_shrink'])
        self._park = load_park(srdf)

        self._geom = None            # ArmGeometry, from /robot_description
        self._joints = {}            # latest arm joint positions (rad)
        self._joints_time = None     # node-clock time of the latest arm joint state
        # [(steady time, 5 positions)] for settled(): steady wall time like spray_controller,
        # so both nodes look at the same window (the sim clock only ticks every 0.1 s)
        self._joint_hist = []
        self._e_stop = False         # /e_stop right now
        self._e_stop_seen = False    # /e_stop went true since this cycle / park started
        self._relay = False          # /spray/relay: only for the spray cone marker
        self._busy = False           # a cycle or an /aim/park is running
        self._error = ''             # '' = ready; else why targets are dropped until /aim/park
        self._goal_handle = None     # the running MoveGroup goal, for the e-stop cancel
        self._goal_result = None     # its result future (done = MoveIt finished with it)
        self._halting = None         # (result future, until): repeat "stop" until it is done
        self._moveit_since = None    # node-clock time MoveIt was first seen up (None = not up)
        self._ground_ok = False      # the ground box is in MoveIt's world
        self._ground_future, self._ground_asked = None, 0.0
        self._marker_target = None   # target point (base_link) drawn in RViz

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # /e_stop VOLATILE (and reliable): a transient-local reader would not hear the owner's
        # `ros2 topic pub` (volatile writer); twist_mux and spray_controller read it the same way.
        volatile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, '/robot_description', self._on_robot_description,
                                 latched)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states,
                                 qos_profile_sensor_data)
        self.create_subscription(Bool, '/e_stop', self._on_e_stop, volatile)
        self.create_subscription(Bool, '/spray/relay', self._on_relay, latched)
        self.create_subscription(PointStamped, '/aim_target', self._on_target, volatile)
        self.create_subscription(PointStamped, '/clicked_point', self._on_target, volatile)
        self._result_pub = self.create_publisher(String, '/aim/result', 10)
        self._stop_pub = self.create_publisher(String, STOP_TOPIC, volatile)
        self._marker_pub = self.create_publisher(MarkerArray, '/aim/markers', 10)

        self._validity = self.create_client(GetStateValidity, '/check_state_validity')
        self._scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._trigger = self.create_client(Trigger, '/spray/trigger')
        self._move_group = ActionClient(self, MoveGroup, '/move_action')
        # Reentrant: a second /aim/park call while one runs must get its "busy" answer at once
        # instead of waiting in line.
        self.create_service(Trigger, '/aim/park', self._on_park,
                            callback_group=ReentrantCallbackGroup())

        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(TICK_S, self._on_tick, clock=steady)
        self.create_timer(1.0, self._on_moveit_timer, clock=steady)
        self.create_timer(0.2, self._publish_markers, clock=steady)
        self.get_logger().info(
            f'weed_aim ready for targets on /aim_target and /clicked_point; park '
            f'{[round(math.degrees(v), 1) for v in self._park]} deg, keep-clear box '
            f'{self._keep_box} (waiting for /robot_description and MoveIt)')

    # ------------------------------------------------------------------ small tools

    def _read_params(self) -> dict:
        return {name: self.get_parameter(name).value for name in PARAMS}

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stopped(self) -> bool:
        """E-stop on now, or pressed (even briefly) since this cycle / park started."""
        return self._e_stop or self._e_stop_seen

    async def _sleep(self, seconds):
        """Wait `seconds` without blocking the node; ends early on an e-stop."""
        end = self._now() + seconds
        while self._now() < end and not self._stopped():
            await _OneRound()

    async def _wait(self, future, timeout, stop_on_e_stop=True) -> bool:
        """Wait (without blocking the node) until `future` is done: True. False on timeout, or
        at once on an e-stop if stop_on_e_stop."""
        end = self._now() + timeout
        while not future.done():
            if (stop_on_e_stop and self._stopped()) or self._now() >= end:
                return False
            await _OneRound()
        return True

    def _lookup(self, target, source):
        """Latest TF target <- source, or refuse with `no transform <source> -> <target>`.
        Latest, not at the message stamp: the robot stands still while it aims."""
        try:
            return self._tf.lookup_transform(target, source, Time())
        except TransformException:
            raise _Refused(f'no transform {source} -> {target}') from None

    def _fresh_joints(self) -> bool:
        return (self._joints_time is not None
                and self._now() - self._joints_time <= JOINTS_MAX_AGE_S)

    def _parked(self, p) -> bool:
        return self._fresh_joints() and reached(self._joints, self._park_goal(),
                                                math.radians(p['arrive_tolerance_deg']))

    def _park_goal(self) -> tuple:
        return clamp_to_limits(self._park, self._geom.lower, self._geom.upper)

    def _halt(self, handle=None, result=None):
        """Stop the arm NOW (see "How the arm is stopped" in the module docstring): MoveIt's
        "stop" event, plus a cancel of the MoveGroup goal; repeat "stop" until MoveIt is done."""
        if handle is None:
            handle, result = self._goal_handle, self._goal_result
        self._stop_pub.publish(String(data='stop'))
        if handle is None:
            return
        handle.cancel_goal_async()
        if result is not None and not result.done():
            self._halting = (result, self._now() + STOP_REPEAT_S)
        self.get_logger().warn('arm stopped: MoveIt "stop" sent and the MoveIt goal cancelled')

    # ------------------------------------------------------------------ callbacks

    def _on_tick(self):
        """20 Hz (steady): wakes the executor; repeats "stop" while an e-stopped goal runs."""
        if self._halting is None:
            return
        result, until = self._halting
        if result.done() or self._now() > until:
            self._halting = None
        else:
            self._stop_pub.publish(String(data='stop'))

    def _on_robot_description(self, msg):
        try:
            geom = ArmGeometry.from_urdf(msg.data)
        except Exception as e:      # keep the old model (if any); cycles say what is missing
            self.get_logger().error(f'/robot_description has no usable arm: {e}')
            return
        self._geom = geom
        self.get_logger().info('arm model read from /robot_description')

    def _on_joint_states(self, msg):
        # /joint_states can have several publishers (wheels, caster ...): pick the arm by name
        seen = False
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self._joints[name] = float(pos)
                seen = True
        if seen:
            self._joints_time = self._now()
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in JOINT_NAMES):      # a full arm sample (not wheels only)
            # at least 1 s, so a changed arrive_hold still finds its whole window
            keep = max(float(self.get_parameter('arrive_hold').value), 1.0)
            add_joint_sample(self._joint_hist, time.monotonic(),
                             [pos[n] for n in JOINT_NAMES], keep, JOINTS_MAX_AGE_S)

    def _on_e_stop(self, msg):
        on = bool(msg.data)
        was_on, self._e_stop = self._e_stop, on
        if on and not was_on:
            self.get_logger().warn('/e_stop ON')
            self._e_stop_seen = True    # remembered even if it is released a moment later
            # Stop the arm right here, not only when the cycle looks next. Only on the change
            # to true: a publisher repeating "true" must not flood MoveIt with "stop" events
            # (no move can start while the e-stop stays on anyway).
            self._halt()
        elif not on and was_on:
            self.get_logger().info('/e_stop released' + (
                ' (ERROR stays until /aim/park succeeds)' if self._error else ''))

    def _on_relay(self, msg):
        changed = bool(msg.data) != self._relay
        self._relay = bool(msg.data)
        if changed:
            self._publish_markers()     # cone on / off without waiting for the marker timer

    def _on_target(self, msg):
        where = (f'({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}) in '
                 f'"{msg.header.frame_id}"')
        if self._busy:
            self.get_logger().info(f'target {where} ignored: busy (a cycle is running)')
            return
        if self._error:
            self.get_logger().warn(f'target {where} ignored: ERROR ({self._error}); release '
                                   f'the e-stop if needed, then call /aim/park')
            return
        self._busy = True
        # Forget e-stops from before this cycle. Here, not in the task: an e-stop between now
        # and the task's first step must count for this cycle.
        self._e_stop_seen = False
        self.get_logger().info(f'target {where}: cycle starts')
        self.executor.create_task(self._run_cycle, msg)

    def _on_moveit_timer(self):
        """Once a second: is MoveIt up? Add the ground box to its world (again after a restart)."""
        up = (self._validity.service_is_ready() and self._scene.service_is_ready()
              and self._move_group.server_is_ready())
        if not up:
            if self._moveit_since is not None:
                self.get_logger().warn('MoveIt (move_group) is gone')
            self._moveit_since, self._ground_ok = None, False
            return
        now = self._now()
        if self._moveit_since is None:
            self._moveit_since = now
            self.get_logger().info('MoveIt is up')
        if self._ground_ok:
            return
        if self._ground_future is not None:
            if now - self._ground_asked < GROUND_RETRY_S:
                return                              # still waiting for the answer
            self._scene.remove_pending_request(self._ground_future)
        self._ground_asked = now
        self._ground_future = self._scene.call_async(self._ground_request())
        self._ground_future.add_done_callback(self._on_ground_answer)

    def _ground_request(self):
        """The ground as a flat box in MoveIt's world (top at ground_top, in base_link): MoveIt
        then never plans the nozzle (or any arm part) into the ground. It is in base_link, so
        it moves with the robot. Only the arm is checked against it (group "arm"), so the
        wheels standing on the ground do not count as a collision."""
        box = CollisionObject()
        box.id = 'ground'
        box.header.frame_id = self._base
        box.operation = CollisionObject.ADD
        box.pose.position.z = self._ground_top - GROUND_SIZE[2] / 2.0
        box.pose.orientation.w = 1.0
        box.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(GROUND_SIZE))]
        box.primitive_poses = [Pose()]
        box.primitive_poses[0].orientation.w = 1.0
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        req.scene.world.collision_objects = [box]
        return req

    def _on_ground_answer(self, future):
        self._ground_future = None
        try:
            ok = bool(future.result().success)
        except Exception:
            ok = False
        if ok:
            self._ground_ok = True
            self.get_logger().info(f'ground box added to MoveIt (top at z {self._ground_top} m)')
        else:
            self.get_logger().warn('MoveIt did not take the ground box; trying again')

    # ------------------------------------------------------------------ one cycle

    async def _run_cycle(self, msg):
        res = {
            'target': {'frame': msg.header.frame_id, 'base_link_xyz': None},
            'reachable': False, 'reason': '', 'joints_deg': None, 'tilt_deg': None,
            'standoff_m': None, 'aim_error_m': None, 'sprayed': False, 'spray_reason': '',
            'durations': {'solve': None, 'move': None, 'spray': None, 'park': None},
            'state': 'refused',
        }
        try:
            await self._cycle(msg, self._read_params(), res)
            res['state'] = 'ok'
        except _Refused as e:
            res['reason'], res['state'] = str(e), 'refused'
        except _EStop:
            res['reason'], res['state'] = 'e-stop', 'error'
            self._error = 'e-stop'
        except _ParkFailed as e:
            self.get_logger().error(f'PARK FAILED ({e}): call /aim/park when the cause is fixed')
            res['reason'], res['state'] = 'PARK FAILED', 'error'
            self._error = 'PARK FAILED'
        except Exception as e:      # a bug must never leave the node stuck "busy" or silent
            self.get_logger().error(f'cycle failed: {e!r}')
            self._halt()
            res['reason'], res['state'] = f'error: {type(e).__name__}: {e}', 'error'
            self._error = res['reason']
        finally:
            self._busy = False
            self._goal_handle = self._goal_result = None
            self._publish_result(res)

    async def _cycle(self, msg, p, res):
        """The recipe (module docstring). Raises _Refused / _EStop / _ParkFailed."""
        if self._stopped():
            raise _EStop()                      # R17: an e-stop in any phase is ERROR
        if self.get_parameter('use_sim_time').value and self._now() == 0.0:
            raise _Refused('no /clock (use_sim_time is true but no simulation clock arrives)')

        # 2. WHERE: target in base_link, then in arm_base
        target_bl = _tf_point(self._lookup(self._base, msg.header.frame_id),
                              (msg.point.x, msg.point.y, msg.point.z))
        if not np.all(np.isfinite(target_bl)):
            raise _Refused('invalid target')
        res['target']['base_link_xyz'] = [_num(v, 4) for v in target_bl]
        self._marker_target = target_bl
        self._publish_markers()
        await self._wait_ready()
        if self._stopped():
            raise _EStop()
        t_solve = time.monotonic()      # "solve" = solver + MoveIt validity checks
        geom = self._geom
        target_arm = _tf_point(self._lookup(self._arm_base, self._base), target_bl)
        scan_z = float(self._lookup(self._base, self._laser).transform.translation.z)

        # 3. SOLVE (plain maths, ~50 ms; the arm is parked meanwhile, so nothing can go wrong)
        lim = AimLimits(tilt_max_deg=p['tilt_max_deg'], standoff_min=p['standoff_min'],
                        standoff_max=p['standoff_max'], tip_clearance=p['tip_clearance'],
                        scan_z=scan_z, keep_box=self._keep_box, scan_band=p['scan_band'])
        solutions, why = solve(target_arm, geom, lim)
        if not solutions:
            res['durations']['solve'] = _num(time.monotonic() - t_solve, 3)
            raise _Refused(why)

        # 4. VALIDATE: the first candidate MoveIt calls collision-free
        chosen = goal = None
        for i, sol in enumerate(solutions):
            candidate = clamp_to_limits(sol.joints, geom.lower, geom.upper)
            if await self._is_valid(candidate):
                chosen, goal = sol, candidate
                self.get_logger().info(f'candidate {i + 1}/{len(solutions)} is collision-free')
                break
        res['durations']['solve'] = _num(time.monotonic() - t_solve, 3)
        if chosen is None:
            raise _Refused('blocked by the robot body')
        res['reachable'] = True
        res['joints_deg'] = [_num(math.degrees(v), 2) for v in goal]
        res['tilt_deg'] = _num(math.degrees(chosen.tilt), 1)
        res['standoff_m'] = _num(chosen.standoff, 3)
        if self._stopped():
            raise _EStop()

        # 5 + 6. MOVE and ARRIVED?
        t_move = time.monotonic()
        try:
            reason = await self._move_to(goal, p, 'aim')
            if reason == 'timeout':
                reason = 'did not reach the aim pose'
            elif not reason and not await self._arrive(goal, p):
                reason = 'did not reach the aim pose'
        finally:
            res['durations']['move'] = _num(time.monotonic() - t_move, 3)
        if reason:
            await self._park_after_failure(p, res)
            raise _Refused(reason)

        # 7. MEASURE
        res['aim_error_m'] = _num(self._aim_error(target_bl), 4)

        # 8. SPRAY
        if p['spray']:
            t_spray = time.monotonic()
            res['sprayed'], res['spray_reason'] = await self._spray()
            res['durations']['spray'] = _num(time.monotonic() - t_spray, 3)
        else:
            res['spray_reason'] = 'not sprayed: parameter spray is false'
        if self._stopped():         # also an e-stop already released while the spray ran
            raise _EStop()

        # 9. PARK
        t_park = time.monotonic()
        try:
            reason = await self._go_park(p)
        finally:
            res['durations']['park'] = _num(time.monotonic() - t_park, 3)
        if reason:
            raise _ParkFailed(reason)

    async def _wait_ready(self):
        """Wait (up to MOVEIT_WAIT_S) for the arm model and MoveIt, else refuse."""
        end = self._now() + MOVEIT_WAIT_S
        while True:
            if self._geom is None:
                why = 'no robot description'
            elif (self._moveit_since is None or not self._ground_ok
                  or self._now() - self._moveit_since < MOVEIT_SETTLE_S):
                why = 'MoveIt not ready'
            else:
                return
            if self._now() >= end:
                raise _Refused(why)
            await _OneRound()

    async def _is_valid(self, joints) -> bool:
        """MoveIt: is the arm at `joints` free of collisions (robot body, ground box)?"""
        if not self._validity.service_is_ready():
            raise _Refused('MoveIt not ready')
        req = GetStateValidity.Request()
        req.group_name = GROUP
        req.robot_state.joint_state.name = list(JOINT_NAMES)
        req.robot_state.joint_state.position = list(joints)
        req.robot_state.is_diff = True      # joints not named (wheels) keep MoveIt's values
        future = self._validity.call_async(req)
        if not await self._wait(future, ANSWER_S, stop_on_e_stop=False):
            self._validity.remove_pending_request(future)
            raise _Refused('MoveIt not ready')
        return bool(future.result().valid)

    def _move_goal(self, joints, p):
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = GROUP
        r.goal_constraints = [joint_goal_constraints(joints)]
        r.num_planning_attempts = 3
        r.allowed_planning_time = float(p['planning_time'])
        r.max_velocity_scaling_factor = float(p['velocity_scaling'])
        r.max_acceleration_scaling_factor = float(p['acceleration_scaling'])
        r.start_state.is_diff = True                # start from where the arm is now
        goal.planning_options.plan_only = False     # plan AND execute
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    async def _move_to(self, joints, p, label) -> str:
        """Plan + execute a joint goal. '' = MoveIt says SUCCESS; else 'MoveIt not ready',
        'planning failed' or 'timeout' (no MoveIt answer in time: goal cancelled).
        Raises _EStop (goal cancelled) on /e_stop."""
        # An e-stopped goal that MoveIt has not finished yet still gets "stop" repeated (_halt):
        # that would stop this new move too, so wait until the old one is over.
        end = self._now() + ANSWER_S
        while self._halting is not None:
            if self._stopped():
                raise _EStop()
            if self._now() >= end:
                return 'MoveIt not ready'
            await _OneRound()
        for attempt in (1, 2):
            if self._stopped():
                raise _EStop()                      # never start a move after an e-stop
            if not self._move_group.server_is_ready():
                return 'MoveIt not ready'
            send = self._move_group.send_goal_async(self._move_goal(joints, p))
            await self._wait(send, ANSWER_S, stop_on_e_stop=False)
            if not send.done():
                self._cancel_if_accepted_later(send)
                return 'MoveIt not ready'
            handle = send.result()
            if not handle.accepted:
                self.get_logger().error(f'{label}: move_group rejected the goal')
                return 'MoveIt not ready'
            result = handle.get_result_async()
            self._goal_handle, self._goal_result = handle, result
            try:
                if self._stopped():                 # came in while the goal was being accepted
                    self._halt()                    # (nothing was running for its own "stop")
                    raise _EStop()
                self.get_logger().info(f'{label}: moving')
                done = await self._wait(result, p['planning_time'] + p['arrive_timeout'])
                if self._stopped():
                    raise _EStop()                  # the /e_stop callback already stopped it
                if not done:
                    self.get_logger().error(f'{label}: no MoveIt result in time')
                    self._halt()
                    return 'timeout'
            finally:
                self._goal_handle = self._goal_result = None
            answer = result.result()
            code = answer.result.error_code.val if answer is not None else None
            if code == MoveItErrorCodes.SUCCESS:
                return ''
            if code == MoveItErrorCodes.CONTROL_FAILED and attempt == 1:
                self.get_logger().warn(
                    f'{label}: MoveIt CONTROL_FAILED (right after start move_group may not have '
                    f'found arm_controller yet); trying once more in {RETRY_PAUSE_S:.0f} s')
                await self._sleep(RETRY_PAUSE_S)
                if self._stopped():
                    raise _EStop()
                continue
            self.get_logger().error(f'{label}: MoveIt error {error_name(code)}')
            return 'planning failed'
        return 'planning failed'

    def _cancel_if_accepted_later(self, send_future):
        """move_group did not answer the goal in time: stop it if it is accepted after all."""
        def done(f):
            try:
                handle = f.result()
                if handle is not None and handle.accepted:
                    self._halt(handle, handle.get_result_async())
            except Exception as e:
                self.get_logger().warn(f'late MoveIt goal not stopped: {e!r}')
        send_future.add_done_callback(done)

    async def _arrive(self, joints, p) -> bool:
        """True once /joint_states shows every joint within arrive_tolerance_deg of `joints`
        without a break for arrive_hold s AND the arm still (settled(): every joint slower than
        arrive_still_speed over the last arrive_hold s, rule R18); False after arrive_timeout.
        Raises _EStop."""
        tol = math.radians(p['arrive_tolerance_deg'])
        hold, still = p['arrive_hold'], p['arrive_still_speed']
        end = self._now() + p['arrive_timeout']
        since = None
        while True:
            if self._stopped():
                raise _EStop()
            now = self._now()
            if self._fresh_joints() and reached(self._joints, joints, tol):
                since = now if since is None else since
                if (now - since >= hold
                        and settled(self._joint_hist, time.monotonic(), hold, still)):
                    return True
            else:
                since = None
            if now >= end:
                return False
            await _OneRound()

    def _aim_error(self, target_bl):
        """Distance (m) from the target to the nozzle's centre line, from the spray_tip TF."""
        try:
            tf = self._tf.lookup_transform(self._base, self._nozzle, Time())
        except TransformException as e:
            self.get_logger().warn(f'aim error not measured: {e}')
            return None
        tip = _tf_point(tf, (0.0, 0.0, 0.0))
        direction = _tf_point(tf, (0.0, 0.0, 1.0)) - tip
        error = ray_point_distance(target_bl, tip, direction / np.linalg.norm(direction))
        self.get_logger().info(f'arrived; aim error {error * 1000:.1f} mm')
        return error

    async def _spray(self):
        """(sprayed, the spray controller's message). Waits for its answer also on an e-stop:
        the controller switches the relay off itself and answers "aborted: e-stop"."""
        if not self._trigger.service_is_ready():
            return False, 'spray controller not running (no /spray/trigger)'
        future = self._trigger.call_async(Trigger.Request())
        if not await self._wait(future, SPRAY_ANSWER_S, stop_on_e_stop=False):
            self._trigger.remove_pending_request(future)
            return False, 'no answer from /spray/trigger'
        answer = future.result()
        self.get_logger().info(f'spray: {answer.message}')
        return bool(answer.success), answer.message

    async def _go_park(self, p) -> str:
        """Move to park and check arrival: '' = parked, else why not. Raises _EStop."""
        goal = self._park_goal()
        if self._parked(p):
            return ''                               # already there (within the tolerance)
        reason = await self._move_to(goal, p, 'park')
        if reason:
            return reason
        if not await self._arrive(goal, p):
            return 'did not reach the park pose'
        self.get_logger().info('parked')
        return ''

    async def _park_after_failure(self, p, res):
        """After a failed aim: park again if the arm has left park."""
        if self._parked(p):
            return
        t_park = time.monotonic()
        try:
            reason = await self._go_park(p)
        finally:
            res['durations']['park'] = _num(time.monotonic() - t_park, 3)
        if reason:
            raise _ParkFailed(reason)

    # ------------------------------------------------------------------ /aim/park

    async def _on_park(self, request, response):
        response.success = False
        if self._busy:
            response.message = 'busy'
            return response
        if self._e_stop:
            response.message = 'e-stop'
            return response
        self._busy = True
        self._e_stop_seen = False   # e-stops before this call are over; new ones count
        reason = ''
        try:
            await self._wait_ready()
            reason = await self._go_park(self._read_params())
        except _Refused as e:
            reason = str(e)
        except _EStop:
            reason = 'e-stop'
            self._error = 'e-stop'
        except Exception as e:
            self.get_logger().error(f'/aim/park failed: {e!r}')
            self._halt()
            reason = f'error: {type(e).__name__}: {e}'
        finally:
            self._busy = False
            self._goal_handle = self._goal_result = None
        if not reason:
            if self._error:
                self.get_logger().info(f'/aim/park: parked, ERROR ({self._error}) cleared')
            self._error = ''
            response.success, response.message = True, 'parked'
            return response
        if reason == 'e-stop':
            response.message = 'e-stop'
        else:
            response.message = f'PARK FAILED: {reason}'
            if self._geom is None or not self._parked(self._read_params()):
                self._error = 'PARK FAILED'
        self.get_logger().error(f'/aim/park: {response.message}')
        return response

    # ------------------------------------------------------------------ outputs

    def _publish_result(self, res):
        self._result_pub.publish(String(data=json.dumps(res, allow_nan=False)))
        if res['state'] == 'ok':
            err = res['aim_error_m']
            self.get_logger().info(
                f'result ok: aim error {"?" if err is None else f"{err * 1000:.1f} mm"}, '
                f'sprayed {res["sprayed"]} ({res["spray_reason"]}), durations {res["durations"]}')
        else:
            self.get_logger().warn(f'result {res["state"]}: {res["reason"]}')

    def _nozzle_ray(self):
        """(tip, unit direction) of the nozzle in base_link from TF, or None."""
        try:
            tf = self._tf.lookup_transform(self._base, self._nozzle, Time())
        except TransformException:
            return None
        tip = _tf_point(tf, (0.0, 0.0, 0.0))
        d = _tf_point(tf, (0.0, 0.0, 1.0)) - tip
        return tip, d / np.linalg.norm(d)

    def _publish_markers(self):
        """Target sphere, nozzle centre line, and the spray cone while the relay is on."""
        if self._marker_target is None:
            return
        arr = MarkerArray()

        def marker(mid, mtype, rgba):
            m = Marker()
            m.header.frame_id = self._base      # stamp 0 = "latest" for RViz
            m.ns, m.id, m.type, m.action = 'aim', mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            arr.markers.append(m)
            return m

        sphere = marker(0, Marker.SPHERE, (1.0, 0.8, 0.0, 0.9))
        sphere.pose.position = Point(x=float(self._marker_target[0]),
                                     y=float(self._marker_target[1]),
                                     z=float(self._marker_target[2]))
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.03
        ray = self._nozzle_ray()
        if ray is not None:
            tip, d = ray
            # line from the tip to where it meets the ground (or 0.3 m if it points up or out)
            length = -tip[2] / d[2] if d[2] < -0.05 else 0.3
            length = min(max(length, 0.0), 0.6)
            line = marker(1, Marker.LINE_STRIP, (0.2, 0.6, 1.0, 1.0))
            line.scale.x = 0.004
            line.points = [Point(x=float(a), y=float(b), z=float(c))
                           for a, b, c in (tip, tip + length * d)]
            if self._relay:
                cone = marker(2, Marker.TRIANGLE_LIST, (0.1, 0.9, 1.0, 0.45))
                cone.scale.x = cone.scale.y = cone.scale.z = 1.0
                cone.points = self._cone_points(tip, d, length)
        if ray is None or not self._relay:
            gone = Marker()
            gone.header.frame_id = self._base
            gone.ns, gone.id, gone.action = 'aim', 2, Marker.DELETE
            arr.markers.append(gone)
        self._marker_pub.publish(arr)

    @staticmethod
    def _cone_points(tip, d, length, segments=16):
        """Triangles of a cone from the tip along d (spray picture only)."""
        side = np.cross(d, (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0))
        u = side / np.linalg.norm(side)
        v = np.cross(d, u)
        radius = length * math.tan(CONE_HALF_ANGLE)
        rim = [tip + length * d + radius * (math.cos(a) * u + math.sin(a) * v)
               for a in np.linspace(0.0, 2.0 * math.pi, segments + 1)]
        pts = []
        for a, b in zip(rim[:-1], rim[1:]):
            pts += [tip, a, b]
        return [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]

    # ------------------------------------------------------------------ shutdown

    def stop_motion(self, executor, timeout=1.0):
        """On shutdown: stop a running MoveIt move, so the arm does not move on its own."""
        handle = self._goal_handle
        if handle is None:
            return
        self.get_logger().warn('shutting down during a move: stopping the arm')
        self._stop_pub.publish(String(data='stop'))
        future = handle.cancel_goal_async()
        try:
            executor.spin_until_future_complete(future, timeout_sec=timeout)
        except Exception as e:
            self.get_logger().warn(f'cancel not confirmed: {e!r}')


def main():
    # Our own Ctrl-C / SIGTERM handling instead of rclpy's: rclpy's handler shuts ROS down first,
    # and then a running MoveIt goal could no longer be cancelled. Here the signal only stops the
    # executor; stop_motion() cancels the goal, then ROS ends. (The 20 Hz wake-up timer makes sure
    # Python sees the signal quickly.)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    log = rclpy.logging.get_logger('weed_aim')
    node = None
    exit_code = 0
    executor = SingleThreadedExecutor()
    try:
        node = WeedAim()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        log.fatal(f'stopped by an error: {e!r}' if node is not None else
                  f'Refusing to start: {e}')
        exit_code = 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            if rclpy.ok():
                node.stop_motion(executor)
            node.destroy_node()
        executor.shutdown(timeout_sec=1.0)
        rclpy.try_shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

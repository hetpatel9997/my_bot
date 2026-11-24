#!/usr/bin/env python3
"""
spray_controller_node.py  -  the ONLY program allowed to switch the spray relay (the pump).

Services
  /spray/arm      std_srvs/SetBool   true = armed, false = disarmed (relay off at once)
  /spray/trigger  std_srvs/Trigger   spray once for `spray_seconds` (hard cap 3.0 s). Answers when
                                     the spray is over or was refused: success = it sprayed,
                                     message = what happened ("sprayed 1.00 s", "robot moving",
                                     "aborted: e-stop", ...).
Topics
  /spray/relay    std_msgs/Bool      relay state, latched (true only while the pump is on)
  /spray/status   std_msgs/String    JSON once a second: {"armed", "relay", "last"}

Before switching on, SIX checks must pass, in this order (the first failure is the answer):
  1. armed                 (not armed)
  2. arm in place          (no joint states | arm moving | nozzle not pointing at the ground |
                            nozzle aimed at the robot)
  3. robot still           (no odom | robot moving)
  4. not in a keepout zone (inside keepout zone <name> | keepout check impossible: ...)
  5. no e-stop             (e-stop)
  6. /safety_state clear   (safety_state <state> | safety_state missing or old)
While spraying, a 20 Hz monitor keeps checking (armed, e-stop, safety state, robot and arm
still). E-stop, a /safety_state other than clear and a disarm switch the relay off right in their
own callbacks, without waiting for the monitor.

The relay is OFF whenever we are not sure: first thing at start, at shutdown, after every spray
(in a `finally`, so also after any exception), and on every abort.

Backends (parameter `backend`):
  sim   no hardware; the relay state is only published on /spray/relay (the sim has no pump).
  gpio  a Raspberry Pi GPIO pin through `lgpio` (gpio_chip, gpio_pin, active_low). Written now,
        first run on the real robot in sub-project 4. lgpio is not installed on the VM, so this
        backend refuses to start there.
`start_armed` works ONLY with the sim backend: on the real robot the node always starts
disarmed and the owner arms it on purpose (CLAUDE.md: no spraying without the owner asking).

Run:  ros2 run my_bot spray_controller_node.py --ros-args --params-file config/arm_aim.yaml
"""
import json
import math
import signal
import sys
import threading
import time

import numpy as np
import yaml

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException

from aim_solver import JOINT_NAMES
from keepout_zones_node import polygon_distance   # the pool guard's code: one copy of the maths

MAX_SPRAY_SECONDS = 3.0     # hard cap: `spray_seconds` can lower it, never raise it
MONITOR_HZ = 20.0           # checks while the relay is on
NOZZLE_MIN_DOWN = -0.1      # spray direction (spray_tip +z) must have z below this in base_link

# Parameters: name -> (default, description). Only spray_seconds and zones_file may change while
# the node runs (Task 6 changes them with `ros2 param set`); all others are fixed at start and
# read-only, so a `ros2 param set` on them fails loudly instead of silently doing nothing.
PARAMS = {
    'backend': ('sim', 'sim (no hardware, relay state only on /spray/relay) or gpio (Pi pin)'),
    'start_armed': (False, 'start armed; honoured ONLY with backend sim'),
    'spray_seconds': (1.0, 'spray time per trigger (s), read at every trigger, capped at 3.0'),
    'zones_file': ('', 'keepout zones YAML (config/backyard_zones.yaml format); '' = no check'),
    'odom_topic': ('/odom', 'robot odometry (nav_msgs/Odometry)'),
    'joint_states_topic': ('/joint_states', 'arm joint positions (sensor_msgs/JointState)'),
    'e_stop_topic': ('/e_stop', 'e-stop (std_msgs/Bool, true = stop)'),
    'safety_state_topic': ('/safety_state', 'clear | slow | stop (std_msgs/String)'),
    'safety_max_age': (0.5, 'oldest /safety_state accepted (s)'),
    'still_linear': (0.01, 'robot still = speed below this (m/s) ...'),
    'still_angular': (0.02, '... and turn rate below this (rad/s) ...'),
    'still_time': (0.5, '... for this long (s)'),
    'odom_timeout': (1.0, 'no /odom for this long (s) = refuse'),
    'arm_still_speed': (0.02, 'arm still = every joint slower than this (rad/s) ...'),
    'arm_still_time': (0.3, '... over this long (s), from joint positions'),
    'joint_states_timeout': (0.5, 'no arm joint states for this long (s) = refuse'),
    'ground_reach': (0.8, 'nozzle line must hit the ground this close to arm_base (m)'),
    'robot_box': ([-0.56, 0.21, -0.19, 0.19], 'refuse if the nozzle line hits the ground inside '
                  'this base_link box [xmin, xmax, ymin, ymax] (m): the robot itself'),
    'base_frame': ('base_link', 'robot frame, z = 0 is the ground'),
    'map_frame': ('map', 'frame of the keepout zones'),
    'arm_base_frame': ('arm_base', 'arm base frame'),
    'nozzle_frame': ('spray_tip', 'nozzle frame, +z = spray direction'),
    'gpio_chip': (0, 'gpio backend: gpiochip number'),
    'gpio_pin': (-1, 'gpio backend: relay pin (BCM number); -1 = not set (refuses to start)'),
    'active_low': (False, 'gpio backend: true if the relay module switches on at a LOW pin'),
}
RUNTIME_PARAMS = ('spray_seconds', 'zones_file')


def parse_box(value) -> tuple:
    """robot_box [xmin, xmax, ymin, ymax] -> 4 floats, or ValueError if it is not a usable box.
    A wrong box must stop the node, not be ignored: with no box it would spray the robot."""
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f'not 4 numbers: {value!r}') from None
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError(f'not 4 numbers: {list(box)}')
    if not (box[0] < box[1] and box[2] < box[3]):
        raise ValueError(f'needs xmin < xmax and ymin < ymax: {list(box)}')
    return box


def inside_box(box, x, y) -> bool:
    """True if (x, y) is in box [xmin, xmax, ymin, ymax], edges included. Written as "not
    outside" so that a NaN point counts as inside (= refuse), never as a pass."""
    return not (x < box[0] or x > box[1] or y < box[2] or y > box[3])


def initial_armed(backend: str, start_armed: bool) -> bool:
    """Armed at start? Only in the sim, and only if asked: the real robot always starts disarmed."""
    return backend == 'sim' and start_armed is True


class SimRelay:
    """Sim backend: no hardware. The node publishes the state on /spray/relay for every backend."""

    def __init__(self):
        self.on = False

    def set(self, on: bool) -> None:
        self.on = bool(on)

    def close(self) -> None:
        self.on = False


class GpioRelay:
    """Relay on a Raspberry Pi GPIO pin, through lgpio. NOT run yet (sub-project 4).

    The pin is claimed as an output already at the OFF level, so it is never on for a moment
    while the node starts. On close it is written OFF before it is released.
    """

    def __init__(self, chip: int, pin: int, active_low: bool):
        if pin < 0:
            raise RuntimeError('gpio backend: gpio_pin is not set (-1); set the relay pin')
        try:
            import lgpio
        except ImportError as e:
            raise RuntimeError('gpio backend needs the Python module lgpio '
                               '(on the Pi: sudo apt install python3-lgpio)') from e
        self._lgpio = lgpio
        self._pin = pin
        self._on_level = 0 if active_low else 1
        self._off_level = 1 - self._on_level
        self._handle = lgpio.gpiochip_open(chip)
        try:
            lgpio.gpio_claim_output(self._handle, pin, self._off_level)
        except Exception:
            lgpio.gpiochip_close(self._handle)
            self._handle = None
            raise

    def set(self, on: bool) -> None:
        if self._handle is None:
            raise RuntimeError('gpio relay already closed')
        self._lgpio.gpio_write(self._handle, self._pin, self._on_level if on else self._off_level)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self.set(False)
        finally:
            try:
                self._lgpio.gpio_free(self._handle, self._pin)
            finally:
                self._lgpio.gpiochip_close(self._handle)
                self._handle = None


def load_zones(path: str, map_frame: str) -> list:
    """Read a keepout zones file (config/backyard_zones.yaml format).

    Returns [(name, polygon as an (N, 2) array, margin)]. Raises ValueError with a plain reason if
    the file cannot be used: a half-read zones file must never count as "no zones".
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError('not a YAML mapping')
    frame = cfg.get('frame_id', map_frame)
    if frame != map_frame:
        raise ValueError(f'zones are in frame {frame}, this node checks in {map_frame}')
    zones = []
    for i, z in enumerate(cfg.get('zones') or []):
        name = str(z.get('name', f'zone {i}'))
        poly = np.array(z['polygon'], dtype=float)
        margin = float(z.get('margin', 0.0))
        if (poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2
                or not np.all(np.isfinite(poly)) or not (0.0 <= margin < math.inf)):
            raise ValueError(f'zone {name}: polygon needs >= 3 finite [x, y] corners and '
                             f'margin >= 0')
        zones.append((name, poly, margin))
    return zones


def _rotate(q, v):
    """Rotate vector v by the unit quaternion q = (x, y, z, w)."""
    u, w = np.array(q[:3]), q[3]
    v = np.asarray(v, dtype=float)
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _tf_parts(tf):
    """(translation, normalised quaternion) of a TransformStamped, as numpy arrays."""
    t, r = tf.transform.translation, tf.transform.rotation
    q = np.array([r.x, r.y, r.z, r.w])
    n = np.linalg.norm(q)
    if not (n > 1e-9):           # also catches NaN
        raise ValueError('invalid rotation in TF')
    return np.array([t.x, t.y, t.z]), q / n


class SprayController(Node):
    def __init__(self, relay=None, **node_kwargs):
        # SAFETY FIRST: an injected relay is switched off before anything else can go wrong.
        if relay is not None:
            relay.set(False)
        super().__init__('spray_controller', **node_kwargs)
        # This node always runs on the steady (wall) clock: the real relay times real seconds,
        # and sim time can run slow or stop (gzserver froze before, CLAUDE.md). use_sim_time
        # would also subscribe /clock with rclpy's default QoS event handler, which brings back
        # the executor stalls described in _subscribe. So refuse it (rule R15).
        if self.get_parameter('use_sim_time').value:
            Node.destroy_node(self)
            raise RuntimeError('use_sim_time must be false: spray_controller runs on wall time '
                               '(do not pass use_sim_time to this node)')
        for name, (default, doc) in PARAMS.items():
            self.declare_parameter(name, default, ParameterDescriptor(
                description=doc, read_only=name not in RUNTIME_PARAMS))
        p = {name: self.get_parameter(name).value for name in PARAMS}
        try:
            self._robot_box = parse_box(p['robot_box'])
        except ValueError as e:
            Node.destroy_node(self)
            raise RuntimeError(f'robot_box {e}') from None
        backend = p['backend']
        if relay is None:
            relay = self._make_relay(backend, p)   # GpioRelay claims its pin at the OFF level
            relay.set(False)
        self._relay = relay

        # Fixed settings (read-only parameters).
        self._base, self._map = p['base_frame'], p['map_frame']
        self._arm_base, self._nozzle = p['arm_base_frame'], p['nozzle_frame']
        self._safety_max_age = p['safety_max_age']
        self._still_linear, self._still_angular = p['still_linear'], p['still_angular']
        self._still_time, self._odom_timeout = p['still_time'], p['odom_timeout']
        self._arm_still_speed, self._arm_still_time = p['arm_still_speed'], p['arm_still_time']
        self._js_timeout, self._ground_reach = p['joint_states_timeout'], p['ground_reach']

        # Everything below is shared between threads: the trigger (blocks during a spray), the
        # subscriptions, the monitor and the parameter callback. ONE lock guards it, and every
        # relay switch happens while holding it, so a check can never pass and then switch the
        # relay on after an e-stop / disarm callback ran in between.
        self._lock = threading.RLock()
        self._closed = False         # destroy_node ran: relay released, nothing may switch it
        self._relay_on = False
        self._relay_fault = ''       # an OFF write failed: refuse everything from then on
        self._off_failing = False    # the latest OFF write failed: the monitor keeps retrying
        self._armed = initial_armed(backend, p['start_armed'])
        self._busy = False           # a trigger is being handled (checks or spray)
        self._idle = threading.Event()  # set while no trigger is being handled
        self._idle.set()
        self._spraying = False       # relay is on for a spray
        self._spray_on_since = 0.0
        self._spray_duration = 0.0   # s, of the running spray (<= MAX_SPRAY_SECONDS)
        self._abort_reason = None
        self._abort_event = threading.Event()   # wakes the trigger thread at once on an abort
        self._last = None            # last trigger result, for /spray/status
        self._e_stop = False
        self._safety, self._safety_time = None, None
        self._odom_last, self._still_since = None, None
        self._joint_hist = []        # [(time, positions in JOINT_NAMES order)], arm_still_time+
        self._zones_file, self._zones, self._zones_error = '', [], ''

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._relay_pub = self.create_publisher(Bool, '/spray/relay', latched)
        self._publish_relay(False)
        self._status_pub = self.create_publisher(String, '/spray/status', 10)

        self._load_zones(p['zones_file'])
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._tf = Buffer()          # fed by our own /tf subscriptions below (see _subscribe)
        # child frame -> (parent frame, arrival time; None = static, never old). tf2 keeps the
        # last sample of a link that stopped updating with no time limit, so we track ages.
        self._tf_links = {}
        self._e_stop_topic = p['e_stop_topic']
        self._warned_publishers = set()

        # Subscriptions, /spray/arm and the timers share a reentrant group: they never wait for
        # each other, so an e-stop is handled even while other callbacks run. The trigger has its
        # own (reentrant) group: it blocks for the whole spray, and a second trigger must still
        # get an answer ("busy") instead of queueing behind it.
        group = ReentrantCallbackGroup()
        self._trigger_group = ReentrantCallbackGroup()
        # QoS: best effort for the sensor streams (matches any publisher); reliable + volatile for
        # /e_stop and /safety_state (matches reliable publishers, latched or not, like twist_mux
        # does; volatile = no stale history: /safety_state is fresh at 10 Hz anyway).
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self._subscribe(Odometry, p['odom_topic'], self._on_odom, qos_profile_sensor_data, group)
        self._subscribe(JointState, p['joint_states_topic'], self._on_joint_states,
                        qos_profile_sensor_data, group)
        self._subscribe(Bool, p['e_stop_topic'], self._on_e_stop, reliable, group)
        self._subscribe(String, p['safety_state_topic'], self._on_safety_state, reliable, group)
        # TF: the same two subscriptions (and QoS) as tf2_ros.TransformListener, made here so
        # they too have no QoS event handlers.
        self._subscribe(TFMessage, '/tf', self._on_tf, QoSProfile(
            depth=100, durability=DurabilityPolicy.VOLATILE), group)
        self._subscribe(TFMessage, '/tf_static', self._on_tf_static, QoSProfile(
            depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL), group)
        self.create_service(SetBool, '/spray/arm', self._on_arm, callback_group=group)
        self.create_service(Trigger, '/spray/trigger', self._on_trigger,
                            callback_group=self._trigger_group)
        steady = Clock(clock_type=ClockType.STEADY_TIME)   # never sim time (see above)
        self.create_timer(1.0 / MONITOR_HZ, self._on_monitor, callback_group=group, clock=steady)
        self.create_timer(1.0, self._on_status, callback_group=group, clock=steady)

        if backend != 'sim' and p['start_armed']:
            self.get_logger().warn(f'start_armed ignored with backend {backend}: starting '
                                   f'DISARMED, arm with /spray/arm')
        self.get_logger().info(f'spray_controller: backend {backend}, '
                               f'{"ARMED" if self._armed else "disarmed"}, relay off')

    def _subscribe(self, msg_type, topic, callback, qos, group):
        """create_subscription WITHOUT QoS event handlers.

        Why: in rclpy Humble (3.3.19, measured on this VM) the MultiThreadedExecutor busy-loops
        while a message waits for a pool thread if the subscription has a QoS event handler
        (rclpy adds an "incompatible QoS" one by default): rcl_wait keeps returning at once.
        That loop hogs Python's GIL and stalled every thread of the node for 0.1-0.35 s with
        a single 10 Hz topic, far over the 0.1 s the relay may stay on after an e-stop. Without
        the handler: no busy loop, no stalls. What we lose, rclpy's "incompatible QoS" warning,
        matters only for /e_stop (the other topics refuse with a clear reason when nothing
        arrives), so _check_e_stop_publishers warns about that one instead.
        """
        return self.create_subscription(
            msg_type, topic, callback, qos, callback_group=group,
            event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False))

    def _check_e_stop_publishers(self):
        """Warn (once each) about /e_stop publishers this reliable subscription cannot hear."""
        for info in self.get_publishers_info_by_topic(self._e_stop_topic):
            key = (info.node_namespace, info.node_name)
            if (info.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
                    and key not in self._warned_publishers):
                self._warned_publishers.add(key)
                self.get_logger().error(
                    f'{info.node_name} publishes {self._e_stop_topic} BEST EFFORT: this node '
                    f'(reliable, like twist_mux) never receives its e-stops; fix the publisher')

    # ------------------------------------------------------------------ relay (hold self._lock)

    @staticmethod
    def _make_relay(backend, p):
        if backend == 'sim':
            return SimRelay()
        if backend == 'gpio':
            return GpioRelay(p['gpio_chip'], p['gpio_pin'], p['active_low'])
        raise RuntimeError(f'unknown backend {backend!r} (use sim or gpio)')

    def _publish_relay(self, on):
        try:
            self._relay_pub.publish(Bool(data=on))
        except Exception as e:      # e.g. ROS already shut down: the relay itself is what counts
            self.get_logger().warn(f'could not publish /spray/relay {on}: {e}')

    def _set_relay_locked(self, on, always_publish=False):
        """Switch the relay and report it on /spray/relay. Caller holds self._lock.

        Order: ON is announced before the pump starts, OFF after it stopped, so /spray/relay
        never says false while the pump might be running. An OFF write never raises: it logs,
        marks a relay fault (every later trigger is refused) and leaves /spray/relay as it was;
        the monitor then re-issues OFF on every tick until a write works.
        """
        if self._closed:
            return                  # destroy_node already switched it off and released it
        if on:
            self._publish_relay(True)
            self._relay_on = True
            self._relay.set(True)   # may raise: the caller's finally switches it off again
            return
        try:
            self._relay.set(False)
        except Exception as e:
            self._relay_fault = f'{type(e).__name__}: {e}'
            if not self._off_failing:
                self.get_logger().fatal(f'RELAY OFF FAILED ({self._relay_fault}): the pump may '
                                        f'still be on, check the hardware; retrying at 20 Hz')
            else:
                self.get_logger().error(f'relay OFF still failing: {self._relay_fault}',
                                        throttle_duration_sec=5.0)
            self._off_failing = True
            return
        if self._off_failing:
            self._off_failing = False
            always_publish = True   # the relay is off again: say so
            self.get_logger().error('relay OFF write worked again; every trigger stays refused '
                                    'until the node is restarted and the hardware checked')
        if self._relay_on or always_publish:
            self._relay_on = False
            self._publish_relay(False)

    def _abort_locked(self, reason):
        """Relay off NOW, and tell a running spray why it ended. Caller holds self._lock."""
        self._set_relay_locked(False)
        if self._spraying and self._abort_reason is None:
            self._abort_reason = reason
            self._abort_event.set()

    # ------------------------------------------------------------------ checks (hold self._lock)

    def _arm_reason(self, now):
        """'' if every arm joint moved less than arm_still_speed for the last arm_still_time.

        Uses joint POSITIONS, not the velocity field (servo drivers often report no velocity).
        Needs samples covering the whole window, so "just started" or "data gap" is not still.
        """
        hist = self._joint_hist
        if not hist or now - hist[-1][0] > self._js_timeout:
            return 'no joint states'
        start = now - self._arm_still_time
        first = max((i for i, (t, _) in enumerate(hist) if t <= start), default=None)
        if first is None:
            return 'arm moving'     # not seen still for long enough yet
        pos = np.array([p for _, p in hist[first:]])
        spread = pos.max(axis=0) - pos.min(axis=0)
        # `not (a <= b)` so that NaN counts as moving
        if not np.all(spread <= self._arm_still_speed * self._arm_still_time):
            return 'arm moving'
        return ''

    def _nozzle_hit(self, now):
        """Where the nozzle line meets the ground (z = 0 in base_link): (point, '') or (None, why).

        The nozzle must point down (spray_tip +z has z < -0.1 in base_link) and the line must hit
        the ground within ground_reach of arm_base: pointing up or out across the yard -> refuse.
        The hit must also lie OUTSIDE robot_box (the robot seen from above + 3 cm, final review
        I2 / ruling R24): in the PARK pose the nozzle points straight down at x 0.107, onto the
        phone holder 5 cm under the tip, and every other check passes there. A weed is never
        under the robot (the aim patch starts at x 0.25, the box ends at x 0.21), so this only
        stops sprays at the robot itself (parked arm, or a pose pointing back over the body).
        Every moving TF link used must be fresh (joint_states_timeout): a frozen arm TF (e.g.
        robot_state_publisher died) would otherwise pass with an old pose.
        """
        try:
            p, q = _tf_parts(self._tf.lookup_transform(self._base, self._nozzle, Time()))
        except (TransformException, ValueError):
            return None, f'nozzle check impossible: no transform {self._base} <- {self._nozzle}'
        for frame in (self._nozzle, self._arm_base):
            old = self._old_link(self._base, frame, self._js_timeout, now)
            if old:
                return None, f'nozzle check impossible: transform {old} is old'
        d = _rotate(q, (0.0, 0.0, 1.0))
        if not (d[2] < NOZZLE_MIN_DOWN) or not (p[2] >= 0.0):
            return None, 'nozzle not pointing at the ground'
        hit = p + (-p[2] / d[2]) * d
        try:
            a, _ = _tf_parts(self._tf.lookup_transform(self._base, self._arm_base,
                                                       Time()))
        except (TransformException, ValueError):
            return None, f'nozzle check impossible: no transform {self._base} <- {self._arm_base}'
        if not (math.hypot(hit[0] - a[0], hit[1] - a[1]) <= self._ground_reach):
            return None, 'nozzle not pointing at the ground'
        if inside_box(self._robot_box, hit[0], hit[1]):
            return None, 'nozzle aimed at the robot'
        return hit, ''

    def _robot_reason(self, now):
        if self._odom_last is None or now - self._odom_last > self._odom_timeout:
            return 'no odom'
        if self._still_since is None or now - self._still_since < self._still_time:
            return 'robot moving'
        return ''

    def _keepout_reason(self, hit, now):
        """'' if the ground hit point (base_link) is outside every keepout zone + its margin."""
        if self._zones_error:
            return f'keepout check impossible: {self._zones_error}'
        if not self._zones_file:
            return ''               # no zones (indoor / sim): skipped, the trigger logs it
        if not self._zones:
            return ''               # a zones file without zones: nothing to keep out of
        try:
            t, q = _tf_parts(self._tf.lookup_transform(self._map, self._base, Time()))
        except (TransformException, ValueError):
            # zones set but we cannot tell where we are: never guess, refuse
            return f'keepout check impossible: no transform {self._map} <- {self._base}'
        old = self._old_link(self._map, self._base, self._odom_timeout, now)
        if old:                     # e.g. localisation stopped: the robot may be elsewhere
            return f'keepout check impossible: transform {old} is old'
        x, y, _ = _rotate(q, hit) + t
        for name, poly, margin in self._zones:
            d = polygon_distance(poly, np.array([x]), np.array([y]))[0]
            if not (d > margin):    # inside = distance <= margin (0 inside the polygon)
                return f'inside keepout zone {name}'
        return ''

    def _safety_reason(self, now):
        # CLAUDE.md hard rule: spray only while /safety_state is clear, and only a fresh one
        if self._safety_time is None or now - self._safety_time > self._safety_max_age:
            return 'safety_state missing or old'
        if self._safety != 'clear':
            return f'safety_state {self._safety}'
        return ''

    def _check_before_locked(self, now):
        """The six checks before switching on, in the spec's order: '' = all pass."""
        if self._closed:
            return 'shutting down'
        if self._relay_fault:
            return f'relay fault: {self._relay_fault}'
        # 1. Armed: the owner's explicit OK (sim: start_armed; robot: /spray/arm).
        if not self._armed:
            return 'not armed'
        # 2. Arm in place: joints still (a moving arm sprays a line, not the weed) and the nozzle
        #    points at the ground close to the robot (never up or across the yard).
        reason = self._arm_reason(now)
        if reason:
            return reason
        hit, reason = self._nozzle_hit(now)
        if reason:
            return reason
        # 3. Robot still: driving while spraying misses the weed and wets everything on the way.
        reason = self._robot_reason(now)
        if reason:
            return reason
        # 4. Keepout (CLAUDE.md hard rule): never spray into a flower bed / the pool zone.
        reason = self._keepout_reason(hit, now)
        if reason:
            return reason
        # 5. No e-stop.
        if self._e_stop:
            return 'e-stop'
        # 6. /safety_state clear and fresh (person / obstacle near, or the safety chain dead).
        return self._safety_reason(now)

    def _check_during_locked(self, now):
        """What can change while spraying (the monitor's checks): '' = keep spraying."""
        if self._e_stop:
            return 'e-stop'
        if not self._armed:
            return 'not armed'
        return self._safety_reason(now) or self._robot_reason(now) or self._arm_reason(now)

    def check(self) -> str:
        """Run the six checks now, without spraying: '' = a trigger would spray, else why not."""
        with self._lock:
            return self._check_before_locked(time.monotonic())

    # ------------------------------------------------------------------ callbacks
    # All times are time.monotonic() at arrival: independent of sim time and of message stamps.

    def _on_e_stop(self, msg):
        with self._lock:
            self._e_stop = bool(msg.data)
            if self._e_stop:
                self._abort_locked('e-stop')        # off here, not only in the 20 Hz monitor

    def _on_safety_state(self, msg):
        now = time.monotonic()
        with self._lock:
            self._safety, self._safety_time = msg.data, now
            if msg.data != 'clear':
                self._abort_locked(f'safety_state {msg.data}')

    def _on_arm(self, request, response):
        with self._lock:
            self._armed = bool(request.data)
            if not self._armed:
                self._abort_locked('not armed')
        response.success = True
        response.message = 'armed' if request.data else 'disarmed'
        self.get_logger().info(f'spray {response.message}')
        return response

    def _on_tf(self, msg):
        self._set_transforms(msg, self._tf.set_transform)

    def _on_tf_static(self, msg):
        self._set_transforms(msg, self._tf.set_transform_static, static=True)

    def _set_transforms(self, msg, setter, static=False):
        now = time.monotonic()
        for t in msg.transforms:
            try:
                setter(t, 'default_authority')
            except Exception as e:   # one bad transform from elsewhere must not stop this node
                self.get_logger().warn(f'ignored a bad transform {t.header.frame_id} -> '
                                       f'{t.child_frame_id}: {e}', throttle_duration_sec=5.0)
                continue
            # arrival time (not the stamp): independent of sim time and of the sender's clock
            self._tf_links[t.child_frame_id] = (t.header.frame_id, None if static else now)

    def _old_link(self, target, source, max_age, now):
        """The first dynamic TF link between two frames not updated for max_age, or ''.

        Walks both frames up to their common ancestor using the parents we received, and
        checks the arrival time of every dynamic link on that path (static links never age).
        """
        def line(frame):            # frame, parent, grandparent, ...
            out = [frame]
            while out[-1] in self._tf_links and len(out) < 100:
                out.append(self._tf_links[out[-1]][0])
            return out
        line_s, line_t = line(source), line(target)
        common = next((f for f in line_s if f in line_t), None)
        if common is None:
            return ''               # not connected: the lookup itself has already failed
        for child in line_s[:line_s.index(common)] + line_t[:line_t.index(common)]:
            parent, arrived = self._tf_links[child]
            if arrived is not None and not (now - arrived <= max_age):
                return f'{parent} -> {child}'
        return ''

    def _on_odom(self, msg):
        now = time.monotonic()
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        # `not (a < b)` so that NaN counts as moving
        moving = not (math.sqrt(lin.x ** 2 + lin.y ** 2 + lin.z ** 2) < self._still_linear and
                      math.sqrt(ang.x ** 2 + ang.y ** 2 + ang.z ** 2) < self._still_angular)
        with self._lock:
            gap = self._odom_last is None or now - self._odom_last > self._odom_timeout
            if moving:
                self._still_since = None
            elif gap or self._still_since is None:
                self._still_since = now     # after a gap we do not know: count from here
            self._odom_last = now

    def _on_joint_states(self, msg):
        now = time.monotonic()
        pos = dict(zip(msg.name, msg.position))
        if not all(n in pos for n in JOINT_NAMES):
            return                  # e.g. a wheels-only message: not about the arm
        sample = tuple(float(pos[n]) for n in JOINT_NAMES)
        with self._lock:
            hist = self._joint_hist
            if hist and now - hist[-1][0] > self._js_timeout:
                hist.clear()        # a gap: we do not know what the arm did in it
            hist.append((now, sample))
            # keep the newest sample at or before the window start, and everything after it
            while len(hist) >= 2 and hist[1][0] <= now - self._arm_still_time:
                hist.pop(0)

    def _on_monitor(self):
        try:
            with self._lock:
                if self._spraying:
                    now = time.monotonic()
                    reason = self._check_during_locked(now)
                    if reason:
                        self._abort_locked(reason)
                    elif now - self._spray_on_since >= self._spray_duration:
                        # Time is up. The trigger thread normally ends the spray itself; this is
                        # the backup in case it is late (the relay must never outlast the time).
                        self._spraying = False
                        self._set_relay_locked(False)
                        self._abort_event.set()
                elif self._relay_fault or self._relay_on:
                    # Not spraying, yet an OFF write failed (gpio) or the relay may be on:
                    # re-issue OFF on every tick. Triggers stay refused ('relay fault').
                    self._set_relay_locked(False)
        except Exception as e:
            with self._lock:
                self._abort_locked(f'error: {type(e).__name__}: {e}')
            self.get_logger().error(f'monitor failed, relay off: {e!r}')

    def _on_status(self):
        with self._lock:
            if self._closed:
                return
            msg = String(data=json.dumps({'armed': self._armed, 'relay': self._relay_on,
                                          'last': self._last}))
            self._status_pub.publish(msg)
        try:
            self._check_e_stop_publishers()
        except Exception as e:      # only a diagnostic (e.g. node being destroyed right now)
            self.get_logger().debug(f'e-stop publisher check skipped: {e!r}')

    # ------------------------------------------------------------------ parameters

    def _load_zones(self, path):
        """(Re)read the zones file. A file that cannot be used refuses every spray (fail-safe)."""
        try:
            zones = load_zones(path, self._map) if path else []
            error = ''
        except Exception as e:
            zones, error = [], f'zones file {path}: {e}'
        with self._lock:
            self._zones_file, self._zones, self._zones_error = path, zones, error
        if error:
            self.get_logger().error(f'{error} -> every spray is refused until this is fixed')
        elif not path:
            self.get_logger().info('keepout check OFF: no zones_file set')
        else:
            self.get_logger().info(f'keepout zones from {path}: '
                                   f'{[name for name, _, _ in zones] or "none"}')

    def _on_set_parameters(self, params):
        for prm in params:
            if prm.name == 'use_sim_time' and prm.value:
                # runs before rclpy's TimeSource callback, so /clock is never subscribed
                return SetParametersResult(successful=False,
                                           reason='spray_controller always runs on wall time')
            if prm.name == 'spray_seconds':
                v = prm.value
                if (isinstance(v, bool) or not isinstance(v, (int, float))
                        or not (0.0 < v < math.inf)):
                    return SetParametersResult(successful=False,
                                               reason='spray_seconds must be a number > 0')
        for prm in params:
            if prm.name == 'zones_file':
                self._load_zones(prm.value)
        return SetParametersResult(successful=True)

    def _spray_time(self):
        """(seconds, '') from spray_seconds, capped at MAX_SPRAY_SECONDS; or (0, why not)."""
        v = self.get_parameter('spray_seconds').value
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < v < math.inf):
            return 0.0, f'bad spray_seconds {v!r}'
        return min(float(v), MAX_SPRAY_SECONDS), ''

    # ------------------------------------------------------------------ trigger

    def _on_trigger(self, request, response):
        with self._lock:
            if self._busy:
                response.success, response.message = False, 'busy'
                self.get_logger().info('trigger refused: busy (a spray is running)')
                return response
            self._busy = True
            self._idle.clear()
            had_fault = bool(self._relay_fault)
        try:
            response.success, response.message = self._spray_once()
        except Exception as e:
            response.success, response.message = False, f'error: {type(e).__name__}: {e}'
            self.get_logger().error(f'spray failed: {e!r}')
        finally:
            with self._lock:
                self._spraying = False
                self._set_relay_locked(False)   # SAFETY: whatever happened, it ends OFF
                if self._relay_fault and not had_fault:
                    # the spray ended but the relay did not switch off: never report success
                    response.success = False
                    response.message = f'RELAY OFF FAILED: {self._relay_fault}'
                self._busy = False
                self._last = response.message
                self._idle.set()
        if response.success:
            self.get_logger().info(response.message)
        elif response.message.startswith('RELAY'):
            self.get_logger().error(f'spray ended but {response.message}')
        elif response.message.startswith(('aborted', 'error')):
            self.get_logger().warn(f'spray {response.message} (relay off)')
        else:
            self.get_logger().info(f'spray refused: {response.message}')
        return response

    def _spray_once(self):
        seconds, reason = self._spray_time()   # read at every trigger: `ros2 param set` works
        if reason:
            return False, reason
        with self._lock:
            reason = self._check_before_locked(time.monotonic())
            if reason:
                return False, reason
            self._abort_reason = None
            self._abort_event.clear()
            self._spraying = True
            self._spray_on_since = time.monotonic()
            self._spray_duration = seconds
            self._set_relay_locked(True)
        if not self._zones_file:
            self.get_logger().info('keepout check skipped: no zones_file set')
        try:
            # Sleep until the time is up, or until an abort (callback or monitor) wakes us.
            self._abort_event.wait(seconds)
        finally:
            with self._lock:
                self._spraying = False
                self._set_relay_locked(False)
                reason = self._abort_reason
        if reason:
            return False, f'aborted: {reason}'
        return True, f'sprayed {seconds:.2f} s'

    # ------------------------------------------------------------------ shutdown

    def destroy_node(self):
        """Relay off and released FIRST, then the ROS side. Safe to call twice."""
        with self._lock:
            if self._closed:
                return
            self._set_relay_locked(False, always_publish=True)   # last /spray/relay = false
            self._abort_locked('shutdown')                       # wakes a running spray
            try:
                self._relay.close()
            except Exception as e:
                self.get_logger().error(f'relay close failed: {e!r}')
            self._closed = True     # from here on nothing switches or publishes the relay
        # The relay is already off. Give a running trigger a moment to return, so its caller
        # gets "aborted: shutdown" instead of a service that vanished mid-call.
        self._idle.wait(timeout=1.0)
        super().destroy_node()


def main():
    # Our own Ctrl-C / SIGTERM handling instead of rclpy's: rclpy's handler shuts ROS down
    # first, and then the node could no longer publish the final /spray/relay false. Here the
    # signal stops the executor, destroy_node switches the relay off and publishes, then ROS ends.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    log = rclpy.logging.get_logger('spray_controller')
    node = None
    exit_code = 0
    executor = MultiThreadedExecutor(num_threads=6)
    try:
        node = SprayController()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        log.fatal(f'stopped by an error: {e!r}' if node is not None else
                  f'Refusing to start: {e}')
        exit_code = 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)    # do not interrupt the relay-off below
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            node.destroy_node()
        executor.shutdown(timeout_sec=2.0)
        rclpy.try_shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

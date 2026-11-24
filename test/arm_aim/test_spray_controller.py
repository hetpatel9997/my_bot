"""Tests for scripts/spray_controller_node.py: the only program allowed to switch the spray relay.

Run (isolated in ROS domain 42, cleans up after itself; no Gazebo needed):
    bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_spray_controller.py

How the tests work:
  - `world` is a helper node that fakes everything the controller listens to: /odom (robot still,
    20 Hz), /joint_states (arm parked, 20 Hz), /safety_state `clear` (10 Hz), /e_stop false, and
    static TF base_link -> arm_base, base_link -> spray_tip (nozzle pointing down), optionally
    map -> base_link. Each test changes one of these and checks the controller's answer.
  - `ctl` is the real SprayController with a FakeRelay instead of a pump relay. FakeRelay writes
    down every switch with its time, so the spray times are measured right here in this process
    (no topic delays in the numbers).
  - Both nodes are spun by one MultiThreadedExecutor in a background thread. Any exception a
    callback raises is collected and fails the test at teardown, so nothing can go wrong silently.
"""
import math
import threading
import time

import pytest
import yaml

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.exceptions import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos_event import SubscriptionEventCallbacks
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from rclpy.clock import ClockType
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from aim_solver import JOINT_NAMES
import spray_controller_node as scn

PARK = (0.0, 0.0, math.pi / 2, math.pi / 2)   # JOINT_NAMES order
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def wait_until(cond, timeout, period=0.02):
    """Poll cond() until it is true or `timeout` s have passed; returns its last value."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(period)
    return bool(cond())


class FakeRelay:
    """Stands in for the pump relay: records every switch as (time.monotonic(), on).

    Like a GPIO relay it can fail: `fail_on = True` makes every ON write raise, `fail_off = n`
    makes the next n OFF writes raise (a passing lgpio error). Failed writes switch nothing,
    so they are only counted (`failed`), not recorded in `calls`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = []
        self.closed = False
        self.fail_on = False
        self.fail_off = 0
        self.failed = 0

    def set(self, on):
        with self._lock:
            if (on and self.fail_on) or (not on and self.fail_off > 0):
                self.failed += 1
                if not on:
                    self.fail_off -= 1
                raise OSError('fake GPIO write error')
            self._calls.append((time.monotonic(), bool(on)))

    def close(self):
        self.closed = True

    @property
    def calls(self):
        with self._lock:
            return list(self._calls)

    def on_periods(self):
        """Every time the relay was on: [(t_on, t_off)], t_off None while still on."""
        periods, t_on = [], None
        for t, on in self.calls:
            if on and t_on is None:
                t_on = t
            elif not on and t_on is not None:
                periods.append((t_on, t))
                t_on = None
        if t_on is not None:
            periods.append((t_on, None))
        return periods

    def ever_on(self):
        return any(on for _, on in self.calls)


class World(Node):
    """Fake robot around the spray controller (see the module docstring)."""

    def __init__(self, dynamic=(), with_map=False):
        """dynamic: child frames sent on /tf at 20 Hz (while dyn_tf_on) instead of /tf_static."""
        super().__init__('spray_test_world')
        group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self.dynamic, self.with_map = tuple(dynamic), with_map
        self.dyn_tf_on = True
        self.dyn_tfs = []
        self.vx = 0.0                    # robot speed in /odom (m/s)
        self.odom_on = self.joints_on = self.safety_on = True
        self.safety = 'clear'
        self.yaw_rate, self.yaw_t0 = 0.0, None
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.safety_pub = self.create_publisher(String, '/safety_state', LATCHED)
        self.estop_pub = self.create_publisher(Bool, '/e_stop', LATCHED)
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_dynamic = TransformBroadcaster(self)
        self.trigger_cli = self.create_client(Trigger, '/spray/trigger', callback_group=group)
        self.arm_cli = self.create_client(SetBool, '/spray/arm', callback_group=group)
        self.relay_msgs = []             # every /spray/relay message, in order
        # no QoS event handler: avoids rclpy Humble's executor busy loop (see the controller's
        # _subscribe), which would stall this whole test process.
        # Depth 10, not 1: when the ON write fails, the controller publishes true and false within
        # ~1 ms; a depth-1 reader keeps only the newer one if both arrive before this process
        # takes the first (seen under CPU load: 2 of 10 runs of test_relay_on_write_fails lost
        # the `true`). Latched like the publisher, so a late join still gets the current state.
        self.create_subscription(Bool, '/spray/relay', lambda m: self.relay_msgs.append(m.data),
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL),
                                 callback_group=group, event_callbacks=(
                                     SubscriptionEventCallbacks(use_default_callbacks=False)))
        self.create_timer(0.05, self._tick_20hz, callback_group=group)
        self.create_timer(0.1, self._tick_10hz, callback_group=group)
        self.estop_pub.publish(Bool(data=False))
        self.send_tf()

    def send_tf(self, nozzle_down=True, with_map=None, tip_xyz=(0.30, 0.0, 0.20)):
        """(Re)publish all static transforms in one latched message (dynamic ones: 20 Hz).

        tip_xyz = where the nozzle tip is in base_link; pointing down it sprays the ground
        straight under it, so the default sprays (0.30, 0), inside the aim patch."""
        with_map = self.with_map if with_map is None else with_map

        def tf(parent, child, xyz, quat=(0.0, 0.0, 0.0, 1.0)):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = quat
            return t
        # spray_tip +z = the spray direction; 180 deg about x turns it to point straight down
        tip_quat = (1.0, 0.0, 0.0, 0.0) if nozzle_down else (0.0, 0.0, 0.0, 1.0)
        tfs = [tf('base_link', 'arm_base', (0.0066, 0.0, 0.2336)),
               tf('base_link', 'spray_tip', tuple(float(v) for v in tip_xyz), tip_quat)]
        if with_map:
            tfs.append(tf('map', 'base_link', (0.0, 0.0, 0.0)))
        with self._lock:
            self.dyn_tfs = [t for t in tfs if t.child_frame_id in self.dynamic]
        self.tf_static.sendTransform([t for t in tfs if t.child_frame_id not in self.dynamic])

    def start_yaw_ramp(self, rate):
        with self._lock:
            self.yaw_rate, self.yaw_t0 = rate, time.monotonic()

    def set_safety(self, state):
        """Change /safety_state and publish it at once; returns the publish time."""
        with self._lock:
            self.safety = state
            t = time.monotonic()
            self.safety_pub.publish(String(data=state))
        return t

    def set_estop(self, on):
        """Publish /e_stop; returns the publish time."""
        t = time.monotonic()
        self.estop_pub.publish(Bool(data=on))
        return t

    def _tick_20hz(self):
        now = self.get_clock().now().to_msg()
        if self.odom_on:
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = now, 'odom', 'base_link'
            o.pose.pose.orientation.w = 1.0
            o.twist.twist.linear.x = self.vx
            self.odom_pub.publish(o)
        if self.joints_on:
            with self._lock:
                yaw = PARK[0] + (self.yaw_rate * (time.monotonic() - self.yaw_t0)
                                 if self.yaw_t0 is not None else 0.0)
            js = JointState()
            js.header.stamp = now
            # the wheels are in the same message on the robot; the controller must ignore them
            js.name = list(JOINT_NAMES) + ['left_wheel_joint', 'right_wheel_joint']
            js.position = [yaw] + list(PARK[1:]) + [0.0, 0.0]
            self.js_pub.publish(js)
        with self._lock:
            dyn = self.dyn_tfs if self.dyn_tf_on else []
        if dyn:
            for t in dyn:
                t.header.stamp = now
            self.tf_dynamic.sendTransform(dyn)

    def _tick_10hz(self):
        with self._lock:
            if self.safety_on:
                self.safety_pub.publish(String(data=self.safety))


class Spinner:
    """One MultiThreadedExecutor spinning in a background thread; collects callback exceptions."""

    def __init__(self):
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.errors = []
        self.allowed = ()                # exception types a test expects (shutdown test only)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception as e:       # keep spinning; teardown reports it
                self.errors.append(e)

    def add(self, node):
        self.executor.add_node(node)

    def stop(self):
        """Stop spinning and wait for callbacks still running (safe to call twice).

        Nodes may only be destroyed after this: a callback the pool picked up earlier would
        otherwise run against an already destroyed node (rclpy Humble raises InvalidHandle).
        """
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.executor.shutdown(timeout_sec=5.0)     # waits only for callbacks already running
        # rclpy Humble's MultiThreadedExecutor never stops its thread pool: do it here (private
        # attribute), so callbacks still queued finish now, while the nodes exist, and the pool
        # threads end instead of piling up test after test.
        self.executor._executor.shutdown(wait=True)

    def collect_unread_errors(self):
        """After shutdown: exceptions of callbacks that finished after the loop stopped.

        rclpy keeps them in the executor's futures list (private, rclpy Humble) and would print
        "exception was never retrieved" at exit; reading them here reports them properly.
        """
        for future in list(getattr(self.executor, '_futures', [])):
            if future.done() and future.exception() is not None:
                self.errors.append(future.exception())


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def spinner(ros):
    s = Spinner()
    yield s
    s.stop()
    s.collect_unread_errors()
    bad = [e for e in s.errors if not isinstance(e, s.allowed)]
    assert not bad, f'a callback raised: {bad!r}'


@pytest.fixture
def world(spinner, request):
    # tests may pass World options with @pytest.mark.parametrize('world', [...], indirect=True)
    w = World(**getattr(request, 'param', {}))
    spinner.add(w)
    yield w
    spinner.stop()
    w.destroy_node()


@pytest.fixture
def relay():
    return FakeRelay()


@pytest.fixture
def ctl(spinner, world, relay):
    node = scn.SprayController(relay=relay,
                               parameter_overrides=[Parameter('start_armed', value=True)])
    spinner.add(node)
    # generous setup wait: discovery, TF, 0.5 s of still odom, 0.3 s of still joints
    assert wait_until(lambda: node.check() == '', 15.0), f'never ready: {node.check()!r}'
    assert world.trigger_cli.wait_for_service(timeout_sec=10.0)
    assert world.arm_cli.wait_for_service(timeout_sec=10.0)
    yield node
    spinner.stop()                       # stop spinning before destroying (see Spinner.stop)
    node.destroy_node()


def result(future, timeout=10.0):
    assert wait_until(future.done, timeout), 'service never answered'
    return future.result()


def trigger(world, timeout=10.0):
    r = result(world.trigger_cli.call_async(Trigger.Request()), timeout)
    return r.success, r.message


def set_param(node, name, value):
    res = node.set_parameters([Parameter(name, value=value)])
    assert res[0].successful, res[0].reason


def start_spray(world, relay):
    """Send a trigger and wait until the relay is on. Returns (future, t_on)."""
    future = world.trigger_cli.call_async(Trigger.Request())
    assert wait_until(lambda: relay.on_periods(), 10.0), 'relay never went on'
    return future, relay.on_periods()[0][0]


def sleep_until(t):
    time.sleep(max(0.0, t - time.monotonic()))


@pytest.fixture
def zones_file(tmp_path):
    """A zones file with one flower bed: a 0.2 m square around (0.30, 0) in map."""
    p = tmp_path / 'zones.yaml'
    p.write_text(yaml.safe_dump({
        'frame_id': 'map',
        'zones': [{'name': 'test_bed', 'kind': 'flower_bed', 'margin': 0.0,
                   'polygon': [[0.2, -0.1], [0.4, -0.1], [0.4, 0.1], [0.2, 0.1]]}]}))
    return str(p)


@pytest.fixture
def far_zones_file(tmp_path):
    """A zones file whose only zone is 5 m away: the keepout check is on and passes."""
    p = tmp_path / 'far_zones.yaml'
    p.write_text(yaml.safe_dump({
        'frame_id': 'map',
        'zones': [{'name': 'far_bed', 'kind': 'flower_bed', 'margin': 0.5,
                   'polygon': [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]]}]}))
    return str(p)


# ---------------------------------------------------------------------------------------------

def test_relay_off_at_start(ctl, world, relay):
    assert relay.calls[0][1] is False
    assert wait_until(lambda: world.relay_msgs, 5.0)
    assert world.relay_msgs[0] is False


def test_initial_armed():
    assert scn.initial_armed('sim', True) is True
    assert scn.initial_armed('sim', False) is False
    assert scn.initial_armed('gpio', True) is False
    assert scn.initial_armed('gpio', False) is False


def test_refuses_when_disarmed(ctl, world, relay):
    r = result(world.arm_cli.call_async(SetBool.Request(data=False)))
    assert r.success
    assert trigger(world) == (False, 'not armed')
    assert not relay.ever_on()


def test_sprays_for_spray_seconds(ctl, world, relay):
    assert trigger(world) == (True, 'sprayed 1.00 s')
    (t_on, t_off), = relay.on_periods()
    assert t_off - t_on == pytest.approx(1.0, abs=0.15)
    assert wait_until(lambda: world.relay_msgs[-2:] == [True, False], 5.0)


def test_robot_moving(ctl, world, relay):
    world.vx = 0.05
    assert wait_until(lambda: ctl.check() == 'robot moving', 5.0)
    assert trigger(world) == (False, 'robot moving')
    assert not relay.ever_on()


def test_no_odom(ctl, world, relay):
    world.odom_on = False
    time.sleep(1.5)
    assert trigger(world) == (False, 'no odom')
    assert not relay.ever_on()


def test_arm_moving(ctl, world, relay):
    world.start_yaw_ramp(0.1)
    assert wait_until(lambda: ctl.check() == 'arm moving', 5.0)
    assert trigger(world) == (False, 'arm moving')
    assert not relay.ever_on()


def test_no_joint_states(ctl, world, relay):
    world.joints_on = False
    time.sleep(1.0)
    assert trigger(world) == (False, 'no joint states')
    assert not relay.ever_on()


def test_nozzle_up(ctl, world, relay):
    world.send_tf(nozzle_down=False)
    assert wait_until(lambda: ctl.check() == 'nozzle not pointing at the ground', 5.0)
    assert trigger(world) == (False, 'nozzle not pointing at the ground')
    assert not relay.ever_on()


# --- final review I2 / ruling R24: never spray the robot itself (the phone under the parked arm)

@pytest.mark.parametrize('tip_xyz', [(0.107, 0.0, 0.347),     # the PARK pose: hit at x 0.107
                                     (-0.30, 0.10, 0.20)],    # pointing down over the body
                         ids=['parked', 'over_the_body'])
def test_nozzle_aimed_at_robot(ctl, world, relay, tip_xyz):
    world.send_tf(tip_xyz=tip_xyz)
    assert wait_until(lambda: ctl.check() == 'nozzle aimed at the robot', 5.0), ctl.check()
    assert trigger(world) == (False, 'nozzle aimed at the robot')
    assert not relay.ever_on()


@pytest.mark.parametrize('tip_xyz', [(0.30, 0.0, 0.20),       # the middle of the aim patch
                                     (0.25, 0.10, 0.20)],     # its nearest corner
                         ids=['x0.30', 'x0.25_y0.10'])
def test_aim_patch_outside_robot_box_sprays(ctl, world, relay, tip_xyz):
    world.send_tf(tip_xyz=tip_xyz)
    time.sleep(0.3)                      # the new TF is in
    assert ctl.check() == ''
    assert trigger(world) == (True, 'sprayed 1.00 s')
    assert len(relay.on_periods()) == 1


def test_robot_box_read_only(ctl):
    assert list(ctl.get_parameter('robot_box').value) == [-0.56, 0.21, -0.19, 0.19]
    assert not ctl.set_parameters([Parameter('robot_box', value=[0.0, 0.0, 0.0, 0.0])])[0] \
        .successful


@pytest.mark.parametrize('box', [[0.21, -0.56, -0.19, 0.19], [-0.56, 0.21, -0.19],
                                 [-0.56, float('nan'), -0.19, 0.19]],
                         ids=['xmin_above_xmax', 'three_numbers', 'nan'])
def test_bad_robot_box_refuses_to_start(ros, box):
    relay = FakeRelay()
    with pytest.raises(RuntimeError, match='robot_box'):
        scn.SprayController(relay=relay, parameter_overrides=[
            Parameter('start_armed', value=True), Parameter('robot_box', value=box)])
    assert [on for _, on in relay.calls] == [False]


def test_keepout(ctl, world, relay, zones_file):
    world.send_tf(with_map=True)
    set_param(ctl, 'zones_file', zones_file)
    assert wait_until(lambda: ctl.check() == 'inside keepout zone test_bed', 5.0)
    assert trigger(world) == (False, 'inside keepout zone test_bed')
    assert not relay.ever_on()


def test_keepout_no_map(ctl, world, relay, zones_file):
    set_param(ctl, 'zones_file', zones_file)
    reason = 'keepout check impossible: no transform map <- base_link'
    assert wait_until(lambda: ctl.check() == reason, 5.0)
    assert trigger(world) == (False, reason)
    assert not relay.ever_on()


@pytest.mark.parametrize('state, reason', [('slow', 'safety_state slow'),
                                           ('stop', 'safety_state stop'),
                                           (None, 'safety_state missing or old')],
                         ids=['slow', 'stop', 'missing'])
def test_safety_not_clear(ctl, world, relay, state, reason):
    if state is None:
        world.safety_on = False
        time.sleep(1.0)
    else:
        world.set_safety(state)
        assert wait_until(lambda: ctl.check() == reason, 5.0)
    assert trigger(world) == (False, reason)
    assert not relay.ever_on()


def test_estop_during_spray(ctl, world, relay):
    set_param(ctl, 'spray_seconds', 2.0)
    future, t_on = start_spray(world, relay)
    sleep_until(t_on + 0.5)
    t_pub = world.set_estop(True)
    r = result(future)
    assert (r.success, r.message) == (False, 'aborted: e-stop')
    (_, t_off), = relay.on_periods()
    assert 0.0 <= t_off - t_pub <= 0.1
    assert wait_until(lambda: world.relay_msgs[-2:] == [True, False], 5.0)


def test_safety_stop_during_spray(ctl, world, relay):
    set_param(ctl, 'spray_seconds', 2.0)
    future, t_on = start_spray(world, relay)
    sleep_until(t_on + 0.5)
    t_pub = world.set_safety('stop')
    r = result(future)
    assert (r.success, r.message) == (False, 'aborted: safety_state stop')
    (_, t_off), = relay.on_periods()
    assert 0.0 <= t_off - t_pub <= 0.1


@pytest.mark.parametrize('change, reason', [('disarm', 'not armed'),
                                            ('odom', 'robot moving'),
                                            ('yaw', 'arm moving')],
                         ids=['disarm', 'robot_moving', 'arm_moving'])
def test_abort_during_spray(ctl, world, relay, change, reason):
    set_param(ctl, 'spray_seconds', 2.0)
    future, t_on = start_spray(world, relay)
    sleep_until(t_on + 0.5)
    t_change = time.monotonic()
    if change == 'disarm':
        world.arm_cli.call_async(SetBool.Request(data=False))
    elif change == 'odom':
        world.vx = 0.05
    else:
        world.start_yaw_ramp(0.1)
    r = result(future)
    assert (r.success, r.message) == (False, f'aborted: {reason}')
    (_, t_off), = relay.on_periods()
    assert t_off - t_change <= 0.5


def test_cap_3s(ctl, world, relay):
    set_param(ctl, 'spray_seconds', 10.0)
    assert trigger(world, timeout=15.0) == (True, 'sprayed 3.00 s')
    (t_on, t_off), = relay.on_periods()
    assert 2.85 <= t_off - t_on <= 3.1


def test_busy(ctl, world, relay):
    future, t_on = start_spray(world, relay)
    sleep_until(t_on + 0.3)
    assert trigger(world) == (False, 'busy')
    r = result(future)
    assert (r.success, r.message) == (True, 'sprayed 1.00 s')
    (t_on, t_off), = relay.on_periods()
    assert t_off - t_on == pytest.approx(1.0, abs=0.15)


def test_shutdown_during_spray(spinner, ctl, world, relay):
    # Destroying a node while the executor still spins it can leave one already-picked callback
    # (e.g. the trigger's answer) with a dead handle in rclpy Humble: expected here, harmless.
    spinner.allowed = (InvalidHandle,)
    _, t_on = start_spray(world, relay)
    sleep_until(t_on + 0.3)
    ctl.destroy_node()
    assert relay.calls[-1][1] is False
    assert relay.closed
    # the subscriber saw the spray start, and the last word on /spray/relay is "off"
    assert wait_until(lambda: True in world.relay_msgs and world.relay_msgs[-1] is False, 5.0), \
        world.relay_msgs


# --- fix round 1: wall time only, TF age, failing relay writes -------------------------------

def test_refuses_use_sim_time(ros):
    relay = FakeRelay()
    with pytest.raises(RuntimeError, match='use_sim_time must be false'):
        scn.SprayController(relay=relay, parameter_overrides=[
            Parameter('start_armed', value=True), Parameter('use_sim_time', value=True)])
    assert [on for _, on in relay.calls] == [False]


def test_wall_time_only(ctl):
    # both timers run on the steady clock, and use_sim_time cannot be switched on later
    assert [t.clock.clock_type for t in ctl.timers] == [ClockType.STEADY_TIME] * 2
    assert not ctl.set_parameters([Parameter('use_sim_time', value=True)])[0].successful
    assert ctl.get_parameter('use_sim_time').value is False
    assert not ctl.get_clock().ros_time_is_active
    assert ctl.check() == ''


@pytest.mark.parametrize('world', [{'dynamic': ('spray_tip',)}], indirect=True,
                         ids=['dynamic_tip'])
def test_nozzle_tf_old(ctl, world, relay):
    t_stop = time.monotonic()
    world.dyn_tf_on = False              # e.g. robot_state_publisher died
    reason = 'nozzle check impossible: transform base_link -> spray_tip is old'
    assert wait_until(lambda: ctl.check() == reason, 5.0)
    assert time.monotonic() - t_stop >= 0.4   # only after joint_states_timeout (0.5 s)
    assert trigger(world) == (False, reason)
    assert not relay.ever_on()


@pytest.mark.parametrize('world', [{'dynamic': ('base_link',), 'with_map': True}],
                         indirect=True, ids=['dynamic_map'])
def test_keepout_tf_old(ctl, world, relay, far_zones_file):
    set_param(ctl, 'zones_file', far_zones_file)
    assert wait_until(lambda: ctl.check() == '', 5.0)   # fresh map TF: keepout passes
    t_stop = time.monotonic()
    world.dyn_tf_on = False              # e.g. localisation stopped
    reason = 'keepout check impossible: transform map -> base_link is old'
    assert wait_until(lambda: ctl.check() == reason, 5.0)
    assert time.monotonic() - t_stop >= 0.9   # only after odom_timeout (1.0 s)
    assert trigger(world) == (False, reason)
    assert not relay.ever_on()


@pytest.mark.parametrize('world', [{'dynamic': ('spray_tip', 'base_link'), 'with_map': True}],
                         indirect=True, ids=['dynamic_tip_and_map'])
def test_fresh_dynamic_tf_sprays(ctl, world, relay, far_zones_file):
    set_param(ctl, 'zones_file', far_zones_file)
    assert trigger(world) == (True, 'sprayed 1.00 s')
    (t_on, t_off), = relay.on_periods()
    assert t_off - t_on == pytest.approx(1.0, abs=0.15)


def test_relay_on_write_fails(ctl, world, relay):
    relay.fail_on = True
    ok, message = trigger(world)
    assert not ok and message.startswith('error: OSError')
    assert not relay.ever_on() and relay.calls[-1][1] is False
    assert wait_until(lambda: True in world.relay_msgs and world.relay_msgs[-1] is False, 5.0), \
        world.relay_msgs


def test_relay_off_write_fails_then_retried(ctl, world, relay):
    future, t_on = start_spray(world, relay)
    relay.fail_off = 4                   # the next 4 OFF writes fail (a passing lgpio error)
    r = result(future)
    assert not r.success and r.message.startswith('RELAY OFF FAILED')
    # the monitor re-issues OFF every tick until a write works, then reports the relay off
    assert wait_until(lambda: relay.failed == 4 and relay.calls[-1][1] is False, 3.0)
    (_, t_off), = relay.on_periods()
    assert t_off - t_on <= 1.0 + 0.5
    assert wait_until(lambda: world.relay_msgs[-1] is False, 5.0)
    ok, message = trigger(world)        # and it never sprays again until restarted
    assert not ok and message.startswith('relay fault: OSError')
    assert len(relay.on_periods()) == 1

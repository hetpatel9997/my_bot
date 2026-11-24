"""Tests for weed_aim's e-stop rules, with MoveIt, the spray controller and the robot faked.

Run (isolated in ROS domain 42, cleans up after itself; no Gazebo, no move_group needed):
    bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_weed_aim_estop.py

The rules checked here (spec 4.4, rulings R5, R17 and R23):
  - an e-stop at ANY point of a cycle gives state `error`, reason `e-stop`, and the arm is never
    moved again by the node (no retry, no park) until /aim/park succeeds;
  - a SHORT e-stop (true, then false 0.3 s later) counts too: weed_aim remembers it for the rest
    of the cycle, even when it happens while weed_aim is only waiting (the 2 s pause before a
    CONTROL_FAILED retry, the spray) and is already released when weed_aim looks next;
  - R23 (final review I1, an e-stop pressed BEFORE weed_aim started): a cycle starts only while
    the latest /safety_state is fresh (<= 0.5 s) and not `stop` (`slow` is allowed), else
    `refused` with `safety_state stop` / `safety_state missing or old` and no motion; an /e_stop
    true from a still-running latched publisher (`ros2 topic pub /e_stop ...` without --once)
    is seen even if it was published before weed_aim started; /aim/park does NOT look at
    /safety_state (a stopped arm can block the LiDAR itself and make it say `stop`).

How the tests work:
  - `aim` is the real WeedAim node, spun alone by a SingleThreadedExecutor in a background thread
    (exactly as in weed_aim_node.main()).
  - `world` fakes everything weed_aim talks to: /robot_description (the real URDF), static TF
    base_link -> arm_base / laser_frame, /safety_state (`clear` at 10 Hz unless a test changes
    it), MoveIt (/check_state_validity says "valid",
    /apply_planning_scene says "ok", /move_action answers with the error codes a test asks for
    and otherwise "moves" the arm at once by publishing the goal on /joint_states), the spray
    controller's /spray/trigger (answers when the test says so), and it records every MoveGroup
    goal and every /aim/result. A MoveGroup goal is the only way weed_aim can move the arm, so
    "no further goal" = "the arm was not moved".
"""
import json
import math
import os
import threading
import time

import pytest

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos_event import SubscriptionEventCallbacks
from geometry_msgs.msg import PointStamped, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from aim_solver import JOINT_NAMES
import weed_aim_node as wan

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SAFETY_YAML = os.path.join(PKG, 'config', 'safety.yaml')
SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
PARK = (0.0, 0.0, math.pi / 2, math.pi / 2, 0.0)   # JOINT_NAMES order
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
VOLATILE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)
# no QoS event handler: avoids rclpy Humble's MultiThreadedExecutor busy loop (see
# spray_controller_node._subscribe), which would stall the fake world
NO_EVENTS = SubscriptionEventCallbacks(use_default_callbacks=False)


def wait_until(cond, timeout, period=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(period)
    return bool(cond())


class FakeWorld(Node):
    """MoveIt, the spray controller and the robot, faked just enough for weed_aim's cycle."""

    def __init__(self, urdf_xml, geom, safety='clear', latched_estop=None):
        """safety: the /safety_state published at 10 Hz (None = nothing published).
        latched_estop: True/False = also publish /e_stop once at start on a TRANSIENT_LOCAL
        publisher that stays alive (like `ros2 topic pub /e_stop ...` without --once)."""
        super().__init__('weed_aim_test_world')
        group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.goals = []              # joint goal of every MoveGroup goal received, in order
        self.codes = []              # MoveIt error code to answer per goal (then SUCCESS)
        self.joints = list(PARK)     # what /joint_states says the arm is at
        self.results = []            # every /aim/result, as a dict
        self.trigger_calls = 0
        self.trigger_started = threading.Event()
        self.trigger_go = threading.Event()      # the fake spray controller answers once set
        self.trigger_go.set()
        self.trigger_answer = (True, 'sprayed 1.00 s')
        self.safety = safety         # /safety_state published at 10 Hz; None = silent

        self._desc = self.create_publisher(String, '/robot_description', LATCHED)
        self._desc.publish(String(data=urdf_xml))
        self._tf = StaticTransformBroadcaster(self)
        self._tf.sendTransform([self._static_tf('arm_base', geom.base_in_base_link),
                                self._static_tf('laser_frame', (-0.2495, 0.0, 0.378))])
        self.create_service(GetStateValidity, '/check_state_validity', self._on_validity,
                            callback_group=group)
        self.create_service(ApplyPlanningScene, '/apply_planning_scene', self._on_scene,
                            callback_group=group)
        self.create_service(Trigger, '/spray/trigger', self._on_trigger, callback_group=group)
        self._move = ActionServer(self, MoveGroup, '/move_action', self._on_move_goal,
                                  callback_group=group)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.estop_pub = self.create_publisher(Bool, '/e_stop', VOLATILE)
        # latched like safety_state_node's own publisher
        self.safety_pub = self.create_publisher(String, '/safety_state', LATCHED)
        self.latched_estop_pub = None
        if latched_estop is not None:
            self.latched_estop_pub = self.create_publisher(Bool, '/e_stop', LATCHED)
            self.latched_estop_pub.publish(Bool(data=bool(latched_estop)))
        self.target_pub = self.create_publisher(PointStamped, '/aim_target', VOLATILE)
        self.create_subscription(String, '/aim/result', self._on_result, VOLATILE,
                                 callback_group=group, event_callbacks=NO_EVENTS)
        self.park_cli = self.create_client(Trigger, '/aim/park', callback_group=group)
        self.create_timer(0.05, self._publish_joints, callback_group=group)
        self.create_timer(0.1, self._publish_safety, callback_group=group)

    @staticmethod
    def _static_tf(child, xyz):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = 'base_link', child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in xyz)
        t.transform.rotation.w = 1.0
        return t

    # ---- fakes
    def _on_validity(self, request, response):
        response.valid = True
        return response

    def _on_scene(self, request, response):
        response.success = True
        return response

    def _on_trigger(self, request, response):
        with self.lock:
            self.trigger_calls += 1
        self.trigger_started.set()
        self.trigger_go.wait(10.0)
        response.success, response.message = self.trigger_answer
        return response

    def _on_move_goal(self, goal_handle):
        target = [j.position for j in goal_handle.request.request.goal_constraints[0]
                  .joint_constraints]
        with self.lock:
            self.goals.append(target)
            code = self.codes.pop(0) if self.codes else MoveItErrorCodes.SUCCESS
        time.sleep(0.2)                         # a short "move"
        result = MoveGroup.Result()
        result.error_code.val = code
        if code == MoveItErrorCodes.SUCCESS:
            with self.lock:
                self.joints = list(target)      # the arm is there
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _publish_joints(self):
        with self.lock:
            pos = list(self.joints)
        self.js_pub.publish(JointState(name=list(JOINT_NAMES), position=pos))

    def _publish_safety(self):
        with self.lock:
            state = self.safety
        if state is not None:
            self.safety_pub.publish(String(data=state))

    def _on_result(self, msg):
        with self.lock:
            self.results.append(json.loads(msg.data))

    # ---- what the tests do
    def send_target(self, x, y, z=0.0):
        msg = PointStamped()
        msg.header.frame_id = 'base_link'
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
        self.target_pub.publish(msg)

    def estop(self, on):
        self.estop_pub.publish(Bool(data=on))

    def pulse_estop(self, seconds):
        """/e_stop true, then false `seconds` later (a short press of the e-stop)."""
        self.estop(True)
        time.sleep(seconds)
        self.estop(False)

    def call_park(self):
        future = self.park_cli.call_async(Trigger.Request())
        assert wait_until(future.done, 20.0), '/aim/park did not answer'
        r = future.result()
        return r.success, r.message

    def set_safety(self, state):
        """/safety_state from now on (None = stop publishing it)."""
        with self.lock:
            self.safety = state

    def goal_count(self):
        with self.lock:
            return len(self.goals)


class Spinner:
    """Spins one executor in a background thread; collects exceptions from callbacks."""

    def __init__(self, executor):
        self.executor = executor
        self.errors = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception as e:
                self.errors.append(e)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.executor.shutdown(timeout_sec=5.0)
        pool = getattr(self.executor, '_executor', None)    # MultiThreadedExecutor's pool
        if pool is not None:
            pool.shutdown(wait=True)


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def world(ros, urdf_xml, geom, request):
    # tests may pass FakeWorld options with @pytest.mark.parametrize('world', [...], indirect=True)
    w = FakeWorld(urdf_xml, geom, **getattr(request, 'param', {}))
    spinner = Spinner(MultiThreadedExecutor(num_threads=6))
    spinner.executor.add_node(w)
    yield w
    spinner.stop()
    w.destroy_node()
    assert not spinner.errors, f'a fake-world callback raised: {spinner.errors!r}'


@pytest.fixture
def aim(world, monkeypatch):
    # No 3 s start-up settle: the fake MoveIt has no race with arm_controller.
    monkeypatch.setattr(wan, 'MOVEIT_SETTLE_S', 0.0)
    node = wan.WeedAim(parameter_overrides=[Parameter('safety_yaml', value=SAFETY_YAML),
                                            Parameter('srdf_path', value=SRDF)])
    spinner = Spinner(SingleThreadedExecutor())
    spinner.executor.add_node(node)
    assert wait_until(lambda: (world.target_pub.get_subscription_count() > 0
                               and world.estop_pub.get_subscription_count() > 0
                               and world.park_cli.service_is_ready()), 10.0)
    if world.safety is not None:
        # a target sent before the first /safety_state arrived would be refused (R23)
        assert wait_until(lambda: node._safety_time is not None, 10.0), 'no /safety_state'
    yield node
    spinner.stop()
    node.destroy_node()
    assert not spinner.errors, f'a weed_aim callback raised: {spinner.errors!r}'


def first_result(world, timeout):
    assert wait_until(lambda: world.results, timeout), 'no /aim/result'
    return world.results[0]


def test_estop_pulse_during_retry_pause(world, aim):
    """A short e-stop while weed_aim waits 2 s before retrying a CONTROL_FAILED move: no retry."""
    world.codes = [MoveItErrorCodes.CONTROL_FAILED]
    world.send_target(0.34, 0.0)
    assert wait_until(lambda: world.goal_count() == 1, 20.0), 'the aim goal never came'
    time.sleep(0.5)                 # weed_aim has the CONTROL_FAILED answer: in its retry pause
    world.pulse_estop(0.3)          # over before the 2 s pause ends
    r = first_result(world, 10.0)
    time.sleep(3.0)                 # well past the end of the retry pause
    assert (r['state'], r['reason']) == ('error', 'e-stop'), r
    assert world.goal_count() == 1, 'the arm was moved again after the e-stop'
    assert world.trigger_calls == 0
    assert len(world.results) == 1


def test_estop_pulse_during_spray(world, aim):
    """A short e-stop during the spray, released before the spray controller answers:
    error, and the arm is not parked by itself."""
    world.trigger_go.clear()
    world.trigger_answer = (False, 'aborted: e-stop')
    world.send_target(0.34, 0.0)
    assert world.trigger_started.wait(20.0), 'weed_aim never asked for the spray'
    world.pulse_estop(0.3)
    time.sleep(0.2)
    world.trigger_go.set()          # the answer arrives after the e-stop was released
    r = first_result(world, 10.0)
    time.sleep(2.0)
    assert (r['state'], r['reason']) == ('error', 'e-stop'), r
    assert r['sprayed'] is False and r['spray_reason'] == 'aborted: e-stop', r
    assert world.goal_count() == 1, 'weed_aim parked the arm by itself after the e-stop'
    # targets are dropped while in ERROR; /aim/park clears it
    world.send_target(0.34, 0.0)
    time.sleep(1.0)
    assert len(world.results) == 1, 'a target was accepted while in ERROR'
    assert world.call_park() == (True, 'parked')
    assert world.goal_count() == 2 and world.goals[1] == pytest.approx(list(PARK), abs=1e-9)


def test_estop_before_motion_is_error(world, aim):
    """R17: an e-stop that is on when the cycle starts (the arm never moves) is ERROR too."""
    world.estop(True)
    time.sleep(0.3)
    world.send_target(0.34, 0.0)
    r = first_result(world, 15.0)
    assert (r['state'], r['reason']) == ('error', 'e-stop'), r
    assert world.goal_count() == 0
    assert world.call_park() == (False, 'e-stop')       # refused while the e-stop is on
    world.estop(False)
    time.sleep(0.3)
    assert world.call_park() == (True, 'parked')        # already parked: no motion needed
    assert world.goal_count() == 0
    # ERROR cleared: the next target runs the full cycle (aim move + park move)
    world.send_target(0.34, 0.0)
    assert wait_until(lambda: len(world.results) == 2, 20.0), 'second target gave no result'
    assert world.results[1]['state'] == 'ok', world.results[1]
    assert world.goal_count() == 2


# --- final review I1 / ruling R23: an e-stop pressed before weed_aim started ------------------

@pytest.mark.parametrize('state', ['stop', None], ids=['stop', 'missing'])
def test_target_refused_without_safe_safety_state(world, aim, state):
    """/safety_state `stop` (e.g. an e-stop pressed with --once before weed_aim started: only
    safety_state_node remembers it) or no /safety_state at all: refused, the arm never moves."""
    world.set_safety(state)
    time.sleep(0.8)                 # `stop` has arrived / the last `clear` is older than 0.5 s
    world.send_target(0.34, 0.0)
    r = first_result(world, 15.0)
    time.sleep(1.0)
    want = 'safety_state stop' if state else 'safety_state missing or old'
    assert (r['state'], r['reason']) == ('refused', want), r
    assert world.goal_count() == 0, 'the arm was moved'
    assert world.trigger_calls == 0
    # refused, not ERROR: once /safety_state is clear again the next target runs normally
    world.set_safety('clear')
    time.sleep(0.5)
    world.send_target(0.34, 0.0)
    assert wait_until(lambda: len(world.results) == 2, 20.0), 'second target gave no result'
    assert world.results[1]['state'] == 'ok', world.results[1]


@pytest.mark.parametrize('world', [{'safety': None}], indirect=True, ids=['never_published'])
def test_target_refused_safety_state_never_seen(world, aim):
    world.send_target(0.34, 0.0)
    r = first_result(world, 15.0)
    assert (r['state'], r['reason']) == ('refused', 'safety_state missing or old'), r
    assert world.goal_count() == 0 and world.trigger_calls == 0


@pytest.mark.parametrize('state', ['clear', 'slow'])
def test_target_runs_on_clear_and_slow(world, aim, state):
    """`slow` (an unknown obstacle near) is allowed: the spray controller refuses the spray
    itself then; weed_aim only refuses to start on `stop` or no fresh state."""
    world.set_safety(state)
    time.sleep(0.5)
    world.send_target(0.34, 0.0)
    r = first_result(world, 20.0)
    assert r['state'] == 'ok', r
    assert world.goal_count() == 2          # aim move + park move
    assert world.trigger_calls == 1


@pytest.mark.parametrize('world', [{'latched_estop': True}], indirect=True, ids=['latched'])
def test_latched_estop_from_before_start_is_seen(world, aim):
    """/e_stop true was published BEFORE weed_aim started, by a latched publisher that is still
    running: weed_aim's TRANSIENT_LOCAL /e_stop subscription gets it at start."""
    assert wait_until(lambda: aim._e_stop, 5.0), 'weed_aim never saw the latched e-stop'
    world.send_target(0.34, 0.0)
    r = first_result(world, 15.0)
    assert (r['state'], r['reason']) == ('error', 'e-stop'), r
    assert world.goal_count() == 0
    assert world.call_park() == (False, 'e-stop')
    world.latched_estop_pub.publish(Bool(data=False))       # released
    assert wait_until(lambda: not aim._e_stop, 5.0)
    assert world.call_park() == (True, 'parked')


def test_park_not_gated_on_safety_state(world, aim):
    """/aim/park works while /safety_state says `stop`: an arm stopped mid-move can be seen by
    the LiDAR, and its own `stop` must not make parking impossible."""
    with world.lock:
        world.joints = [0.3, 0.2, 1.2, 1.0, 0.0]            # somewhere off park
    world.set_safety('stop')
    time.sleep(0.5)
    assert world.call_park() == (True, 'parked')
    assert world.goal_count() == 1
    assert world.goals[0] == pytest.approx(list(PARK), abs=1e-9)


def test_safety_and_estop_subscriptions_have_no_qos_event_handlers(aim):
    """The rclpy busy-loop workaround (spray_controller_node._subscribe): no default handlers.
    For the TRANSIENT_LOCAL /e_stop one it also avoids an "incompatible QoS" warning for every
    volatile /e_stop publisher (those are heard by the volatile subscription)."""
    subs = [s for s in aim.subscriptions if s.topic_name in ('/safety_state', '/e_stop')]
    assert sorted(s.topic_name for s in subs) == ['/e_stop', '/e_stop', '/safety_state']
    durabilities = sorted(s.qos_profile.durability.name for s in subs if s.topic_name == '/e_stop')
    assert durabilities == ['TRANSIENT_LOCAL', 'VOLATILE']
    for s in subs:
        if s.topic_name == '/safety_state' or \
                s.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
            assert s.event_handlers == [], s.topic_name

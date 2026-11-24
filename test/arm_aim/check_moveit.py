#!/usr/bin/env python3
"""
check_moveit.py  -  SIMULATION ONLY: MoveIt 2 knows the arm, finds the parked pose collision-free,
and really moves the Gazebo arm.

Why this test exists: the aim node (next task) will ask MoveIt two things: "is this arm pose
collision-free?" (/check_state_validity) and "plan and move there" (/move_action). Both only work
if the MoveIt config (config/moveit/, launch/arm_aim.launch.py) matches the robot. Checked here, in
this order:
  1. PARK VALID   - the parked pose (SRDF group_state "park") is valid with NO contacts. If it is
                    not, two robot parts overlap in that pose. Neighbouring parts (bolted together
                    or one joint apart) are already excluded in the SRDF, so a contact reported
                    here is between parts that should NOT touch: a robot model problem. Do not
                    hide it by adding the pair to the SRDF; fix the model.
  2. G VALID      - test pose G = (0.3, 0.3, 1.2, 1.2, 0.0) rad (yaw, shoulder, elbow, wrist, roll):
                    arm turned a little left and leaning forward, nozzle down; also valid, no
                    contacts.
  3. MOVE TO G    - MoveGroup goal (joint goal, 0.001 rad tolerance, plan + execute): MoveIt answers
                    SUCCESS (error_code 1) and /joint_states shows every joint within 1 deg of G,
                    both within 15 s (sim time) of sending the goal.
  4. BACK TO PARK - the same three checks for park (valid, SUCCESS, within 1 deg in 15 s).

Run (sim from test/arm_aim/start_sim.sh and arm_aim.launch.py running, same terminal settings):
  source test/arm_aim/sim_env.sh && python3 test/arm_aim/check_moveit.py
Last line: "MOVEIT PASS" (exit 0) or "MOVEIT FAIL: <which check>" (exit 1).
Never run it on the real robot: it moves the arm. It refuses to run without a simulation clock.

Cleanup afterwards: ~/dev_ws/src/my_bot/scripts/sim_cleanup.sh --domain 42 (must print "Clean"),
then `ros2 daemon stop` with sim_env.sh sourced. sim_cleanup.sh does not look for move_group
(2026-11-23), and stopping its `ros2 launch` did not stop it in the first run, so check with
`pgrep -af moveit_ros_move_group` and stop only the test's own one (domain 42):
  for p in $(pgrep -f moveit_ros_move_group/move_group); do
    tr '\\0' '\\n' < /proc/$p/environ | grep -qx ROS_DOMAIN_ID=42 && kill -INT $p; done
"""
import math
import os
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
from aim_solver import JOINT_NAMES  # noqa: E402
from arm_config import load_park  # noqa: E402

SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
GOAL_G = (0.3, 0.3, 1.2, 1.2, 0.0)      # rad, JOINT_NAMES order
GOAL_TOL = 0.001                        # rad, MoveIt joint goal tolerance
REACH_TOL = math.radians(1.0)           # rad, /joint_states must get this close
REACH_S = 15.0                          # s sim time from sending the goal
VEL_SCALE = ACC_SCALE = 0.5             # the same scaling the aim node will use
PLAN_TIME = 5.0                         # s, OMPL planning time limit
WAIT_MOVEIT_S = 120.0                   # s wall, for move_group to come up after the launch


class Fail(Exception):
    pass


def error_name(val):
    """MoveItErrorCodes number -> its name (e.g. -1 -> PLANNING_FAILED), for readable messages."""
    for name in dir(MoveItErrorCodes):
        if name.isupper() and getattr(MoveItErrorCodes, name) == val:
            return name
    return str(val)


class MoveItCheck(Node):
    def __init__(self):
        super().__init__('check_moveit',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.joints = {}            # latest arm joint positions from /joint_states
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 50)
        self.validity = self.create_client(GetStateValidity, '/check_state_validity')
        self.move_action = ActionClient(self, MoveGroup, '/move_action')

    def _joint_cb(self, msg):
        # /joint_states has several publishers (joint_broad: wheels + arm; the stub: caster),
        # so pick the arm joints by name.
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self.joints[name] = pos

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_until(self, done, sim_s=None, wall_s=60.0):
        """Spin until done() is true. Gives up after sim_s of sim time or wall_s of wall time
        (the wall limit only stops the test from hanging if the sim freezes)."""
        t_sim, t_wall = self.sim_now(), time.monotonic()
        while rclpy.ok():
            if done():
                return True
            if sim_s is not None and self.sim_now() - t_sim > sim_s:
                return False
            if time.monotonic() - t_wall > wall_s:
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def wait_ready(self):
        # With use_sim_time the node clock stays at 0 until the first /clock message arrives.
        if not self.spin_until(lambda: self.sim_now() > 0.0, wall_s=30.0):
            raise Fail('no /clock: is the simulation running? (refusing to run without it)')
        if not self.spin_until(lambda: all(j in self.joints for j in JOINT_NAMES), wall_s=30.0):
            raise Fail(f'/joint_states has no arm joints after 30 s (got {sorted(self.joints)})')
        t0 = time.monotonic()
        if not self.validity.wait_for_service(timeout_sec=WAIT_MOVEIT_S):
            raise Fail(f'no service /check_state_validity after {WAIT_MOVEIT_S:.0f} s '
                       '(is arm_aim.launch.py / move_group running? see $ARM_TEST_LOG/arm_aim.log)')
        rest = max(1.0, WAIT_MOVEIT_S - (time.monotonic() - t0))
        if not self.move_action.wait_for_server(timeout_sec=rest):
            raise Fail('no action server /move_action (move_group not ready?)')
        print(f'  move_group ready after {time.monotonic() - t0:.1f} s')

    # ---- validity -----------------------------------------------------------------------
    def check_valid(self, target, label):
        req = GetStateValidity.Request()
        req.group_name = 'arm'
        req.robot_state.joint_state.name = list(JOINT_NAMES)
        req.robot_state.joint_state.position = [float(v) for v in target]
        # is_diff: joints not named here (the wheels) keep MoveIt's current values
        req.robot_state.is_diff = True
        future = self.validity.call_async(req)
        if not self.spin_until(future.done, wall_s=20.0):
            raise Fail(f'{label}: /check_state_validity did not answer')
        res = future.result()
        pairs = sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in res.contacts})
        if not res.valid or pairs:
            raise Fail(f'{label}: MoveIt says valid={res.valid}, contacts between {pairs}. '
                       'Neighbouring links are already disabled in the SRDF, so these parts '
                       'overlap although they should not: a robot model problem (do not hide it '
                       'in the SRDF)')
        print(f'  {label}: valid, no contacts')

    # ---- move ---------------------------------------------------------------------------
    def move_to(self, target, label):
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = 'arm'
        r.num_planning_attempts = 1
        r.allowed_planning_time = PLAN_TIME
        r.max_velocity_scaling_factor = VEL_SCALE
        r.max_acceleration_scaling_factor = ACC_SCALE
        r.start_state.is_diff = True             # start from where the arm is now
        c = Constraints()
        for name, pos in zip(JOINT_NAMES, target):
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(pos),
                tolerance_above=GOAL_TOL, tolerance_below=GOAL_TOL, weight=1.0))
        r.goal_constraints = [c]
        goal.planning_options.plan_only = False  # plan AND execute
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        t_send = self.sim_now()
        send = self.move_action.send_goal_async(goal)
        if not self.spin_until(send.done, wall_s=20.0):
            raise Fail(f'{label}: MoveGroup goal not answered')
        handle = send.result()
        if not handle.accepted:
            raise Fail(f'{label}: MoveGroup goal rejected')
        result = handle.get_result_async()

        def off():
            return {j: self.joints[j] - t for j, t in zip(JOINT_NAMES, target)}

        def arrived():
            return result.done() and all(abs(d) <= REACH_TOL for d in off().values())

        # 10x wall margin: only there so a frozen sim cannot hang the test.
        ok = self.spin_until(arrived, sim_s=REACH_S, wall_s=10 * REACH_S)
        t_reach = self.sim_now() - t_send
        if not result.done():
            raise Fail(f'{label}: no MoveGroup result within {REACH_S:.0f} s sim')
        res = result.result().result
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise Fail(f'{label}: MoveGroup error_code {res.error_code.val} '
                       f'({error_name(res.error_code.val)})')
        if not ok:
            bad = {j: round(math.degrees(d), 2) for j, d in off().items() if abs(d) > REACH_TOL}
            raise Fail(f'{label}: MoveIt said SUCCESS but /joint_states is off by more than 1 deg '
                       f'after {REACH_S:.0f} s: {bad} (deg)')
        points = res.planned_trajectory.joint_trajectory.points
        duration = (points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
                    if points else 0.0)
        worst = max(abs(d) for d in off().values())
        # (res.planning_time is not filled in for plan + execute, so it is not printed)
        print(f'  {label}: SUCCESS, trajectory {duration:.2f} s ({len(points)} points), '
              f'all joints within 1 deg after {t_reach:.2f} s sim, '
              f'max error {math.degrees(worst):.3f} deg')


def main():
    rclpy.init()
    node = MoveItCheck()
    ok = False
    try:
        park = load_park(SRDF)
        node.wait_ready()
        node.check_valid(park, 'park valid')
        node.check_valid(GOAL_G, 'G valid')
        node.move_to(GOAL_G, 'move to G')
        node.check_valid(park, 'park valid (again)')
        node.move_to(park, 'move back to park')
        ok = True
        print('MOVEIT PASS')
    except Fail as e:
        print(f'MOVEIT FAIL: {e}')
    except (KeyboardInterrupt, ExternalShutdownException):
        print('MOVEIT FAIL: interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

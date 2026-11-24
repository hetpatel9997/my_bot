#!/usr/bin/env python3
"""
check_arm_hold.py  -  SIMULATION ONLY: the arm holds its parked pose in Gazebo and moves when told.

Why this test exists: in Gazebo the four arm joints are position-controlled by ros2_control
(description/ros2_control.xacro sim block + arm_controller, config/arm_controllers_sim.yaml).
Before that they were welded in the parked pose. The later arm tasks rely on two things, checked
here in this order:
  1. HOLD  - for 30 s (sim time) the spray tip stays where the parked arm puts it: base_link
             x 0.107, z 0.347 (6 cm nozzle), every sample within 5 mm in x and z. If the arm sagged
             (gravity) or jittered, the tip would move.
  2. MOVE  - a FollowJointTrajectory goal on /arm_controller/follow_joint_trajectory turns the
             yaw joint to 0.5 rad (other joints parked) in 3 s; /joint_states must show the yaw
             within 0.02 rad of 0.5 within 6 s, and the controller must report success. Then the
             same back to the parked pose (yaw 0).

Run (sim started by test/arm_aim/start_sim.sh, same terminal settings):
  source test/arm_aim/sim_env.sh && python3 test/arm_aim/check_arm_hold.py
Last line: "ARM HOLD + MOVE PASS" (exit 0) or "ARM HOLD + MOVE FAIL: <which check>" (exit 1).
Never run it on the real robot: it moves the arm. It refuses to run without a simulation clock.
"""
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

# Joint order used everywhere in the arm code.
JOINTS = ['arm_base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch']
# Parked pose (weedbot_properties.xacro arm_park_*): elbow and wrist +90 deg = nozzle pointing down.
PARK = [0.0, 0.0, math.pi / 2, math.pi / 2]
TIP_PARKED_X, TIP_PARKED_Z = 0.107, 0.347   # spray tip in base_link, arm parked (wrist axis 0.4073 - 6 cm nozzle)
TIP_TOL = 0.005        # m, hold check
HOLD_S = 30.0          # s sim time, hold check
YAW_GOAL = 0.5         # rad, move check
MOVE_S = 3.0           # s, trajectory duration
REACH_S = 6.0          # s sim time from sending the goal until the yaw must be there
JOINT_TOL = 0.02       # rad


class Fail(Exception):
    pass


class ArmCheck(Node):
    def __init__(self):
        super().__init__('check_arm_hold',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joints = {}            # latest arm joint positions from /joint_states
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 50)
        self.action = ActionClient(self, FollowJointTrajectory,
                                   '/arm_controller/follow_joint_trajectory')

    def _joint_cb(self, msg):
        # /joint_states has several publishers (joint_broad: wheels + arm; the stub: caster),
        # so pick the arm joints by name.
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
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

    # ---- 1. hold ------------------------------------------------------------------------
    def check_hold(self):
        # With use_sim_time the node clock stays at 0 until the first /clock message arrives.
        if not self.spin_until(lambda: self.sim_now() > 0.0, wall_s=30.0):
            raise Fail('no /clock: is the simulation running? (refusing to run without it)')
        tip_ready = lambda: self.tf_buffer.can_transform('base_link', 'spray_tip', Time())
        if not self.spin_until(tip_ready, wall_s=30.0):
            raise Fail('no TF base_link <- spray_tip after 30 s')
        worst_dx = worst_dz = 0.0
        stamps = set()
        n = 0
        t0 = self.sim_now()
        t_wall = time.monotonic()
        next_sample = t0
        while self.sim_now() - t0 < HOLD_S:
            if time.monotonic() - t_wall > 6 * HOLD_S:
                raise Fail(f'hold check: sim time too slow ({self.sim_now() - t0:.1f} s sim in '
                           f'{6 * HOLD_S:.0f} s wall)')
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.sim_now() < next_sample:
                continue
            next_sample += 0.1        # 10 samples per sim second
            t = self.tf_buffer.lookup_transform('base_link', 'spray_tip', Time())
            p = t.transform.translation
            dx, dz = p.x - TIP_PARKED_X, p.z - TIP_PARKED_Z
            n += 1
            stamps.add((t.header.stamp.sec, t.header.stamp.nanosec))
            worst_dx, worst_dz = max(worst_dx, abs(dx)), max(worst_dz, abs(dz))
            if abs(dx) > TIP_TOL or abs(dz) > TIP_TOL:
                raise Fail(f'hold check: spray tip at x {p.x:.4f} z {p.z:.4f} after '
                           f'{self.sim_now() - t0:.1f} s, more than {TIP_TOL * 1000:.0f} mm from '
                           f'parked x {TIP_PARKED_X} z {TIP_PARKED_Z}')
        # The TF must really update (fresh joint states), not just repeat an old value.
        if len(stamps) < HOLD_S:
            raise Fail(f'hold check: only {len(stamps)} different TF stamps in {HOLD_S:.0f} s '
                       f'(joint states not updating?)')
        print(f'  hold: {n} samples over {HOLD_S:.0f} s sim, {len(stamps)} TF updates, '
              f'worst tip offset x {worst_dx * 1000:.2f} mm, z {worst_dz * 1000:.2f} mm')

    # ---- 2. move ------------------------------------------------------------------------
    def move_to(self, target, label):
        if not self.action.wait_for_server(timeout_sec=10.0):
            raise Fail('no action server /arm_controller/follow_joint_trajectory '
                       '(arm_controller not running?)')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        point.velocities = [0.0] * len(JOINTS)      # stop at the end (smooth start and stop)
        point.time_from_start = DurationMsg(sec=int(MOVE_S), nanosec=int((MOVE_S % 1) * 1e9))
        goal.trajectory.points = [point]

        t_send = self.sim_now()
        send = self.action.send_goal_async(goal)
        if not self.spin_until(send.done, wall_s=20.0):
            raise Fail(f'{label}: goal not answered by arm_controller')
        handle = send.result()
        if not handle.accepted:
            raise Fail(f'{label}: goal rejected by arm_controller')
        result = handle.get_result_async()

        yaw_ok = lambda: ('arm_base_yaw' in self.joints
                          and abs(self.joints['arm_base_yaw'] - target[0]) <= JOINT_TOL)
        if not self.spin_until(yaw_ok, sim_s=REACH_S, wall_s=10 * REACH_S):
            raise Fail(f'{label}: /joint_states yaw {self.joints.get("arm_base_yaw", float("nan")):.3f}'
                       f' rad, not within {JOINT_TOL} of {target[0]} within {REACH_S:.0f} s')
        t_reach = self.sim_now() - t_send

        # The controller must also say "done": later, MoveIt treats anything else as a failure.
        if not self.spin_until(result.done, sim_s=MOVE_S + 5.0, wall_s=60.0):
            raise Fail(f'{label}: arm_controller sent no result')
        res = result.result().result
        if res.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise Fail(f'{label}: arm_controller result error_code {res.error_code} '
                       f'"{res.error_string}"')
        off = {j: self.joints[j] - t for j, t in zip(JOINTS, target)}
        bad = {j: round(d, 3) for j, d in off.items() if abs(d) > JOINT_TOL}
        if bad:
            raise Fail(f'{label}: joints off target by more than {JOINT_TOL} rad: {bad}')
        print(f'  {label}: yaw within {JOINT_TOL} rad of {target[0]} after {t_reach:.2f} s sim, '
              f'result SUCCESSFUL, max joint error {max(abs(d) for d in off.values()):.4f} rad')


def main():
    rclpy.init()
    node = ArmCheck()
    ok = False
    try:
        node.check_hold()
        node.move_to([YAW_GOAL] + PARK[1:], 'move to yaw 0.5')
        node.move_to(PARK, 'move back to park')
        ok = True
        print('ARM HOLD + MOVE PASS')
    except Fail as e:
        print(f'ARM HOLD + MOVE FAIL: {e}')
    except (KeyboardInterrupt, ExternalShutdownException):
        print('ARM HOLD + MOVE FAIL: interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

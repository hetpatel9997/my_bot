"""Tests for the small helpers in scripts/weed_aim_node.py (no ROS graph, no Gazebo needed).

Run (isolated in ROS domain 42, cleans up after itself):
    bash test/arm_aim/run_unit_tests.sh test/arm_aim/test_weed_aim_helpers.py

What is checked:
  - reached(): "has the arm arrived?" is true only if EVERY arm joint is within the tolerance,
    and false when a joint is missing from /joint_states (never assume a joint we cannot see).
  - joint_goal_constraints(): the MoveIt joint goal names the five arm joints in the usual order,
    with the goal positions and a small tolerance.
  - clamp_to_limits(): every goal sent to MoveIt is pulled inside the URDF joint limits (the SRDF
    park value is 5e-12 rad above the elbow / wrist limit; MoveIt would refuse it as out of bounds).
"""
import math

from moveit_msgs.msg import Constraints

from aim_solver import JOINT_NAMES
from weed_aim_node import clamp_to_limits, joint_goal_constraints, reached

PARK = (0.0, 0.0, math.pi / 2, math.pi / 2, 0.0)   # JOINT_NAMES order
ONE_DEG = math.radians(1.0)


def test_reached_within_tolerance():
    close = {n: g + math.radians(0.9) for n, g in zip(JOINT_NAMES, PARK)}
    assert reached(close, PARK, ONE_DEG) is True
    # 1.1 deg off on a single joint is enough to say "not there"
    for name in JOINT_NAMES:
        off = dict(close)
        off[name] = dict(zip(JOINT_NAMES, PARK))[name] + math.radians(1.1)
        assert reached(off, PARK, ONE_DEG) is False, name
    # also in the other direction
    below = {n: g - math.radians(0.9) for n, g in zip(JOINT_NAMES, PARK)}
    assert reached(below, PARK, ONE_DEG) is True


def test_reached_missing_joint():
    current = {n: g for n, g in zip(JOINT_NAMES, PARK) if n != 'wrist_roll'}
    assert reached(current, PARK, ONE_DEG) is False
    # a NaN position (broken driver) is not "arrived" either
    nan = dict(zip(JOINT_NAMES, PARK))
    nan['elbow_pitch'] = float('nan')
    assert reached(nan, PARK, ONE_DEG) is False


def test_joint_goal_constraints():
    goal = (0.1, -0.2, 0.3, -0.4, 0.5)
    c = joint_goal_constraints(goal)
    assert isinstance(c, Constraints)
    assert [j.joint_name for j in c.joint_constraints] == list(JOINT_NAMES)
    assert len(c.joint_constraints) == 5
    for j, g in zip(c.joint_constraints, goal):
        assert j.position == g
        assert j.tolerance_above == 0.001
        assert j.tolerance_below == 0.001
        assert j.weight == 1.0
    # nothing else is constrained (no pose / orientation goals)
    assert not c.position_constraints and not c.orientation_constraints
    # a different tolerance is used when asked
    assert all(j.tolerance_above == 0.01 and j.tolerance_below == 0.01
               for j in joint_goal_constraints(goal, tol=0.01).joint_constraints)


def test_clamp_to_limits():
    lower = (-math.pi / 2,) * 5
    upper = (math.pi / 2,) * 5
    srdf_park = (0.0, 0.0, 1.5707963268, 1.5707963268, 0.0)   # 5e-12 above the limit
    out = clamp_to_limits(srdf_park, lower, upper)
    assert out == (0.0, 0.0, math.pi / 2, math.pi / 2, 0.0)
    assert all(lo <= v <= hi for v, lo, hi in zip(out, lower, upper))
    assert clamp_to_limits((-1.6, 0.2, 2.0, -0.3, 0.0), lower, upper) == (
        -math.pi / 2, 0.2, math.pi / 2, -0.3, 0.0)

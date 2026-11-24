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
  - add_joint_sample() + settled() (ruling R18): "arrived" also needs the arm to be STILL, judged
    like the spray controller does it (position spread of each joint over the last window, from
    /joint_states samples; NaN = moving; a gap in the samples restarts the window). Without it
    weed_aim asked for the spray while the arm still crept at 0.1-0.2 rad/s and the spray
    controller refused "arm moving" (Task 6).
"""
import math

from moveit_msgs.msg import Constraints

from aim_solver import JOINT_NAMES
from weed_aim_node import (add_joint_sample, clamp_to_limits, joint_goal_constraints, reached,
                           settled)

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


# ---- settled (R18): samples every 0.1 s like /joint_states in the sim (10 Hz)
WINDOW = 0.3            # s, arrive_hold
STILL = 0.01            # rad/s, arrive_still_speed (half of the spray controller's 0.02)
MAX_GAP = 0.5           # s, no joint states for longer = the window starts again
EPS = 1e-6              # ask just after the last sample (float rounding of 0.1 * k)


def ramp(hist, t0, t1, speed, joint=1):
    """Add samples from t0 to t1 (0.1 s apart): park, with one joint moving at `speed` rad/s."""
    n = int(round((t1 - t0) / 0.1))
    for i in range(n + 1):
        t = t0 + 0.1 * i
        q = list(PARK)
        q[joint] += speed * t
        add_joint_sample(hist, t, q, keep=WINDOW, max_gap=MAX_GAP)


def test_settled_still_arm():
    hist = []
    ramp(hist, 0.0, 0.5, 0.0)
    assert settled(hist, 0.5 + EPS, WINDOW, STILL) is True


def test_settled_slow_creep_is_not_still():
    # 0.015 rad/s: inside 1 deg for a long time, but faster than the 0.01 rad/s limit
    hist = []
    ramp(hist, 0.0, 1.0, 0.015)
    assert settled(hist, 1.0 + EPS, WINDOW, STILL) is False


def test_settled_real_deceleration():
    """The end of a move as Task 6 measured it (rad/s per 0.1 s step): the old rule ("within
    1 deg for 0.3 s") was already met while the spray controller still saw "arm moving"."""
    speeds = (0.50, 0.49, 0.50, 0.39, 0.31, 0.21, 0.12, 0.02, 0.0, 0.0, 0.0)
    final = sum(speeds) * 0.1
    hist, q, since = [], 0.0, None
    for k in range(len(speeds) + 1):
        t = 0.1 * k
        add_joint_sample(hist, t, (0.0, q, PARK[2], PARK[3], 0.0), keep=WINDOW, max_gap=MAX_GAP)
        near = abs(q - final) <= ONE_DEG
        since = (t if since is None else since) if near else None
        old_rule = since is not None and t - since >= WINDOW - EPS
        if k == 9:
            assert old_rule, 'the old rule should already say "arrived" here'
            # spread 0.012 + 0.002 rad over the window: more than even the spray controller's
            # 0.02 rad/s x 0.3 s = 0.006 rad -> it would refuse "arm moving"
            assert settled(hist, t + EPS, WINDOW, STILL) is False
            assert settled(hist, t + EPS, WINDOW, 0.02) is False
        if k == 8:
            assert settled(hist, t + EPS, WINDOW, STILL) is False
        if k == 10:
            assert settled(hist, t + EPS, WINDOW, STILL) is True    # 0.002 rad <= 0.003
        if k < len(speeds):
            q += speeds[k] * 0.1


def test_settled_slower_than_limit_is_still():
    hist = []
    ramp(hist, 0.0, 1.0, 0.005)     # 0.0015 rad over the window, below 0.003
    assert settled(hist, 1.0 + EPS, WINDOW, STILL) is True


def test_settled_needs_the_whole_window():
    hist = []
    ramp(hist, 0.0, 0.2, 0.0)       # only 0.2 s of samples: not seen still long enough
    assert settled(hist, 0.2 + EPS, WINDOW, STILL) is False
    assert settled([], 1.0, WINDOW, STILL) is False


def test_settled_nan_is_moving():
    hist = []
    ramp(hist, 0.0, 0.5, 0.0)
    q = list(PARK)
    q[3] = float('nan')
    add_joint_sample(hist, 0.6, q, keep=WINDOW, max_gap=MAX_GAP)
    assert settled(hist, 0.6 + EPS, WINDOW, STILL) is False


def test_gap_restarts_the_window():
    hist = []
    ramp(hist, 0.0, 0.5, 0.0)
    add_joint_sample(hist, 1.2, PARK, keep=WINDOW, max_gap=MAX_GAP)   # 0.7 s without samples
    assert len(hist) == 1
    # we do not know what the arm did in the gap: not still until a whole new window is seen
    assert settled(hist, 1.2 + EPS, WINDOW, STILL) is False
    for t in (1.3, 1.4, 1.5):
        add_joint_sample(hist, t, PARK, keep=WINDOW, max_gap=MAX_GAP)
    assert settled(hist, 1.5 + EPS, WINDOW, STILL) is True


def test_history_stays_short():
    hist = []
    ramp(hist, 0.0, 20.0, 0.0)
    # only the newest sample at or before the window start, and the ones after it
    assert len(hist) <= 5
    assert settled(hist, 20.0 + EPS, WINDOW, STILL) is True

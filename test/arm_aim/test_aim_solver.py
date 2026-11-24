import math

import numpy as np
import pytest

from aim_solver import AimLimits, arm_points, nozzle_ray, ray_point_distance, solve
from conftest import bl_to_arm


def test_from_urdf_matches_cad(geom):
    assert geom.l_shoulder == pytest.approx(0.0707, abs=1e-6)
    assert geom.l_upper == pytest.approx(0.100, abs=1e-6)
    assert geom.l_fore == pytest.approx(0.100, abs=1e-6)
    assert geom.l_tip == pytest.approx(0.131, abs=1e-6)
    for v in geom.lower:
        assert v == pytest.approx(-math.pi / 2, abs=1e-6)
    for v in geom.upper:
        assert v == pytest.approx(math.pi / 2, abs=1e-6)


def test_random_points_exact(geom):
    rng = np.random.default_rng(0)
    lim = AimLimits()
    for _ in range(50):
        p_bl = (rng.uniform(0.27, 0.42), rng.uniform(-0.12, 0.12), 0.0)
        target = bl_to_arm(p_bl, geom)
        sols, reason = solve(target, geom, lim)
        assert sols, f'no solution for {p_bl}: {reason}'
        for s in sols:
            assert ray_point_distance(target, *nozzle_ray(s.joints, geom)) < 1e-6
            for q, lo, hi in zip(s.joints, geom.lower, geom.upper):
                assert lo <= q <= hi
            assert s.tip[2] >= target[2] + 0.03 - 1e-9
            assert s.joints[4] == 0.0


def test_ahead_point(geom):
    sols, _ = solve(bl_to_arm((0.30, 0, 0), geom), geom, AimLimits())
    assert sols
    assert sols[0].tilt < math.radians(30)


def test_behind(geom):
    assert solve(bl_to_arm((-0.30, 0, 0), geom), geom, AimLimits()) == ([], 'behind the arm')


def test_too_far(geom):
    assert solve(bl_to_arm((1.0, 0, 0), geom), geom, AimLimits()) == ([], 'too far')


def test_under_arm(geom):
    assert solve(bl_to_arm((0.05, 0, 0), geom), geom, AimLimits()) == (
        [], 'too close / under the arm')


def test_invalid_target(geom):
    assert solve((float('nan'), 0, 0), geom, AimLimits()) == ([], 'invalid target')
    assert solve((0.3, float('inf'), 0), geom, AimLimits()) == ([], 'invalid target')


def test_ranking(geom):
    sols, _ = solve(bl_to_arm((0.34, 0.05, 0), geom), geom, AimLimits())
    assert 0 < len(sols) <= 20
    keys = [(s.tilt, s.standoff) for s in sols]
    assert keys == sorted(keys)


def test_keep_clear_grid(geom, keep):
    xmin, xmax, ymin, ymax = keep.keep_box
    for x in (0.28, 0.34, 0.40):
        for y in (-0.10, 0.0, 0.10):
            sols, reason = solve(bl_to_arm((x, y, 0), geom), geom, keep)
            assert sols, f'({x}, {y}): {reason}'
            for s in sols:
                pts = arm_points(s.joints, geom) + np.asarray(geom.base_in_base_link)
                near = np.abs(pts[:, 2] - keep.scan_z) < keep.scan_band
                for px, py, _ in pts[near]:
                    assert xmin <= px <= xmax and ymin <= py <= ymax


def test_from_urdf_ignores_ros2_control_joints(urdf_xml, geom):
    """<joint> tags inside <ros2_control> must not override the real URDF joints."""
    from aim_solver import ArmGeometry, JOINT_NAMES
    block = '<ros2_control name="arm_sim" type="system"><hardware><plugin>x</plugin></hardware>'
    for n in JOINT_NAMES:
        block += (f'<joint name="{n}"><command_interface name="position"/>'
                  f'<state_interface name="position"/></joint>')
    block += '</ros2_control>'
    injected = urdf_xml.replace('</robot>', block + '</robot>')
    assert injected != urdf_xml
    assert ArmGeometry.from_urdf(injected) == geom

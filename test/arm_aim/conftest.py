"""Shared test setup for the arm aim tests."""
import os
import sys

import numpy as np
import pytest

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))

from aim_solver import AimLimits, ArmGeometry  # noqa: E402


@pytest.fixture(scope='session')
def urdf_xml():
    """The full robot URDF (real-robot variant: revolute arm joints, all at 0)."""
    import xacro
    doc = xacro.process_file(
        os.path.join(PKG, 'description', 'robot.urdf.xacro'),
        mappings={'use_ros2_control': 'true', 'sim_mode': 'false',
                  'wheel_mu': '1.0', 'sim_camera': 'false'})
    return doc.toxml()


@pytest.fixture(scope='session')
def geom(urdf_xml):
    return ArmGeometry.from_urdf(urdf_xml)


@pytest.fixture
def keep():
    """Limits with the LiDAR keep-clear rule switched on."""
    return AimLimits(scan_z=0.378, keep_box=(-0.51, 0.16, -0.14, 0.14))


def bl_to_arm(p, geom):
    """Convert a point from base_link to arm_base coordinates."""
    return np.asarray(p, dtype=float) - np.asarray(geom.base_in_base_link)

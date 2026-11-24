"""Checks on the MoveIt 2 config for the arm (config/moveit/), no ROS nodes needed.

Why: MoveIt keeps its own copy of a few facts that also live elsewhere in the package (the
parked pose, the joint list, which link pairs may touch). If one copy changes and the other
does not, the arm would park in a different place in MoveIt than on the robot, or MoveIt
would reject a pose for a "collision" between parts that are bolted together. These tests
catch that before anything moves.

Run (workspace sourced, so xacro can find my_bot):
  cd ~/dev_ws/src/my_bot && python3 -m pytest test/arm_aim/test_moveit_config.py -v
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
sys.path.insert(0, HERE)

from aim_solver import JOINT_NAMES  # noqa: E402
from arm_config import load_keep_box, load_park  # noqa: E402
import gen_disable_collisions  # noqa: E402

SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
XACRO_NS = '{http://www.ros.org/wiki/xacro}'


def test_srdf_group():
    """Group `arm` is the chain arm_base -> spray_tip (the 5 arm joints, nothing else)."""
    root = ET.parse(SRDF).getroot()
    groups = [g for g in root.findall('group') if g.get('name') == 'arm']
    assert len(groups) == 1, 'exactly one group named arm'
    chains = groups[0].findall('chain')
    assert len(chains) == 1
    assert chains[0].get('base_link') == 'arm_base'
    assert chains[0].get('tip_link') == 'spray_tip'


def _xacro_park():
    """arm_park_* from weedbot_properties.xacro, evaluated like xacro does (${pi/2} etc.)."""
    root = ET.parse(os.path.join(PKG, 'description', 'weedbot_properties.xacro')).getroot()
    props = {p.get('name'): p.get('value') for p in root.iter(XACRO_NS + 'property')}
    out = []
    for key in ('yaw', 'shoulder', 'elbow', 'wrist', 'roll'):
        text = props['arm_park_' + key].strip()
        if text.startswith('${') and text.endswith('}'):
            text = text[2:-1]
        out.append(float(eval(text, {'__builtins__': {}}, {'pi': math.pi})))
    return out


def test_park_matches_xacro_and_stub():
    """MoveIt's `park` = the URDF's parked pose = what the real-robot stub publishes."""
    park = load_park(SRDF)
    assert len(park) == 5
    assert list(park) == pytest.approx(_xacro_park(), abs=1e-6)

    with open(os.path.join(PKG, 'config', 'arm_joint_state_stub.yaml')) as f:
        stub = yaml.safe_load(f)['arm_joint_state_stub']['ros__parameters']
    assert tuple(stub['joints'][:5]) == JOINT_NAMES
    assert list(park) == pytest.approx(stub['positions'][:5], abs=1e-6)


def test_joint_limits_all_joints():
    """Every arm joint has the slow, gentle limits (1 rad/s, 2 rad/s^2)."""
    with open(os.path.join(PKG, 'config', 'moveit', 'joint_limits.yaml')) as f:
        limits = yaml.safe_load(f)['joint_limits']
    assert set(limits) == set(JOINT_NAMES)
    for name in JOINT_NAMES:
        j = limits[name]
        assert j['has_velocity_limits'] is True, name
        assert j['max_velocity'] == 1.0, name
        assert j['has_acceleration_limits'] is True, name
        assert j['max_acceleration'] == 2.0, name


def test_disable_list_up_to_date():
    """The SRDF's generated block is exactly what gen_disable_collisions.py makes from the URDF
    today. If this fails after a URDF change, run:
      python3 test/arm_aim/gen_disable_collisions.py --write"""
    expected = gen_disable_collisions.disable_lines(gen_disable_collisions.sim_urdf())
    with open(SRDF) as f:
        in_srdf = gen_disable_collisions.read_block(f.read())
    assert expected, 'generator produced no pairs'
    assert in_srdf == expected


def test_load_keep_box():
    """scan_self_filter box (x -0.53..0.18, y -0.16..0.16) shrunk 2 cm on every side."""
    box = load_keep_box(os.path.join(PKG, 'config', 'safety.yaml'), 0.02)
    assert box == pytest.approx((-0.51, 0.16, -0.14, 0.14))

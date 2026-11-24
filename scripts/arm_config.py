#!/usr/bin/env python3
"""Shared arm settings, read from the files that own them (no ROS needed).

Why this file exists: the arm nodes (aim node, spray controller) and the tests need a few numbers
that already live in other files. Reading them from there means each number exists ONCE:
  - the parked pose: from MoveIt's SRDF (group_state `park`), so "park" means exactly the
    same pose to MoveIt and to every node;
  - the LiDAR keep-clear box: from config/safety.yaml (`scan_self_filter` box), the area where
    the scan filter throws away the robot's own returns. The arm may cross the scan plane only
    inside it, otherwise the LiDAR sees the arm as an obstacle and the safety chain stops.
"""
import xml.etree.ElementTree as ET

import yaml

from aim_solver import JOINT_NAMES


def load_park(srdf_path: str) -> tuple:
    """Parked arm pose from the SRDF's group_state `park` (group `arm`).

    Returns 5 angles in rad, in JOINT_NAMES order. Raises ValueError if the state is missing
    or does not name all five joints (a half-defined park pose must never be used).
    """
    root = ET.parse(srdf_path).getroot()
    for state in root.findall('group_state'):
        if state.get('name') != 'park' or state.get('group') != 'arm':
            continue
        values = {j.get('name'): float(j.get('value')) for j in state.findall('joint')}
        missing = [n for n in JOINT_NAMES if n not in values]
        if missing:
            raise ValueError(f'{srdf_path}: group_state park is missing joints {missing}')
        return tuple(values[n] for n in JOINT_NAMES)
    raise ValueError(f'{srdf_path}: no group_state "park" for group "arm"')


def load_keep_box(safety_yaml_path: str, shrink: float) -> tuple:
    """The scan_self_filter box from safety.yaml, made `shrink` metres smaller on every side.

    Returns (xmin, xmax, ymin, ymax) in base_link. Why smaller: the filter box is only a hair
    bigger than the robot, and the arm is drawn as thin lines in the aim solver; the margin keeps
    the real (thicker) arm parts well inside the area the filter ignores.
    """
    with open(safety_yaml_path) as f:
        data = yaml.safe_load(f)
    xmin, xmax, ymin, ymax = (float(v) for v in
                              data['scan_self_filter']['ros__parameters']['box'])
    if shrink < 0.0 or xmax - xmin <= 2 * shrink or ymax - ymin <= 2 * shrink:
        raise ValueError(f'cannot shrink box {[xmin, xmax, ymin, ymax]} by {shrink} m')
    return (xmin + shrink, xmax - shrink, ymin + shrink, ymax - shrink)

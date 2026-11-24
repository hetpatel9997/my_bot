#!/usr/bin/env python3
"""Aim solver for the WeedBot spray arm (pure maths, no ROS).

The arm has 5 joints: a yaw (turns the whole arm about the vertical), three
pitch joints (shoulder, elbow, wrist) that all bend in ONE vertical plane, and
a roll that spins the nozzle about its own axis (it does not change where the
nozzle points, so it stays 0).

The question answered here: "which joint angles make the nozzle point at a
weed, from a short distance away, without the arm blocking the LiDAR?"

How it works, in plain words
----------------------------
1. Yaw is easy: turn the arm so its vertical plane contains the weed.
   yaw = atan2(y, x).
2. In that plane everything is 2-D: r = distance outward from the arm axis,
   up = height. The three pitch joints are a 2-link arm (upper arm, forearm)
   plus a wrist that sets the nozzle direction.
3. We CHOOSE the nozzle direction ("tilt" = angle from straight down, leaning
   outward) and the "standoff" (distance from the nozzle tip to the weed along
   the nozzle direction). That fixes where the tip must be, and where the
   wrist must be (one nozzle length back from the tip).
4. The shoulder/elbow then have a closed-form answer (law of cosines), with
   two mirror solutions (elbow up / elbow down). The wrist angle is whatever
   is left over to make the nozzle point in the chosen direction.
5. We try tilts from 0 upward and standoffs from near to far, drop anything
   outside the joint limits (or whose tip would be too low, or that would put
   part of the arm in the LiDAR scan plane over the robot), and rank what is
   left (the nozzle tip must also stay on the target side of the arm axis;
   otherwise a target under the arm "works" by folding the arm over backwards):
   smallest tilt first (spray from straight above), then smallest
   standoff, then lowest elbow.

Angle convention (same as the URDF): pitch joints rotate about +y, 0 = link
straight up, positive tilts the link toward +x (outward). Radians everywhere.
All positions are in the `arm_base` frame unless stated otherwise.
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import numpy as np

JOINT_NAMES = ('arm_base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll')


def _wrap(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class ArmGeometry:
    base_in_base_link: tuple   # (x, y, z) of arm_base in base_link (arm_base_joint origin)
    yaw_z: float               # arm_base -> yaw joint (arm_base_yaw origin z)
    l_shoulder: float          # yaw joint -> shoulder axis   (0.0707)
    l_upper: float             # shoulder -> elbow            (0.100)
    l_fore: float              # elbow -> wrist pitch         (0.100)
    l_tip: float               # wrist pitch -> spray tip     (0.131 = wrist_roll z + spray_tip_joint z)
    lower: tuple               # 5 joint limits, rad, JOINT_NAMES order
    upper: tuple

    @staticmethod
    def from_urdf(urdf_xml: str) -> 'ArmGeometry':
        """Read the arm lengths and joint limits out of the robot URDF text."""
        root = ET.fromstring(urdf_xml)
        joints = {j.get('name'): j for j in root.iter('joint')}

        def origin(name):
            if name not in joints:
                raise ValueError(f'joint {name} missing from the URDF')
            o = joints[name].find('origin')
            xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()] if o is not None else [0.0] * 3
            rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()] if o is not None else [0.0] * 3
            if any(abs(v) > 1e-9 for v in rpy):
                raise ValueError(f'joint {name} origin has non-zero rpy (arm not at zero pose)')
            return xyz

        base = tuple(origin('arm_base_joint'))
        yaw_z = origin('arm_base_yaw')[2]
        l_shoulder = origin('shoulder_pitch')[2]
        l_upper = origin('elbow_pitch')[2]
        l_fore = origin('wrist_pitch')[2]
        l_tip = origin('wrist_roll')[2] + origin('spray_tip_joint')[2]

        lower, upper = [], []
        for name in JOINT_NAMES:
            if name not in joints:
                raise ValueError(f'joint {name} missing from the URDF')
            lim = joints[name].find('limit')
            if lim is None:
                raise ValueError(f'joint {name} has no limit')
            lower.append(float(lim.get('lower')))
            upper.append(float(lim.get('upper')))
        return ArmGeometry(base, yaw_z, l_shoulder, l_upper, l_fore, l_tip,
                           tuple(lower), tuple(upper))


@dataclass(frozen=True)
class AimLimits:
    tilt_max_deg: float = 50.0
    tilt_step_deg: float = 1.0
    standoff_min: float = 0.08
    standoff_max: float = 0.25
    standoff_step: float = 0.01
    tip_clearance: float = 0.03
    max_candidates: int = 20
    scan_z: Optional[float] = None       # laser_frame height in base_link; None = no keep-clear check
    keep_box: Optional[tuple] = None     # (xmin, xmax, ymin, ymax) in base_link, ALREADY shrunk
    scan_band: float = 0.03


@dataclass(frozen=True)
class AimSolution:
    joints: tuple        # 5 rad, JOINT_NAMES order, roll = 0.0
    tilt: float          # rad from straight down
    standoff: float      # m, tip -> target
    tip: tuple           # (x, y, z) in arm_base


def _plane_points(q, geom):
    """(radial, up) of yaw joint, shoulder, elbow, wrist, tip, plus the nozzle angle."""
    qs, qe, qw = q[1], q[2], q[3]
    a1, a2, a3 = qs, qs + qe, qs + qe + qw        # absolute angle of each link from "up"
    p0 = np.array([0.0, geom.yaw_z])
    p1 = p0 + [0.0, geom.l_shoulder]
    p2 = p1 + geom.l_upper * np.array([math.sin(a1), math.cos(a1)])
    p3 = p2 + geom.l_fore * np.array([math.sin(a2), math.cos(a2)])
    p4 = p3 + geom.l_tip * np.array([math.sin(a3), math.cos(a3)])
    return [p0, p1, p2, p3, p4], a3


def _to_3d(p2, yaw):
    return np.array([p2[0] * math.cos(yaw), p2[0] * math.sin(yaw), p2[1]])


def nozzle_ray(joints, geom: ArmGeometry):
    """Forward kinematics: (tip position, unit pointing direction) in arm_base."""
    pts, a3 = _plane_points(joints, geom)
    yaw = joints[0]
    tip = _to_3d(pts[4], yaw)
    direction = np.array([math.sin(a3) * math.cos(yaw), math.sin(a3) * math.sin(yaw), math.cos(a3)])
    return tip, direction


def arm_points(joints, geom: ArmGeometry, step: float = 0.01) -> np.ndarray:
    """(N, 3) points spaced ~`step` apart along yaw joint -> shoulder -> elbow
    -> wrist -> tip, in arm_base. Used to see which heights the arm occupies."""
    pts, _ = _plane_points(joints, geom)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for i in range(n):
            out.append(_to_3d(a + (b - a) * i / n, joints[0]))
    out.append(_to_3d(pts[-1], joints[0]))
    return np.array(out)


def ray_point_distance(point, origin, direction) -> float:
    """Perpendicular distance from `point` to the ray; inf if it is behind the origin."""
    v = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    if float(np.dot(v, d)) < 0.0:
        return math.inf
    return float(np.linalg.norm(np.cross(v, d)))


def _blocks_lidar(joints, geom, lim) -> bool:
    """True if any part of the arm sits in the LiDAR scan band outside the keep box."""
    if lim.scan_z is None or lim.keep_box is None:
        return False
    xmin, xmax, ymin, ymax = lim.keep_box
    pts = arm_points(joints, geom) + np.asarray(geom.base_in_base_link)
    in_band = np.abs(pts[:, 2] - lim.scan_z) < lim.scan_band
    for x, y in pts[in_band][:, :2]:
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return True
    return False


def solve(target, geom: ArmGeometry, lim: AimLimits):
    """Find joint angles that aim the nozzle at `target` (x, y, z in arm_base).

    Returns (solutions ranked best first, '') or ([], reason).
    """
    t = np.asarray(target, dtype=float)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        return [], 'invalid target'
    x, y, z = t
    yaw = math.atan2(y, x)
    if yaw < geom.lower[0] or yaw > geom.upper[0]:
        return [], 'behind the arm'

    r = math.hypot(x, y)
    h = z - (geom.yaw_z + geom.l_shoulder)       # target height relative to the shoulder axis
    goal = np.array([r, h])
    shoulder_z = geom.yaw_z + geom.l_shoulder
    lu, lf = geom.l_upper, geom.l_fore

    found = []            # (tilt, standoff, elbow_up, AimSolution)
    blocked_only = False  # some candidate was fine except for the keep-clear rule
    n_tilt = int(round(lim.tilt_max_deg / lim.tilt_step_deg))
    n_stand = int(round((lim.standoff_max - lim.standoff_min) / lim.standoff_step))
    for i in range(n_tilt + 1):
        tilt = math.radians(i * lim.tilt_step_deg)
        th = math.pi - tilt                                  # nozzle angle from straight up
        d = np.array([math.sin(th), math.cos(th)])           # (radial, up) nozzle direction
        for j in range(n_stand + 1):
            s = lim.standoff_min + j * lim.standoff_step
            tip = goal - s * d
            if shoulder_z + tip[1] < t[2] + lim.tip_clearance:
                continue                                      # tip too low (too near the weed's height)
            if tip[0] < 0.0:
                continue                                      # tip would cross over the arm's own axis
            wr = tip - geom.l_tip * d
            c = (wr @ wr - lu * lu - lf * lf) / (2 * lu * lf)
            if abs(c) > 1.0:
                continue                                      # wrist out of reach
            for qe in (math.acos(c), -math.acos(c)):
                qs = math.atan2(wr[0], wr[1]) - math.atan2(lf * math.sin(qe), lu + lf * math.cos(qe))
                qw = th - qs - qe
                q = (yaw, _wrap(qs), _wrap(qe), _wrap(qw), 0.0)
                if any(v < lo - 1e-12 or v > hi + 1e-12
                       for v, lo, hi in zip(q, geom.lower, geom.upper)):
                    continue
                if _blocks_lidar(q, geom, lim):
                    blocked_only = True
                    continue
                pts, _ = _plane_points(q, geom)
                tip3 = _to_3d(pts[4], yaw)
                sol = AimSolution(q, tilt, s, tuple(float(v) for v in tip3))
                found.append((tilt, s, pts[2][1], sol))

    if found:
        found.sort(key=lambda f: f[:3])
        return [f[3] for f in found[:lim.max_candidates]], ''
    if blocked_only:
        return [], 'would block the LiDAR'
    if r >= lu + lf:
        return [], 'too far'
    return [], 'too close / under the arm'

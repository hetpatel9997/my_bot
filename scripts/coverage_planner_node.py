#!/usr/bin/env python3
"""
coverage_planner_node.py  -  robot-vacuum style coverage (edge pass + smooth zig-zag lanes).

What it does (indoor maps; outdoor use needs the pool keepout first, see CLAUDE.md)
  1. Reads /map. Computes, for every free cell, the distance to the nearest obstacle/unknown cell.
  2. Works out where the robot's CENTRE may go, from the robot outline in nav2_params.yaml
     (local_costmap footprint + footprint_padding; one source of truth with Nav2):
       lane_clear = half width + clearance_margin            driving straight
       turn_clear = farthest corner + clearance_margin       turning (corners swing out)
     and, once Nav2 runs, keeps every path point where the global costmap cost is below max_cost
     (so the planner/controller never get a point they consider unsafe).
     Gaps the robot physically fits through stay usable; nothing is shrunk by a rough circle.
  3. Perimeter laps (perimeter_laps = 0, 1 or 2): smooth loops along the walls at the safe
     distance (lap 2 one lane spacing further in), plus one loop around every free-standing
     obstacle; corners rounded and pushed inward where the robot has to turn.
  4. Lanes at lane_angle (auto = along the longest side of the area, or degrees), spaced by the
     pattern preset (dense 0.25 m, medium 0.35 m, wide 0.50 m, or custom = lane_spacing).
     Boustrophedon cell decomposition (every obstacle splits the area into cells that can each
     be swept without crossing it); inside a cell the lanes are joined by smooth half-circle
     U-turns (radius = spacing / 2) whenever the turn fits, so the robot never stops.
  5. Cells are visited nearest-first; Nav2's planner (Smac Hybrid-A*, footprint aware) plans the
     short transitions between edge pass and cells.
  6. Everything becomes ONE dense path (every path_resolution m, with headings) that is sent to
     Nav2's controller (FollowPath -> Regulated Pure Pursuit). The robot drives it continuously.
     If the controller gives up (a person, a moved chair...), the node waits blocked_wait s for
     the way to clear, then re-plans from the robot to a point 0.3 m further along (Nav2 goes
     around the obstacle if there is room). If it is still blocked there, it skips 1 m and puts
     that piece in a revisit queue, which is retried at the end. The final report (also on
     /coverage/report) lists anything still skipped with its map location.
  7. /coverage_path (nav_msgs/Path) shows the plan, /coverage_grid what has been covered.

Services (std_srvs/Trigger):
  /coverage/plan    compute and publish the path, do not move
  /coverage/start   plan if needed, then drive it
  /coverage/stop    cancel and stop

Choosing the pattern per run (any change discards the current plan):
  ros2 param set /coverage_planner pattern dense          # dense | medium | wide | custom
  ros2 param set /coverage_planner lane_spacing 0.30      # used when pattern is custom
  ros2 param set /coverage_planner lane_angle 30          # degrees, or auto
  ros2 param set /coverage_planner perimeter_laps 2       # 0, 1 or 2
  ros2 service call /coverage/plan std_srvs/srv/Trigger   # preview /coverage_path in RViz
  ros2 service call /coverage/start std_srvs/srv/Trigger
or at launch: ros2 launch my_bot coverage.launch.py pattern:=wide lane_angle:=90 perimeter_laps:=0

Parameters: see config/coverage_params.yaml
"""
import ast
import math
import os
import threading
import time

import yaml

import numpy as np
from scipy import ndimage

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


PRESETS = {'dense': 0.25, 'medium': 0.35, 'wide': 0.50}   # lane spacing (m) per pattern


# =========================================================================== grid helpers
MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # clockwise


def trace_outer_boundary(mask):
    """Ordered list of (r, c) cells on the outer boundary of the (single) region in mask
    (Moore-neighbour tracing, starting at the top-left-most cell)."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return []
    i = np.lexsort((cols, rows))[0]
    start = (int(rows[i]), int(cols[i]))
    h, w = mask.shape

    def inside(p):
        return 0 <= p[0] < h and 0 <= p[1] < w and mask[p]

    out = [start]
    cur, back_dir = start, 6          # we "came from" the west of the top-left cell
    for _ in range(4 * mask.sum() + 10):
        found = False
        for k in range(1, 9):
            d = (back_dir + k) % 8
            nxt = (cur[0] + MOORE[d][0], cur[1] + MOORE[d][1])
            if inside(nxt):
                back_dir = (d + 4) % 8       # direction pointing back to cur
                cur = nxt
                found = True
                break
        if not found or cur == start:
            break
        out.append(cur)
    return out


def runs(line, min_len):
    """(start, end) inclusive index pairs of True runs at least min_len long."""
    out, start = [], None
    for i, v in enumerate(line):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(line) - start >= min_len:
        out.append((start, len(line) - 1))
    return out


def decompose_cells(lane_runs):
    """Boustrophedon cell decomposition. lane_runs: list over lanes of [(a, b), ...].
    Returns cells, each a list of (lane_index, a, b) with consecutive lane indices, such that
    each run overlaps exactly one run in the neighbouring lane of the same cell."""
    cells, open_cells = [], {}            # open_cells: run (a,b) in previous lane -> cell index
    prev = []
    for li, cur in enumerate(lane_runs):
        new_open = {}
        for (a, b) in cur:
            ov_prev = [(pa, pb) for (pa, pb) in prev if pa <= b and a <= pb]
            if len(ov_prev) == 1:
                pa, pb = ov_prev[0]
                ov_cur = [(ca, cb) for (ca, cb) in cur if ca <= pb and pa <= cb]
                if len(ov_cur) == 1 and (pa, pb) in open_cells:
                    ci = open_cells[(pa, pb)]
                    cells[ci].append((li, a, b))
                    new_open[(a, b)] = ci
                    continue
            cells.append([(li, a, b)])
            new_open[(a, b)] = len(cells) - 1
        open_cells, prev = new_open, cur
    return cells


def smooth_closed(pts, w):
    """Circular moving average of an (N, 2) array."""
    if len(pts) < w or w < 2:
        return pts
    k = np.ones(w) / w
    pad = w // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    return np.column_stack([np.convolve(ext[:, i], k, mode='valid')[:len(pts)] for i in range(2)])


def resample(pts, step, closed=False):
    """Points every `step` metres along a polyline (N, 2)."""
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step:
        return pts
    t = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def headings(pts):
    """Direction of travel at each point, from neighbours two samples away (less jitter)."""
    n = len(pts)
    if n < 5:
        d = np.gradient(pts, axis=0)
        return np.arctan2(d[:, 1], d[:, 0])
    ahead = pts[np.minimum(np.arange(n) + 2, n - 1)]
    behind = pts[np.maximum(np.arange(n) - 2, 0)]
    d = ahead - behind
    return np.arctan2(d[:, 1], d[:, 0])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# =========================================================================== node
class CoveragePlanner(Node):
    def __init__(self):
        super().__init__('coverage_planner')
        p = self.declare_parameter
        p('pattern', 'medium')                  # dense 0.25 | medium 0.35 | wide 0.50 | custom
        p('lane_spacing', 0.35)                 # m between lanes, used when pattern = custom
        p('lane_angle', 'auto', ParameterDescriptor(dynamic_typing=True))   # 'auto' or degrees
        p('perimeter_laps', 1)                  # 0, 1 or 2 loops along walls/obstacles first
        p('nav2_params_file', os.path.join(get_package_share_directory('my_bot'),
                                           'config', 'nav2_params.yaml'))
        p('clearance_margin', 0.09)             # m extra gap to obstacles (on top of padding)
        p('costmap_topic', '/global_costmap/costmap')
        p('max_cost', 50)                       # 0-100 (costmap topic scale); path points stay below
        p('coverage_radius', 0.225)             # m, half the swath that one pass "covers"
        p('min_segment_len', 0.30)              # m, skip lane pieces shorter than this
        p('blocked_wait', 5.0)                  # s to wait for a blocked path to clear before re-planning
        p('path_resolution', 0.05)              # m between path poses
        p('map_topic', '/map')
        p('global_frame', 'map')
        p('base_frame', 'base_link')
        p('planner_id', 'GridBased')
        p('controller_id', 'FollowPath')
        p('goal_checker_id', 'general_goal_checker')

        self.gp = lambda n: self.get_parameter(n).value
        self.add_on_set_parameters_callback(self._on_params)
        self.half_width, self.circ_radius = self._robot_size()
        self.global_frame = self.gp('global_frame')
        self.base_frame = self.gp('base_frame')
        self.cb = ReentrantCallbackGroup()

        self.map_msg = None
        self.costmap = None
        self.aborts = 0
        self.grid = None
        self.coverable = None
        self.covered = None
        self.path = []                  # list of (x, y, yaw)
        self.worker = None
        self.lane_angle_used = 0.0
        self.stop_flag = threading.Event()
        self.goal_handle = None
        self.lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, self.gp('map_topic'), self._map_cb, latched,
                                 callback_group=self.cb)
        self.create_subscription(OccupancyGrid, self.gp('costmap_topic'),
                                 lambda m: setattr(self, 'costmap', m), latched, callback_group=self.cb)
        self.path_pub = self.create_publisher(Path, 'coverage_path', latched)
        self.report_pub = self.create_publisher(String, 'coverage/report', latched)
        self.cov_pub = self.create_publisher(OccupancyGrid, 'coverage_grid', latched)
        self.create_service(Trigger, 'coverage/plan', self._srv_plan, callback_group=self.cb)
        self.create_service(Trigger, 'coverage/start', self._srv_start, callback_group=self.cb)
        self.create_service(Trigger, 'coverage/stop', self._srv_stop, callback_group=self.cb)
        self.plan_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose',
                                        callback_group=self.cb)
        self.follow_client = ActionClient(self, FollowPath, 'follow_path', callback_group=self.cb)
        self.create_timer(0.3, self._track_coverage, callback_group=self.cb)
        self.create_timer(2.0, self._publish_coverage_grid, callback_group=self.cb)
        self.get_logger().info('coverage_planner ready: call /coverage/plan or /coverage/start')

    # ------------------------------------------------------------------ coverage options
    PLAN_PARAMS = ('pattern', 'lane_spacing', 'lane_angle', 'perimeter_laps', 'clearance_margin',
                   'min_segment_len', 'max_cost')

    def _on_params(self, params):
        for prm in params:
            if prm.name == 'pattern' and prm.value not in (*PRESETS, 'custom'):
                return SetParametersResult(successful=False,
                                           reason='pattern must be dense, medium, wide or custom')
            if prm.name == 'perimeter_laps' and prm.value not in (0, 1, 2):
                return SetParametersResult(successful=False, reason='perimeter_laps must be 0, 1 or 2')
            if prm.name == 'lane_spacing' and not 0.1 <= float(prm.value) <= 2.0:
                return SetParametersResult(successful=False, reason='lane_spacing must be 0.1-2.0 m')
            if prm.name == 'lane_angle' and str(prm.value).strip().lower() != 'auto':
                try:
                    float(prm.value)
                except ValueError:
                    return SetParametersResult(successful=False,
                                               reason="lane_angle must be 'auto' or degrees")
        if any(prm.name in self.PLAN_PARAMS for prm in params):
            if self.worker is not None and self.worker.is_alive():
                return SetParametersResult(successful=False,
                                           reason='coverage is running; call /coverage/stop first')
            self.path = []                       # next /coverage/plan or /start re-plans
        return SetParametersResult(successful=True)

    def spacing(self):
        pat = self.gp('pattern')
        return float(self.gp('lane_spacing')) if pat == 'custom' else PRESETS[pat]

    # ------------------------------------------------------------------ robot size from Nav2
    def _robot_size(self):
        """(half width, circumscribed radius) incl. footprint_padding, from the LOCAL costmap in
        nav2_params.yaml: the real outline the controller collision-checks with (the global
        costmap uses a half-width circle for the point-based planner)."""
        f = self.gp('nav2_params_file')
        cm = yaml.safe_load(open(f))['local_costmap']['local_costmap']['ros__parameters']
        pad = float(cm.get('footprint_padding', 0.01))
        if 'footprint' in cm:
            pts = ast.literal_eval(cm['footprint']) if isinstance(cm['footprint'], str) else cm['footprint']
            half = max(abs(y) for _, y in pts) + pad
            circ = max(math.hypot(x, y) for x, y in pts) + pad
        else:
            half = circ = float(cm['robot_radius']) + pad
        self.get_logger().info(f'Robot size from {os.path.basename(f)}: half width {half:.3f} m, '
                               f'farthest corner {circ:.3f} m (incl. padding {pad:.2f} m)')
        return half, circ

    def _clearances(self):
        m = self.gp('clearance_margin')
        return self.half_width + m, self.circ_radius + m

    def cost_at(self, x, y):
        """Global costmap value (0-100, -1 unknown) at a world point; 0 if no costmap yet."""
        c = self.costmap
        if c is None:
            return 0
        i = c.info
        cx = int((x - i.origin.position.x) / i.resolution)
        cy = int((y - i.origin.position.y) / i.resolution)
        if 0 <= cx < i.width and 0 <= cy < i.height:
            return c.data[cy * i.width + cx]
        return 100

    def cost_ok(self, x, y):
        v = self.cost_at(x, y)
        return 0 <= v < self.gp('max_cost')

    # ------------------------------------------------------------------ map / pose / conversions
    def _map_cb(self, msg):
        self.map_msg = msg
        self.grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        if self.covered is None or self.covered.shape != self.grid.shape:
            self.covered = np.zeros(self.grid.shape, dtype=bool)
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height} '
                               f'@ {msg.info.resolution:.3f} m')

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def cell_of(self, x, y):
        i = self.map_msg.info
        return (int((y - i.origin.position.y) / i.resolution),
                int((x - i.origin.position.x) / i.resolution))

    def world_of(self, r, c):
        i = self.map_msg.info
        return (i.origin.position.x + (c + 0.5) * i.resolution,
                i.origin.position.y + (r + 0.5) * i.resolution)

    def dist_at(self, x, y):
        """Clearance (m) at a world point, bilinear-free nearest lookup; 0 outside the map."""
        r, c = self.cell_of(x, y)
        if 0 <= r < self.dist.shape[0] and 0 <= c < self.dist.shape[1]:
            return float(self.dist[r, c])
        return 0.0

    def pose_msg(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = self.global_frame
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    # ------------------------------------------------------------------ action helpers
    @staticmethod
    def _wait(fut, timeout):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            time.sleep(0.02)
        return fut.done()

    def plan_between(self, start, goal):
        """Nav2 planner path (list of (x, y, yaw)) from start (or the robot if None) to goal."""
        if not self.plan_client.wait_for_server(timeout_sec=2.0):
            return None
        g = ComputePathToPose.Goal()
        g.goal = self.pose_msg(*goal)
        g.planner_id = self.gp('planner_id')
        if start is not None:
            g.start = self.pose_msg(*start)
            g.use_start = True
        f = self.plan_client.send_goal_async(g)
        if not self._wait(f, 5.0) or not f.result().accepted:
            return None
        rf = f.result().get_result_async()
        if not self._wait(rf, 10.0) or rf.result().status != GoalStatus.STATUS_SUCCEEDED:
            return None
        poses = rf.result().result.path.poses
        if len(poses) < 2:
            return None
        pts = np.array([[q.pose.position.x, q.pose.position.y] for q in poses])
        pts = resample(pts, self.gp('path_resolution'))
        yaw = headings(pts)
        return [(x, y, a) for (x, y), a in zip(pts, yaw)]

    # ------------------------------------------------------------------ planning
    def _plan(self):
        if self.map_msg is None:
            return False, 'No map received yet (is map_server or SLAM running?)'
        res = self.map_msg.info.resolution
        step = self.gp('path_resolution')
        lane_clear, turn_clear = self._clearances()

        free = self.grid == 0                       # unknown (-1) counts as blocked
        padded = np.pad(free, 1, constant_values=False)
        # distance from each free cell centre to the nearest blocked cell's EDGE (half a cell less
        # than centre-to-centre)
        self.dist = np.maximum(ndimage.distance_transform_edt(padded)[1:-1, 1:-1] - 0.5, 0.0) * res

        # Region the robot centre may sweep: clearance >= lane_clear, cost below max_cost (when the
        # global costmap is available), connected to the robot.
        ok = self.dist >= lane_clear
        if self.costmap is not None:
            h, w = ok.shape
            rr, cc = np.mgrid[0:h, 0:w]
            ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
            ci = self.costmap.info
            cx = ((ox + (cc + 0.5) * res - ci.origin.position.x) / ci.resolution).astype(int)
            cy = ((oy + (rr + 0.5) * res - ci.origin.position.y) / ci.resolution).astype(int)
            inside = (cx >= 0) & (cx < ci.width) & (cy >= 0) & (cy < ci.height)
            cdata = np.array(self.costmap.data, dtype=np.int16).reshape(ci.height, ci.width)
            cost = np.full(ok.shape, 100, dtype=np.int16)
            cost[inside] = cdata[cy[inside], cx[inside]]
            ok &= (cost >= 0) & (cost < self.gp('max_cost'))
        else:
            self.get_logger().warn('No global costmap yet: planning on the map only (start Nav2 '
                                   'first for the cost check)')
        labels, n = ndimage.label(ok)
        if n == 0:
            return False, (f'No space with {lane_clear:.2f} m clearance: the robot does not fit '
                           'anywhere on this map')
        pose = self.robot_pose()
        region_id = None
        if pose is not None:
            r, c = self.cell_of(pose[0], pose[1])
            if 0 <= r < ok.shape[0] and 0 <= c < ok.shape[1] and labels[r, c] > 0:
                region_id = labels[r, c]
            else:   # robot slightly outside (e.g. near a wall): nearest region cell
                rr, cc = np.nonzero(ok)
                k = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
                region_id = labels[rr[k], cc[k]]
        if region_id is None:
            self.get_logger().warn('Robot pose unknown; using the largest free region')
            region_id = 1 + int(np.argmax(ndimage.sum(ok, labels, range(1, n + 1))))
        region = labels == region_id

        # Cells the robot can cover = free cells within coverage_radius of the region.
        near = ndimage.distance_transform_edt(~region) * res <= self.gp('coverage_radius')
        self.coverable = near & free

        segments = []                               # list of lists of (x, y, yaw)
        here = pose[:2] if pose else None
        for loop in self._perimeter_loops(region, lane_clear, turn_clear, step, here):
            segments.append(('perimeter', loop))
            here = loop[-1][:2]
        lanes = self._lane_cells(region, res)
        for cell_path in self._order_cells(lanes, here):
            segments.append(('lanes', cell_path))
        if not segments:
            return False, 'Region too small for any lane'

        # Join everything into one path; Nav2's planner fills the gaps between segments.
        path, joins_failed = list(segments[0][1]), 0
        for _, seg in segments[1:]:
            gap = math.hypot(seg[0][0] - path[-1][0], seg[0][1] - path[-1][1])
            if gap > 3 * step:
                # A segment start can be a tight spot (e.g. a lane end near a wall corner); if the
                # planner cannot arrive there, join a little further into the segment instead.
                for skip_m in (0.0, 0.3, 0.6, 0.9):
                    k = min(len(seg) - 1, int(skip_m / step))
                    t = self.plan_between(path[-1], seg[k])
                    if t is not None:
                        path.extend(t[1:-1])
                        seg = seg[k:]
                        break
                else:
                    joins_failed += 1
                    self.get_logger().warn(f'No planner path to segment at ({seg[0][0]:.2f}, '
                                           f'{seg[0][1]:.2f}); the robot will re-plan there.')
            path.extend(seg)
        # Final safety pass: every pose must be in low-cost space.
        bad = [p for p in path if not self.cost_ok(p[0], p[1])] if self.costmap is not None else []
        if bad:
            self.get_logger().warn(f'{len(bad)} of {len(path)} poses above max_cost (dropped)')
            path = [p for p in path if self.cost_ok(p[0], p[1])]
        self.path = path
        self._publish_path()
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        n_cells = sum(1 for k, _ in segments if k == 'lanes')   # lane pieces (cells or sub-paths)
        n_loops = sum(1 for k, _ in segments if k == 'perimeter')
        msg = (f'pattern {self.gp("pattern")} (spacing {self.spacing():.2f} m), lane angle '
               f'{math.degrees(self.lane_angle_used):.0f} deg, {n_loops} perimeter loop(s), '
               f'{n_cells} lane piece(s): {len(path)} poses, {length:.1f} m; clearance '
               f'{lane_clear:.2f}/{turn_clear:.2f} m' +
               (f', {joins_failed} joins left to re-plan' if joins_failed else ''))
        self.get_logger().info('Plan: ' + msg)
        return True, msg

    def _push_out(self, pts, need):
        """Move points along the clearance gradient until clearance >= need (per point)."""
        res = self.map_msg.info.resolution
        gy, gx = np.gradient(self.dist)
        out = pts.copy()
        for i, (x, y) in enumerate(out):
            for _ in range(40):
                if self.dist_at(x, y) >= need[i] and self.cost_ok(x, y):
                    break
                r, c = self.cell_of(x, y)
                if not (0 <= r < self.dist.shape[0] and 0 <= c < self.dist.shape[1]):
                    break
                g = np.array([gx[r, c], gy[r, c]])
                nrm = np.linalg.norm(g)
                if nrm < 1e-6:
                    break
                x, y = (x, y) + 0.5 * res * g / nrm
            out[i] = (x, y)
        return out

    def _perimeter_loops(self, region, lane_clear, turn_clear, step, here):
        """perimeter_laps loops along the outside of the region (lap 2 one spacing further in),
        then (if laps > 0) one loop around every obstacle that stands free inside the region."""
        laps = int(self.gp('perimeter_laps'))
        loops = []
        for lap in range(laps):
            edge_clear = lane_clear + lap * self.spacing()
            mask = region & (self.dist >= edge_clear - 1e-6)
            lab, n = ndimage.label(mask)
            if n == 0:
                break
            mask = lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))
            loop = self._edge_loop(mask, edge_clear, turn_clear, step, here)
            if loop:
                loops.append(loop)
                here = loop[-1][:2]
        if laps > 0:                                   # free-standing obstacles (holes)
            lab, n = ndimage.label(~region)
            for i in range(1, n + 1):
                hole = lab == i
                ring = ndimage.binary_dilation(hole) & ~hole
                if hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any():
                    continue                           # touches the map edge: outside, not a hole
                if not region[ring].all():
                    continue
                loop = self._edge_loop(ndimage.binary_dilation(hole), lane_clear, turn_clear, step, here)
                if loop:
                    loops.append(loop)
                    here = loop[-1][:2]
        return loops

    def _edge_loop(self, region, lane_clear, turn_clear, step, pose):
        """Smooth closed loop along the outer boundary of mask `region`; lane_clear is the
        clearance required on straight parts, turn_clear where it bends."""
        cells = trace_outer_boundary(region)
        if len(cells) < 10:
            return None
        pts = np.array([self.world_of(r, c) for r, c in cells])
        pts = resample(smooth_closed(pts, 9), step, closed=True)
        # Round the corners (~0.35 m radius) and, where the loop bends, push it inward to the
        # turning clearance (the footprint corners swing out). Repeat until the shape settles.
        k = max(2, int(0.2 / step))
        for _ in range(4):
            yaw = headings(np.vstack([pts[-3:], pts, pts[:3]]))[3:-3]
            bend = np.array([abs(wrap(yaw[(i + k) % len(yaw)] - yaw[i - k])) for i in range(len(yaw))])
            need = np.where(bend > math.radians(25), turn_clear, lane_clear)
            pts = self._push_out(pts, need)
            pts = resample(smooth_closed(pts, 13), step, closed=True)
        yaw = headings(np.vstack([pts[-3:], pts, pts[:3]]))[3:-3]
        bend = np.array([abs(wrap(yaw[(i + k) % len(yaw)] - yaw[i - k])) for i in range(len(yaw))])
        pts = self._push_out(pts, np.where(bend > math.radians(25), turn_clear, lane_clear))
        pts = resample(smooth_closed(pts, 3), step, closed=True)
        if pose is not None:                         # start the loop nearest to the robot / last point
            i0 = int(np.argmin(np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])))
            pts = np.vstack([pts[i0:], pts[:i0]])
        pts = np.vstack([pts, pts[:1]])              # close the loop
        yaw = headings(pts)
        return [(x, y, a) for (x, y), a in zip(pts, yaw)]

    def _cells_xy(self, region):
        rr, cc = np.nonzero(region)
        o, res = self.map_msg.info.origin.position, self.map_msg.info.resolution
        return np.column_stack([o.x + (cc + 0.5) * res, o.y + (rr + 0.5) * res])

    def _lane_runs(self, region, a, res):
        """Lane offsets, sample positions and runs for lanes at angle a (radians)."""
        d = np.array([math.cos(a), math.sin(a)])        # along the lanes
        nrm = np.array([-math.sin(a), math.cos(a)])     # across the lanes
        pts = self._cells_xy(region)
        U, V = pts @ d, pts @ nrm
        k = self.spacing()
        nl = int((V.max() - V.min()) // k)
        first = V.min() + ((V.max() - V.min()) - nl * k) / 2
        lane_v = [first + i * k for i in range(nl + 1)]
        ds = res / 2                                    # sample every half cell along a lane
        us = np.arange(U.min(), U.max() + ds, ds)
        o = self.map_msg.info.origin.position
        h, w = region.shape
        min_len = max(1, int(round(self.gp('min_segment_len') / ds)))
        lane_runs = []
        for v in lane_v:
            xy = us[:, None] * d + v * nrm
            r = np.floor((xy[:, 1] - o.y) / res).astype(int)
            c = np.floor((xy[:, 0] - o.x) / res).astype(int)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            inside = np.zeros(len(us), dtype=bool)
            inside[ok] = region[r[ok], c[ok]]
            lane_runs.append(runs(inside, min_len))
        return d, nrm, lane_v, us, lane_runs

    def _lane_angle(self, region, res):
        """lane_angle parameter in radians. auto = the angle (5 deg steps) that needs the fewest
        lane pieces, i.e. the fewest turns and joins; for a rectangular room that is along the
        longest side. Ties go to the angle nearest to the map axes."""
        val = str(self.gp('lane_angle')).strip().lower()
        if val != 'auto':
            return math.radians(float(val))
        best = None
        for deg in range(0, 180, 5):
            a = math.radians(deg)
            pieces = sum(len(r) for r in self._lane_runs(region, a, res)[4])
            off_axis = min(deg % 90, 90 - deg % 90)
            key = (pieces, off_axis)
            if best is None or key < best[0]:
                best = (key, a)
        return best[1]

    def _lane_cells(self, region, res):
        """Cells (lists of lanes (v, u_start, u_end) in metres) for lanes at lane_angle; v is the
        offset across the lanes, u the position along them. self.to_world maps (v, u) -> (x, y)."""
        a = self._lane_angle(region, res)
        self.lane_angle_used = a
        d, nrm, lane_v, us, lane_runs = self._lane_runs(region, a, res)
        self.to_world = lambda v, u: (float(u * d[0] + v * nrm[0]), float(u * d[1] + v * nrm[1]))
        return [[(lane_v[li], float(us[ia]), float(us[ib])) for (li, ia, ib) in cell]
                for cell in decompose_cells(lane_runs)]

    def _cell_path(self, lanes, reverse_order, first_dir, lane_clear, turn_clear, step):
        """Smooth path through one cell: lanes alternate direction, joined by half-circle
        U-turns where they fit (else a straight join is left for the planner)."""
        lanes = lanes[::-1] if reverse_order else lanes
        dirs = [first_dir * (1 if i % 2 == 0 else -1) for i in range(len(lanes))]
        # lane start/end along u, in travel direction
        se = [[(a, b) if d > 0 else (b, a)][0] for (v, a, b), d in zip(lanes, dirs)]
        se = [list(x) for x in se]

        def clear(v, u):
            x, y = self.to_world(v, u)
            return self.dist_at(x, y) if self.cost_ok(x, y) else 0.0

        # entry / exit of the cell need room to turn: move them inward until turn_clear
        def trim(i, which, d):
            v = lanes[i][0]
            u = se[i][which]
            sign = d if which == 0 else -d
            limit = se[i][1 - which]
            while clear(v, u) < turn_clear and (limit - u) * sign > self.gp('min_segment_len'):
                u += sign * step
            se[i][which] = u
        trim(0, 0, dirs[0])
        trim(len(lanes) - 1, 1, dirs[-1])

        pieces, subpaths = [], []                     # arrays of (v, u); finished sub-paths
        for i in range(len(lanes)):
            v = lanes[i][0]
            u0, u1 = se[i]
            if i + 1 < len(lanes):
                d = dirs[i]
                v2 = lanes[i + 1][0]
                R = abs(v2 - v) / 2
                uT0 = min(u1 * d, se[i + 1][0] * d) * d   # inner-most of the two lane ends
                phis = np.linspace(-math.pi / 2, math.pi / 2, max(8, int(math.pi * R / step)))
                fitted = None
                uT = uT0 - d * R
                while (uT - u0) * d > self.gp('min_segment_len') / 2:
                    arc = np.column_stack([(v + v2) / 2 + (v2 - v) / 2 * np.sin(phis),
                                           uT + d * R * np.cos(phis)])
                    if all(clear(a, b) >= turn_clear for a, b in arc):
                        fitted = (uT, arc)
                        break
                    uT -= d * step
                if fitted:
                    uT, arc = fitted
                    pieces.append(self._line(v, u0, uT, step))
                    pieces.append(arc)
                    se[i + 1][0] = uT
                    continue
                # No room for a smooth U-turn here: finish this sub-path after the lane; Nav2's
                # planner joins it to the next lane live (no sharp sideways jump in the path).
                if (u1 - u0) * dirs[i] > step:
                    pieces.append(self._line(v, u0, u1, step))
                if pieces:
                    subpaths.append(pieces)
                pieces = []
                continue
            if (u1 - u0) * dirs[i] > step:
                pieces.append(self._line(v, u0, u1, step))
            # else: the U-turn already brought the robot past this (trimmed) lane end; a line
            # would run backwards and leave a hook in the path, so the lane is skipped.
        if pieces:
            subpaths.append(pieces)
        out = []
        for pcs in subpaths:
            pcs = [p for p in pcs if len(p)]
            if not pcs:
                continue
            xy = np.array([self.to_world(v, u) for v, u in np.vstack(pcs)])
            xy = resample(xy, step)
            if len(xy) >= 2:
                out.append([(x, y, a) for (x, y), a in zip(xy, headings(xy))])
        return out

    @staticmethod
    def _line(v, ua, ub, step):
        n = max(2, int(abs(ub - ua) / step) + 1)
        return np.column_stack([np.full(n, v), np.linspace(ua, ub, n)])

    def _order_cells(self, cells, here):
        """Nearest-first order; each cell entered at whichever corner is closest."""
        lane_clear, turn_clear = self._clearances()
        step = self.gp('path_resolution')
        todo = list(cells)
        out = []
        while todo:
            best = None
            for ci, lanes in enumerate(todo):
                for rev in (False, True):
                    first = lanes[-1] if rev else lanes[0]
                    for d in (1, -1):
                        u = first[1] if d > 0 else first[2]
                        x, y = self.to_world(first[0], u)
                        dd = 0.0 if here is None else math.hypot(x - here[0], y - here[1])
                        if best is None or dd < best[0]:
                            best = (dd, ci, rev, d)
            _, ci, rev, d = best
            for p in self._cell_path(todo.pop(ci), rev, d, lane_clear, turn_clear, step):
                out.append(p)
                here = p[-1][:2]
        return out

    def _publish_path(self):
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = [self.pose_msg(*p) for p in self.path]
        self.path_pub.publish(msg)

    # ------------------------------------------------------------------ services
    def _srv_plan(self, req, resp):
        resp.success, resp.message = self._plan()
        return resp

    def _srv_start(self, req, resp):
        if self.worker is not None and self.worker.is_alive():
            resp.success, resp.message = False, 'Coverage already running (call /coverage/stop)'
            return resp
        if not self.path:
            ok, msg = self._plan()
            if not ok:
                resp.success, resp.message = False, msg
                return resp
        if not self.follow_client.wait_for_server(timeout_sec=3.0):
            resp.success, resp.message = False, 'Nav2 follow_path action server not available'
            return resp
        self.stop_flag.clear()
        if self.grid is not None:
            self.covered = np.zeros(self.grid.shape, dtype=bool)   # each run counts from zero
        self.worker = threading.Thread(target=self._execute, daemon=True)
        self.worker.start()
        resp.success, resp.message = True, f'Started: {len(self.path)} poses'
        return resp

    def _srv_stop(self, req, resp):
        self.stop_flag.set()
        with self.lock:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
        resp.success, resp.message = True, 'Stopping'
        return resp

    # ------------------------------------------------------------------ execution
    def _nearest_index(self, start, window):
        pose = self.robot_pose()
        if pose is None:
            return start
        seg = self.path[start:start + window]
        if not seg:
            return start
        d = [math.hypot(p[0] - pose[0], p[1] - pose[1]) for p in seg]
        return start + int(np.argmin(d))

    def _follow(self, poses):
        """Send poses to the controller and block until it finishes. Returns status."""
        g = FollowPath.Goal()
        g.path.header.frame_id = self.global_frame
        g.path.header.stamp = self.get_clock().now().to_msg()
        g.path.poses = [self.pose_msg(*p) for p in poses]
        g.controller_id = self.gp('controller_id')
        g.goal_checker_id = self.gp('goal_checker_id')
        f = self.follow_client.send_goal_async(g)
        if not self._wait(f, 10.0) or not f.result().accepted:
            return GoalStatus.STATUS_ABORTED
        with self.lock:
            self.goal_handle = f.result()
        rf = self.goal_handle.get_result_async()
        while not rf.done():
            if self.stop_flag.is_set():
                self.goal_handle.cancel_goal_async()
            time.sleep(0.1)
            self.progress = self._nearest_index(self.progress, 80)
        with self.lock:
            self.goal_handle = None
        return rf.result().status

    def _sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.stop_flag.is_set():
            time.sleep(0.1)

    def _xy(self, i):
        p = self.path[min(i, len(self.path) - 1)]
        return f'({p[0]:.2f}, {p[1]:.2f})'

    def _execute(self):
        step = self.gp('path_resolution')
        wait_s = float(self.gp('blocked_wait'))
        self.progress = 0
        self.aborts = 0
        t0 = time.time()
        skipped = []                                    # revisit queue: (first, last) path index
        retried_at = None
        approach = self.plan_between(None, self.path[0]) or []      # get onto the path
        target = 0
        while not self.stop_flag.is_set():
            status = self._follow(approach + self.path[target:])
            if status == GoalStatus.STATUS_SUCCEEDED or self.stop_flag.is_set():
                break
            self.aborts += 1
            here = self._nearest_index(self.progress, 100)
            # Blocked (a person, a moved chair...): wait for the way to clear, then let Nav2 plan
            # from the robot to a point a bit further on - around the obstacle if there is room.
            self.get_logger().warn(f'Controller aborted (status {status}) at {self._xy(here)} '
                                   f'(abort #{self.aborts}); waiting {wait_s:.0f} s for the way to clear')
            self._sleep(wait_s)
            if retried_at is not None and abs(here - retried_at) < int(1.0 / step):
                target = here + int(1.0 / step)                     # still blocked: skip 1 m
                what = 'still blocked: skipping 1 m (queued for a revisit at the end)'
                retried_at = None
            else:
                target = here + int(0.3 / step)                     # first time: retry
                what = 're-planning to 0.3 m further along (around the obstacle if there is room)'
                retried_at = here
            self.get_logger().warn(what)
            # If Nav2 cannot plan to the target, try further along instead of sending the path
            # blindly (the controller would only abort again straight away).
            approach = None
            while target < len(self.path) - 1 and not self.stop_flag.is_set():
                approach = self.plan_between(None, self.path[target])
                if approach is not None:
                    break
                target += int(0.5 / step)
            if target > here + int(0.3 / step):
                skipped.append((here, min(target, len(self.path) - 1)))
            if approach is None:
                self.get_logger().error('Cannot plan back onto the coverage path; stopping.')
                break
            if self.aborts >= 10 and self.progress == target:
                self.get_logger().error('Too many aborts without progress; stopping.')
                break
            self.progress = target

        # Revisit queue: try every skipped piece once more (the obstacle may have gone).
        merged = []
        for a_, b_ in sorted(skipped):
            if merged and a_ <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b_))
            else:
                merged.append((a_, b_))
        still, revisited = [], 0
        for a_, b_ in merged:
            if self.stop_flag.is_set():
                still.append((a_, b_))
                continue
            self.get_logger().info(f'Revisiting skipped piece {self._xy(a_)} -> {self._xy(b_)}')
            appr = self.plan_between(None, self.path[a_])
            ok = appr is not None and self._follow(appr + self.path[a_:b_ + 1]) == GoalStatus.STATUS_SUCCEEDED
            if ok:
                revisited += 1
            else:
                still.append((a_, b_))
        mins = (time.time() - t0) / 60.0
        state = 'stopped' if self.stop_flag.is_set() else 'finished'
        report = (f'Coverage {state} after {mins:.1f} min, {self.aborts} controller aborts, covered '
                  f'{self._percent():.0f}% of the reachable area; revisited {revisited} of '
                  f'{len(merged)} skipped piece(s)')
        if still:
            report += '; STILL SKIPPED: ' + ', '.join(f'{self._xy(a_)} -> {self._xy(b_)}' for a_, b_ in still)
            self.get_logger().warn(report)
        else:
            self.get_logger().info(report)
        self.report_pub.publish(String(data=report))

    # ------------------------------------------------------------------ coverage tracking
    def _track_coverage(self):
        if self.map_msg is None or self.covered is None:
            return
        pose = self.robot_pose()
        if pose is None:
            return
        r0, c0 = self.cell_of(pose[0], pose[1])
        rad = int(round(self.gp('coverage_radius') / self.map_msg.info.resolution))
        h, w = self.covered.shape
        r_lo, r_hi, c_lo, c_hi = max(0, r0 - rad), min(h, r0 + rad + 1), max(0, c0 - rad), min(w, c0 + rad + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return
        rr, cc = np.ogrid[r_lo:r_hi, c_lo:c_hi]
        self.covered[r_lo:r_hi, c_lo:c_hi] |= (rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2

    def _percent(self):
        if self.coverable is None or not self.coverable.any():
            return 0.0
        return 100.0 * (self.covered & self.coverable).sum() / self.coverable.sum()

    def _publish_coverage_grid(self):
        if self.map_msg is None or self.coverable is None:
            return
        out = OccupancyGrid()
        out.header.frame_id = self.global_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = self.map_msg.info
        data = np.full(self.grid.shape, -1, dtype=np.int8)
        data[self.coverable] = 0
        data[self.coverable & self.covered] = 100
        out.data = data.flatten().tolist()
        self.cov_pub.publish(out)
        if self.worker is not None and self.worker.is_alive():
            self.get_logger().info(f'Covered {self._percent():.0f}%', throttle_duration_sec=10.0)


def main():
    rclpy.init()
    node = CoveragePlanner()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_flag.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

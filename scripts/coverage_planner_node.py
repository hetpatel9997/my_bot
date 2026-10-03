#!/usr/bin/env python3
"""
coverage_planner_node.py  -  lawn-mower (boustrophedon) coverage for a mapped room.

How it works
  1. Reads the map from /map (the one map_server publishes for AMCL / Nav2).
  2. Shrinks free space by robot_radius + margin so lanes stay clear of walls.
  3. Keeps only the free region the robot is standing in (never plans into other rooms).
  4. Lays parallel lanes along the longer side of that region, lane_spacing apart,
     splits each lane wherever an obstacle crosses it, and orders the pieces so the
     robot sweeps back and forth like a mower.
  5. Publishes the plan on /coverage_path (nav_msgs/Path) for RViz.
  6. On /coverage/start it drives the lanes one NavigateToPose goal at a time, skipping
     a goal if Nav2 gives up on it, and keeps a /coverage_grid showing what has been
     covered (cells within coverage_radius of the robot).

Services (std_srvs/Trigger):
  /coverage/plan    compute and publish the path, do not move
  /coverage/start   plan if needed, then start driving
  /coverage/stop    cancel the current goal and stop

Parameters: lane_spacing, robot_radius, margin, coverage_radius, min_segment_len,
            map_topic, global_frame, base_frame
"""
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


# --------------------------------------------------------------------------- geometry
def erode(mask, r):
    """Shrink a boolean mask by r cells (8-connected)."""
    m = mask.copy()
    for _ in range(r):
        p = np.pad(m, 1, constant_values=False)
        m = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
             & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:])
    return m


def component(mask, seed):
    """Boolean mask of the 4-connected region of `mask` containing seed=(row, col)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    r0, c0 = seed
    if not (0 <= r0 < h and 0 <= c0 < w) or not mask[r0, c0]:
        return out
    q = deque([(r0, c0)])
    out[r0, c0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def largest_component(mask):
    best = None
    seen = np.zeros_like(mask)
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        comp = component(mask, (r, c))
        seen |= comp
        if best is None or comp.sum() > best.sum():
            best = comp
    return best if best is not None else np.zeros_like(mask)


def runs(line):
    """Start/end (inclusive) indices of consecutive True runs in a 1-D boolean array."""
    out = []
    start = None
    for i, v in enumerate(line):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(line) - 1))
    return out


def plan_lanes(free, spacing_cells, min_len_cells):
    """
    free: boolean grid [row, col] of plannable cells.
    Returns a list of segments ((r0, c0), (r1, c1)) in sweep order, each one lane piece,
    already oriented in the direction of travel.
    """
    rows, cols = np.nonzero(free)
    if len(rows) == 0:
        return []
    r_min, r_max, c_min, c_max = rows.min(), rows.max(), cols.min(), cols.max()
    along_cols = (c_max - c_min) >= (r_max - r_min)   # lanes parallel to the longer side

    segments = []
    if along_cols:
        lane_rows = range(r_min + spacing_cells // 2, r_max + 1, spacing_cells)
        for i, r in enumerate(lane_rows):
            pieces = [(a, b) for a, b in runs(free[r, :]) if b - a + 1 >= min_len_cells]
            if i % 2 == 1:
                pieces = [(b, a) for a, b in reversed(pieces)]
            for a, b in pieces:
                segments.append(((r, a), (r, b)))
    else:
        lane_cols = range(c_min + spacing_cells // 2, c_max + 1, spacing_cells)
        for i, c in enumerate(lane_cols):
            pieces = [(a, b) for a, b in runs(free[:, c]) if b - a + 1 >= min_len_cells]
            if i % 2 == 1:
                pieces = [(b, a) for a, b in reversed(pieces)]
            for a, b in pieces:
                segments.append(((a, c), (b, c)))
    return segments


# --------------------------------------------------------------------------- node
class CoveragePlanner(Node):
    def __init__(self):
        super().__init__('coverage_planner')
        self.declare_parameter('lane_spacing', 0.40)      # m between lanes
        self.declare_parameter('robot_radius', 0.18)      # m, half of the widest dimension
        self.declare_parameter('margin', 0.10)            # m extra distance from walls
        self.declare_parameter('coverage_radius', 0.20)   # m, how wide a pass "covers"
        self.declare_parameter('min_segment_len', 0.30)   # m, skip tiny lane pieces
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.map_msg = None
        self.grid = None
        self.plannable = None       # bool grid used for the plan
        self.covered = None         # bool grid of visited cells
        self.waypoints = []         # list of PoseStamped in sweep order
        self.next_index = 0
        self.active = False
        self.goal_handle = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value,
                                 self._map_cb, map_qos)
        self.path_pub = self.create_publisher(Path, 'coverage_path', map_qos)
        self.cov_pub = self.create_publisher(OccupancyGrid, 'coverage_grid', map_qos)

        self.create_service(Trigger, 'coverage/plan', self._srv_plan)
        self.create_service(Trigger, 'coverage/start', self._srv_start)
        self.create_service(Trigger, 'coverage/stop', self._srv_stop)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(0.5, self._track_coverage)
        self.create_timer(2.0, self._publish_coverage_grid)
        self.get_logger().info('coverage_planner ready: call /coverage/plan or /coverage/start')

    # ----------------------------------------------------------------- map / robot pose
    def _map_cb(self, msg):
        self.map_msg = msg
        self.grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        if self.covered is None or self.covered.shape != self.grid.shape:
            self.covered = np.zeros(self.grid.shape, dtype=bool)
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height} '
                               f'@ {msg.info.resolution:.3f} m')

    def _robot_cell(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        except Exception:
            return None
        x, y = t.transform.translation.x, t.transform.translation.y
        info = self.map_msg.info
        c = int((x - info.origin.position.x) / info.resolution)
        r = int((y - info.origin.position.y) / info.resolution)
        return (r, c)

    def _cell_to_pose(self, r, c, yaw):
        info = self.map_msg.info
        p = PoseStamped()
        p.header.frame_id = self.global_frame
        p.pose.position.x = info.origin.position.x + (c + 0.5) * info.resolution
        p.pose.position.y = info.origin.position.y + (r + 0.5) * info.resolution
        p.pose.orientation.z = math.sin(yaw / 2.0)
        p.pose.orientation.w = math.cos(yaw / 2.0)
        return p

    # ----------------------------------------------------------------- planning
    def _plan(self):
        if self.map_msg is None:
            return False, 'No map received yet (is map_server running?)'
        res = self.map_msg.info.resolution
        free = self.grid == 0                                   # unknown (-1) counts as blocked
        shrink = int(math.ceil((self.get_parameter('robot_radius').value
                                + self.get_parameter('margin').value) / res))
        free = erode(free, shrink)
        seed = self._robot_cell()
        region = component(free, seed) if seed is not None else None
        if region is None or not region.any():
            self.get_logger().warn('Robot pose unknown or outside free space; using the largest free region')
            region = largest_component(free)
        if not region.any():
            return False, 'No plannable free space after shrinking for the robot size'
        self.plannable = region

        spacing = max(1, int(round(self.get_parameter('lane_spacing').value / res)))
        min_len = max(1, int(round(self.get_parameter('min_segment_len').value / res)))
        segments = plan_lanes(region, spacing, min_len)
        if not segments:
            return False, 'Region too small for even one lane'

        self.waypoints = []
        for (r0, c0), (r1, c1) in segments:
            yaw = math.atan2(r1 - r0, c1 - c0)      # rows = y, cols = x
            self.waypoints.append(self._cell_to_pose(r0, c0, yaw))
            self.waypoints.append(self._cell_to_pose(r1, c1, yaw))
        self.next_index = 0

        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = self.waypoints
        self.path_pub.publish(path)
        total_m = sum(math.hypot(b.pose.position.x - a.pose.position.x,
                                 b.pose.position.y - a.pose.position.y)
                      for a, b in zip(self.waypoints[0::2], self.waypoints[1::2]))
        msg = f'{len(segments)} lane pieces, {total_m:.1f} m of lanes, {len(self.waypoints)} waypoints'
        self.get_logger().info('Plan: ' + msg)
        return True, msg

    # ----------------------------------------------------------------- services
    def _srv_plan(self, req, resp):
        resp.success, resp.message = self._plan()
        return resp

    def _srv_start(self, req, resp):
        if not self.waypoints:
            ok, msg = self._plan()
            if not ok:
                resp.success, resp.message = False, msg
                return resp
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            resp.success, resp.message = False, 'Nav2 navigate_to_pose action server not available'
            return resp
        self.active = True
        self.next_index = 0
        self._send_next()
        resp.success, resp.message = True, f'Started: {len(self.waypoints)} waypoints'
        return resp

    def _srv_stop(self, req, resp):
        self.active = False
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        resp.success, resp.message = True, 'Stopped'
        return resp

    # ----------------------------------------------------------------- execution
    def _send_next(self):
        if not self.active:
            return
        if self.next_index >= len(self.waypoints):
            self.active = False
            self.get_logger().info(f'Coverage finished. Covered {self._percent():.0f}% of the planned area.')
            return
        goal = NavigateToPose.Goal()
        goal.pose = self.waypoints[self.next_index]
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f'Goal {self.next_index + 1}/{len(self.waypoints)}: '
                               f'({goal.pose.pose.position.x:.2f}, {goal.pose.pose.position.y:.2f})')
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._goal_response)

    def _goal_response(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected, skipping')
            self.next_index += 1
            self._send_next()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result)

    def _goal_result(self, fut):
        status = fut.result().status
        if status != 4:   # 4 = SUCCEEDED
            self.get_logger().warn(f'Goal ended with status {status}, skipping to the next one')
        self.next_index += 1
        self.goal_handle = None
        self._send_next()

    # ----------------------------------------------------------------- coverage tracking
    def _track_coverage(self):
        if self.map_msg is None:
            return
        cell = self._robot_cell()
        if cell is None:
            return
        r0, c0 = cell
        rad = int(round(self.get_parameter('coverage_radius').value / self.map_msg.info.resolution))
        h, w = self.covered.shape
        rr, cc = np.ogrid[max(0, r0 - rad):min(h, r0 + rad + 1), max(0, c0 - rad):min(w, c0 + rad + 1)]
        disc = (rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2
        self.covered[max(0, r0 - rad):min(h, r0 + rad + 1), max(0, c0 - rad):min(w, c0 + rad + 1)] |= disc

    def _percent(self):
        if self.plannable is None or not self.plannable.any():
            return 0.0
        return 100.0 * (self.covered & self.plannable).sum() / self.plannable.sum()

    def _publish_coverage_grid(self):
        if self.map_msg is None or self.plannable is None:
            return
        out = OccupancyGrid()
        out.header.frame_id = self.global_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = self.map_msg.info
        data = np.full(self.grid.shape, -1, dtype=np.int8)
        data[self.plannable] = 0
        data[self.plannable & self.covered] = 100
        out.data = data.flatten().tolist()
        self.cov_pub.publish(out)
        if self.active:
            self.get_logger().info(f'Covered {self._percent():.0f}%', throttle_duration_sec=10.0)


def main():
    rclpy.init()
    node = CoveragePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
keepout_zones_node.py  -  pool guard: keepout zones file -> Nav2 keepout filter mask.

Reads a zones file (e.g. config/backyard_zones.yaml: polygons in the map frame, each with a band
`margin` around it) and publishes, latched:
  /keepout_filter_mask   nav_msgs/OccupancyGrid: 100 inside zone + band, -1 (= leave alone)
                         everywhere else; covers only the zones' bounding box
  /costmap_filter_info   nav2_msgs/CostmapFilterInfo: type 0 (keepout), points to the mask
Nav2's KeepoutFilter (global + local costmap, nav2_params.yaml) then marks those cells lethal:
the planner never routes through them, RPP's footprint check stops before the outline enters,
and the coverage planner treats the mask as a wall.

A zone of kind `pool` needs margin >= min_pool_margin (1.0 m, CLAUDE.md hard rule: the LiDAR
cannot see water); otherwise the node refuses to start and Nav2 has no keepout at all (the
coverage planner with require_keepout then refuses to plan). With no zones file (indoor) it
publishes an empty mask, so the filter is satisfied and changes nothing.

Started by launch/keepout.launch.py (included by navigation_launch.py, arg zones_file).
"""
import math
import os

import numpy as np
import yaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid


def polygon_distance(poly, x, y):
    """Distance (m) from points (x, y arrays) to a closed polygon; 0 inside it."""
    d = np.full(x.shape, np.inf)
    inside = np.zeros(x.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        d = np.minimum(d, np.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
        inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * dx / (dy or 1e-12))   # even-odd
    d[inside] = 0.0
    return d


class KeepoutZones(Node):
    def __init__(self):
        super().__init__('keepout_zones')
        self.declare_parameter('zones_file', '')            # '' = no zones (indoor)
        self.declare_parameter('min_pool_margin', 1.0)      # m, refuse a pool zone with less
        self.declare_parameter('mask_topic', '/keepout_filter_mask')
        self.declare_parameter('filter_info_topic', '/costmap_filter_info')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        mask_topic = self.get_parameter('mask_topic').value
        self.mask_pub = self.create_publisher(OccupancyGrid, mask_topic, latched)
        self.info_pub = self.create_publisher(CostmapFilterInfo,
                                              self.get_parameter('filter_info_topic').value, latched)
        mask = self._make_mask(self.get_parameter('zones_file').value)
        info = CostmapFilterInfo()
        info.header = mask.header
        info.type = 0                                       # keepout
        info.filter_mask_topic = mask_topic
        info.base, info.multiplier = 0.0, 1.0
        self.info_pub.publish(info)
        self.mask_pub.publish(mask)

    def _make_mask(self, path):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        cfg = yaml.safe_load(open(path)) if path else {}
        zones = (cfg or {}).get('zones') or []
        res = float((cfg or {}).get('resolution', 0.05))
        msg.header.frame_id = (cfg or {}).get('frame_id', 'map')
        msg.info.resolution = res
        msg.info.origin.orientation.w = 1.0
        if not zones:
            msg.info.width = msg.info.height = 1
            msg.data = [-1]
            self.get_logger().info('No keepout zones' + (f' in {path}' if path else ' (no zones_file)') +
                                   ': empty keepout mask (indoor)')
            return msg
        polys = []
        for z in zones:
            poly = np.array(z['polygon'], dtype=float)
            margin = float(z.get('margin', 0.0))
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2 or margin < 0.0:
                raise ValueError(f'zone {z.get("name")}: polygon needs >= 3 [x, y] corners and '
                                 f'margin >= 0')
            if z.get('kind') == 'pool' and margin < self.get_parameter('min_pool_margin').value:
                raise ValueError(f'pool zone {z.get("name")}: margin {margin:.2f} m is less than '
                                 f'min_pool_margin {self.get_parameter("min_pool_margin").value:.2f} m '
                                 f'(CLAUDE.md: keepout around the pool with at least 1 m)')
            polys.append((z.get('name', '?'), poly, margin))
        lo = np.min([p.min(0) - m for _, p, m in polys], axis=0) - 2 * res
        hi = np.max([p.max(0) + m for _, p, m in polys], axis=0) + 2 * res
        w, h = (np.ceil((hi - lo) / res)).astype(int)
        xs = lo[0] + (np.arange(w) + 0.5) * res
        ys = lo[1] + (np.arange(h) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        keep = np.zeros(X.shape, dtype=bool)
        for name, poly, margin in polys:
            # a cell is keepout if ANY part of it is within the band (centre within margin + half
            # a cell diagonal): the costmap's cells do not line up with the mask's
            zone = polygon_distance(poly, X, Y) <= margin + res / math.sqrt(2)
            keep |= zone
            span = poly.max(0) - poly.min(0) + 2 * margin
            self.get_logger().info(f'Keepout zone {name}: {poly.max(0)[0] - poly.min(0)[0]:.1f} x '
                                   f'{poly.max(0)[1] - poly.min(0)[1]:.1f} m + {margin:.3f} m band '
                                   f'-> {span[0]:.1f} x {span[1]:.1f} m, {zone.sum()} cells')
        msg.info.width, msg.info.height = int(w), int(h)
        msg.info.origin.position.x, msg.info.origin.position.y = float(lo[0]), float(lo[1])
        msg.data = np.where(keep, 100, -1).astype(np.int8).flatten().tolist()
        self.get_logger().info(f'Keepout mask from {os.path.basename(path)}: {w}x{h} cells @ {res} m, '
                               f'origin ({lo[0]:.2f}, {lo[1]:.2f}) in {msg.header.frame_id}')
        return msg


def main():
    rclpy.init()
    try:
        node = KeepoutZones()
    except (ValueError, KeyError, OSError, yaml.YAMLError) as e:
        rclpy.logging.get_logger('keepout_zones').fatal(f'Refusing to start: {e}')
        rclpy.shutdown()
        raise SystemExit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

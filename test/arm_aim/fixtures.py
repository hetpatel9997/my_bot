#!/usr/bin/env python3
"""fixtures.py - test helpers for the arm tests in the empty Gazebo test world (SIMULATION ONLY).

Why a fake map: safety_state_node reports `slow` while there is no /map (by design: without a map
every obstacle is "unknown"), and the spray controller sprays only while /safety_state is `clear`.
In the empty test world (test/arm_aim/arm_test.world) there is nothing to map, so the test hands
it an empty map instead of running SLAM:
  - /map: a 10 x 10 m OccupancyGrid, 0.05 m cells, every cell free (0), origin (-5, -5), frame map,
    published latched (transient local), so safety_state gets it whenever it (re)starts;
  - TF map -> odom: identity, on /tf_static. Static on purpose: the spray controller refuses a
    DYNAMIC map -> odom older than 1 s ("keepout check impossible: transform ... is old").
With nothing near the robot, /safety_state then turns `clear`.

Use from a test:   start_map_fixture(node)   (keep the node spinning; the latched messages live
                                             as long as the node does)
Owner demo:        python3 ~/dev_ws/src/my_bot/test/arm_aim/fixtures.py     (Ctrl-C to stop)
Never next to a real map (SLAM / AMCL / map_server): two map sources would fight.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import StaticTransformBroadcaster

MAP_SIZE_M = 10.0
MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-5.0, -5.0)


def start_map_fixture(node: Node) -> None:
    """Publish the empty latched /map and the static TF map -> odom from `node`."""
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    map_pub = node.create_publisher(OccupancyGrid, '/map', latched)
    cells = int(round(MAP_SIZE_M / MAP_RESOLUTION))
    grid = OccupancyGrid()
    grid.header.frame_id = 'map'
    grid.header.stamp = node.get_clock().now().to_msg()
    grid.info.map_load_time = grid.header.stamp
    grid.info.resolution = MAP_RESOLUTION
    grid.info.width = grid.info.height = cells
    grid.info.origin.position.x, grid.info.origin.position.y = MAP_ORIGIN
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * (cells * cells)          # 0 = free
    map_pub.publish(grid)

    tf_pub = StaticTransformBroadcaster(node)  # publishes on /tf_static (latched)
    t = TransformStamped()
    t.header.stamp = grid.header.stamp
    t.header.frame_id, t.child_frame_id = 'map', 'odom'
    t.transform.rotation.w = 1.0               # identity: map = odom
    tf_pub.sendTransform(t)
    # Keep the publishers alive with the node: latched data disappears with its publisher.
    node._map_fixture = (map_pub, tf_pub)


def main():
    rclpy.init()
    node = Node('arm_test_map_fixture')
    start_map_fixture(node)
    node.get_logger().info('empty /map (10 x 10 m) and TF map -> odom published; Ctrl-C to stop')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

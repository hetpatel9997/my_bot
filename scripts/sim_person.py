#!/usr/bin/env python3
"""
sim_person.py  -  SIMULATION ONLY: drop, move and remove a "person" in Gazebo.

The person is a cylinder 0.4 m wide and 1.7 m tall that the LiDAR sees and the robot cannot push
(kinematic). Coordinates are Gazebo world metres (in room.world the robot spawns at 0, 0 and the
saved room_map is aligned with it).

Relative to the robot (uses Gazebo's true robot pose, /ground_truth/odom):
  ros2 run my_bot sim_person.py spawn --ahead 0.5          # 0.5 m in front of the robot's centre
  ros2 run my_bot sim_person.py spawn --ahead 1.0 --left 0.3   # 1 m ahead, 0.3 m to its left
  ros2 run my_bot sim_person.py cross --ahead 1.0 --speed 0.2  # walk slowly across the robot's
                                     # path, 1 m ahead, from 1 m left to 1 m right (--width 1.0)
Absolute (Gazebo world metres):
  ros2 run my_bot sim_person.py spawn X Y                  # drop the person at X, Y
  ros2 run my_bot sim_person.py move X1 Y1 X2 Y2 [SPEED]   # walk from X1,Y1 to X2,Y2 (default 0.5 m/s)
  ros2 run my_bot sim_person.py remove                     # take it away
The person stands still wherever it was put or stopped walking; `spawn` on an existing person
moves it there. Add --name NAME to use several people (default: person).

Needs the sim started with worlds/room.world (it has the gazebo_ros_state plugin) and the same
ROS environment as the sim (run `ros_local` first in that terminal).
"""
import argparse
import math
import sys
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from rclpy.node import Node

SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <link name="body">
      <kinematic>true</kinematic>
      <gravity>false</gravity>
      <pose>0 0 0.85 0 0 0</pose>
      <collision name="c"><geometry><cylinder><radius>0.2</radius><length>1.7</length></cylinder></geometry></collision>
      <visual name="v"><geometry><cylinder><radius>0.2</radius><length>1.7</length></cylinder></geometry>
        <material><ambient>0.9 0.4 0.1 1</ambient><diffuse>0.9 0.4 0.1 1</diffuse></material></visual>
    </link>
  </model>
</sdf>"""


class Person(Node):
    def __init__(self, name):
        super().__init__('sim_person')
        self.name = name
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        self.state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')

    def _call(self, cli, req, what, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            sys.exit(f'{what}: service {cli.srv_name} not available (is the sim running, same ROS env?)')
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=timeout)
        if not f.done():
            sys.exit(f'{what}: no answer from {cli.srv_name}')
        return f.result()

    def robot_pose(self, timeout=5.0):
        """Gazebo's true robot pose (x, y, yaw) from /ground_truth/odom (sim only)."""
        got = []
        sub = self.create_subscription(Odometry, '/ground_truth/odom', got.append, 10)
        end = time.time() + timeout
        while not got and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        if not got:
            sys.exit('No /ground_truth/odom: is the sim running (launch_sim) in this ROS environment?')
        p, q = got[-1].pose.pose.position, got[-1].pose.pose.orientation
        return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def relative(self, ahead, left):
        """World point `ahead` m in front of and `left` m to the left of the robot's centre."""
        x, y, yaw = self.robot_pose()
        return (x + ahead * math.cos(yaw) - left * math.sin(yaw),
                y + ahead * math.sin(yaw) + left * math.cos(yaw))

    def spawn(self, x, y):
        if self.place(x, y, quiet=True):            # already in the world: just move it there
            print(f'moved {self.name} to ({x:.2f}, {y:.2f})')
            return True
        r = SpawnEntity.Request()
        r.name, r.xml = self.name, SDF.format(name=self.name)
        r.initial_pose.position.x, r.initial_pose.position.y = float(x), float(y)
        res = self._call(self.spawn_cli, r, 'spawn')
        ok = res.success or 'already exists' in res.status_message
        print(f'spawn {self.name} at ({x:.2f}, {y:.2f}): {"ok" if ok else res.status_message}')
        return ok

    def place(self, x, y, quiet=False):
        r = SetEntityState.Request()
        r.state = EntityState(name=self.name)
        r.state.pose.position.x, r.state.pose.position.y = float(x), float(y)
        r.state.pose.orientation.w = 1.0
        r.state.reference_frame = 'world'
        return self._call(self.state_cli, r, 'move', timeout=5.0).success

    def move(self, x1, y1, x2, y2, speed):
        if not self.place(x1, y1):
            self.spawn(x1, y1)
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(dist / max(speed, 0.05) / 0.1))
        t0 = time.time()
        for i in range(1, steps + 1):
            f = i / steps
            self.place(x1 + f * (x2 - x1), y1 + f * (y2 - y1))
            time.sleep(max(0.0, t0 + i * 0.1 - time.time()))
        print(f'moved {self.name} ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}) in {time.time() - t0:.1f} s')

    def remove(self):
        r = DeleteEntity.Request()
        r.name = self.name
        res = self._call(self.delete_cli, r, 'remove')
        print(f'remove {self.name}: {"ok" if res.success else res.status_message}')


def main():
    ap = argparse.ArgumentParser(description='Drop, move and remove a person in Gazebo (sim only).')
    ap.add_argument('--name', default='person')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('spawn', help='drop the person at X Y, or --ahead/--left of the robot')
    sp.add_argument('x', type=float, nargs='?'); sp.add_argument('y', type=float, nargs='?')
    sp.add_argument('--ahead', type=float, help='m in front of the robot centre')
    sp.add_argument('--left', type=float, default=0.0, help='m to the robot\'s left (negative = right)')
    cr = sub.add_parser('cross', help='walk across the robot\'s path')
    cr.add_argument('--ahead', type=float, default=1.0, help='m in front of the robot (default 1.0)')
    cr.add_argument('--width', type=float, default=1.0, help='start/end this far left/right (default 1.0)')
    cr.add_argument('--speed', type=float, default=0.2, help='m/s (default 0.2, slow walk)')
    mv = sub.add_parser('move')
    for a in ('x1', 'y1', 'x2', 'y2'):
        mv.add_argument(a, type=float)
    mv.add_argument('speed', type=float, nargs='?', default=0.5)
    sub.add_parser('remove')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    rclpy.init()
    node = Person(args.name)
    try:
        if args.cmd == 'spawn':
            if args.ahead is not None:
                x, y = node.relative(args.ahead, args.left)
            elif args.x is not None and args.y is not None:
                x, y = args.x, args.y
            else:
                sys.exit('spawn needs X Y or --ahead D [--left L]')
            node.spawn(x, y)
        elif args.cmd == 'cross':
            x1, y1 = node.relative(args.ahead, args.width)
            x2, y2 = node.relative(args.ahead, -args.width)
            node.move(x1, y1, x2, y2, args.speed)
        elif args.cmd == 'move':
            node.move(args.x1, args.y1, args.x2, args.y2, args.speed)
        else:
            node.remove()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

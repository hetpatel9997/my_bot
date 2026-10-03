#!/usr/bin/env python3
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import time

def create_pose(nav, x, y):
    """
    Helper function to cleanly format 2D coordinates 
    into a ROS 2 PoseStamped message.
    """
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.w = 1.0  # Tells the robot to point forward
    return pose

def main():
    rclpy.init()
    nav = BasicNavigator()

    print("=== Connecting to Nav2 Stack... ===")
    # This waits until AMCL, the planner, and the controllers are fully up and running
    nav.waitUntilNav2Active(localizer='bt_navigator')

    # ====================================================================
    # LAWNMOWER PARAMETERS: Change these values to match your room size!
    # ====================================================================
    start_x = 0.5       # X coordinate on your map where the grid should start
    start_y = 0.5       # Y coordinate on your map where the grid should start
    width = 2.0         # Total width of the sweeping grid (in meters)
    height = 3.0        # Total length/depth of the sweeping grid (in meters)
    row_spacing = 0.35  # Distance between sweeps (keep it slightly smaller than your robot's width)
    # ====================================================================

    print("=== Generating Lawnmower Grid Pattern Waypoints ===")
    waypoints = []
    current_y = start_y
    moving_right = True

    # Generates the zigzag rows mathematically
    while current_y <= start_y + height:
        if moving_right:
            waypoints.append(create_pose(nav, start_x, current_y))
            waypoints.append(create_pose(nav, start_x + width, current_y))
        else:
            waypoints.append(create_pose(nav, start_x + width, current_y))
            waypoints.append(create_pose(nav, start_x, current_y))
        
        current_y += row_spacing
        moving_right = not moving_right  # Alternate tracking directions for the next sweep row

    print(f"Generated {len(waypoints)} grid waypoints.")
    print("Launching Autonomous Lawnmower Coverage!")
    
    # Hand the entire array of points over to Nav2
    nav.goThroughPoses(waypoints)

    # Monitor the navigation actively
    loop_counter = 0
    while not nav.isTaskComplete():
        loop_counter += 1
        feedback = nav.getFeedback()
        
        # Print status updates every few seconds
        if feedback and loop_counter % 6 == 0:
            print(f"Distance remaining to finish pattern: {feedback.distance_remaining:.2f} meters.")
        
        time.sleep(0.5)
    
    # Check if we succeeded or if something stopped us
    result = nav.getResult()
    if result == 1: # TaskState.SUCCEEDED
        print("=== Success! Lawnmower path completed perfectly. ===")
    else:
        print("=== Mission Failed or Aborted by Nav2. ===")

    rclpy.shutdown()

if __name__ == '__main__':
    main()

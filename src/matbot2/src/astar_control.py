#!/usr/bin/env python3

import rospy
import tf2_ros
import tf
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist, Point
from tf.transformations import quaternion_from_euler
from tf2_geometry_msgs import do_transform_pose
from astar_rviz_test import AStarPlanner #AStarPlanner   Import your A* planner

# Initialize the A* planner
#planner = AStarPlanner()

def controller(waypoint):
    """
    Controls the TurtleBot to move to a specific waypoint.
    :param waypoint: Tuple (x, y) in world coordinates
    """
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    r = rospy.Rate(10)  # 10 Hz control loop

    # PID controller parameters
    Kp = np.diag([2.0, 0.8])  # Proportional gains
    Kd = np.diag([-0.5, 0.5])  # Derivative gains
    Ki = np.diag([0.0, 0.0])  # Integral gains

    prev_time = rospy.get_time()
    integ = np.array([0.0, 0.0])
    previous_error = np.array([0.0, 0.0])

    while not rospy.is_shutdown():
        try:
            # Transform between odom and base_link frames
            trans_odom_to_base_link = tfBuffer.lookup_transform('base_footprint', 'odom', rospy.Time(), rospy.Duration(5))
            (roll, pitch, baselink_yaw) = tf.transformations.euler_from_quaternion(
                [trans_odom_to_base_link.transform.rotation.x, trans_odom_to_base_link.transform.rotation.y,
                 trans_odom_to_base_link.transform.rotation.z, trans_odom_to_base_link.transform.rotation.w])

            # Transform waypoint to base_link frame
            waypoint_pose = PoseStamped()
            waypoint_pose.header.frame_id = "odom"
            waypoint_pose.pose.position.x = waypoint.pose.position.x
            waypoint_pose.pose.position.y = waypoint.pose.position.y
            waypoint_in_base_link = do_transform_pose(waypoint_pose, trans_odom_to_base_link)
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [waypoint_in_base_link.pose.orientation.x, waypoint_in_base_link.pose.orientation.y,
                 waypoint_in_base_link.pose.orientation.z, waypoint_in_base_link.pose.orientation.w])

            # Calculate errors
            x_error = waypoint_in_base_link.pose.position.x
            z_error = yaw
            error = np.array([-x_error, z_error])

            # Proportional term
            proportional = np.dot(Kp, error).squeeze()

            # Integral term
            curr_time = rospy.get_time()
            dt = curr_time - prev_time
            integ += error * dt
            integral = np.dot(Ki, integ).squeeze()

            # Derivative term
            error_deriv = (error - previous_error) / dt
            derivative = np.dot(Kd, error_deriv).squeeze()

            # Control command
            msg = Twist()
            msg.linear.x = float(proportional[0] + derivative[0] + integral[0])
            msg.angular.z = proportional[1] + derivative[1] + integral[1]

            # Publish control command
            pub.publish(msg)

            # Update previous error and time
            previous_error = error
            prev_time = curr_time

            # Check if we reached the waypoint
            if np.abs(error[0]) < 0.1:  # Tolerance for reaching the waypoint
                rospy.loginfo("Reached waypoint")
                return  # Move to the next waypoint

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
            pass

        r.sleep()


def planning_callback(msg):
    """
    Handles the planning request by generating a trajectory using A*.
    :param msg: Point message specifying the goal position
    """
    try:
        # Start position in world coordinates
        start = (0.6, 0.0)  # Assume the robot starts at (0, 0) in odom frame
        goal = (msg.x, msg.y)

        # Plan path using A*
        trajectory = planner.plan(start, goal)

        if trajectory:
            rospy.loginfo("Trajectory computed. Moving to waypoints...")
            for waypoint in trajectory.poses:
                controller(waypoint)  # Send each waypoint to the controller
        else:
            rospy.logwarn("No path found by A* planner.")

    except rospy.ROSInterruptException as e:
        rospy.logwarn(f"Exception in planning callback: {e}")


if __name__ == '__main__':
    #rospy.init_node('turtlebot_controller', anonymous=True)
    planner=AStarPlanner()
    # Subscribe to goal points
    rospy.Subscriber("goal_point", Point, planning_callback)

    rospy.spin()

#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from collections import deque
from geometry_msgs.msg import PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Float64MultiArray
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

class ARStateCollector:
    def __init__(self, agent_id):
        # Initialize parameters
        self.agent_id = agent_id
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)
        self.history_length = rospy.get_param('~history_length', 50)  # 5 seconds at 10Hz
        
        # Initialize state storage
        self.state_history = deque(maxlen=self.history_length)
        self.marker_id = agent_id  # Assuming marker ID matches agent ID
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers and Subscribers
        self.state_pub = rospy.Publisher(
            f'/agent/{self.agent_id}/state_history',
            Float64MultiArray,
            queue_size=10
        )
        
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )
        
        # Timer for publishing state history
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.publish_state_history)

    def ar_callback(self, msg):
        """Process AR tag detections and update state history"""
        matching_markers = [m for m in msg.markers if m.id == self.marker_id]
        if not matching_markers:
            return
            
        marker = matching_markers[0]
        
        try:
            # Transform from camera to odom frame
            transform = self.tf_buffer.lookup_transform(
                'odom',
                marker.header.frame_id,
                marker.header.stamp,
                rospy.Duration(1.0)
            )
            
            # Convert marker pose
            marker_pose = PoseStamped()
            marker_pose.header = marker.header
            marker_pose.pose = marker.pose.pose
            
            # Transform to world frame
            pose_world = tf2_geometry_msgs.do_transform_pose(marker_pose, transform)
            
            # Extract position and orientation
            pos = pose_world.pose.position
            ori = pose_world.pose.orientation
            euler = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            
            # Create state dictionary
            state = {
                'position': np.array([pos.x, pos.y]),  # 2D position
                'velocity': np.zeros(2),  # Will be calculated if history exists
                'heading': euler[2],  # yaw angle
                'timestamp': pose_world.header.stamp.to_sec()
            }
            
            # Calculate velocity if we have previous states
            if self.state_history:
                prev_state = self.state_history[-1]
                dt = state['timestamp'] - prev_state['timestamp']
                if dt > 0:
                    state['velocity'] = (state['position'] - prev_state['position']) / dt
            
            self.state_history.append(state)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF2 error: {e}")

    def publish_state_history(self, event=None):
        """Publish state history in Trajectron++ compatible format"""
        if not self.state_history:
            return
            
        # Format for Trajectron++: [x, y, vx, vy, heading, timestamp]
        history_array = []
        for state in self.state_history:
            state_entry = [
                *state['position'],  # x, y
                *state['velocity'],  # vx, vy
                state['heading'],    # heading
                state['timestamp']   # timestamp
            ]
            history_array.extend(state_entry)
        
        msg = Float64MultiArray()
        msg.data = history_array
        self.state_pub.publish(msg)
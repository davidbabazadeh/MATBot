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
    def __init__(self):
        # Initialize parameters
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)
        self.history_length = rospy.get_param('~history_length', 50)  # 5 seconds at 10Hz
        
        # Get list of AR tag IDs to track
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 2])  # List of AR tag IDs to track
        
        # Initialize state storage for each AR tag
        self.state_histories = {tag_id: deque(maxlen=self.history_length) for tag_id in self.ar_tag_ids}
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Create publishers for each AR tag
        self.state_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/{tag_id}/state_history',
                Float64MultiArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        # Single subscriber for all AR tags
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )
        
        # Timer for publishing state histories
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.publish_state_histories)

    def ar_callback(self, msg):
        """Process AR tag detections and update state histories"""
        for marker in msg.markers:
            # Only process markers we're interested in
            if marker.id not in self.ar_tag_ids:
                continue
                
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
                if self.state_histories[marker.id]:
                    prev_state = self.state_histories[marker.id][-1]
                    dt = state['timestamp'] - prev_state['timestamp']
                    if dt > 0:
                        state['velocity'] = (state['position'] - prev_state['position']) / dt
                
                self.state_histories[marker.id].append(state)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF2 error for AR tag {marker.id}: {e}")

    def publish_state_histories(self, event=None):
        """Publish state histories for all tracked AR tags"""
        for tag_id in self.ar_tag_ids:
            if not self.state_histories[tag_id]:
                continue
                
            # Format for Trajectron++: [x, y, vx, vy, heading, timestamp]
            history_array = []
            for state in self.state_histories[tag_id]:
                state_entry = [
                    *state['position'],  # x, y
                    *state['velocity'],  # vx, vy
                    state['heading'],    # heading
                    state['timestamp']   # timestamp
                ]
                history_array.extend(state_entry)
            
            msg = Float64MultiArray()
            msg.data = history_array
            self.state_pubs[tag_id].publish(msg)

def main():
    try:
        rospy.init_node('ar_state_collector')
        collector = ARStateCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
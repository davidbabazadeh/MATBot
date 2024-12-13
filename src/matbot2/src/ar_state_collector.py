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
        print("running script")

        # Initialize parameters
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)
        self.history_length = rospy.get_param('~history_length', 1)
        
        # Get reference frame AR tag ID and list of AR tags to track
        self.reference_frame_id = rospy.get_param('~reference_frame_id', 0)  # ID of the AR tag serving as reference frame
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 2])
        
        # Remove reference frame ID from tracked tags if it's in there
        if self.reference_frame_id in self.ar_tag_ids:
            self.ar_tag_ids.remove(self.reference_frame_id)
        
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

        print("initialized publisher")
        
        # Single subscriber for all AR tags
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )

        print("initialized subscriber")
        
        # Timer for publishing state histories
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.publish_state_histories)

    def ar_callback(self, msg):
        """Process AR tag detections and update state histories"""
        reference_frame = f"ar_marker_{self.reference_frame_id}"
        
        # First check if reference marker is visible
        reference_visible = any(marker.id == self.reference_frame_id for marker in msg.markers)
        if not reference_visible:
            rospy.logwarn("Reference frame marker not visible")
            return

        for marker in msg.markers:
            if marker.id not in self.ar_tag_ids:
                continue
                
            try:
                # Look up the transform from this marker to our reference frame
                transform = self.tf_buffer.lookup_transform(
                    reference_frame,                    # target frame
                    f"ar_marker_{marker.id}",          # source frame
                    marker.header.stamp,               # time
                    rospy.Duration(1.0)                # timeout
                )
                
                # Extract position from transform
                position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y
                ])
                
                # Extract heading (yaw) from transform
                quat = transform.transform.rotation
                euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
                heading = euler[2]  # yaw angle
                
                # Create state dictionary
                state = {
                    'position': position,  # 2D position relative to reference frame
                    'velocity': np.zeros(2),  # Will be calculated if history exists
                    'acceleration': np.zeros(2),
                    'heading': heading,  # heading relative to reference frame
                    'timestamp': marker.header.stamp.to_sec()
                }
                
                # Calculate velocity if we have previous states
                if self.state_histories[marker.id]:
                    prev_state = self.state_histories[marker.id][-1]
                    dt = state['timestamp'] - prev_state['timestamp']
                    if dt > 0:
                        state['velocity'] = (state['position'] - prev_state['position']) / dt
                    if len(self.state_histories[marker.id]) > 1:
                        state['acceleration'] = (state['velocity'] - prev_state['velocity']) / dt
                
                self.state_histories[marker.id].append(state)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF2 error for AR tag {marker.id}: {e}")

    def publish_state_histories(self, event=None):
        """Publish state histories for all tracked AR tags"""
        for tag_id in self.ar_tag_ids:
            if not self.state_histories[tag_id]:
                continue
                
            history_array = []
            for state in self.state_histories[tag_id]:
                state_entry = [
                    *state['position'],  # x, y in reference frame
                    *state['velocity'],  # vx, vy
                    *state['acceleration'], # ax, ay
                    state['heading'],    # heading relative to reference frame
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
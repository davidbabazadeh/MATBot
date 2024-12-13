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

class StateCollector:
    def __init__(self):
        # Initialize parameters
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)
        self.history_length = rospy.get_param('~history_length', 1)
        
        # Get list of AR tag IDs to track
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 2, 3])
        
        # Initialize state storage
        self.ar_state_histories = {tag_id: deque(maxlen=self.history_length) for tag_id in self.ar_tag_ids}
        self.ego_state_history = deque(maxlen=self.history_length)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Create publishers for AR agents
        self.ar_state_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/ar_{tag_id}/state_history',
                Float64MultiArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        # Create publisher for ego agent (turtlebot)
        self.ego_state_pub = rospy.Publisher(
            '/agent/ego/state_history',
            Float64MultiArray,
            queue_size=10
        )

        # Subscribe to AR markers
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )
        
        # Timer for state collection and publishing
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.collect_and_publish_states)

    def collect_ego_state(self, timestamp):
        """Collect ego agent state using TF"""
        try:
            # Look up transform from base_footprint to odom frame
            transform = self.tf_buffer.lookup_transform(
                'odom',              # target frame (fixed frame)
                'base_footprint',    # source frame (ego robot frame)
                timestamp,           # time
                rospy.Duration(0.1)  # timeout
            )
            
            # Extract position from transform
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            
            # Extract heading from transform
            quat = transform.transform.rotation
            euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            heading = euler[2]
            
            # Create state dictionary
            state = {
                'position': position,
                'velocity': np.zeros(2),  # Will be calculated from position history
                'acceleration': np.zeros(2),
                'heading': heading,
                'timestamp': timestamp.to_sec()
            }
            
            # Calculate velocity and acceleration if we have previous states
            if self.ego_state_history:
                prev_state = self.ego_state_history[-1]
                dt = state['timestamp'] - prev_state['timestamp']
                if dt > 0:
                    state['velocity'] = (state['position'] - prev_state['position']) / dt
                    if len(self.ego_state_history) > 1:
                        state['acceleration'] = (state['velocity'] - prev_state['velocity']) / dt
            
            self.ego_state_history.append(state)
            return True
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Error collecting ego state: {e}")
            return False

    def ar_callback(self, msg):
        """Process AR tag detections and update state histories"""
        for marker in msg.markers:
            if marker.id not in self.ar_tag_ids:
                continue
                
            try:
                # Look up transform from marker to odom frame
                transform = self.tf_buffer.lookup_transform(
                    'odom',                          # target frame (fixed frame)
                    f"ar_marker_{marker.id}",        # source frame
                    marker.header.stamp,             # time
                    rospy.Duration(1.0)              # timeout
                )
                
                # Extract position from transform
                position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y
                ])
                
                # Extract heading from transform
                quat = transform.transform.rotation
                euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
                heading = euler[2]
                
                # Create state dictionary
                state = {
                    'position': position,
                    'velocity': np.zeros(2),
                    'acceleration': np.zeros(2),
                    'heading': heading,
                    'timestamp': marker.header.stamp.to_sec()
                }
                
                # Calculate velocity and acceleration if we have previous states
                if self.ar_state_histories[marker.id]:
                    prev_state = self.ar_state_histories[marker.id][-1]
                    dt = state['timestamp'] - prev_state['timestamp']
                    if dt > 0:
                        state['velocity'] = (state['position'] - prev_state['position']) / dt
                        if len(self.ar_state_histories[marker.id]) > 1:
                            state['acceleration'] = (state['velocity'] - prev_state['velocity']) / dt
                
                self.ar_state_histories[marker.id].append(state)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF2 error for AR tag {marker.id}: {e}")

    def collect_and_publish_states(self, event=None):
        """Collect ego state and publish all states"""
        # Collect ego state
        current_time = rospy.Time.now()
        self.collect_ego_state(current_time)
        
        # Publish ego state
        if self.ego_state_history:
            history_array = []
            for state in self.ego_state_history:
                state_entry = [
                    *state['position'],     # x, y in odom frame
                    *state['velocity'],     # vx, vy
                    *state['acceleration'], # ax, ay
                    state['heading'],       # heading in odom frame
                    state['timestamp']      # timestamp
                ]
                history_array.extend(state_entry)
            
            msg = Float64MultiArray()
            msg.data = history_array
            self.ego_state_pub.publish(msg)
        
        # Publish AR agents states
        for tag_id in self.ar_tag_ids:
            if not self.ar_state_histories[tag_id]:
                continue
                
            history_array = []
            for state in self.ar_state_histories[tag_id]:
                state_entry = [
                    *state['position'],
                    *state['velocity'],
                    *state['acceleration'],
                    state['heading'],
                    state['timestamp']
                ]
                history_array.extend(state_entry)
            
            msg = Float64MultiArray()
            msg.data = history_array
            self.ar_state_pubs[tag_id].publish(msg)

def main():
    try:
        rospy.init_node('state_collector')
        collector = StateCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
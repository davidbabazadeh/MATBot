#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from collections import deque
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Float64MultiArray
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion


class StateCollector:
    ar_to_map_length = 0
    ar_to_map_width = 0

    def __init__(self):
        # Initialize parameters
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)
        self.history_length = rospy.get_param('~history_length', 1)
        self.robot_ar_tag_id = rospy.get_param('~robot_ar_tag_id', 2)
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 6])
        self.ar_tag_ids_ref = rospy.get_param('~ar_tag_ids', [8, 7, 6, 5])
        self.tf_buffer_duration = rospy.Duration(10.0)  # Increased buffer size
        
        # Initialize state storage
        self.ar_state_histories = {tag_id: deque(maxlen=self.history_length) 
                                 for tag_id in self.ar_tag_ids if tag_id != self.robot_ar_tag_id}
        
        # TF setup with larger buffer
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.tf_buffer_duration)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Store latest transforms and timestamps
        self.latest_transforms = {}
        self.transform_cache = {}
        
        # Create publishers
        self.state_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/ar_{tag_id}/state_history',
                Float64MultiArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids if tag_id != self.robot_ar_tag_id
        }
        
        # Subscribe to AR markers
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )

        self.bot_right = [0, 0] # AR 8
        self.bot_left = [0, 0] # AR 7
        self.top_right = [0, 0] # AR 6
        self.calibrate_board_dims()
        
        # Timer for publishing state histories
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.publish_state_histories)
    
    # AR tags:
    # bot right = 8
    # bot left = 7
    # top right = 6
    def calibrate_board_dims(self, length=1.1049, width=.7874, depth=1.2192):
        ar_to_map_length = length / (self.bot_right[0] - self.bot_left[0]) 
        ar_to_map_width = width / (self.bot_right[1] - self.top_right[1]) 
        


    def get_transform_with_fallback(self, target_frame, source_frame, timestamp):
        """Attempt to get transform with fallback to latest available"""
        try:
            # First try: Exact timestamp
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                timestamp,
                rospy.Duration(0.1)
            )
            return transform
        except tf2_ros.ExtrapolationException:
            try:
                # Second try: Latest available transform
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),  # Latest available transform
                    rospy.Duration(0.1)
                )
                rospy.logdebug(f"Using latest available transform for {source_frame} to {target_frame}")
                return transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform failed for {source_frame} to {target_frame}: {e}")
                return None

    def update_camera_to_base_transform(self, robot_ar_marker):
        """Update transform between camera and base_footprint using robot's AR tag"""
        try:
            # Get latest transform from odom to base_footprint
            odom_to_base = self.get_transform_with_fallback(
                'ar_marker_5',
                'usb_cam',
                robot_ar_marker.header.stamp
            )
            
            if odom_to_base is None:
                return
            
            # Create transform from camera to AR marker
            transform_stamped = TransformStamped()
            transform_stamped.header.stamp = robot_ar_marker.header.stamp
            transform_stamped.header.frame_id = 'ar_marker_5'
            transform_stamped.child_frame_id = 'usb_cam'
            
            # Use the AR marker pose to compute camera position relative to odom
            marker_pose = robot_ar_marker.pose.pose
            
            # Simplified transform computation - adjust based on your setup
            transform_stamped.transform.translation.x = odom_to_base.transform.translation.x
            transform_stamped.transform.translation.y = odom_to_base.transform.translation.y
            transform_stamped.transform.translation.z = odom_to_base.transform.translation.z
            transform_stamped.transform.rotation = odom_to_base.transform.rotation
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform_stamped)
            
            # Cache the transform
            self.transform_cache['camera_to_ar_marker_5'] = transform_stamped
            
        except Exception as e:
            rospy.logwarn(f"Error updating camera transform: {e}")

    def ar_callback(self, msg):
        """Process AR tag detections and update state histories"""
        if not msg.markers:
            return

        # Update camera transform if robot AR tag is visible
        for marker in msg.markers:
            if marker.id == self.robot_ar_tag_id:
                self.update_camera_to_base_transform(marker)
                break
        
        # Process other markers
        for marker in msg.markers:
            if marker.id == self.robot_ar_tag_id or marker.id not in self.ar_tag_ids + self.ar_tag_ids_ref:
                continue

            # Create a slightly earlier timestamp to avoid extrapolation
            timestamp = marker.header.stamp - rospy.Duration(0.1)
            
            try:
                # Get transform using fallback method
                transform = self.get_transform_with_fallback(
                    'odom',
                    f"ar_marker_{marker.id}",
                    timestamp
                )
                
                if transform is None:
                    continue
                
                # Extract position and orientation
                position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y
                ])
                
                quat = transform.transform.rotation
                euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
                heading = euler[2]

                if marker.id in self.ar_tag_ids:
                
                    # Create state dictionary
                    state = {
                        'position': position,
                        'velocity': np.zeros(2),
                        'acceleration': np.zeros(2),
                        'heading': heading,
                        'timestamp': timestamp.to_sec()
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

                else:
                    if marker.id == 8:
                        self.bot_right = position
                    elif marker.id == 7:
                        self.bot_left = position
                    elif marker.id == 6:
                        self.top_right = position
                
            except Exception as e:
                rospy.logwarn(f"Error processing AR tag {marker.id}: {e}")

    def publish_state_histories(self, event=None):
        """Publish state histories for all tracked AR tags"""
        for tag_id in self.ar_state_histories:
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
            self.state_pubs[tag_id].publish(msg)

def main():
    try:
        rospy.init_node('state_collector')
        collector = StateCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
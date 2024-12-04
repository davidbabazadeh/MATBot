#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Float64MultiArray
import tf2_geometry_msgs

class ARStateCollector:
    def __init__(self):
        rospy.init_node('ar_state_collector')
        
        # Parameters
        self.agent_id = rospy.get_param('~agent_id', 0)
        self.poll_rate = rospy.get_param('~poll_rate', 10.0)  

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
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
        
        # State storage
        self.latest_pose = None
        self.marker_id = rospy.get_param('~marker_id', 0)  # AR tag ID to track
        
        rospy.Timer(rospy.Duration(1.0/self.poll_rate), self.timer_callback)
        
    def ar_callback(self, msg):
        """Process incoming AR tag detections"""
        # Find marker with matching ID
        matching_markers = [m for m in msg.markers if m.id == self.marker_id]
        if not matching_markers:
            return
            
        marker = matching_markers[0]
        
        try:
            # Transform from camera frame (/usb_cam) to world frame
            transform = self.tf_buffer.lookup_transform(
                'odom',  # target frame - the fixed world coordinate frame
                marker.header.frame_id,  # source frame - from the AR marker detection
                marker.header.stamp,
                rospy.Duration(1.0)
            )
            #################### refer to lab 8
            # Convert marker pose to PoseStamped
            marker_pose = PoseStamped()
            marker_pose.header = marker.header
            marker_pose.pose = marker.pose.pose
            
            # Transform pose to world frame
            pose_world = tf2_geometry_msgs.do_transform_pose(
                marker_pose,
                transform
            )
            ####################
            self.latest_pose = pose_world
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF2 error: {e}")
    
    def timer_callback(self, event):
        """Publish state at regular intervals"""
        if self.latest_pose is None:
            return
            
        #################### refer to lab 8
        # Extract position and orientation
        pos = self.latest_pose.pose.position
        ori = self.latest_pose.pose.orientation
        
        # Create state array [x, y, z, qx, qy, qz, qw, timestamp, marker_id]
        state = [
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w,
            self.latest_pose.header.stamp.to_sec(),
            float(self.marker_id)
        ]
        ####################
        # Publish state
        msg = Float64MultiArray()
        msg.data = state
        self.state_pub.publish(msg)

def main():
    try:
        collector = ARStateCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
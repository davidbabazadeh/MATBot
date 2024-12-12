#!/usr/bin/env python
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Float64MultiArray
import tf2_ros
import numpy as np

class ARTrajectronBridge:
    """Bridge between AR tag detection and Trajectron prediction"""
    def __init__(self):
        # Parameters
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 2])
        self.history_length = rospy.get_param('~history_length', 20)
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Create publishers for Trajectron input
        self.trajectron_pubs = {
            tag_id: rospy.Publisher(
                f'/trajectron/agent_{tag_id}/input',
                Float64MultiArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        # Subscribe to AR state collector output
        self.state_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/{tag_id}/state_history',
                Float64MultiArray,
                lambda msg, id=tag_id: self.state_callback(msg, id),
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }

    def state_callback(self, msg, tag_id):
        """Convert AR state to Trajectron format and publish"""
        # Reshape incoming data [x, y, vx, vy, ax, ay, heading, t]
        num_states = len(msg.data) // 8
        states = np.reshape(msg.data, (num_states, 8))
        
        # Format for Trajectron: [x, y, vx, vy]
        trajectron_data = []
        for state in states:
            trajectron_state = [
                state[0],  # x
                state[1],  # y
                state[2],  # vx
                state[3],  # vy
            ]
            trajectron_data.extend(trajectron_state)
        
        # Publish formatted data
        msg = Float64MultiArray()
        msg.data = trajectron_data
        self.trajectron_pubs[tag_id].publish(msg)

def main():
    rospy.init_node('ar_trajectron_bridge')
    bridge = ARTrajectronBridge()
    rospy.spin()

if __name__ == '__main__':
    main()
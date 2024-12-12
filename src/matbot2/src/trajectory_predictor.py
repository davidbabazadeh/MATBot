#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import euler_from_quaternion

class TrajectoryPredictor:
    def __init__(self):
        # Parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 3.0)  # seconds
        self.time_step = rospy.get_param('~time_step', 0.1)  # 10Hz predictions
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 6])
        
        # Create subscribers for each AR tag's state history
        self.state_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/{tag_id}/state_history',
                Float64MultiArray,
                self.state_callback,
                callback_args=tag_id
            ) for tag_id in self.ar_tag_ids
        }
        
        # Create publishers for predicted trajectories
        self.trajectory_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/{tag_id}/predicted_trajectory',
                PoseArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }

    def state_callback(self, msg, tag_id):
        """Process incoming state history and publish trajectory prediction"""
        # Parse state history
        # Format: [x, y, vx, vy, ax, ay, heading, timestamp] repeated for each point
        state_length = 8  # number of elements per state
        num_states = len(msg.data) // state_length
        
        if num_states == 0:
            return
            
        # Get latest state
        latest_idx = (num_states - 1) * state_length
        current_state = {
            'position': np.array(msg.data[latest_idx:latest_idx+2]),
            'velocity': np.array(msg.data[latest_idx+2:latest_idx+4]),
            'acceleration': np.array(msg.data[latest_idx+4:latest_idx+6]),
            'heading': msg.data[latest_idx+6],
            'timestamp': msg.data[latest_idx+7]
        }
        
        # Generate prediction
        predicted_poses = self.predict_trajectory(current_state)
        
        # Publish prediction
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'usb_cam'  # Use same frame as AR tag detection
        msg.poses = predicted_poses
        self.trajectory_pubs[tag_id].publish(msg)

    def predict_trajectory(self, current_state):
        """Generate predicted trajectory using constant acceleration model"""
        poses = []
        num_steps = int(self.prediction_horizon / self.time_step)
        
        for i in range(num_steps):
            t = i * self.time_step
            
            # Predict position using physics equations
            # p = p0 + v0*t + 0.5*a*t^2
            predicted_position = (
                current_state['position'] + 
                current_state['velocity'] * t +
                0.5 * current_state['acceleration'] * t**2
            )
            
            # Create pose for this prediction
            pose = Pose()
            pose.position.x = predicted_position[0]
            pose.position.y = predicted_position[1]
            pose.position.z = 0.0
            
            # Use current heading for orientation (could be improved)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = np.sin(current_state['heading'] / 2)
            pose.orientation.w = np.cos(current_state['heading'] / 2)
            
            poses.append(pose)
            
        return poses

def main():
    try:
        rospy.init_node('trajectory_predictor')
        predictor = TrajectoryPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
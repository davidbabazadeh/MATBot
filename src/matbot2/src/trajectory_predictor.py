#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class TrajectoryPredictor:
    def __init__(self):
        # Parameters
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 3.0)  # seconds
        self.time_step = rospy.get_param('~time_step', 0.1)  # 10Hz predictions
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 2, 3])
        
        # Create subscribers for AR agents
        self.ar_state_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/ar_{tag_id}/state_history',
                Float64MultiArray,
                self.state_callback,
                callback_args=('ar', tag_id)
            ) for tag_id in self.ar_tag_ids
        }
        
        # Create subscriber for ego agent
        self.ego_state_sub = rospy.Subscriber(
            '/agent/ego/state_history',
            Float64MultiArray,
            self.state_callback,
            callback_args=('ego', 'ego')
        )
        
        # Create publishers for AR agents' predicted trajectories
        self.ar_trajectory_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/ar_{tag_id}/predicted_trajectory',
                PoseArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        # Create publisher for ego agent's predicted trajectory
        self.ego_trajectory_pub = rospy.Publisher(
            '/agent/ego/predicted_trajectory',
            PoseArray,
            queue_size=10
        )

    def state_callback(self, msg, args):
        """Process incoming state history and publish trajectory prediction"""
        agent_type, agent_id = args
        
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
        
        # Create and publish prediction message
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'odom'  # All predictions in odom frame
        msg.poses = predicted_poses
        
        # Publish to appropriate topic
        if agent_type == 'ego':
            self.ego_trajectory_pub.publish(msg)
        else:  # ar tag
            self.ar_trajectory_pubs[agent_id].publish(msg)

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
            
            # Predict velocity (could be used for heading prediction)
            predicted_velocity = (
                current_state['velocity'] + 
                current_state['acceleration'] * t
            )
            
            # Create pose for this prediction
            pose = Pose()
            pose.position.x = predicted_position[0]
            pose.position.y = predicted_position[1]
            pose.position.z = 0.0
            
            # Update heading based on velocity direction if speed is significant
            speed = np.linalg.norm(predicted_velocity)
            if speed > 0.1:  # Only update heading if moving
                predicted_heading = np.arctan2(predicted_velocity[1], predicted_velocity[0])
            else:
                predicted_heading = current_state['heading']
            
            # Convert heading to quaternion
            q = quaternion_from_euler(0, 0, predicted_heading)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
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
#!/usr/bin/env python
import rospy
import torch
import numpy as np
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose

class TrajectronPlusPlus:
    def __init__(self):
        rospy.init_node('trajectron_plus_plus_node')
        
        # Load Trajectron++ model
        self.model_path = rospy.get_param('~model_path', 'model_registrar-20.pt')
        self.model = torch.load(self.model_path)
        self.model.eval()
        
        # Parameters
        self.num_agents = rospy.get_param('~num_agents', 3)
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 30)  # 3 seconds at 10Hz
        
        # Store state histories for all agents
        self.agent_histories = {i: [] for i in range(self.num_agents)}
        
        # Publishers
        self.prediction_pubs = {
            i: rospy.Publisher(f'/agent/{i}/predicted_trajectory', PoseArray, queue_size=10)
            for i in range(self.num_agents)
        }
        
        # Subscribers
        self.history_subs = {
            i: rospy.Subscriber(f'/agent/{i}/state_history', Float64MultiArray, 
                              lambda msg, aid=i: self.history_callback(msg, aid))
            for i in range(self.num_agents)
        }
        
        # Timer for predictions
        self.prediction_timer = rospy.Timer(
            rospy.Duration(0.1),  # 10Hz
            self.predict_trajectories
        )

    def history_callback(self, msg, agent_id):
        """Process incoming state history for an agent"""
        # Reshape data into [N, 6] array where each row is [x, y, vx, vy, heading, timestamp]
        data = np.array(msg.data).reshape(-1, 6)
        self.agent_histories[agent_id] = data

    def prepare_model_input(self, agent_histories):
        """Format agent histories for Trajectron++ input"""
        # Create input tensors according to Trajectron++ requirements
        # This will depend on your specific Trajectron++ model configuration
        batch = {
            'positions': [],
            'velocities': [],
            'headings': [],
            'timestamps': []
        }
        
        for agent_id, history in agent_histories.items():
            if len(history) > 0:
                batch['positions'].append(history[:, :2])
                batch['velocities'].append(history[:, 2:4])
                batch['headings'].append(history[:, 4])
                batch['timestamps'].append(history[:, 5])
        
        # Convert to tensors
        for key in batch:
            if batch[key]:
                batch[key] = torch.tensor(np.stack(batch[key]), dtype=torch.float32)
            
        return batch

    def predict_trajectories(self, event=None):
        """Generate trajectory predictions for all agents"""
        if not all(self.agent_histories.values()):
            return
            
        # Prepare input for Trajectron++
        model_input = self.prepare_model_input(self.agent_histories)
        
        with torch.no_grad():
            # Generate predictions
            predictions = self.model(model_input, prediction_horizon=self.prediction_horizon)
            
            # Process predictions for each agent
            for agent_id in range(self.num_agents):
                if agent_id in predictions:
                    # Convert prediction to PoseArray
                    pose_array = PoseArray()
                    pose_array.header.frame_id = 'odom'
                    pose_array.header.stamp = rospy.Time.now()
                    
                    # Extract mean trajectory prediction
                    trajectory = predictions[agent_id]['mean']
                    
                    for pos in trajectory:
                        pose = Pose()
                        pose.position.x = pos[0]
                        pose.position.y = pos[1]
                        pose_array.poses.append(pose)
                    
                    # Publish prediction
                    self.prediction_pubs[agent_id].publish(pose_array)

def main():
    try:
        node = TrajectronPlusPlus()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
#!/usr/bin/env python
import rospy
import numpy as np
import torch
import dill
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseArray, Pose
from collections import defaultdict
import os

class TrajactronPredictor:
    def __init__(self):
        # Load parameters
        self.model_path = rospy.get_param('~model_path', 'trajectron_model.pkl')
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 3.0)
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 2])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load Trajectron++ model
        self.model = self.load_model(self.model_path)
        rospy.loginfo(f"Model loaded from {self.model_path}")
        
        # State storage
        self.agent_histories = defaultdict(list)
        self.current_scene = None
        self.prediction_timesteps = int(self.prediction_horizon * 10)  # Assuming 10Hz
        
        # Publishers for predicted trajectories
        self.prediction_pubs = {
            tag_id: rospy.Publisher(
                f'/agent/{tag_id}/predicted_trajectory',
                PoseArray,
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        # Subscribers for agent states
        self.state_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/{tag_id}/state_history',
                Float64MultiArray,
                lambda msg, id=tag_id: self.state_callback(msg, id),
                queue_size=10
            ) for tag_id in self.ar_tag_ids
        }
        
        self.prediction_timer = rospy.Timer(
            rospy.Duration(0.1),
            self.predict_trajectories
        )

    def load_model(self, model_path):
        """Load the Trajectron++ model properly handling the checkpoint structure"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            rospy.loginfo("Checkpoint loaded successfully")
            
            # Initialize model with proper configuration
            model_config = {
                'node_types': ['VEHICLE', 'PEDESTRIAN'],
                'edge_types': [('VEHICLE', 'VEHICLE'), ('VEHICLE', 'PEDESTRIAN')],
                'state_dim': 4,  # [x, y, vx, vy]
                'pred_state_dim': 2,  # [x, y]
            }
            
            # Create new Trajectron model
            from trajectron import Trajectron  # Import here to avoid potential circular imports
            model = Trajectron(model_config)
            
            # Load components from checkpoint
            for key, value in checkpoint.items():
                if hasattr(model, key.split('/')[-1]):
                    setattr(model, key.split('/')[-1], value)
            
            # Move model to device
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            return model
            
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            raise

    def create_scene(self):
        """Create a Trajectron++ scene from current agent histories"""
        from trajectron.environment import Scene, Node, Environment
        
        try:
            # Initialize scene
            scene = Scene(timesteps=self.prediction_timesteps, dt=0.1)
            scene.env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'])
            
            for agent_id, history in self.agent_histories.items():
                if len(history) < 2:
                    continue
                    
                # Create node for agent (treating as VEHICLE type)
                node_data = self.prepare_node_data(history)
                if node_data is not None:
                    node = Node(node_type='VEHICLE',
                              node_id=f'agent_{agent_id}',
                              data=node_data,
                              first_timestep=0)
                    scene.nodes.append(node)
            
            return scene
            
        except Exception as e:
            rospy.logwarn(f"Error creating scene: {str(e)}")
            return None

    def prepare_node_data(self, history):
        """Prepare agent history data in Trajectron++ format"""
        try:
            # Extract relevant features: [x, y, vx, vy]
            data = np.zeros((len(history), 4))
            data[:, 0:2] = history[:, 0:2]  # x, y
            data[:, 2:4] = history[:, 2:4]  # vx, vy
            
            # Create data dictionary with additional required fields
            node_data = {
                'pos': torch.FloatTensor(data[:, 0:2]).to(self.device),
                'vel': torch.FloatTensor(data[:, 2:4]).to(self.device),
                'frames': torch.arange(len(history), device=self.device),
                'first_timestep': 0,
                'last_timestep': len(history) - 1,
                'timesteps': torch.arange(len(history), device=self.device),
                'type': 'VEHICLE'
            }
            
            return node_data
            
        except Exception as e:
            rospy.logwarn(f"Error preparing node data: {str(e)}")
            return None

    def predict_trajectories(self, event=None):
        """Generate and publish trajectory predictions for all agents"""
        if not all(self.agent_histories.values()):
            return
            
        try:
            # Create scene from current state
            scene = self.create_scene()
            if not scene or not scene.nodes:
                return
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model.predict(
                    scene,
                    timesteps=self.prediction_timesteps,
                    num_samples=1,
                    min_future_timesteps=self.prediction_timesteps,
                    z_mode=True,
                    gmm_mode=True,
                    full_dist=False
                )
            
            # Process and publish predictions
            self.publish_predictions(predictions, scene)
            
        except Exception as e:
            rospy.logwarn(f"Error in prediction: {str(e)}")

    def publish_predictions(self, predictions, scene):
        """Process and publish trajectory predictions"""
        current_time = rospy.Time.now()
        
        try:
            for node in scene.nodes:
                agent_id = int(node.id.split('_')[1])
                
                if node.id not in predictions:
                    continue
                    
                pred = predictions[node.id]
                
                pose_array = PoseArray()
                pose_array.header.frame_id = "usb_cam"
                pose_array.header.stamp = current_time
                
                # Convert predicted trajectory to poses
                trajectory = pred.squeeze()
                for pos in trajectory:
                    pose = Pose()
                    pose.position.x = float(pos[0])
                    pose.position.y = float(pos[1])
                    pose.position.z = 0.0
                    
                    if len(pos) >= 4:
                        heading = np.arctan2(float(pos[3]), float(pos[2]))
                        pose.orientation.z = np.sin(heading/2)
                        pose.orientation.w = np.cos(heading/2)
                    
                    pose_array.poses.append(pose)
                
                self.prediction_pubs[agent_id].publish(pose_array)
                
        except Exception as e:
            rospy.logwarn(f"Error publishing predictions: {str(e)}")

def main():
    try:
        rospy.init_node('trajectron_predictor')
        predictor = TrajactronPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
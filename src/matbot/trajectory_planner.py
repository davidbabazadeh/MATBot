#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs

class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('trajectory_planner')
        
        # Parameters
        self.num_agents = rospy.get_param('~num_agents', 3)
        self.planning_horizon = rospy.get_param('~planning_horizon', 30)
        self.safety_margin = rospy.get_param('~safety_margin', 0.5)  # meters
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Store predicted trajectories
        self.agent_predictions = {i: None for i in range(self.num_agents)}
        
        # Subscribe to predicted trajectories and goal
        self.prediction_subs = {
            i: rospy.Subscriber(f'/agent/{i}/predicted_trajectory', PoseArray,
                              lambda msg, aid=i: self.prediction_callback(msg, aid))
            for i in range(self.num_agents)
        }
        
        self.goal_sub = rospy.Subscriber(
            '/goal_pose',
            PoseStamped,
            self.goal_callback
        )
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.planned_path_pub = rospy.Publisher('/planned_path', PoseArray, queue_size=1)
        
        self.goal_pose = None
        self.current_plan = None
        
        # Planning timer
        self.planning_timer = rospy.Timer(
            rospy.Duration(0.1),  # 10Hz
            self.plan_trajectory
        )

    def prediction_callback(self, msg, agent_id):
        """Store predicted trajectories for each agent"""
        self.agent_predictions[agent_id] = msg

    def goal_callback(self, msg):
        """Store goal pose"""
        self.goal_pose = msg

    def get_current_pose(self):
        """Get current robot pose in odom frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_footprint',
                rospy.Time(0)
            )
            
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position = transform.transform.translation
            pose.pose.orientation = transform.transform.rotation
            
            return pose
        except:
            return None

    def check_collision(self, path, predictions):
        """Check if planned path collides with predicted trajectories"""
        for t in range(min(len(path), self.planning_horizon)):
            robot_pos = np.array([path[t].position.x, path[t].position.y])
            
            # Check against each agent's predicted position
            for agent_id, prediction in predictions.items():
                if prediction is None or t >= len(prediction.poses):
                    continue
                    
                agent_pos = np.array([
                    prediction.poses[t].position.x,
                    prediction.poses[t].position.y
                ])
                
                # Check distance
                if np.linalg.norm(robot_pos - agent_pos) < self.safety_margin:
                    return True
                    
        return False

    def plan_trajectory(self, event=None):
        """Generate collision-free trajectory to goal"""
        if not self.goal_pose or not all(self.agent_predictions.values()):
            return
            
        current_pose = self.get_current_pose()
        if not current_pose:
            return
            
        # Simple linear interpolation between current pose and goal
        num_points = self.planning_horizon
        path = PoseArray()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()
        
        for i in range(num_points):
            t = float(i) / (num_points - 1)
            pose = Pose()
            pose.position.x = (1-t) * current_pose.pose.position.x + t * self.goal_pose.pose.position.x
            pose.position.y = (1-t) * current_pose.pose.position.y + t * self.goal_pose.pose.position.y
            path.poses.append(pose)
        
        # Check for collisions
        if not self.check_collision(path.poses, self.agent_predictions):
            self.current_plan = path
            self.planned_path_pub.publish(path)
            self.execute_trajectory()

    def execute_trajectory(self):
        """Execute the planned trajectory"""
        if not self.current_plan or not self.current_plan.poses:
            return
            
        current_pose = self.get_current_pose()
        if not current_pose:
            return
            
        # Get next waypoint
        target = self.current_plan.poses[0]
        
        # Calculate control commands
        dx = target.position.x - current_pose.pose.position.x
        dy = target.position.y - current_pose.pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        
        # Simple proportional control
        cmd = Twist()
        cmd.linear.x = min(0.2, distance)  # max 0.2 m/s
        cmd.angular.z = 0.5 * angle        # proportional control for rotation
        
        self.cmd_vel_pub.publish(cmd)

def main():
    try:
        planner = TrajectoryPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
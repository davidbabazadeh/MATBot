#!/usr/bin/env python
import rospy
import numpy as np
from scipy.sparse import block_diag, csc_matrix, vstack
import osqp
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

class MPCController:
    def __init__(self):
        # MPC parameters
        self.dt = rospy.get_param('~dt', 0.1)  # Time step
        self.N = rospy.get_param('~horizon', 20)  # Prediction horizon
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 6])
        
        # State and control dimensions
        self.nx = 4  # State dimension [x, y, v, theta]
        self.nu = 2  # Control dimension [v_cmd, omega_cmd]
        
        # Weights for the cost function
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost
        self.R = np.diag([1.0, 0.5])  # Control cost
        self.P = self.Q  # Terminal cost
        
        # Control constraints
        self.v_max = 0.5  # Maximum linear velocity
        self.omega_max = 1.0  # Maximum angular velocity
        
        # Collision avoidance parameters
        self.safety_radius = 0.5  # Minimum distance to maintain from other agents
        self.other_agents_predictions = {}  # Store other agents' predicted trajectories
        
        # Initialize OSQP solver
        self.setup_mpc()
        
        # Current state and reference trajectory
        self.current_state = np.zeros(self.nx)
        self.reference_trajectory = None
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribe to robot odometry
        self.odom_sub = rospy.Subscriber(
            '/odom', 
            Odometry, 
            self.odom_callback
        )
        
        # Subscribe to predicted trajectories for each AR tag
        self.traj_subs = {}
        for tag_id in self.ar_tag_ids:
            if tag_id == 0:  # Assuming tag_id 0 is ego vehicle
                continue
            self.traj_subs[tag_id] = rospy.Subscriber(
                f'/agent/{tag_id}/predicted_trajectory',
                PoseArray,
                self.other_agent_trajectory_callback,
                callback_args=tag_id
            )
        
        # Subscribe to ego vehicle's target trajectory
        self.ego_traj_sub = rospy.Subscriber(
            '/agent/0/predicted_trajectory',
            PoseArray,
            self.ego_trajectory_callback
        )
        
        # Control loop timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)

    def setup_mpc(self):
        """Setup the MPC problem matrices"""
        # Linearized discrete-time model matrices
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        
        # Cast MPC problem to QP problem
        self.Q_block = block_diag([self.Q] * self.N + [self.P])
        self.R_block = block_diag([self.R] * self.N)

    def generate_collision_constraints(self, x0):
        """Generate collision avoidance constraints for all predicted timesteps"""
        n_constraints = len(self.other_agents_predictions) * self.N
        A_coll = np.zeros((n_constraints, self.nx * (self.N + 1) + self.nu * self.N))
        b_coll = np.zeros(n_constraints)
        
        constraint_idx = 0
        for agent_id, pred_traj in self.other_agents_predictions.items():
            for t in range(min(self.N, len(pred_traj))):
                # Position of other agent at time t
                other_x = pred_traj[t][0]
                other_y = pred_traj[t][1]
                
                # Add constraint for minimum distance
                # ||p_ego - p_other||^2 >= safety_radius^2
                # Linearized around current position
                x_idx = t * self.nx
                A_coll[constraint_idx, x_idx:x_idx+2] = 2 * np.array([
                    x0[0] - other_x,
                    x0[1] - other_y
                ])
                
                b_coll[constraint_idx] = self.safety_radius**2 - (
                    (x0[0] - other_x)**2 + (x0[1] - other_y)**2
                )
                
                constraint_idx += 1
        
        return A_coll, b_coll

    def solve_mpc(self, x0, x_ref):
        """Solve the MPC problem with collision avoidance"""
        # Generate collision avoidance constraints
        A_coll, b_coll = self.generate_collision_constraints(x0)
        
        # Combine all constraints
        A = np.vstack([
            self.A_eq,  # Dynamic constraints
            np.zeros((self.nu * self.N, self.nx * (self.N + 1) + self.nu * self.N)),  # Control constraints
            A_coll  # Collision constraints
        ])
        
        # Update bounds
        l = np.hstack([
            self.l_eq,  # Dynamic constraints
            np.tile(self.u_min, self.N),  # Control constraints lower bound
            b_coll  # Collision constraints lower bound
        ])
        
        u = np.hstack([
            self.u_eq,  # Dynamic constraints
            np.tile(self.u_max, self.N),  # Control constraints upper bound
            np.inf * np.ones_like(b_coll)  # Collision constraints upper bound
        ])
        
        # Update and solve the QP problem
        self.prob.update(l=l, u=u, Ax=A.data)
        result = self.prob.solve()
        
        if result.info.status != 'solved':
            rospy.logwarn("MPC problem could not be solved!")
            return None
            
        return result.x[:self.nu]  # Return first control input

    def other_agent_trajectory_callback(self, msg, agent_id):
        """Store predicted trajectories of other agents"""
        predictions = []
        for pose in msg.poses:
            predictions.append([
                pose.position.x,
                pose.position.y
            ])
        self.other_agents_predictions[agent_id] = predictions

    def ego_trajectory_callback(self, msg):
        """Process incoming target trajectory for ego vehicle"""
        # Convert PoseArray to state trajectory
        n_poses = len(msg.poses)
        self.reference_trajectory = np.zeros((n_poses, self.nx))
        
        for i, pose in enumerate(msg.poses):
            self.reference_trajectory[i] = [
                pose.position.x,
                pose.position.y,
                0.0,
                np.arctan2(2.0 * (pose.orientation.w * pose.orientation.z),
                          1.0 - 2.0 * pose.orientation.z * pose.orientation.z)
            ]
            
            # Compute velocities from consecutive positions
            if i > 0:
                dt = self.dt
                dx = self.reference_trajectory[i, 0] - self.reference_trajectory[i-1, 0]
                dy = self.reference_trajectory[i, 1] - self.reference_trajectory[i-1, 1]
                self.reference_trajectory[i-1, 2] = np.sqrt(dx*dx + dy*dy) / dt

    def control_loop(self, event=None):
        """Main control loop"""
        if self.reference_trajectory is None or not self.other_agents_predictions:
            return
            
        # Solve MPC problem with collision avoidance
        u = self.solve_mpc(self.current_state, self.reference_trajectory)
        
        if u is None:
            return
            
        # Publish control commands
        cmd_msg = Twist()
        cmd_msg.linear.x = u[0]
        cmd_msg.angular.z = u[1]
        self.cmd_pub.publish(cmd_msg)

def main():
    try:
        rospy.init_node('mpc_controller')
        controller = MPCController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from casadi import *
import tf
from scipy.spatial.distance import cdist
from typing import List, Tuple
import threading

class MPCPlanner:
    def __init__(self):
        # MPC parameters
        self.T = 20  # Prediction horizon
        self.dt = 0.1  # Time step
        
        # Robot parameters
        self.v_max = 0.22  # Maximum linear velocity (m/s)
        self.w_max = 2.84  # Maximum angular velocity (rad/s)
        self.robot_radius = 0.2  # Robot radius for collision checking
        
        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0])  # State cost
        self.R = np.diag([1.0, 0.5])  # Control cost
        self.Q_terminal = np.diag([20.0, 20.0, 2.0])  # Terminal cost
        self.obstacle_weight = 50.0  # Weight for obstacle avoidance
        update_predictions
        # Initialize CasADi solver
        self.setup_solver()
        
        # ROS setup
        rospy.init_node('mpc_planner')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # State variables
        self.current_state = np.zeros(3)  # [x, y, theta]
        self.goal_state = np.zeros(3)
        self.static_obstacles = []  # List of [x, y, radius] for static obstacles
        
        # Dynamic prediction variables
        self.predicted_trajectories = []  # List of predicted trajectories from Trajectron++
        self.prediction_lock = threading.Lock()  # Thread lock for prediction updates
        self.last_prediction_time = rospy.Time.now()
        self.prediction_timeout = rospy.Duration(0.5)  # Maximum age of predictions
        
    def setup_solver(self):
        # State variables
        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')
        states = vertcat(x, y, theta)
        n_states = states.size1()
        
        # Control variables
        v = SX.sym('v')
        omega = SX.sym('omega')
        controls = vertcat(v, omega)
        n_controls = controls.size1()
        
        # System dynamics (unicycle model)
        rhs = vertcat(v*cos(theta), v*sin(theta), omega)
        f = Function('f', [states, controls], [rhs])
        
        # Decision variables
        X = SX.sym('X', n_states, self.T + 1)
        U = SX.sym('U', n_controls, self.T)
        
        # Parameters: initial state + reference state + dynamic obstacles
        # For each timestep, we pass the predicted positions of all agents
        max_agents = 10  # Maximum number of dynamic agents to consider
        P = SX.sym('P', n_states + n_states + max_agents * 2 * self.T)
        
        # Initialize objective and constraints
        obj = 0
        g = []
        
        # Initial condition constraint
        g.append(X[:, 0] - P[:n_states])
        
        # Dynamic constraints and objective
        for k in range(self.T):
            st_next = X[:, k] + self.dt * f(X[:, k], U[:, k])
            g.append(X[:, k+1] - st_next)
            
            # Stage cost
            state_error = X[:, k] - P[n_states:2*n_states]
            control_cost = mtimes(U[:, k].T, mtimes(self.R, U[:, k]))
            state_cost = mtimes(state_error.T, mtimes(self.Q, state_error))
            obj += state_cost + control_cost
            
            # Dynamic obstacle avoidance cost
            for agent in range(max_agents):
                agent_pos_idx = 2*n_states + 2*agent + 2*k
                agent_x = P[agent_pos_idx]
                agent_y = P[agent_pos_idx + 1]
                
                # Distance to agent
                dist = sqrt((X[0,k] - agent_x)**2 + (X[1,k] - agent_y)**2)
                safety_margin = 0.5  # Minimum safe distance
                
                # Soft constraint for obstacle avoidance
                obj += self.obstacle_weight * fmax(0, safety_margin - dist)**2
        
        # Terminal cost
        terminal_error = X[:, -1] - P[n_states:2*n_states]
        obj += mtimes(terminal_error.T, mtimes(self.Q_terminal, terminal_error))
        
        # Create solver
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes'
        }
        nlp = {'x': vertcat(vec(X), vec(U)), 'f': obj, 'g': vertcat(*g), 'p': P}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        # Save dimensions for later use
        self.n_states = n_states
        self.n_controls = n_controls
        self.max_agents = max_agents
    
    def update_predictions(self, trajectories):
        """Update predicted trajectories from Trajectron++
        
        Args:
            trajectories: List of dictionaries containing:
                - agent_id: unique identifier
                - timestamps: list of prediction timestamps
                - positions: list of predicted (x,y) positions
                - probabilities: list of probabilities for each prediction
        """
        with self.prediction_lock:
            self.predicted_trajectories = trajectories
            self.last_prediction_time = rospy.Time.now()
    
    def get_predicted_positions(self, current_time: float) -> List[Tuple[float, float]]:
        """Get predicted positions of all agents at a specific time"""
        positions = []
        
        with self.prediction_lock:
            if (rospy.Time.now() - self.last_prediction_time) > self.prediction_timeout:
                return []  # Return empty list if predictions are too old
            
            for traj in self.predicted_trajectories:
                # Find the closest timestamp
                idx = np.argmin(abs(np.array(traj['timestamps']) - current_time))
                positions.append(traj['positions'][idx])
                
                if len(positions) >= self.max_agents:
                    break
        
        # Pad with far-away positions if we have fewer agents than max_agents
        while len(positions) < self.max_agents:
            positions.append((1000.0, 1000.0))  # Add far-away dummy positions
            
        return positions
    
    def get_control_input(self):
        # Prepare parameters
        current_time = rospy.Time.now().to_sec()
        
        # Get predicted positions for each timestep in the horizon
        prediction_params = []
        for t in range(self.T):
            future_time = current_time + t * self.dt
            positions = self.get_predicted_positions(future_time)
            prediction_params.extend([pos[0] for pos in positions])
            prediction_params.extend([pos[1] for pos in positions])
        
        # Combine all parameters
        p = np.concatenate([
            self.current_state,
            self.goal_state,
            prediction_params
        ])
        
        # Initial guess (warm start from previous solution if available)
        if hasattr(self, 'previous_solution'):
            x0 = self.previous_solution
        else:
            x0 = np.zeros((self.n_states*(self.T+1) + self.n_controls*self.T, 1))
        
        # Bounds
        lbx = -np.inf * np.ones(x0.shape)
        ubx = np.inf * np.ones(x0.shape)
        
        # Control bounds
        lbx[self.n_states*(self.T+1)::2] = -self.v_max
        ubx[self.n_states*(self.T+1)::2] = self.v_max
        lbx[self.n_states*(self.T+1)+1::2] = -self.w_max
        ubx[self.n_states*(self.T+1)+1::2] = self.w_max
        
        try:
            # Solve optimization problem
            sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, p=p)
            self.previous_solution = sol['x']
            u = sol['x'][self.n_states*(self.T+1):self.n_states*(self.T+1)+self.n_controls]
            
            # Create ROS message
            cmd_msg = Twist()
            cmd_msg.linear.x = float(u[0])
            cmd_msg.angular.z = float(u[1])
            return cmd_msg
            
        except:
            rospy.logwarn("MPC solver failed, emergency stop!")
            return Twist()  # Return zero velocity command
    
    def odom_callback(self, msg):
        # Update current state from odometry
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.current_state[2] = euler[2]
    
    def run(self):
        rate = rospy.Rate(int(1/self.dt))
        while not rospy.is_shutdown():
            try:
                cmd_msg = self.get_control_input()
                self.cmd_pub.publish(cmd_msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

if __name__ == '__main__':
    try:
        planner = MPCPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
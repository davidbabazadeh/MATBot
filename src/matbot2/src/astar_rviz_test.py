#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from math import atan2, ceil
from scipy.ndimage import binary_dilation

class AStarPlanner:
    def __init__(self):
        rospy.init_node('astar_planner')
        
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.path_pub = rospy.Publisher('/trajectory', Path, queue_size=1)
        
        self.map = None
        self.inflated_map = None
        self.resolution = None
        self.width = None
        self.height = None
        self.origin = None
        self.robot_radius = 0.105  # meters
        self.downsample_factor = 10  # Adjustable factor for downsampling
        
    def map_callback(self, msg):
        # Convert the flat map data into a 2D grid
        original_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        # Downsample the grid
        self.map = self.downsample_grid(original_grid, self.downsample_factor)
        self.resolution = msg.info.resolution * self.downsample_factor
        self.width = self.map.shape[1]
        self.height = self.map.shape[0]
        self.origin = msg.info.origin.position
        
        self.inflate_obstacles()
        rospy.loginfo(f"Map downsampled to {self.width}x{self.height} with resolution {self.resolution}")

    def downsample_grid(self, grid, factor):
        """
        Downsample a 2D grid by a given factor.
        :param grid: 2D numpy array (original grid)
        :param factor: Downsample factor (e.g., 10 for reducing dimensions by 10x)
        :return: Downsampled 2D numpy array
        """
        new_height = grid.shape[0] // factor
        new_width = grid.shape[1] // factor

        # Reshape and reduce
        downsampled = grid[:new_height * factor, :new_width * factor].reshape(
            new_height, factor, new_width, factor
        ).mean(axis=(1, 3))

        # Convert to binary occupancy (free: 0, occupied: 100)
        return np.where(downsampled > 50, 100, 0)

    def inflate_obstacles(self):
        # Convert robot radius to grid cells
        inflation_radius = ceil(self.robot_radius / self.resolution) * 0.5
        
        # Create a circular kernel for dilation
        y, x = np.ogrid[-inflation_radius:inflation_radius+1, -inflation_radius:inflation_radius+1]
        kernel = x**2 + y**2 <= inflation_radius**2
        
        # Create a binary map where obstacles are 1 and free space is 0
        binary_map = np.where(self.map > 50, 1, 0)
        
        # Dilate the binary map
        inflated_binary_map = binary_dilation(binary_map, kernel)
        
        # Convert back to occupancy grid format
        self.inflated_map = np.where(inflated_binary_map, 100, 0)
        
    def world_to_grid(self, x, y):
        gx = int((x - self.origin.x) / self.resolution)
        gy = int((y - self.origin.y) / self.resolution)
        return gx, gy
        
    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin.x
        y = gy * self.resolution + self.origin.y
        return x, y
        
    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        
    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.inflated_map[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors
        
    def astar(self, start, goal):
        start = self.world_to_grid(start[0], start[1])
        goal = self.world_to_grid(goal[0], goal[1])
        
        if self.inflated_map[start[1]][start[0]] != 0 or self.inflated_map[goal[1]][goal[0]] != 0:
            rospy.logwarn("Start or goal position is too close to an obstacle")
            return None
        
        open_set = set([start])
        closed_set = set()
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current[0], current[1]))
                    current = came_from[current]
                path.append(self.grid_to_world(start[0], start[1]))
                return path[::-1]
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
        
        return None
        
    def create_trajectory(self, path):
        trajectory = Path()
        trajectory.header.frame_id = "map"
        trajectory.header.stamp = rospy.Time.now()
        
        for i in range(len(path)):
            pose = PoseStamped()
            pose.header = trajectory.header
            pose.pose.position = Point(path[i][0], path[i][1], 0)
            
            if i < len(path) - 1:
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                theta = atan2(dy, dx)
            else:
                theta = 0
            
            pose.pose.orientation.z = np.sin(theta / 2)
            pose.pose.orientation.w = np.cos(theta / 2)
            
            trajectory.poses.append(pose)
        
        return trajectory
        
    def plan(self, start, goal):
        if self.inflated_map is None:
            rospy.logwarn("No map received yet")
            return
        
        path = self.astar(start, goal)
        if path:
            trajectory = self.create_trajectory(path)
            self.path_pub.publish(trajectory)
            return trajectory
        else:
            rospy.logwarn("No path found")

if __name__ == '__main__':
    planner = AStarPlanner()

    # Fetch start and goal positions from ROS parameters
    start_position = rospy.get_param('~start_position', [0.6, 0.0])  # Default: (0, 0)
    goal_position = rospy.get_param('~goal_position', [0.6, 0.9])    # Default: (5, 5)

    rospy.sleep(1)  # Allow time for the map to load
    planner.plan(tuple(start_position), tuple(goal_position))

    rospy.spin()

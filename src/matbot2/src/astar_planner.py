#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math

class AStarPlanner:
    def __init__(self):
        """
        A* Planner for TurtleBot3 navigation with dynamic map integration.
        """
        rospy.init_node("astar_planner", anonymous=True)

        # Parameters
        self.grid = None
        self.grid_width = 0
        self.grid_height = 0
        self.resolution = 1.0  # Default grid resolution (meters per cell)
        self.origin = (0, 0)  # Origin of the map in world coordinates

        # ROS publishers and subscribers
        self.occupancy_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.start = None
        self.goal = None

    class Node:
        def __init__(self, position, g=0, h=0, parent=None):
            self.position = position
            self.g = g  # Cost from start to current node
            self.h = h  # Heuristic cost to goal
            self.f = g + h  # Total cost
            self.parent = parent

        def __eq__(self, other):
            return self.position == other.position

    def map_callback(self, msg):
        """Callback for the /map topic."""
        self.resolution = msg.info.resolution
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        # Convert the flat occupancy grid data into a 2D numpy array
        grid_data = np.array(msg.data).reshape((self.grid_height, self.grid_width))
        self.grid = np.where(grid_data == 100, 1, 0)  # Convert to binary grid (1: obstacle, 0: free)

    def heuristic(self, current, goal):
        """Heuristic function: Euclidean distance."""
        return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

    def get_neighbors(self, node):
        """Get valid neighbors for a given node."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connectivity
        neighbors = []
        for dx, dy in directions:
            new_pos = (node.position[0] + dx, node.position[1] + dy)
            if 0 <= new_pos[0] < self.grid_height and 0 <= new_pos[1] < self.grid_width:
                if self.grid[new_pos[0], new_pos[1]] == 0:  # Free space
                    neighbors.append(self.Node(new_pos))
        return neighbors

    def reconstruct_path(self, node):
        """Reconstruct path from the goal node to the start node."""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get the correct order

    def plan(self, start, goal):
        """
        Plan a path from start to goal using A* algorithm.
        :param start: Tuple of (x, y) for the start position
        :param goal: Tuple of (x, y) for the goal position
        :return: List of path coordinates
        """
        if self.grid is None:
            rospy.logwarn("Map not received yet!")
            return None

        start_node = self.Node(start)
        goal_node = self.Node(goal)

        open_set = [start_node]
        closed_set = []

        while open_set:
            # Get node with the lowest f cost
            current_node = min(open_set, key=lambda n: n.f)
            open_set.remove(current_node)
            closed_set.append(current_node)

            # Check if goal is reached
            if current_node == goal_node:
                return self.reconstruct_path(current_node)

            # Expand neighbors
            for neighbor in self.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                # Calculate costs
                tentative_g = current_node.g + 1  # Uniform cost for moving one cell
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g >= neighbor.g:
                    continue  # Not a better path

                # Update neighbor
                neighbor.g = tentative_g
                neighbor.h = self.heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

        return None  # No path found

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates."""
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return x, y

    def publish_path(self, path):
        """
        Publish the planned path as a ROS Path message.
        :param path: List of (x, y) path points
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for point in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x, pose.pose.position.y = self.grid_to_world(point[0], point[1])
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

if __name__ == "__main__":
    planner = AStarPlanner()

    # Define start and goal positions in world coordinates
    start_world = (0.0, 0.0)  # Example start position
    goal_world = (2.0, 2.0)  # Example goal position

    rospy.sleep(1)  # Allow map to be received
    start_grid = planner.world_to_grid(*start_world)
    goal_grid = planner.world_to_grid(*goal_world)

    rospy.loginfo("Planning path...")
    path = planner.plan(start_grid, goal_grid)
    if path:
        rospy.loginfo(f"Path found: {path}")
        planner.publish_path(path)
    else:
        rospy.logwarn("No path found!")
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped


class AStarPlanner:
    def __init__(self, grid):
        """
        A* Planner for TurtleBot3 navigation
        :param grid: 2D numpy array representing the occupancy grid
        """
        self.grid = grid
        self.grid_height, self.grid_width = grid.shape
        self.occupancy_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)

    class Node:
        def __init__(self, position, g=0, h=0, parent=None):
            self.position = position
            self.g = g  # Cost from start to current node
            self.h = h  # Heuristic cost to goal
            self.f = g + h  # Total cost
            self.parent = parent

        def __eq__(self, other):
            return self.position == other.position

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
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_occupancy_grid(self):
        """
        Publish the occupancy grid as a ROS OccupancyGrid message.
        """
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.info.resolution = 1.0  # Grid resolution (1m per cell)
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = 0
        msg.info.origin.position.y = 0
        msg.info.origin.position.z = 0
        msg.data = self.grid.flatten().tolist()

        self.occupancy_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("astar_planner")

    # Example grid (1: obstacle, 0: free space)
    grid = np.zeros((10, 10))
    grid[3:7, 5] = 1  # Example obstacle

    planner = AStarPlanner(grid)

    start = (0, 0)
    goal = (9, 9)

    rospy.loginfo("Planning path...")
    path = planner.plan(start, goal)
    if path:
        rospy.loginfo(f"Path found: {path}")
        planner.publish_path(path)
        planner.publish_occupancy_grid()
    else:
        rospy.logwarn("No path found!")

    rospy.spin()
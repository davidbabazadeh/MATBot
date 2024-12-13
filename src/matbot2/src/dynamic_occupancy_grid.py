#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, TransformStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion, quaternion_matrix
import tf2_ros
import tf2_geometry_msgs
import os

class DynamicOccupancyGridNode:
    def __init__(self):
        rospy.init_node('dynamic_occupancy_grid_node', anonymous=True)
        
        # Initialize parameters
        self.agent_radius = rospy.get_param('~agent_radius', 0.3)  # radius of agents in meters
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 6])
        self.update_rate = rospy.get_param('~update_rate', 10.0)  # Hz
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load static map
        image_path = os.path.expanduser('~/MATBot/saved_maps/imagemap.jpg')
        self.static_map = self.load_static_map(image_path)
        
        if self.static_map is None:
            rospy.logerr("Failed to initialize static map")
            return
            
        # Store map properties
        self.map_resolution = 0.000371  # meters per pixel
        self.map_width = self.static_map.shape[1]
        self.map_height = self.static_map.shape[0]
        
        # Initialize dynamic layer for trajectory predictions
        self.dynamic_layer = np.zeros_like(self.static_map)
        
        # Create subscribers for AR agent trajectories
        self.trajectory_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/ar_{tag_id}/predicted_trajectory',
                PoseArray,
                self.trajectory_callback,
                callback_args=tag_id
            ) for tag_id in self.ar_tag_ids
        }
        
        # Publisher for combined occupancy grid
        self.grid_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        
        # Store transform from odom to map when available
        self.odom_to_map_transform = None
        
        # Wait for transform to become available
        rospy.loginfo("Waiting for transform between map and odom frames...")
        try:
            self.odom_to_map_transform = self.tf_buffer.lookup_transform(
                "map",    # target frame
                "odom",   # source frame
                rospy.Time(0),  # get latest transform
                rospy.Duration(5.0)  # timeout after 5 seconds
            )
            rospy.loginfo("Transform from odom to map frame received")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get transform: {e}")
        
        # Timer for publishing updated grid
        rospy.Timer(rospy.Duration(1.0/self.update_rate), self.publish_grid)
        
        rospy.loginfo("Dynamic Occupancy Grid Node initialized")

    def transform_pose_to_map_frame(self, pose, stamp):
        """Transform a pose from odom frame to map frame"""
        try:
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map",
                "odom",
                stamp,
                rospy.Duration(0.1)
            )
            
            # Create PoseStamped message
            pose_stamped = tf2_geometry_msgs.PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = "odom"
            pose_stamped.header.stamp = stamp
            
            # Transform pose
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            
            return pose_transformed.pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform failed: {e}")
            return None

    def world_to_grid(self, x, y):
        """Convert world coordinates (in map frame) to grid coordinates"""
        # Convert from world coordinates to grid coordinates
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        return grid_x, grid_y

    def trajectory_callback(self, msg, tag_id):
        """Process predicted trajectory and update dynamic layer"""
        # Clear previous predictions for this agent
        self.dynamic_layer = np.zeros_like(self.static_map)
        
        # Process each predicted pose
        for i, pose in enumerate(msg.poses):
            # Transform pose from odom to map frame
            map_pose = self.transform_pose_to_map_frame(pose, msg.header.stamp)
            if map_pose is None:
                continue
                
            # Decay factor based on prediction time
            certainty = max(20, 100 - i * 5)
            
            # Convert position to grid coordinates
            grid_x, grid_y = self.world_to_grid(map_pose.position.x, map_pose.position.y)
            
            # Mark cells as occupied in dynamic layer
            self.mark_cell_occupied(grid_x, grid_y, certainty)

    def mark_cell_occupied(self, x, y, value=100):
        """Mark a cell and its neighbors within agent_radius as occupied"""
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return

        # Calculate cell radius in grid cells
        cell_radius = int(self.agent_radius / self.map_resolution)
        
        # Create a circle mask
        y_indices, x_indices = np.ogrid[-cell_radius:cell_radius+1, -cell_radius:cell_radius+1]
        mask = x_indices**2 + y_indices**2 <= cell_radius**2
        
        # Calculate bounds for the circle
        x_min = max(0, x - cell_radius)
        x_max = min(self.map_width, x + cell_radius + 1)
        y_min = max(0, y - cell_radius)
        y_max = min(self.map_height, y + cell_radius + 1)
        
        # Adjust mask to match bounded region
        mask_x_min = max(0, -(x - cell_radius))
        mask_x_max = mask_x_min + (x_max - x_min)
        mask_y_min = max(0, -(y - cell_radius))
        mask_y_max = mask_y_min + (y_max - y_min)
        
        # Update dynamic layer with mask
        bounded_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        self.dynamic_layer[y_min:y_max, x_min:x_max][bounded_mask] = value

    def load_static_map(self, image_path):
        """Load and process static map from image"""
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            return None
            
        # Convert to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to create a binary image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Convert to occupancy grid format (0-100)
        return np.where(binary_image == 255, 0, 100)

    def publish_grid(self, event=None):
        """Publish combined static and dynamic occupancy grid"""
        if self.odom_to_map_transform is None:
            rospy.logwarn_throttle(1.0, "No transform available between odom and map frames")
            return
            
        # Combine static and dynamic layers
        combined_grid = np.maximum(self.static_map, self.dynamic_layer)
        
        # Create and publish occupancy grid message
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        
        # Set origin
        msg.info.origin.position.x = 0
        msg.info.origin.position.y = 0
        msg.info.origin.position.z = 0
        msg.info.origin.orientation.w = 1.0
        
        # Convert to int8 and flatten
        msg.data = combined_grid.flatten().astype(np.int8).tolist()
        
        self.grid_pub.publish(msg)

def main():
    try:
        node = DynamicOccupancyGridNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
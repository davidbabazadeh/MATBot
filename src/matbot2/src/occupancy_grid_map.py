#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, TransformStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
import os

class OccupancyGridMapNode:
    def __init__(self):
        rospy.init_node('occupancy_grid_map_node', anonymous=True)
        self.bridge = CvBridge()

        # Parameters
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 2, 3])
        self.map_reference_tag = rospy.get_param('~map_reference_tag', 0)
        self.trajectory_width = rospy.get_param('~trajectory_width', 5)
        self.update_rate = rospy.get_param('~update_rate', 10.0)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.map_initialized = False
        self.reference_tag_transform = None
        self.static_occupancy_grid = None  # Store the static grid message

        # Initialize publisher
        self.occupancy_grid_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size=10)

        # Load and process static map first
        if not self.load_static_map():
            rospy.logerr("Failed to load static map. Shutting down node.")
            rospy.signal_shutdown("Failed to load static map")
            return

        # Subscribe to AR tag detections
        self.ar_sub = rospy.Subscriber(
            '/ar_pose_marker',
            AlvarMarkers,
            self.ar_callback
        )

        # Store latest trajectories
        self.latest_trajectories = {}
        
        # Subscribe to predicted trajectories
        self.trajectory_subs = {
            tag_id: rospy.Subscriber(
                f'/agent/ar_{tag_id}/predicted_trajectory',
                PoseArray,
                self.trajectory_callback,
                callback_args=tag_id
            ) for tag_id in self.ar_tag_ids
        }

        # Start periodic publishing
        rospy.Timer(rospy.Duration(1.0/self.update_rate), self.publish_integrated_grid)

        # Publish static map immediately and periodically until AR tag is detected
        rospy.Timer(rospy.Duration(0.5), self.publish_static_map)

    def load_static_map(self):
        """Load and process the static map"""
        try:
            image_path = os.path.expanduser('~/MATBot/saved_maps/imagemap.jpg')
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                rospy.logerr(f"Failed to load image from path: {image_path}")
                return False

            # Convert to grayscale and threshold
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, self.static_binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            
            # Initialize grid info
            self.grid_info = OccupancyGrid().info
            self.grid_info.resolution = 0.000371
            self.grid_info.width = self.static_binary_image.shape[1]
            self.grid_info.height = self.static_binary_image.shape[0]
            self.grid_info.origin.position.x = 0
            self.grid_info.origin.position.y = 0
            self.grid_info.origin.position.z = 0
            self.grid_info.origin.orientation.w = 1.0

            # Create static occupancy grid message
            self.static_occupancy_grid = OccupancyGrid()
            self.static_occupancy_grid.header.frame_id = "map"
            self.static_occupancy_grid.info = self.grid_info
            
            # Convert binary image to occupancy data
            data = []
            for i in range(self.static_binary_image.shape[0]):
                for j in range(self.static_binary_image.shape[1]):
                    data.append(0 if self.static_binary_image[i, j] == 255 else 100)
            
            self.static_occupancy_grid.data = data
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in loading static map: {e}")
            return False

    def publish_static_map(self, event=None):
        """Publish static map until AR tag is detected"""
        if not self.map_initialized and self.static_occupancy_grid is not None:
            self.static_occupancy_grid.header.stamp = rospy.Time.now()
            print("static map")
            self.occupancy_grid_publisher.publish(self.static_occupancy_grid)

    def ar_callback(self, msg):
        """Process AR tag detections and update map alignment"""
        if self.map_initialized:
            return

        for marker in msg.markers:
            if marker.id == self.map_reference_tag:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "odom",
                        f"ar_marker_{self.map_reference_tag}",
                        rospy.Time(0),
                        rospy.Duration(1.0)
                    )
                    self.reference_tag_transform = transform
                    self.map_initialized = True
                    rospy.loginfo(f"Map initialized with reference tag {self.map_reference_tag}")
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"Failed to get transform for reference tag: {e}")
                break

    def broadcast_map_transform(self):
        """Broadcast transform from map to odom based on reference AR tag"""
        if not self.map_initialized:
            # Broadcast identity transform until AR tag is detected
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "odom"
            transform.child_frame_id = "map"
            transform.transform.rotation.w = 1.0
        else:
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "odom"
            transform.child_frame_id = "map"
            transform.transform.translation.x = -self.reference_tag_transform.transform.translation.x
            transform.transform.translation.y = -self.reference_tag_transform.transform.translation.y
            transform.transform.translation.z = 0
            transform.transform.rotation.x = 0
            transform.transform.rotation.y = 0
            transform.transform.rotation.z = 0
            transform.transform.rotation.w = 1

        self.tf_broadcaster.sendTransform(transform)

    def initialize_grid_info(self, binary_image):
        """Initialize the grid info structure"""
        self.grid_info = OccupancyGrid().info
        self.grid_info.resolution = 0.000371  # meters per pixel
        self.grid_info.width = binary_image.shape[1]
        self.grid_info.height = binary_image.shape[0]
        self.grid_info.origin.position.x = 0
        self.grid_info.origin.position.y = 0
        self.grid_info.origin.position.z = 0
        self.grid_info.origin.orientation.w = 1.0

    def transform_pose_to_map(self, pose, from_frame):
        """Transform a pose from given frame to map frame"""
        if not self.map_initialized:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                from_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            pose_stamped = tf2_geometry_msgs.PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = from_frame
            pose_stamped.header.stamp = rospy.Time.now()
            
            transformed_pose = tf2_geometry_msgs.do_transform_pose(
                pose_stamped,
                transform
            )
            return transformed_pose.pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform failed: {e}")
            return None

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.grid_info.origin.position.x) / self.grid_info.resolution)
        grid_y = int((y - self.grid_info.origin.position.y) / self.grid_info.resolution)
        return grid_x, grid_y

    def trajectory_callback(self, msg, ar_tag_id):
        """Process incoming trajectory predictions"""
        if not self.map_initialized:
            return

        trajectory_points = []
        
        for pose in msg.poses:
            # Transform pose from odom to map frame
            transformed_pose = self.transform_pose_to_map(pose, "odom")
            if transformed_pose is None:
                continue
                
            x = transformed_pose.position.x
            y = transformed_pose.position.y
            grid_x, grid_y = self.world_to_grid(x, y)
            
            if 0 <= grid_x < self.grid_info.width and 0 <= grid_y < self.grid_info.height:
                trajectory_points.append((grid_x, grid_y))
        
        self.latest_trajectories[ar_tag_id] = trajectory_points

    def draw_trajectory(self, grid_image, points):
        """Draw trajectory on the grid image"""
        if not points:
            return grid_image
        
        # Convert points to numpy array for opencv
        points = np.array(points, dtype=np.int32)
        
        if len(points) > 1:
            cv2.polylines(grid_image, [points], False, 0, thickness=self.trajectory_width)
        
        return grid_image

    def create_integrated_grid(self):
        """Create occupancy grid with both static and dynamic obstacles"""
        if self.static_binary_image is None or not self.map_initialized:
            return None
            
        # Start with copy of static binary image
        integrated_image = self.static_binary_image.copy()
        
        # Add trajectories
        for trajectory in self.latest_trajectories.values():
            integrated_image = self.draw_trajectory(integrated_image, trajectory)
        
        # Create occupancy grid message
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = "map"
        occupancy_grid.info = self.grid_info
        
        # Convert binary image to occupancy data
        data = []
        for i in range(integrated_image.shape[0]):
            for j in range(integrated_image.shape[1]):
                # 255 (white) is free space, 0 (black) is occupied
                data.append(0 if integrated_image[i, j] == 255 else 100)
        
        occupancy_grid.data = data
        return occupancy_grid

    def publish_integrated_grid(self, event=None):
        """Publish the integrated occupancy grid"""
        if not self.map_initialized:
            return

        try:
            # Broadcast the current map transform
            self.broadcast_map_transform()
            
            # Create and publish the occupancy grid
            occupancy_grid = self.create_integrated_grid()
            if occupancy_grid is not None:
                print("printing dynamic")
                self.occupancy_grid_publisher.publish(occupancy_grid)
        except Exception as e:
            rospy.logerr(f"Error publishing integrated grid: {e}")

if __name__ == '__main__':
    try:
        OccupancyGridMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
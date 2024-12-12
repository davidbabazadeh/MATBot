#!/usr/bin/env python

import rospy
import cv2
#import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import os

class OccupancyGridMapNode:
    def __init__(self):
        rospy.init_node('occupancy_grid_map_node', anonymous=True)
        self.bridge = CvBridge()

         # Initialize publisher
        #self.occupancy_grid_publisher = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)

        # Test initializing publisher to publish to /map
        self.occupancy_grid_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size=10)
        #self.image_subscriber = rospy.Subscriber("/input_image", Image, self.image_callback)
        # Load the image
        image_path = os.path.expanduser('~/MATBot/saved_maps/imagemap.jpg')
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            rospy.logerr(f"Failed to load image from path: {image_path}")
            return

        
        # Convert ROS Image message to OpenCV format
        try:
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Thresholding to create a binary image
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

            # Create occupancy grid from binary image
            occupancy_grid = self.create_occupancy_grid(binary_image)

            # Publish the occupancy grid
            self.occupancy_grid_publisher.publish(occupancy_grid)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

        
        rospy.spin()

    # def image_callback(self, msg):
    #     try:
    #         # Convert ROS Image message to OpenCV format
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
    #         # Convert to grayscale
    #         gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    #         # Thresholding to create a binary image
    #         _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    #         # Create occupancy grid from binary image
    #         occupancy_grid = self.create_occupancy_grid(binary_image)

    #         # Publish the occupancy grid
    #         self.occupancy_grid_publisher.publish(occupancy_grid)

    #     except Exception as e:
    #         rospy.logerr(f"Error in image_callback: {e}")

    def create_occupancy_grid(self, binary_image):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = "map"
        
        # Set resolution and dimensions of the grid
        occupancy_grid.info.resolution = 0.05  # meters per pixel
        occupancy_grid.info.width = binary_image.shape[1]
        occupancy_grid.info.height = binary_image.shape[0]
        
        # Set origin of the grid
        occupancy_grid.info.origin.position.x = 0
        occupancy_grid.info.origin.position.y = 0
        occupancy_grid.info.origin.position.z = 0
        occupancy_grid.info.origin.orientation.w = 1.0
        
        # Fill occupancy grid data based on binary image
        data = []
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if binary_image[i, j] == 255:  # Free space
                    data.append(0)
                else:  # Occupied space
                    data.append(100)

        occupancy_grid.data = data

        return occupancy_grid

if __name__ == '__main__':
    try:
        OccupancyGridMapNode()
    except rospy.ROSInterruptException:
        pass



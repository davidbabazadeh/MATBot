#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from math import ceil

from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

import geometry_msgs.msg
import tf2_geometry_msgs

def get_transform_from_position_orientation(position, orientation, target_frame):
    # Initialize the tf2 buffer and listener
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    # Manually create a TransformStamped message
    transform_stamped = geometry_msgs.msg.TransformStamped()
    
    # Set the header information
    transform_stamped.header.stamp = rospy.Time.now()
    transform_stamped.header.frame_id = target_frame  # This will be the target frame, e.g., 'base_link'
    
    # Set the translation (position)
    transform_stamped.transform.translation.x = position.x
    transform_stamped.transform.translation.y = position.y
    transform_stamped.transform.translation.z = position.z
    
    # Set the rotation (orientation) as quaternion
    transform_stamped.transform.rotation.x = orientation.x
    transform_stamped.transform.rotation.y = orientation.y
    transform_stamped.transform.rotation.z = orientation.z
    transform_stamped.transform.rotation.w = orientation.w
    
    try:
        # You can use the transform you created to look up transforms in the TF tree
        transform = tf_buffer.lookup_transform(target_frame, transform_stamped.header.frame_id, rospy.Time(0))
        rospy.loginfo("Transform from {} to {}: {}".format(transform_stamped.header.frame_id, target_frame, transform))
        return transform
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn("Transform lookup failed: %s", e)
        return None


class OccupancyGridUpdater:
    def __init__(self):
        rospy.init_node('occupancy_grid_updater')
        # Get list of AR tag IDs to track
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [3])  # List of AR tag IDs to track

        
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.future_horizon = 3
        # delete
        # self.obstacle_sub = rospy.Subscriber('/agents/{tag_id}/state_history', Float64MultiArray, self.obstacle_callback)

        # Create publishers for each AR tag
        self.state_pubs = {
            tag_id: rospy.Subscriber(
                f'/agent/{tag_id}/state_history',
                Float64MultiArray,
                self.obstacle_callback(tag_id)
            ) for tag_id in self.ar_tag_ids
        }
        self.map_pub = rospy.Publisher('/dynamic_map', OccupancyGrid, queue_size=1)
        
        self.current_map = None
        self.current_agent_states = {} # agent i : state_hist
        self.obstacle_radius = 0.075  # meters
        
        self.update_rate = rospy.Rate(10)  # 10 Hz
        
    def map_callback(self, msg):
        self.current_map = msg
        
    def obstacle_callback(self, tag_id):
        def callback(self, msg):
            self.current_agent_states[tag_id] = msg[-8:]
        return callback
        
    def update_map(self):
        if self.current_map is None:
            return
        
        updated_map = self.current_map

        # TODO: why is this reshape here?
        map_data = np.array(updated_map.data).reshape((updated_map.info.height, updated_map.info.width))
        
        # Calculate the radius in grid cells
        grid_radius = ceil(self.obstacle_radius / updated_map.info.resolution)

        obstacle_queue = []

        tfBuffer = tf2_ros.Buffer()
        tfListener = tf2_ros.TransformListener(tfBuffer)
        cam_to_map = tfBuffer.lookup_transform(self.current_map.header.frame_id, "usb_cam", rospy.Time(0), rospy.Duration(.5))
        # cam_to_map = get_transform_from_position_orientation(self.current_map.info.origin.position, self.current_map.info.origin.orientation, "usb_cam")
        
        for _, state in self.current_agent_states.items():
            # TODO: generate future datapoints with smooth velocity (from previous timesteps)
            x, y, vx, vy, ax, ay, h, t = state

            ar_pose_cam = PoseStamped()
            ar_pose_cam.pose.position.x = x
            ar_pose_cam.pose.position.y = y
            ar_pose_cam.pose.position.z = 0

            quat = quaternion_from_euler(0, 0, h)
            ar_pose_cam.pose.orientation.x = quat[0]
            ar_pose_cam.pose.orientation.y = quat[1]
            ar_pose_cam.pose.orientation.z = quat[2]
            ar_pose_cam.pose.orientation.w = quat[3]

            ar_pose_map = do_transform_pose(ar_pose_cam, cam_to_map)
            roll, pitch, yaw = euler_from_quaternion(ar_pose_map.pose.orientation.x, ar_pose_map.pose.orientation.y, ar_pose_map.pose.orientation.z, ar_pose_map.pose.orientation.w)

            x_map = ar_pose_map.pose.position.x
            y_map = ar_pose_map.pose.position.y
            vx_map = vx * np.cos(yaw) - vy * np.sin(yaw)
            vy_map = vx * np.sin(yaw) + vy * np.cos(yaw)
            obstacle_queue.append([[x_map + k*vx_map, y_map + k*vy_map, 0, 0, 0, 0, 0, 0] for k in range(self.future_horizon)])

        
        while obstacle_queue:
            obstacle = obstacle_queue.pop(0)
            center_x = int((obstacle[-1][0] - updated_map.info.origin.position.x) / updated_map.info.resolution)
            center_y = int((obstacle[-1][1] - updated_map.info.origin.position.y) / updated_map.info.resolution)
            
            # Create a circular mask
            y, x = np.ogrid[-grid_radius:grid_radius+1, -grid_radius:grid_radius+1]
            mask = x*x + y*y <= grid_radius*grid_radius
            
            # Apply the mask to the map
            for dy in range(-grid_radius, grid_radius+1):
                for dx in range(-grid_radius, grid_radius+1):
                    if mask[dy+grid_radius, dx+grid_radius]:
                        map_y = center_y + dy
                        map_x = center_x + dx
                        if 0 <= map_x < updated_map.info.width and 0 <= map_y < updated_map.info.height:
                            map_data[map_y, map_x] = 100  # Mark as occupied
        
        updated_map.data = map_data.flatten().tolist()
        self.map_pub.publish(updated_map)
        
    def run(self):
        while not rospy.is_shutdown():
            self.update_map()
            self.update_rate.sleep()

if __name__ == '__main__':
    try:
        updater = OccupancyGridUpdater()
        updater.run()
    except rospy.ROSInterruptException:
        pass

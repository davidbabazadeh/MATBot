#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from math import ceil

class OccupancyGridUpdater:
    def __init__(self):
        rospy.init_node('occupancy_grid_updater')
        # Get list of AR tag IDs to track
        self.ar_tag_ids = rospy.get_param('~ar_tag_ids', [0, 1, 2, 3])  # List of AR tag IDs to track

        
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
        self.obstacle_radius = 0.105  # meters
        
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
        
        for _, state in self.current_agent_states.items():
            center_x = int((state[0] - updated_map.info.origin.position.x) / updated_map.info.resolution)
            center_y = int((state[1] - updated_map.info.origin.position.y) / updated_map.info.resolution)

            # TODO: generate future datapoints with smooth velocity (from previous timesteps)
            x, y, vx, vy, ax, ay, h, t = state
            obstacle_queue.append([[x + vx, y + vy, 0, 0, 0, 0, 0, 0] for k in range(self.future_horizon)])
            
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

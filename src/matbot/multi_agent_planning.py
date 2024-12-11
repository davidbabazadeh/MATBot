#!/usr/bin/env python

import math
import tf2_ros
import rospy
import tf
import numpy as np

import matplotlib.pyplot as plt


# T++ imports

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Final, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trajdata.visualization.vis as trajdata_vis
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm
from trajdata import AgentBatch, AgentType, UnifiedDataset

import trajectron.visualization as visualization
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron


if torch.cuda.is_available():
    device = "cuda:0"
    torch.cuda.set_device(0)
else:
    device = "cpu"

base_checkpoint = 20

def load_model(
    model_dir: str,
    device: str,
    epoch: int = 10,
    custom_hyperparams: Optional[Dict] = None,
):
    save_path = Path(model_dir) / f"model_registrar-{epoch}.pt"

    model_registrar = ModelRegistrar(model_dir, device)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams

# importing trajectron++ (tpp)
model_path = '../../pretrained_models/nusc_mm_base_tpp-11_Sep_2022_19_15_45'
tpp, _ = load_model(model_path, device, epoch=base_checkpoint)  # not using custom paramters

# https://github.com/NVlabs/adaptive-prediction/blob/main/experiments/nuScenes/full_per_agent_eval.py#L456
# structure to make call to T++ with tpp, online_batch is of type AgentBatch (in trajdata) and has a lot of inputs
# output is possibly? stored in model_eval_dict: DefaultDict[str, Union[List[int], List[float]]]
# TODO: make call to T++ with out collected data. possibly done using this function

def per_agent_eval(
    curr_agent: str,
    model: Trajectron,
    model_name: str,
    batch: AgentBatch,
    agent_ts: int,
    model_eval_dict: DefaultDict[str, Union[List[int], List[float]]],
    plot=True,
):
    with torch.no_grad():
        if plot:
            plot_outputs(
                online_eval_dataset,
                dataset_idx=batch.data_idx[0].item(),
                model=model,
                model_name=model_name,
                agent_ts=agent_ts,
                subfolder="per_agent_lyft/",
            )

        model_perf = defaultdict(lambda: defaultdict(list))
        eval_results: Dict[
            AgentType, Dict[str, torch.Tensor]
        ] = model.predict_and_evaluate_batch(batch)
        for agent_type, metric_dict in eval_results.items():
            for metric, values in metric_dict.items():
                model_perf[agent_type][metric].append(values.cpu().numpy())

        for idx, metric in enumerate(metrics_list):
            if len(model_perf[AgentType.VEHICLE]) == 0:
                break

            metric_values = np.concatenate(
                model_perf[AgentType.VEHICLE][metric]
            ).tolist()
            if idx == 0:
                model_eval_dict["agent_ts"].extend([agent_ts] * len(metric_values))

            model_eval_dict[metric].extend(metric_values)


# per_agent_eval(
#     curr_agent,
#     tpp,
#     "Base",
#     online_batch,
#     agent_ts,
#     base_dict,
#     plot=plot_per_step,
# )


def plot_trajectory(waypoints):
    # Extracting x, y and theta values from the waypoints
    x_vals = [point[0] for point in waypoints]
    y_vals = [point[1] for point in waypoints]
    theta_vals = [point[2] for point in waypoints]
    
    # Plotting the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, '-o', label='Trajectory')
    
    # Plotting the start and end points
    plt.scatter(x_vals[0], y_vals[0], color='green', s=100, zorder=5, label='Start')
    plt.scatter(x_vals[-1], y_vals[-1], color='red', s=100, zorder=5, label='End')
    
    # Plotting orientation arrows along the trajectory
    for x, y, theta in waypoints:
        plt.arrow(x, y, 0.05 * np.cos(theta), 0.05 * np.sin(theta), head_width=0.01, head_length=0.005, fc='blue', ec='blue')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

def bezier_curve(p0, p1, p2, p3, t):
    """Calculate a point on a cubic Bezier curve defined by p0, p1, p2, and p3 at parameter t."""
    return (1 - t)*(1 - t)*(1 - t)*p0 + 3*(1 - t)*(1 - t)*t*p1 + 3*(1 - t)*t*t*p2 + t*t*t*p3 ## TODO

def generate_bezier_waypoints(x1, y1, theta1, x2, y2, theta2, offset=1.0, num_points=10):
    # 1. Calculate direction vector based on yaw
    direction_start = np.array([np.cos(theta1), np.sin(theta1)])
    direction_end = np.array([-np.cos(theta2), -np.sin(theta2)])  # Opposite direction for the end point

    # 2. Determine control points based on yaw and offset
    control1 = np.array([x1, y1]) + offset * direction_start
    control2 = np.array([x2, y2]) + offset * direction_end

    # 3. Sample points along the Bezier curve
    t_values = np.linspace(0, 1, num_points)
    waypoints = [bezier_curve(np.array([x1, y1]), control1, control2, np.array([x2, y2]), t) for t in t_values]

    # 4. Determine orientation at each point
    thetas = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        thetas.append(np.arctan2(dy, dx))
    thetas.append(thetas[-1])  # Repeat last orientation for the last waypoint

    waypoints_with_theta = [(waypoints[i][0], waypoints[i][1], thetas[i]) for i in range(len(waypoints))]

    return waypoints_with_theta

def plan_curved_trajectory(target_position):
    """
    Plan a curved trajectory for a Roomba-type robot from current_position to target_position using a Bezier curve.
    
    Parameters:
    - target_position: A tuple (x, y) representing in the robot base frame.
    
    Returns:
    - A list of waypoints [(x, y, theta), ...] where type can be 'rotate' or 'move' and value is the amount to rotate in radians or move in meters.
    """
    tfBuffer = tf2_ros.Buffer() #TODO
    tfListener = tf2_ros.TransformListener(tfBuffer) #TODO
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('odom', 'base_footprint', rospy.Time(0), rospy.Duration(5.0)) ## TODO: apply a lookup transform between our world frame and turtlebot frame
            print(trans)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Error: " + e)
            continue
    x1, y1 = trans.transform.translation.x, trans.transform.translation.y
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [trans.transform.rotation.x, trans.transform.rotation.y,
            trans.transform.rotation.z, trans.transform.rotation.w])
    
    x2 = target_position[0] + x1*np.cos(yaw) - y1*np.sin(yaw) ## TODO: how would you get x2 from our target position? remember this is relative to x1 
    y2 = target_position[1] + x1*np.sin(yaw) + y1*np.cos(yaw) ## TODO: how would you get x2 from our target position? remember this is relative to x1 

    waypoints = generate_bezier_waypoints(x1, y1, yaw, x2, y2, yaw, offset=0.2, num_points=50)
    plot_trajectory(waypoints)

    return waypoints



if __name__ == '__main__':
    rospy.init_node('turtlebot_controller', anonymous=True)
    trajectory = plan_curved_trajectory([0.2,0.2])

    # For testing
    trajectory = generate_bezier_waypoints(0.0, 0.0, np.pi/2, 0.2, 0.2, np.pi/2, offset=0.2, num_points=100)
    plot_trajectory(trajectory)

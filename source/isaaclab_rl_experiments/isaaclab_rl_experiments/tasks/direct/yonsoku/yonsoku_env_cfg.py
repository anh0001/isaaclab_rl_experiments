# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Yonsoku quadruped robot in Isaac Lab."""

import math
import numpy as np

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from isaaclab_rl_experiments.assets.robots.yonsoku.robot_config import YONSOKU_CFG

@configclass
class YonsokuSceneCfg(InteractiveSceneCfg):
    """Custom scene configuration for Yonsoku with sensors as direct attributes."""
    num_envs = 4096
    env_spacing = 5.0
    replicate_physics = True
    # Define sensor as a direct attribute
    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/yonsoku_robot/.*3",  # This matches all foot links
        history_length=1,
        update_period=0.0  # Update every simulation step
    )

@configclass
class YonsokuVelocityEnvCfg(DirectRLEnvCfg):
    """Environment configuration for Yonsoku quadruped robot velocity control."""
    
    # Simulation parameters
    decimation = 4
    episode_length_s = 20.0
    
    # Robot configuration
    robot_cfg = YONSOKU_CFG
    
    # Simulation setup
    sim = SimulationCfg(dt=1/120.0, render_interval=decimation)
    
    # Scene configuration - now using the custom scene config
    scene = YonsokuSceneCfg()
    
    # Observation and action spaces
    observation_space = 48  # Base state + joint positions + velocities + commands
    action_space = 12  # 12 joint actuators (3 per leg)
    state_space = 0
    
    # Define joint names for easy access
    leg_joint_names = {
        "RF": ["RF_JOINT1", "RF_JOINT2", "RF_JOINT3"],
        "RB": ["RB_JOINT1", "RB_JOINT2", "RB_JOINT3"],
        "LB": ["LB_JOINT1", "LB_JOINT2", "LB_JOINT3"],
        "LF": ["LF_JOINT1", "LF_JOINT2", "LF_JOINT3"]
    }
    
    # Control parameters
    lin_velocity_scale = 1.0
    ang_velocity_scale = 0.5
    dof_position_scale = 1.0
    dof_velocity_scale = 0.05
    action_scale = 0.25  # Changed from 1.0 to match A1's scale
    
    # Reward scales - Modified to match Unitree A1 approach
    reward_scales = {
        "lin_vel_z": -2.0,            # Penalize vertical movement
        "ang_vel_xy": -0.05,          # Penalize roll and pitch angular velocity
        "orientation": -2.5,          # Penalize non-flat orientation (like flat_orientation_l2)
        "dof_torques": -0.0002,       # Match the A1 dof_torques_l2 weight
        "dof_acceleration": -2.5e-7,  # Match the A1 dof_acc_l2 weight
        "base_height": -2.5,          # Penalize incorrect base height
        "joint_regularization": -0.001, # Regularize joint positions
        "action_rate": -0.01,         # Penalize rapid action changes
        "feet_air_time": 0.25,        # Reward feet air time (from A1 flat config)
        "tracking_lin_vel_xy": 1.5,   # Match A1's track_lin_vel_xy_exp weight
        "tracking_ang_vel_z": 0.75    # Match A1's track_ang_vel_z_exp weight
    }
    
    # Initial randomization
    start_position_noise = 0.1
    start_rotation_noise = 0.1
    
    # Termination conditions
    termination_height = 0.25
    termination_roll = 1.0   # Reduced to be more sensitive to roll problems
    termination_pitch = 1.0  # Reduced to be more sensitive to pitch problems
    
    # Command ranges - set to similar ranges as would be used for A1
    command_x_range = [0.0, 1.0]      # Forward velocity (m/s)
    command_y_range = [-0.5, 0.5]     # Lateral velocity (m/s)
    command_yaw_range = [-1.0, 1.0]   # Yaw velocity (rad/s)
    
    # Command curriculum
    lin_velocity_curriculum = True     # Enable linear velocity curriculum
    curriculum_steps = 5000           # Number of steps for curriculum
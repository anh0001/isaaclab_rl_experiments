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

from isaaclab_rl_experiments.assets.robots.yonsoku.robot_config import YONSOKU_CFG

@configclass
class YonsokuVelocityEnvCfg(DirectRLEnvCfg):
    """Environment configuration for Yonsoku quadruped robot velocity control."""
    
    # Simulation parameters
    decimation = 4
    episode_length_s = 20.0
    
    # Robot configuration
    robot_cfg = YONSOKU_CFG
    
    # Simulation setup
    sim = SimulationCfg(dt=1/120.0, substeps=2, render_interval=decimation)
    
    # Scene configuration
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)
    
    # Observation and action spaces
    observation_space = 48  # To be determined based on robot state
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
    action_scale = 1.0
    
    # Reward scales
    reward_scales = {
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.5,
        "base_height": -3.0,
        "joint_regularization": -0.001,
        "action_rate": -0.01,
        "feet_air_time": 2.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5
    }
    
    # Initial randomization
    start_position_noise = 0.0
    start_rotation_noise = 0.0
    
    # Termination conditions
    termination_height = 0.25
    termination_roll = 1.57  # ~90 degrees
    termination_pitch = 1.57  # ~90 degrees
    
    # Targets
    default_target_lin_velocity = [0.0, 0.0, 0.0]  # x, y, z
    default_target_ang_velocity = [0.0, 0.0, 0.0]  # roll, pitch, yaw
    
    # Command ranges
    command_x_range = [-1.0, 1.0]
    command_y_range = [-1.0, 1.0]
    command_yaw_range = [-1.0, 1.0]
    
    # Command curriculum
    lin_velocity_curriculum = False
    curriculum_steps = 5000
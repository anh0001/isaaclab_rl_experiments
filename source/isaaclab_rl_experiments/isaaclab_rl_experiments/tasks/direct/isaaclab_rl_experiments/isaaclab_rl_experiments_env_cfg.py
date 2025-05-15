# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

@configclass
class A1SceneCfg(InteractiveSceneCfg):
    """Custom scene configuration for A1 with contact sensors."""
    num_envs = 4096
    env_spacing = 5.0
    replicate_physics = True
    # Define foot contact sensor
    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*_foot",  # Matches all foot links
        history_length=1,
        update_period=0.0  # Update every simulation step
    )

@configclass
class A1VelocityEnvCfg(DirectRLEnvCfg):
    """Environment configuration for A1 quadruped robot velocity control."""
    
    # Simulation parameters
    decimation = 4
    episode_length_s = 20.0
    
    # Make sure activate_contact_sensors is explicitly set to True
    robot_cfg = UNITREE_A1_CFG.replace(
        prim_path="/World/envs/env_.*/robot",
        # Ensure contact sensors are activated
        spawn=UNITREE_A1_CFG.spawn.replace(activate_contact_sensors=True)
    )
    
    # Update the foot names to match A1's naming convention
    feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Simulation setup
    sim = SimulationCfg(dt=1/120.0, render_interval=decimation)
    
    # Scene configuration - now using the custom scene config
    scene = A1SceneCfg()
    
    # Observation and action spaces
    observation_space = 48  # Base state + joint positions + velocities + commands
    action_space = 12  # 12 joint actuators (3 per leg)
    state_space = 0
    
    # Define joint names for easy access
    leg_joint_names = {
        "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "RR": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "RL": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
    }
    
    # Control parameters
    lin_velocity_scale = 1.0
    ang_velocity_scale = 0.5
    dof_position_scale = 1.0
    dof_velocity_scale = 0.05
    action_scale = 0.25
    
    # Reward scales based on A1 configuration
    reward_scales = {
        "lin_vel_z": -2.0,            # Penalize vertical movement
        "ang_vel_xy": -0.05,          # Penalize roll and pitch angular velocity
        "orientation": -2.5,          # Penalize non-flat orientation
        "dof_torques": -0.0002,       # Penalize joint torques
        "dof_acceleration": -2.5e-7,  # Penalize joint accelerations
        "base_height": -2.5,          # Penalize incorrect base height
        "joint_regularization": -0.001, # Regularize joint positions
        "action_rate": -0.01,         # Penalize rapid action changes
        "feet_air_time": 0.25,        # Reward feet air time
        "tracking_lin_vel_xy": 1.5,   # Reward for tracking linear velocity
        "tracking_ang_vel_z": 0.75    # Reward for tracking angular velocity
    }
    
    # Initial randomization
    start_position_noise = 0.1
    start_rotation_noise = 0.1
    
    # Termination conditions
    termination_height = 0.25
    termination_roll = 1.0
    termination_pitch = 1.0
    
    # Command ranges
    command_x_range = [0.0, 1.0]      # Forward velocity (m/s)
    command_y_range = [-0.5, 0.5]     # Lateral velocity (m/s)
    command_yaw_range = [-1.0, 1.0]   # Yaw velocity (rad/s)
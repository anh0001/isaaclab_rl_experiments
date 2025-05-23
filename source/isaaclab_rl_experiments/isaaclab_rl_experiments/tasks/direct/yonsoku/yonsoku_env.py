# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yonsoku quadruped robot environment for Isaac Lab."""

from collections.abc import Sequence
import math
import torch
from typing import Dict, Optional, Tuple, Union

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse, sample_uniform

from .yonsoku_env_cfg import YonsokuVelocityEnvCfg


class YonsokuEnv(DirectRLEnv):
    """Yonsoku quadruped robot environment in Isaac Lab."""
    
    cfg: YonsokuVelocityEnvCfg
    
    def __init__(self, cfg: YonsokuVelocityEnvCfg, render_mode: Optional[str] = None, **kwargs):
        """Initialize the Yonsoku environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Add this line to define dt from config
        self.dt = self.cfg.sim.dt
        
        # Extract joint indices
        self.dof_indices = {}
        for leg_name, joint_names in self.cfg.leg_joint_names.items():
            indices, names = self.robot.find_joints(joint_names)
            self.dof_indices[leg_name] = indices
        
        # Flatten all joint indices
        self.all_dof_indices = []
        for indices in self.dof_indices.values():
            self.all_dof_indices.extend(indices)
        
        # Command values
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Command ranges
        self.command_ranges = torch.tensor(
            [
                self.cfg.command_x_range,
                self.cfg.command_y_range,
                self.cfg.command_yaw_range,
            ],
            device=self.device,
        )
        
        # Initialize feet contact buffers

        # Last link in each leg
        # TODO: Check this - it should be the last link in each leg
        # self.feet_names = ["RF_FOOT", "RB_FOOT", "LB_FOOT", "LF_FOOT"]
        self.feet_names = ["RF3", "RB3", "LB3", "LF3"]

        self.feet_indices = []
        for name in self.feet_names:
            self.feet_indices.append(self.robot.find_bodies(name)[0])
        
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_names)), device=self.device)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_names)), device=self.device)
        
        # Store default joint positions
        self.default_dof_pos = self.robot.data.default_joint_pos.clone()
        
        # Save previous actions and joint velocities for acceleration calculation
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.last_joint_vel = torch.zeros_like(self.robot.data.joint_vel)

        # Initialize additional trackers for rewards
        self.last_joint_vel = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create robot - use environment pattern
        self.robot = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/yonsoku_robot"))

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Note: Removed the problematic line that manually added the contact sensor
        # The sensors are now created from the configuration in InteractiveSceneCfg
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Prepare for physics step by applying actions."""
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # Scale actions according to the configured action_scale
        scaled_actions = actions * self.cfg.action_scale
        
        # Get the reference pose (default joint positions)
        # Fix: Make sure default_dof_pos is properly shaped before repeating
        if len(self.default_dof_pos.shape) == 1:
            # If default_dof_pos is a 1D tensor, reshape it for broadcasting
            reference_pose = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1)
        else:
            # If it's already 2D, ensure it has the correct first dimension
            reference_pose = self.default_dof_pos.expand(self.num_envs, -1)
        
        # Apply the actions as offsets to the reference pose
        position_targets = reference_pose + scaled_actions
        
        # Apply actions to the robot's actuators as position targets
        self.robot.set_joint_position_target(position_targets, joint_ids=self.all_dof_indices)
    
    def _apply_action(self):
        """Apply the processed actions to the simulation at each physics step."""
        # DCMotor actuators will handle the physics-based actuation
        # Just ensure the data is written to sim
        self.robot.write_data_to_sim()
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Get robot state
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Access individual components from body_state_w
        # body_state_w shape is [num_envs, num_bodies, 13]
        # where 13 = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        base_pos = self.robot.data.body_state_w[:, 0, 0:3]
        base_quat = self.robot.data.body_state_w[:, 0, 3:7]
        base_lin_vel = self.robot.data.body_state_w[:, 0, 7:10]
        base_ang_vel = self.robot.data.body_state_w[:, 0, 10:13]
        
        # Convert base velocity to base frame
        base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_local = quat_rotate_inverse(base_quat, base_ang_vel)
        
        # Compose observations
        obs = torch.cat(
            [
                base_lin_vel_local,  # 3
                base_ang_vel_local,  # 3
                joint_pos,  # 12
                joint_vel,  # 12
                self.actions,  # 12
                self.commands,  # 3
            ],
            dim=-1,
        )
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on Unitree A1 approach."""
        # Get robot state
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_acc = (joint_vel - self.last_joint_vel) / self.dt
        self.last_joint_vel = joint_vel.clone()
        
        # Access individual components from body_state_w
        base_pos = self.robot.data.body_state_w[:, 0, 0:3]
        base_quat = self.robot.data.body_state_w[:, 0, 3:7]
        base_lin_vel = self.robot.data.body_state_w[:, 0, 7:10]
        base_ang_vel = self.robot.data.body_state_w[:, 0, 10:13]
        
        # Convert base velocity to base frame
        base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_local = quat_rotate_inverse(base_quat, base_ang_vel)
        
        # Get roll and pitch from quaternion
        roll_pitch = torch.zeros((self.num_envs, 2), device=self.device)
        qx, qy, qz, qw = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        
        # Calculate roll (x-axis rotation)
        roll_pitch[:, 0] = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        
        # Calculate pitch (y-axis rotation)
        roll_pitch[:, 1] = torch.asin(2 * (qw * qy - qz * qx))
        
        # Calculate reward components similar to A1
        rewards = dict()
        
        # Tracking linear velocity - exponential reward like A1's tracking reward
        # This implements the track_lin_vel_xy_exp and track_ang_vel_z_exp from A1
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel_local[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel_local[:, 2])
        
        # Exponential rewards for better tracking (matching A1's approach)
        rewards["tracking_lin_vel_xy"] = torch.exp(-lin_vel_error / 0.25) * self.cfg.reward_scales["tracking_lin_vel_xy"]
        rewards["tracking_ang_vel_z"] = torch.exp(-ang_vel_error / 0.25) * self.cfg.reward_scales["tracking_ang_vel_z"]
        
        # Penalize vertical velocity
        rewards["lin_vel_z"] = -torch.square(base_lin_vel_local[:, 2]) * self.cfg.reward_scales["lin_vel_z"]
        
        # Penalize pitch and roll angular velocity (stability)
        rewards["ang_vel_xy"] = -torch.sum(torch.square(base_ang_vel_local[:, :2]), dim=1) * self.cfg.reward_scales["ang_vel_xy"]
        
        # Penalize non-flat orientation (similar to flat_orientation_l2)
        rewards["orientation"] = -torch.sum(torch.square(roll_pitch), dim=1) * self.cfg.reward_scales["orientation"]
        
        # Replace the original torque-based reward with a proxy using scaled actions
        # For DCMotor actuators, the torques will be physically simulated by the motor
        estimated_torques = self.actions * self.cfg.action_scale  # Scale actions as proxy for torques
        rewards["dof_torques"] = -torch.sum(torch.square(estimated_torques), dim=1) * self.cfg.reward_scales["dof_torques"]
        
        # Penalize joint accelerations (similar to dof_acc_l2)
        rewards["dof_acceleration"] = -torch.sum(torch.square(joint_acc), dim=1) * self.cfg.reward_scales["dof_acceleration"]
        
        # Penalize incorrect base height
        rewards["base_height"] = -torch.square(base_pos[:, 2] - 0.52) * self.cfg.reward_scales["base_height"]
        
        # Penalize joint position regularization
        rewards["joint_regularization"] = -torch.sum(torch.square(joint_pos), dim=1) * self.cfg.reward_scales["joint_regularization"]
        
        # Penalize action rate (rapid changes in actions)
        rewards["action_rate"] = -torch.sum(torch.square(self.actions - self.last_actions), dim=1) * self.cfg.reward_scales["action_rate"]
        
        # Update feet air time calculations
        # Get foot contacts from the contact sensor
        feet_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device)
        
        # Check if the foot_contacts sensor exists (safer check)
        if "foot_contacts" in self.scene.sensors:
            # Get contact forces from the sensor
            contact_forces = self.scene.sensors["foot_contacts"].data.net_forces_w
            
            # Map the forces to the appropriate feet indices
            if contact_forces is not None and contact_forces.shape[1] >= len(self.feet_indices):
                for i in range(len(self.feet_indices)):
                    # Check if the force magnitude exceeds the threshold
                    feet_contacts[:, i] = torch.norm(contact_forces[:, i], dim=1) > 1.0
        else:
            # Fallback method if sensor isn't correctly initialized
            # Check for collision states directly:
            for i, foot_name in enumerate(self.feet_names):
                body_indices, _ = self.robot.find_bodies(foot_name)
                if len(body_indices) > 0:
                    # Check if bodies have non-zero contact forces (simplified)
                    feet_contacts[:, i] = self.robot.has_contact(body_ids=body_indices[0])
        
        # Update air time tracker
        self.feet_air_time += self.dt
        # Reset air time when contact is detected
        self.feet_air_time = torch.where(feet_contacts > 0, torch.zeros_like(self.feet_air_time), self.feet_air_time)
        
        # Calculate feet air time reward similar to A1
        feet_air_time_reward = torch.mean(
            torch.minimum(self.feet_air_time, torch.ones_like(self.feet_air_time) * 0.5), dim=1
        ) * self.cfg.reward_scales["feet_air_time"]
        rewards["feet_air_time"] = feet_air_time_reward
        
        # Sum all rewards
        total_reward = torch.zeros_like(rewards["tracking_lin_vel_xy"])
        for name, value in rewards.items():
            total_reward += value
        
        return total_reward
    
    def _reset_feet_air_time(self, env_ids: torch.Tensor):
        """Reset the feet air time tracking."""
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = 0.0
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine if episodes should terminate."""
        # Access individual components from body_state_w
        base_pos = self.robot.data.body_state_w[:, 0, 0:3]
        base_quat = self.robot.data.body_state_w[:, 0, 3:7]
        
        # Check for termination conditions
        base_height = base_pos[:, 2]
        
        # Convert quaternion to euler angles
        # Use roll and pitch for termination check
        # This is a simplified version - in practice, you'd compute proper Euler angles
        roll_pitch = torch.zeros((self.num_envs, 2), device=self.device)
        qx, qy, qz, qw = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        
        # Calculate roll (x-axis rotation)
        roll_pitch[:, 0] = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        
        # Calculate pitch (y-axis rotation)
        roll_pitch[:, 1] = torch.asin(2 * (qw * qy - qz * qx))
        
        # Check termination conditions
        # 1. Base height is too low
        # 2. Robot rolled or pitched too much
        # 3. Episode length exceeded
        too_low = base_height < self.cfg.termination_height
        roll_pitch_limit = torch.any(torch.abs(roll_pitch) > torch.tensor([self.cfg.termination_roll, self.cfg.termination_pitch], device=self.device), dim=1)
        
        # Check timeout condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Combine termination conditions
        terminated = too_low | roll_pitch_limit
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Optional[Sequence[int]] = None):
        """Reset environments with the given indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Reset commands
        self._resample_commands(env_ids)
        
        # Reset air time tracking
        self._reset_feet_air_time(env_ids)
        
        # Get default root states from config's init_state
        default_root_pos = torch.tensor([self.cfg.robot_cfg.init_state.pos], 
                                device=self.device).repeat(len(env_ids), 1)
        default_root_quat = torch.tensor([self.cfg.robot_cfg.init_state.rot], 
                                 device=self.device).repeat(len(env_ids), 1)
        default_root_lin_vel = torch.zeros_like(default_root_pos)
        default_root_ang_vel = torch.zeros_like(default_root_pos)
        
        # Apply position noise
        if self.cfg.start_position_noise > 0:
            position_noise = torch.randn((len(env_ids), 3), device=self.device) * self.cfg.start_position_noise
            default_root_pos += position_noise
        
        # Apply rotation noise around z-axis (yaw)
        if self.cfg.start_rotation_noise > 0:
            # Create random rotations around z-axis
            random_yaw = torch.randn(len(env_ids), device=self.device) * self.cfg.start_rotation_noise
            quat_noise = torch.zeros((len(env_ids), 4), device=self.device)
            quat_noise[:, 0] = torch.cos(random_yaw / 2)  # w
            quat_noise[:, 3] = torch.sin(random_yaw / 2)  # z
            
            # Apply rotation to default quaternion - simplified approach
            default_root_quat = quat_noise
        
        # Set environment origins
        default_root_pos += self.scene.env_origins[env_ids]
        
        # Reset joint positions with some noise
        default_joint_pos = self.default_dof_pos[env_ids].clone()
        
        # Apply joint position noise
        # In a full implementation, you'd add appropriate joint position noise here
        
        # Reset velocities
        default_joint_vel = torch.zeros_like(default_joint_pos)
        
        # Write robot state to simulation
        self.robot.write_root_pose_to_sim(
            torch.cat([default_root_pos, default_root_quat], dim=1), 
            env_ids
        )
        self.robot.write_root_velocity_to_sim(
            torch.cat([default_root_lin_vel, default_root_ang_vel], dim=1), 
            env_ids
        )
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
    
    def _resample_commands(self, env_ids: Union[torch.Tensor, Sequence[int]]):
        """Resample commands for the specified environments."""
        # Create random commands within the specified ranges
        self.commands[env_ids, 0] = sample_uniform(
            self.command_ranges[0, 0], self.command_ranges[0, 1], (len(env_ids),), self.device
        )
        self.commands[env_ids, 1] = sample_uniform(
            self.command_ranges[1, 0], self.command_ranges[1, 1], (len(env_ids),), self.device
        )
        self.commands[env_ids, 2] = sample_uniform(
            self.command_ranges[2, 0], self.command_ranges[2, 1], (len(env_ids),), self.device
        )
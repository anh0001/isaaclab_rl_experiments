# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yonsoku quadruped robot configuration for Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.assets.articulation import JointActuator
from isaaclab.utils import configclass

# Define the path to the Yonsoku robot USD model
_USD_PATH = "source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/yonsoku_robot.usd"

@configclass
class YonsokuBaseCfg(ArticulationCfg):
    """Base configuration for the Yonsoku quadruped robot."""
    
    # Required properties
    usd_path = _USD_PATH
    prim_path = "/World/envs/env_.*/Robot"  # Template for environment replication
    name = "yonsoku"
    
    # Default robot state
    root_position = [0.0, 0.0, 0.52]  # Starting height above ground
    root_orientation = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
    
    # Define joint properties based on your URDF
    default_joint_positions = {
        "RF_JOINT1": 0.0,
        "RF_JOINT2": 90.0,
        "RF_JOINT3": -165.0,
        "RB_JOINT1": 0.0,
        "RB_JOINT2": -90.0,
        "RB_JOINT3": 165.0,
        "LB_JOINT1": 0.0,
        "LB_JOINT2": -90.0,
        "LB_JOINT3": 165.0,
        "LF_JOINT1": 0.0,
        "LF_JOINT2": 90.0,
        "LF_JOINT3": -165.0,
    }
    
    # Control mode settings
    dof_control_mode = "position"
    
    # Define joint actuators
    actuators = {
        "RF_JOINT1": JointActuator(joint_name="RF_JOINT1"),
        "RF_JOINT2": JointActuator(joint_name="RF_JOINT2"),
        "RF_JOINT3": JointActuator(joint_name="RF_JOINT3"),
        "RB_JOINT1": JointActuator(joint_name="RB_JOINT1"),
        "RB_JOINT2": JointActuator(joint_name="RB_JOINT2"),
        "RB_JOINT3": JointActuator(joint_name="RB_JOINT3"),
        "LB_JOINT1": JointActuator(joint_name="LB_JOINT1"),
        "LB_JOINT2": JointActuator(joint_name="LB_JOINT2"),
        "LB_JOINT3": JointActuator(joint_name="LB_JOINT3"),
        "LF_JOINT1": JointActuator(joint_name="LF_JOINT1"),
        "LF_JOINT2": JointActuator(joint_name="LF_JOINT2"),
        "LF_JOINT3": JointActuator(joint_name="LF_JOINT3"),
    }
    
    # Set stiffness and damping
    dof_stiffness = {
        "RF_JOINT1": 2000.0,
        "RF_JOINT2": 1000.0,
        "RF_JOINT3": 1500.0,
        "RB_JOINT1": 2000.0,
        "RB_JOINT2": 1000.0,
        "RB_JOINT3": 1500.0,
        "LB_JOINT1": 2000.0,
        "LB_JOINT2": 1000.0,
        "LB_JOINT3": 1500.0,
        "LF_JOINT1": 2000.0,
        "LF_JOINT2": 1000.0,
        "LF_JOINT3": 1500.0,
    }
    
    dof_damping = {
        "RF_JOINT1": 20.1,
        "RF_JOINT2": 20.1,
        "RF_JOINT3": 20.1,
        "RB_JOINT1": 20.1,
        "RB_JOINT2": 20.1,
        "RB_JOINT3": 20.1,
        "LB_JOINT1": 20.1,
        "LB_JOINT2": 20.1,
        "LB_JOINT3": 20.1,
        "LF_JOINT1": 20.1,
        "LF_JOINT2": 20.1,
        "LF_JOINT3": 20.1,
    }
    
    dof_max_efforts = {
        "RF_JOINT1": 2000.0,
        "RF_JOINT2": 1000.0,
        "RF_JOINT3": 1500.0,
        "RB_JOINT1": 2000.0,
        "RB_JOINT2": 1000.0,
        "RB_JOINT3": 1500.0,
        "LB_JOINT1": 2000.0,
        "LB_JOINT2": 1000.0,
        "LB_JOINT3": 1500.0,
        "LF_JOINT1": 2000.0,
        "LF_JOINT2": 1000.0,
        "LF_JOINT3": 1500.0,
    }

# Export the configuration
YONSOKU_CFG = YonsokuBaseCfg()
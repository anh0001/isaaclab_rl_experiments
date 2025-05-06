# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yonsoku quadruped robot configuration for Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg  # Change this import
from isaaclab.utils import configclass

# Define the path to the Yonsoku robot USD model
_USD_PATH = "/home/dl-box/codes/anhar/isaaclab_rl_experiments/source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/yonsoku_robot.usd"

@configclass
class YonsokuBaseCfg(ArticulationCfg):
    """Base configuration for the Yonsoku quadruped robot."""
    
    # Required properties
    usd_path = _USD_PATH
    # prim_path = "/World/envs/env_.*/yonsoku_robot"
    prim_path = "/yonsoku_robot"
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
    
    # Define actuators using ImplicitActuatorCfg instead of JointActuator
    actuators = {
        "rf_actuators": ImplicitActuatorCfg(
            joint_names_expr=["RF_JOINT[1-3]"],
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "rb_actuators": ImplicitActuatorCfg(
            joint_names_expr=["RB_JOINT[1-3]"],
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "lb_actuators": ImplicitActuatorCfg(
            joint_names_expr=["LB_JOINT[1-3]"],
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "lf_actuators": ImplicitActuatorCfg(
            joint_names_expr=["LF_JOINT[1-3]"],
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        )
    }
    
# Export the configuration
YONSOKU_CFG = YonsokuBaseCfg()
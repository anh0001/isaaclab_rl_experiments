# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yonsoku quadruped robot configuration for Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass

# Define the path to the Yonsoku robot USD model
_USD_PATH = "/home/dl-box/codes/anhar/isaaclab_rl_experiments/source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/yonsoku_robot.usd"

# Create the configuration using the standard pattern
YONSOKU_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/yonsoku_robot",
    # Use spawn with UsdFileCfg to properly load the USD file
    spawn=UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props={
            "enable_gyroscopic_forces": True,
            "max_depenetration_velocity": 100.0,
        },
        articulation_props={
            "solver_position_iteration_count": 4,
            "solver_velocity_iteration_count": 0,
            "sleep_threshold": 0.005,
            "stabilization_threshold": 0.001,
        },
        activate_contact_sensors=True,
    ),
    # Define actuators with appropriate configs
    actuators={
        "RF_JOINT[1-3]": ImplicitActuatorCfg(
            joint_names_expr="RF_JOINT[1-3]",
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "RB_JOINT[1-3]": ImplicitActuatorCfg(
            joint_names_expr="RB_JOINT[1-3]",
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "LB_JOINT[1-3]": ImplicitActuatorCfg(
            joint_names_expr="LB_JOINT[1-3]",
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        ),
        "LF_JOINT[1-3]": ImplicitActuatorCfg(
            joint_names_expr="LF_JOINT[1-3]",
            stiffness=2000.0,
            damping=20.1,
            effort_limit_sim=2000.0
        )
    },
    # Define initial joint positions and robot state
    init_state={
        "joint_pos": {
            "RF_JOINT1": 0.0,
            "RF_JOINT2": 1.57,  # 90 degrees in radians
            "RF_JOINT3": -2.88,  # -165 degrees in radians
            "RB_JOINT1": 0.0,
            "RB_JOINT2": -1.57,  # -90 degrees in radians
            "RB_JOINT3": 2.88,   # 165 degrees in radians
            "LB_JOINT1": 0.0,
            "LB_JOINT2": -1.57,  # -90 degrees in radians
            "LB_JOINT3": 2.88,   # 165 degrees in radians
            "LF_JOINT1": 0.0,
            "LF_JOINT2": 1.57,   # 90 degrees in radians
            "LF_JOINT3": -2.88,  # -165 degrees in radians
        },
        "joint_vel": {},
        "pos": [0.0, 0.0, 0.52],  # Starting height above ground
        "quat": [1.0, 0.0, 0.0, 0.0],  # Quaternion [w, x, y, z]
    },
)
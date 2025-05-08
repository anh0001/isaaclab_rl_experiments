# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yonsoku quadruped robot configuration for Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg  # Changed from ImplicitActuatorCfg to DCMotorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg

# Define the path to the Yonsoku robot USD model
_USD_PATH = "/home/dl-box/codes/anhar/isaaclab_rl_experiments/source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/yonsoku_robot.usd"

# Create a simple configuration class for the initial state
@configclass
class InitState:
    joint_pos = {
        "RF_JOINT1": 0.0,
        "RF_JOINT2": 0.8,  # Reduced from 1.57 to match A1's thigh joint angle
        "RF_JOINT3": -1.5,  # Reduced from -2.88 to match A1's calf joint angle
        "RB_JOINT1": 0.0,
        "RB_JOINT2": -0.8,  # Reduced from -1.57
        "RB_JOINT3": 1.5,   # Reduced from 2.88
        "LB_JOINT1": 0.0,
        "LB_JOINT2": -0.8,  # Reduced from -1.57
        "LB_JOINT3": 1.5,   # Reduced from 2.88
        "LF_JOINT1": 0.0,
        "LF_JOINT2": 0.8,   # Reduced from 1.57
        "LF_JOINT3": -1.5,  # Reduced from -2.88
    }
    joint_vel = {}
    pos = [0.0, 0.0, 0.52]  # Starting height above ground
    rot = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
    lin_vel = [0.0, 0.0, 0.0]  # Linear velocity
    ang_vel = [0.0, 0.0, 0.0]  # Angular velocity

# Create the configuration using the standard pattern
YONSOKU_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/yonsoku_robot",
    # Use spawn with UsdFileCfg to properly load the USD file
    spawn=UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=RigidBodyPropertiesCfg(
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,
    ),
    # Define actuators with DCMotorCfg - parameters similar to A1
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["RF_JOINT[1-3]", "RB_JOINT[1-3]", "LB_JOINT[1-3]", "LF_JOINT[1-3]"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
    # Use our custom InitState class
    init_state=InitState(),
)
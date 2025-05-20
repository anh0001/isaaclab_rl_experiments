# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg

from .rough_env_cfg import UnitreeA1RoughEnvCfg

@configclass
class UnitreeA1SoilGardenEnvCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # Call parent's post_init
        super().__post_init__()
        
        # Override terrain settings to simulate soil garden
        if self.scene.terrain.terrain_generator is not None:
            # Configure a new terrain generator with soil-like properties
            self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
                size=(15, 15),
                sub_terrains={
                    # Loose soil with gentle undulations
                    "loose_soil": HfRandomUniformTerrainCfg(
                        noise_range=(0.02, 0.05),  # Lower height variation for soil
                        noise_step=0.1,           # Medium frequency noise
                        horizontal_scale=0.1,
                        proportion=0.6            # 60% loose soil
                    ),
                    # Compacted soil areas (more flat)
                    "compacted_soil": HfRandomUniformTerrainCfg(
                        noise_range=(0.005, 0.02),  # Very minor height variations
                        noise_step=0.2,            # Lower frequency noise
                        horizontal_scale=0.1,
                        proportion=0.4             # 40% compacted soil
                    )
                },
                curriculum=True,
                difficulty_range=(0.0, 1.0)
            )
        
        # Configure physics materials for soil
        self.scene.terrain.physics_material.static_friction = 0.7
        self.scene.terrain.physics_material.dynamic_friction = 0.5
        self.scene.terrain.physics_material.restitution = 0.1
        
        # Add garden obstacles via spawner
        # Note: We'll leverage the existing obstacle spawner in UnitreeA1RoughEnvCfg
        # but update its parameters to create garden-like objects
        
        # Modify the rewards to optimize for soil locomotion
        self.rewards.flat_orientation_l2.weight = -2.0  # Slightly less penalty for orientation
        self.rewards.feet_air_time.weight = 0.3  # Encourage lifting feet (good for soil)
        self.rewards.dof_torques_l2.weight = -0.0003  # More penalty for high torques (soil is softer)

@configclass
class UnitreeA1SoilGardenEnvCfg_PLAY(UnitreeA1SoilGardenEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
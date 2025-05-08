# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Yonsoku robot."""
    
    # Runner configuration - closer to A1 settings
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1500  # Match A1's max_iterations
    save_interval = 50
    experiment_name = "yonsoku_velocity"
    run_name = ""
    
    # Curriculum settings
    learning_starts = 0
    curriculum_steps = 0
    
    # Normalization
    clip_observations = 5.0
    clip_actions = 1.0
    empirical_normalization = True
    
    # Actor-Critic configuration - match A1's network sizes
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # Match A1's network architecture
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # PPO Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Same as A1
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
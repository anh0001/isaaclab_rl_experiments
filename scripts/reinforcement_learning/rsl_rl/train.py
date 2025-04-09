#!/usr/bin/env python3

"""
Training script for reinforcement learning with Isaac Lab.
"""

import argparse
import os
import sys
import yaml

# Import any necessary IsaacLab modules
# This will depend on the IsaacLab API structure
# The import paths below are examples - you'll need to adjust based on the actual API
try:
    from isaaclab.rl import environments
    from isaaclab.rl.algorithms import PPO
except ImportError:
    print("Error: IsaacLab modules not found. Please ensure IsaacLab is installed correctly.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents in Isaac Lab')
    parser.add_argument('--task', type=str, required=True, 
                        help='Task name (e.g., Isaac-Ant-v0, Isaac-Velocity-Rough-Anymal-C-v0)')
    parser.add_argument('--headless', action='store_true', 
                        help='Run in headless mode (no visualization)')
    parser.add_argument('--num_envs', type=int, default=4096, 
                        help='Number of environments to run in parallel')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load appropriate config file
    config_path = None
    if "Ant" in args.task:
        config_path = os.path.join(os.path.dirname(__file__), "../config/ant.yaml")
    elif "Anymal" in args.task:
        config_path = os.path.join(os.path.dirname(__file__), "../config/anymal.yaml")
    
    if not config_path or not os.path.exists(config_path):
        print(f"Error: Config file not found for task {args.task}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set headless mode in config if specified
    config['headless'] = args.headless
    config['num_envs'] = args.num_envs
    
    # Initialize and run training using the IsaacLab API
    # This code will depend on the actual IsaacLab API structure
    env = environments.create(args.task, num_envs=args.num_envs, headless=args.headless)
    
    # Initialize the PPO algorithm with the config
    ppo = PPO(env, config)
    
    # Run the training
    ppo.train()
    
if __name__ == "__main__":
    main()
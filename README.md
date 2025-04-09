# Isaac Lab Reinforcement Learning Experiments

This repository contains experiments for training and evaluating reinforcement learning agents in Isaac Lab, focusing on locomotion tasks for Ant and Anymal robots.

## Prerequisites

- NVIDIA GPU with CUDA 12.x support
- Ubuntu (tested on Ubuntu 22.04)
- At least 32GB RAM recommended
- Miniconda or Anaconda

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/anh0001/isaaclab-rl-experiments.git
   cd isaaclab-rl-experiments
   ```

2. Run the setup script:
   ```
   ./scripts/utilities/setup_env.sh
   ```

3. Install Isaac Lab:
   ```
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ./isaaclab.sh --install
   cd ..
   ```

4. Set the Isaac Lab path:
   ```
   export ISAAC_LAB_PATH=$PWD/IsaacLab
   ```

## Running Experiments

### Training

Train the Ant model:
```
./scripts/reinforcement_learning/train_ant.sh
```

Train the Anymal model:
```
./scripts/reinforcement_learning/train_anymal.sh
```

To train without headless mode (to visualize training):
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0
```

### Evaluation

To evaluate a trained model:
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 32 --checkpoint logs/rsl_rl/anymal_c_rough/<timestamp>/model_1499.pt
```

## Troubleshooting

If you encounter Vulkan errors, try running:
```
./scripts/utilities/fix_vulkan.sh
```

## Future Work

- Integration with Genesis environment
- Sim-to-real transfer
- Joystick control interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

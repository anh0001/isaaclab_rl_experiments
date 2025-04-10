# Isaac Lab Reinforcement Learning Experiments

## Overview

This project extends Isaac Lab to implement reinforcement learning experiments for locomotion tasks, focusing on Ant and Anymal robots. It builds on the Isaac Lab template to provide a structured environment for developing and evaluating RL algorithms in robotics simulation.

**Key Features:**

- Pre-configured training pipelines for Ant and Anymal locomotion tasks
- Support for multiple RL frameworks (RL-Games, RSL-RL, Stable-Baselines3, SKRL)
- Tools for model evaluation and visualization
- Isolated development environment outside core Isaac Lab

**Keywords:** reinforcement learning, locomotion, isaaclab, ant, anymal

## Prerequisites

- NVIDIA GPU with CUDA 12.x support
- Ubuntu (tested on Ubuntu 22.04)
- At least 32GB RAM recommended
- Miniconda or Anaconda

## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. Clone this repository:
   ```bash
   git clone https://github.com/anh0001/isaaclab-rl-experiments.git
   cd isaaclab-rl-experiments
   ```

3. Create a conda environment using the setup script:
   ```bash
   ./scripts/utilities/setup_env.sh
   ```

4. Activate the conda environment:
   ```bash
   conda activate ./isaaclab_env
   ```
   This allows you to use `python` commands directly. If you don't want to activate the environment, you can use `./isaaclab.sh -p` before every Python command.

5. Clone the Isaac Lab repository:
   ```bash
   git clone git@github.com:isaac-sim/IsaacLab.git
   ```

6. Install Isaac Lab:
   ```bash
   ./IsaacLab/isaaclab.sh --install
   ```

7. Verify the Isaac Lab installation:
   ```bash
   python ./IsaacLab/scripts/tutorials/00_sim/create_empty.py
   ```
   This should launch the Isaac Sim viewer with an empty scene, confirming the installation is working correctly.

8. Install this project's extension in editable mode using:
   ```bash
   python -m pip install -e source/isaaclab_rl_experiments
   ```

9. Verify that the extension is correctly installed by listing the available tasks:
   ```bash
   python scripts/list_envs.py
   ```

## Running Experiments

### Training

Make sure your conda environment is activated (`conda activate ./isaaclab_env`), then you can train the models using one of the supported RL frameworks:

**Using RSL-RL (Recommended for locomotion tasks):**
```bash
# Train Unitree-A1
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs=4096 --headless

# Train Ant
python scripts/rsl_rl/train.py --task=Isaac-Ant-v0 --num_envs=4096 --experiment_name=ant_locomotion

# Train Anymal
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs=2048 --experiment_name=anymal_c_rough
```

**Using other frameworks:**
```bash
# RL-Games
python scripts/rl_games/train.py --task=Isaac-Ant-v0 --num_envs=4096

# Stable-Baselines3
python scripts/sb3/train.py --task=Isaac-Ant-v0 --num_envs=1024

# SKRL
python scripts/skrl/train.py --task=Isaac-Ant-v0 --num_envs=4096 --algorithm=PPO
```

Note: Training in headless mode is faster. To visualize training, add the `--headless=False` flag to the commands.

### Evaluation

To evaluate a trained model (with the conda environment activated):

```bash
# Evaluate Unitree A1
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs=100 --checkpoint=logs/rsl_rl/unitree_a1_flat/2025-04-09_17-59-40/model_299.pt --video --video_length 1000

# Evaluate Ant
python scripts/rsl_rl/play.py --task=Isaac-Ant-v0 --num_envs=16 --checkpoint=logs/rsl_rl/ant_locomotion/<timestamp>/model_<iteration>.pt --video

# Evaluate Anymal
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs=32 --checkpoint=logs/rsl_rl/anymal_c_rough/<timestamp>/model_<iteration>.pt --video
```

Replace `<timestamp>` and `<iteration>` with your actual values.

## Development

### Set up IDE (Optional)

To setup your development environment:

1. Run VSCode Tasks by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task`, and running the `setup_python_env` in the dropdown menu.
2. When prompted, add the absolute path to your Isaac Sim installation.

If successful, it will create a `.python.env` file in the `.vscode` directory containing Python paths to all extensions provided by Isaac Sim and Omniverse.

### Setup as Omniverse Extension (Optional)

An example UI extension will load when you enable the extension:

1. Add the search path of this project to the extension manager:
   - Navigate to `Window` -> `Extensions`
   - Click on the **Hamburger Icon**, then go to `Settings`
   - Add the absolute path to the `source` directory of this repository
   - Add the path to Isaac Lab's extension directory (`IsaacLab/source`) if not present
   - Click on the **Hamburger Icon**, then click `Refresh`

2. Search and enable your extension under the `Third Party` category.

### Code Formatting

A pre-commit template is provided to automatically format code:

```bash
pip install pre-commit
pre-commit run --all-files
```

## Customizing Environments

To create or modify robotic environments:

1. The example cartpole environment is located in: `source/isaaclab_rl_experiments/isaaclab_rl_experiments/tasks/direct/isaaclab_rl_experiments/`
2. To implement Ant or Anymal environments, follow a similar structure and adapt configs accordingly.
3. Register new environments in `__init__.py` files following the template example.

## Troubleshooting

### Vulkan Errors

If you encounter Vulkan errors, try running:
```bash
./scripts/utilities/fix_vulkan.sh
```

### Pylance Missing Indexing of Extensions

If VSCode indexing is incomplete, add the extension path in `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/isaaclab_rl_experiments"
    ]
}
```

### Pylance Crash

If Pylance crashes due to memory issues, exclude unused Omniverse packages in `.vscode/settings.json`:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
```

## Future Work

- Integration with Genesis environment
- Sim-to-real transfer
- Joystick control interface

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
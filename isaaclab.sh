#!/bin/bash

# Main entry point script for Isaac Lab experiments

# Pass all arguments to Isaac Lab
if [ -n "$ISAAC_LAB_PATH" ]; then
    $ISAAC_LAB_PATH/isaaclab.sh "$@"
else
    echo "Error: ISAAC_LAB_PATH environment variable not set."
    echo "Please set it to your Isaac Lab installation directory."
    echo "Example: export ISAAC_LAB_PATH=~/IsaacLab"
    exit 1
fi

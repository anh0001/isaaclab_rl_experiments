#!/bin/bash

# Main entry point script for Isaac Lab experiments

# Define path to IsaacLab relative to this script
ISAAC_LAB_PATH="$(dirname "$(readlink -f "$0")")/IsaacLab"

# Special handling for python script execution
if [[ "$1" == "-p" || "$1" == "--python" ]]; then
    # Use the current Python from the activated environment
    python_exe=$(which python)
    echo "[INFO] Using python from: $python_exe"
    
    # Add IsaacLab to the Python path
    export PYTHONPATH="$ISAAC_LAB_PATH:$ISAAC_LAB_PATH/source:$PYTHONPATH"
    echo "[INFO] Added IsaacLab to PYTHONPATH: $ISAAC_LAB_PATH"
    
    shift # Remove -p from arguments
    
    # Get the script path
    script_path="$1"
    shift # Remove script path from arguments
    
    # Make the script executable if it isn't already
    chmod +x "$script_path" 2>/dev/null || true
    
    # Execute the script with remaining arguments
    $python_exe "$script_path" "$@"
    
    # Exit after running the python script
    exit 0
fi

# For all other commands, pass to the real Isaac Lab script
"$ISAAC_LAB_PATH/isaaclab.sh" "$@"
#!/bin/bash

# Script to fix Vulkan ICD issue with NVIDIA on some systems

# Check for multiple nvidia_icd.json files
if [ -f "/usr/share/vulkan/icd.d/nvidia_icd.json" ] && [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    echo "Found multiple nvidia_icd.json files. This can cause conflicts."
    echo "Backing up the one in /usr/share/vulkan/icd.d/..."
    
    sudo mv /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json.backup
    
    echo "Backup created at /usr/share/vulkan/icd.d/nvidia_icd.json.backup"
    echo "Isaac Sim should now be able to start without Vulkan errors."
else
    echo "No duplicate nvidia_icd.json files found."
    echo "If you're still experiencing Vulkan errors, please check the paths manually:"
    ls -la /etc/vulkan/icd.d/
    ls -la /usr/share/vulkan/icd.d/
fi

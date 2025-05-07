# test_yonsoku_usd.py
import os

# Launch Isaac Sim Simulator first
from isaaclab.app import AppLauncher

# Initialize the app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Import modules after app is launched
import omni.usd
from pxr import Usd, UsdGeom, Sdf

# Path to the Yonsoku robot USD file
usd_path = os.path.abspath("source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/yonsoku_robot.usd")
print(f"Checking if USD file exists at: {usd_path}")
print(f"File exists: {os.path.exists(usd_path)}")

# Let's check if we need to generate the USD file from URDF
urdf_path = os.path.abspath("source/isaaclab_rl_experiments/isaaclab_rl_experiments/assets/robots/yonsoku/urdf/yonsoku_robot.urdf")
print(f"URDF exists: {os.path.exists(urdf_path)}")

# If USD doesn't exist but URDF does, we need to convert it
if not os.path.exists(usd_path) and os.path.exists(urdf_path):
    print("USD file doesn't exist. We need to convert the URDF to USD first.")
    
    # Import URDF modules
    from omni.isaac.urdf import _urdf
    
    # Create a new USD stage
    stage = omni.usd.get_context().new_stage()
    
    # Import URDF
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = True
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.default_drive_strength = 1500.0
    import_config.default_position_drive_damping = 20.0
    
    urdf_importer = _urdf.acquire_urdf_interface()
    result, prim_path = urdf_importer.import_urdf(urdf_path, "/yonsoku_robot", import_config)
    
    if result:
        print(f"Successfully imported URDF to prim path: {prim_path}")
        # Save the USD file
        omni.usd.get_context().save_stage_as(usd_path)
        print(f"Saved USD file to: {usd_path}")
    else:
        print("Failed to import URDF")
else:
    # If USD file exists, try to load it and print the prim path
    if os.path.exists(usd_path):
        # Create a new stage
        stage = omni.usd.get_context().new_stage()
        
        # Get the stage
        stage = omni.usd.get_context().get_stage()
        
        # Create the target prim if it doesn't exist
        if not stage.GetPrimAtPath("/yonsoku_robot"):
            stage.DefinePrim("/yonsoku_robot", "Xform")
        
        # Add a reference to the prim
        prim = stage.GetPrimAtPath("/yonsoku_robot")
        prim.GetReferences().AddReference(usd_path)
        
        print("Successfully referenced USD file")
        # Print all available prims
        print("Available prims in the stage:")
        for prim in stage.Traverse():
            print(f"  - {prim.GetPath()}")
    else:
        print("Failed to reference USD file")

# Keep the app running so we can see the result
import time
print("Keeping the app running for 10 seconds to view the result...")
time.sleep(10)

# Close the app
simulation_app.close()
# test_yonsoku_usd.py
import os
from isaaclab.app import AppLauncher
import omni.usd

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)  # Set to False to see the viewer
simulation_app = app_launcher.app

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
    from pxr import Usd, UsdGeom
    from omni.isaac.urdf import _urdf
    
    # Create a new USD stage
    stage = omni.usd.get_context().create_new_stage()
    
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
        omni.usd.get_context().save_as_stage(usd_path, None)
        print(f"Saved USD file to: {usd_path}")
    else:
        print("Failed to import URDF")
else:
    # If USD file exists, try to load it and print the prim path
    if os.path.exists(usd_path):
        # Create a new stage
        stage = omni.usd.get_context().create_new_stage()
        
        # Add a reference to the USD file
        success = omni.usd.get_context().add_reference_to_stage(usd_path, "/yonsoku_robot")
        
        if success:
            print("Successfully referenced USD file")
            # Print all available prims
            stage = omni.usd.get_context().get_stage()
            print("Available prims in the stage:")
            for prim in stage.Traverse():
                print(f"  - {prim.GetPath()}")
        else:
            print("Failed to reference USD file")

# Keep the app running so we can see the result
while simulation_app.is_running():
    simulation_app.update()

# Close the app
simulation_app.close()
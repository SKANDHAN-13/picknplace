# Updated pick-and-place script with orientation-aware IK, joint smoothing, weld constraints, and safer targets

import mujoco_py
import time
import os
import numpy as np
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import traceback

# Configuration
os.environ["MUJOCO_GL"] = "osmesa"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = mujoco_py.load_model_from_path("ur5e_mjcf.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjRenderContextOffscreen(sim)
camera_id = model.camera_name2id("fixed") if "fixed" in model.camera_names else -1

sim.data.ctrl[:] = np.zeros(model.nu)

# Body IDs
gripper_body_id = model.body_name2id("wrist_3_link")
cube_body_id = model.body_name2id("cube")
grasp_site_id = model.site_name2id("grasp_site")

frame_count = 0
frame_paths = []

def save_frame():
    global frame_count
    viewer.render(1280, 960)
    img = viewer.read_pixels(1280, 960, depth=False)
    img = img[400:680, 400:680]
    img = np.flipud(img)
    path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    Image.fromarray(img).save(path)
    frame_paths.append(path)
    frame_count += 1

# IK with position + orientation objective and joint continuity

def compute_ik(target_pos, initial_guess, target_quat=None, last_q=None, max_iter=100):
    def objective(q):
        sim.data.qpos[:6] = q
        sim.forward()
        ee_pos = sim.data.site_xpos[grasp_site_id] if grasp_site_id != -1 else sim.data.body_xpos[gripper_body_id]
        pos_err = np.linalg.norm(ee_pos - target_pos)
        err = pos_err

        if target_quat is not None:
            ee_quat = sim.data.body_xquat[gripper_body_id]
            q1 = R.from_quat(ee_quat[[1,2,3,0]])
            q2 = R.from_quat(target_quat[[1,2,3,0]])
            angle_err = q1.inv() * q2
            rot_err = np.linalg.norm(angle_err.as_rotvec())
            err += rot_err

        if last_q is not None:
            err += 0.5 * np.linalg.norm(q - last_q)

        return err

    bounds = [(-2.8, 2.8)] * 6
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': max_iter})
    sim.data.qpos[:6] = result.x
    sim.forward()
    for _ in range(5): sim.step()
    return result.x if result.success else initial_guess

# Motion controller

def move_to(target_q, steps=100):
    current_q = sim.data.qpos[:6].copy()
    Kp, Kd = 100.0, 10.0
    for i in range(steps):
        alpha = i / steps
        q_des = (1 - alpha) * current_q + alpha * target_q
        q = sim.data.qpos[:6]
        qd = sim.data.qvel[:6]
        error = q_des - q
        sim.data.ctrl[:6] = Kp * error - Kd * qd
        sim.step()
        if i % 5 == 0:
            save_frame()
    for _ in range(10):
        sim.step()
        save_frame()

# Gripper control
GRIP_OPEN = np.zeros(6)
GRIP_CLOSE = np.array([0.8, -0.8, 0.8, 0.8, -0.8, -0.8])

def control_gripper(state):
    q_des = GRIP_CLOSE if state == "close" else GRIP_OPEN
    Kp, Kd = 200.0, 5.0
    for _ in range(30):
        q = sim.data.qpos[6:12]
        qd = sim.data.qvel[6:12]
        error = q_des - q
        sim.data.ctrl[6:12] = Kp * error - Kd * qd
        sim.step()
        if _ % 5 == 0: save_frame()

# Grasping with weld
class WeldGrasper:
    def __init__(self, sim, model):
        self.sim = sim
        self.model = model
        self.eq_id = None
        self.grasped = False

    def grasp(self, cube_id, gripper_id):
        control_gripper("close")
        for _ in range(30):
            self.sim.step(); save_frame()
        pos1 = self.sim.data.body_xpos[gripper_id]
        pos2 = self.sim.data.body_xpos[cube_id]
        relpos = pos2 - pos1
        if np.linalg.norm(relpos) < 0.15:
            self.model.eq_type[0] = mujoco_py.const.EQ_WELD
            self.model.eq_obj1id[0] = gripper_id
            self.model.eq_obj2id[0] = cube_id
            self.model.eq_active[0] = 1
            self.sim.forward()
            self.grasped = True
            return True
        return False

    def release(self):
        self.model.eq_active[0] = 0
        control_gripper("open")
        for _ in range(30): self.sim.step(); save_frame()
        self.grasped = False

# Circular target

def get_circular_target(base_pos, angle_shift_deg, radius=None):
    angle = np.radians(angle_shift_deg)
    if radius is None:
        radius = np.clip(np.linalg.norm(base_pos[:2]), 0.2, 0.5)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.array([x, y, base_pos[2]])

# Main sequence

def main():
    q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    sim.data.qpos[:6] = q_home
    sim.forward()
    save_frame()

    grasper = WeldGrasper(sim, model)
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    cube_quat = sim.data.body_xquat[cube_body_id].copy()

    place_pos = get_circular_target(cube_pos, angle_shift_deg=90)

    pre_grasp = cube_pos + np.array([0, 0, 0.1])
    grasp = cube_pos + np.array([0, 0, 0.015])
    pre_place = place_pos + np.array([0, 0, 0.1])
    place = place_pos.copy()

    try:
        q1 = compute_ik(pre_grasp, q_home, last_q=q_home)
        move_to(q1)
        q2 = compute_ik(grasp, q1, last_q=q1)
        move_to(q2)

        if grasper.grasp(cube_body_id, gripper_body_id):
            q3 = compute_ik(pre_grasp, q2, last_q=q2)
            move_to(q3)

            q4 = compute_ik(pre_place, q3, last_q=q3)
            move_to(q4)
            q5 = compute_ik(place, q4, last_q=q4)
            move_to(q5)

            grasper.release()
            move_to(q4)
            move_to(q_home)
        else:
            print("Grasp failed")

    except Exception:
        traceback.print_exc()

    if frame_paths:
        with imageio.get_writer(os.path.join(output_dir, "final_output.mp4"), fps=30) as writer:
            for p in frame_paths:
                writer.append_data(imageio.v2.imread(p))

if __name__ == '__main__':
    main()


"""
--------------------------------------------------------------------------------------------------------------------
import mujoco_py
import time
import os
import numpy as np
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import traceback

# Configuration
os.environ["MUJOCO_GL"] = "osmesa"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Load model with enhancements
model = mujoco_py.load_model_from_path("ur5e_mjcf.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjRenderContextOffscreen(sim)
camera_id = model.camera_name2id("fixed") if "fixed" in model.camera_names else -1

# Initialize controls
sim.data.ctrl[:] = np.zeros(model.nu)

# Body IDs
try:
    gripper_body_id = model.body_name2id("wrist_3_link")
    cube_body_id = model.body_name2id("cube")
    grasp_site_id = model.site_name2id("grasp_site")
except:
    print("Warning: Body/site names not found. Using default IDs.")
    gripper_body_id = 0
    cube_body_id = 1
    grasp_site_id = -1

# Frame saving
frame_count = 0
frame_paths = []

def save_frame():
    global frame_count
    try:
        viewer.render(1280, 960)
        img = viewer.read_pixels(1280, 960, depth=False)
        img = img[400:680, 400:680]
        img = np.flipud(img)
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        Image.fromarray(img).save(frame_path)
        frame_paths.append(frame_path)
        frame_count += 1
    except Exception as e:
        print(f"Error saving frame: {e}")

# Enhanced IK solver
def compute_ik(target_pos, initial_guess, max_iter=100):
    old_qpos = sim.data.qpos.copy()
    
    def objective(q):
        sim.data.qpos[:6] = q
        try:
            sim.forward()
            if grasp_site_id != -1:
                ee_pos = sim.data.site_xpos[grasp_site_id]
            else:
                ee_pos = sim.data.body_xpos[gripper_body_id]
            pos_err = np.linalg.norm(ee_pos - target_pos)
            
            # Joint limit penalty
            joint_penalty = sum(max(0, val - 2.8) + max(0, -2.8 - val) for val in q)
            
            return pos_err + 10 * joint_penalty
        except:
            return float('inf')
        finally:
            sim.data.qpos[:] = old_qpos
            sim.forward()
    
    bounds = [(-2.8, 2.8)] * 6
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', 
                      options={'maxiter': max_iter})
    
    # Apply solution and settle physics
    sim.data.qpos[:6] = result.x
    sim.forward()
    for _ in range(10):
        sim.step()
    
    return result.x if result.success else initial_guess

# Motion control
def move_to(target_q, steps=100):
    current_q = sim.data.qpos[:6].copy()
    
    Kp, Kd = 100.0, 2.0
    for i in range(steps):
        alpha = i/steps
        q_des = current_q*(1-alpha) + target_q*alpha
        
        q   = sim.data.qpos[:6]
        qd  = sim.data.qvel[:6]
        error = q_des - q
        
        # PD control:
        sim.data.ctrl[:6] = Kp*error - Kd*qd
        
        sim.step()
        if i % 5 == 0:
            save_frame()
    
    # Settle
    for _ in range(10):
        sim.step()
        save_frame()

# Gripper control
GRIP_OPEN  = np.array([0, 0, 0, 0, 0, 0])
GRIP_CLOSE = np.array([0.8, -0.8, 0.8, 0.8, -0.8, -0.8])

def control_gripper(state):
    q_des = GRIP_CLOSE if state == "close" else GRIP_OPEN
    Kp, Kd = 200.0, 5.0
    
    for _ in range(30):
        q   = sim.data.qpos[6:12]
        qd  = sim.data.qvel[6:12]
        error = q_des - q
        
        # PD on fingers:
        sim.data.ctrl[6:12] = Kp*error - Kd*qd
        
        sim.step()
        if _ % 5 == 0:
            save_frame()

# ============== IMPROVED GRASPING SYSTEM ==============
class ConstraintGrasper:
    def __init__(self, sim, model):
        self.sim = sim
        self.model = model
        self.constraint_id = None
        self.is_grasping = False
        self.cube_qpos_start = None
        self.cube_qvel_start = None
        
        # Find cube's joint indices
        try:
            cube_joint_names = [name for name in model.joint_names if 'cube' in name.lower()]
            if cube_joint_names:
                cube_joint_id = model.joint_name2id(cube_joint_names[0])
                self.cube_qpos_start = model.jnt_qposadr[cube_joint_id]
                self.cube_qvel_start = model.jnt_dofadr[cube_joint_id]
                print(f"Cube joint indices found - qpos: {self.cube_qpos_start}, qvel: {self.cube_qvel_start}")
            else:
                print("Warning: No cube joint found for constraint grasping")
        except Exception as e:
            print(f"Error finding cube joint indices: {e}")
        
    def create_weld_constraint(self, body1_id, body2_id, relative_pos=None):
        #Create a weld constraint between two bodies
        if relative_pos is None:
            # Calculate relative position
            pos1 = self.sim.data.body_xpos[body1_id]
            pos2 = self.sim.data.body_xpos[body2_id]
            relative_pos = pos2 - pos1
        
        # Create constraint data
        constraint_data = np.zeros(11)  # MuJoCo constraint format
        constraint_data[0] = 3  # mjCNSTR_WELD
        constraint_data[1] = body1_id
        constraint_data[2] = body2_id
        constraint_data[3:6] = relative_pos  # relative position
        constraint_data[6:10] = [1, 0, 0, 0]  # relative quaternion (identity)
        constraint_data[10] = 1  # constraint active
        
        return constraint_data
    
    def grasp_cube(self, cube_body_id, gripper_body_id, grasp_distance=0.15):
        #Attempt to grasp the cube using constraints
        print("Attempting constraint-based grasp...")
        
        # Close gripper
        control_gripper("close")
        
        # Wait for gripper to close
        for _ in range(50):
            self.sim.step()
            save_frame()
        
        # Check if cube is within grasp distance
        gripper_pos = self.sim.data.body_xpos[gripper_body_id]
        cube_pos = self.sim.data.body_xpos[cube_body_id]
        distance = np.linalg.norm(cube_pos - gripper_pos)
        
        print(f"Distance between gripper and cube: {distance:.4f}")
        
        if distance < grasp_distance:
            # Create weld constraint
            try:
                # Method 1: Direct body position manipulation (simpler approach)
                self.attach_cube_to_gripper(cube_body_id, gripper_body_id)
                self.is_grasping = True
                print("Cube attached to gripper successfully!")
                return True
            except Exception as e:
                print(f"Constraint creation failed: {e}")
                return False
        else:
            print(f"Cube too far from gripper (distance: {distance:.4f})")
            return False
    
    def attach_cube_to_gripper(self, cube_body_id, gripper_body_id):
        #Attach cube to gripper by maintaining relative position
        gripper_pos = self.sim.data.body_xpos[gripper_body_id]
        cube_pos = self.sim.data.body_xpos[cube_body_id]
        self.relative_pos = cube_pos - gripper_pos
        self.cube_body_id = cube_body_id
        self.gripper_body_id = gripper_body_id
        print(f"Relative position stored: {self.relative_pos}")
    
    def update_grasp(self):
        #Update cube position to maintain grasp
        if self.is_grasping and self.cube_qpos_start is not None:
            try:
                # Get current gripper position
                gripper_pos = self.sim.data.body_xpos[self.gripper_body_id]
                # Calculate target cube position
                target_cube_pos = gripper_pos + self.relative_pos
                
                # Apply position directly (this is a simplification)
                # In a real implementation, you'd use forces or constraints
                if self.cube_qpos_start + 3 <= len(self.sim.data.qpos):
                    self.sim.data.qpos[self.cube_qpos_start:self.cube_qpos_start+3] = target_cube_pos
                
                # Also zero out cube velocity to prevent drift
                if self.cube_qvel_start is not None and self.cube_qvel_start + 3 <= len(self.sim.data.qvel):
                    self.sim.data.qvel[self.cube_qvel_start:self.cube_qvel_start+3] = 0
                
            except Exception as e:
                print(f"Error updating grasp: {e}")
    
    def get_cube_qpos_idx(self):
        #Get the starting index of cube position in qpos array
        return self.cube_qpos_start if self.cube_qpos_start is not None else 12
    
    def get_cube_qvel_idx(self):
        #Get the starting index of cube velocity in qvel array
        return self.cube_qvel_start if self.cube_qvel_start is not None else 12
    
    def release_cube(self):
        #Release the cube
        print("Releasing cube...")
        control_gripper("open")
        self.is_grasping = False
        
        # Wait for gripper to open
        for _ in range(30):
            self.sim.step()
            save_frame()
        
        return True

# Alternative approach using equality constraints
def create_equality_constraint(sim, body1_id, body2_id):
    #"Create equality constraint between two bodies
    # This is a more advanced approach that would require
    # modifying the MuJoCo model or using the constraint API
    pass

# ============== MAIN EXECUTION ==============
def main():
    # Initialize constraint grasper
    grasper = ConstraintGrasper(sim, model)
    
    # Initialization
    q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    sim.data.qpos[:6] = q_home
    
    # Debug: Print qpos array info
    print(f"Total qpos size: {sim.data.qpos.shape}")
    print(f"Total qvel size: {sim.data.qvel.shape}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    
    # Try to find cube's joint and set position
    cube_initial_pos = np.array([0.2, 0.1, 0.65])  # x, y, z coordinates
    
    try:
        # Look for cube joint
        cube_joint_names = [name for name in model.joint_names if 'cube' in name.lower()]
        print(f"Found cube joints: {cube_joint_names}")
        
        if cube_joint_names:
            cube_joint_id = model.joint_name2id(cube_joint_names[0])
            cube_qpos_start = model.jnt_qposadr[cube_joint_id]
            cube_qvel_start = model.jnt_dofadr[cube_joint_id]
            
            print(f"Cube joint ID: {cube_joint_id}")
            print(f"Cube qpos start: {cube_qpos_start}")
            print(f"Cube qvel start: {cube_qvel_start}")
            
            # Set cube position if it has a free joint (7 DOF: 3 pos + 4 quat)
            if cube_qpos_start + 3 <= len(sim.data.qpos):
                sim.data.qpos[cube_qpos_start:cube_qpos_start+3] = cube_initial_pos
                print(f"Set cube position to: {cube_initial_pos}")
                
                # Set quaternion to identity if it's a free joint
                if cube_qpos_start + 7 <= len(sim.data.qpos):
                    sim.data.qpos[cube_qpos_start+3:cube_qpos_start+7] = [1, 0, 0, 0]  # w, x, y, z
                
                # Reset velocities
                if cube_qvel_start + 6 <= len(sim.data.qvel):
                    sim.data.qvel[cube_qvel_start:cube_qvel_start+6] = 0
                    
            else:
                print("Warning: Cannot set cube position - insufficient qpos space")
        else:
            print("Warning: No cube joint found. Cube position may be fixed in XML.")
            
    except Exception as e:
        print(f"Error setting cube position: {e}")
        print("Proceeding with default cube position...")
    
    sim.forward()
    save_frame()
    
    # Get cube position
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    print(f"Cube position: {cube_pos}")
    
    # Placement location
    place_pos = np.array([0.0, 0.5, cube_pos[2]])
    
    # Approach positions - adjust these offsets as needed
    approach_height = 0.1
    
    # Grasp position: slightly above cube center
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    print(f"Cube position: {cube_pos}")

# Calculate grasp position - approach from directly above
    grasp_pos = cube_pos.copy()
    grasp_pos[2] += 0.1
    # Adjust the Z offset (0.02) based on your gripper geometry
    #grasp_pos = cube_pos.copy() + np.array([0, 0, 0.02])
    
    # You can also offset in X,Y if needed for better gripper alignment
    # grasp_pos = cube_pos.copy() + np.array([0.01, 0, 0.02])  # slight X offset
    
    pre_grasp_pos = grasp_pos.copy() + np.array([0, 0, approach_height])
    pre_place_pos = place_pos.copy() + np.array([0, 0, approach_height])
    
    # Execute sequence
    try:
        print("Moving to pre-grasp position...")
        q_pre_grasp = compute_ik(pre_grasp_pos, q_home)
        move_to(q_pre_grasp)
        
        print("Moving to grasp position...")
        q_grasp = compute_ik(grasp_pos, q_pre_grasp)
        move_to(q_grasp)
        
        # Debug: Check final gripper and cube positions
        final_gripper_pos = sim.data.body_xpos[gripper_body_id]
        final_cube_pos = sim.data.body_xpos[cube_body_id]
        distance = np.linalg.norm(final_cube_pos - final_gripper_pos)
        print(f"Final gripper position: {final_gripper_pos}")
        print(f"Final cube position: {final_cube_pos}")
        print(f"Distance: {distance:.4f}")
        
        print("Attempting to grasp cube...")
        if grasper.grasp_cube(cube_body_id, gripper_body_id, grasp_distance=0.2):
            print("Cube grasped successfully!")
            
            # Update grasp during movement
            print("Lifting cube...")
            q_lifted = compute_ik(pre_grasp_pos, q_grasp)
            
            # Move with grasp updates
            current_q = sim.data.qpos[:6].copy()
            Kp, Kd = 100.0, 2.0
            
            for i in range(100):
                alpha = i/100
                q_des = current_q*(1-alpha) + q_lifted*alpha
                
                q   = sim.data.qpos[:6]
                qd  = sim.data.qvel[:6]
                error = q_des - q
                
                sim.data.ctrl[:6] = Kp*error - Kd*qd
                sim.step()
                
                # Update grasp constraint
                grasper.update_grasp()
                
                if i % 5 == 0:
                    save_frame()
            
            print("Moving to pre-place position...")
            q_pre_place = compute_ik(pre_place_pos, q_lifted)
            
            # Move to pre-place with grasp updates
            current_q = sim.data.qpos[:6].copy()
            for i in range(100):
                alpha = i/100
                q_des = current_q*(1-alpha) + q_pre_place*alpha
                
                q   = sim.data.qpos[:6]
                qd  = sim.data.qvel[:6]
                error = q_des - q
                
                sim.data.ctrl[:6] = Kp*error - Kd*qd
                sim.step()
                
                # Update grasp constraint
                grasper.update_grasp()
                
                if i % 5 == 0:
                    save_frame()
            
            print("Moving to place position...")
            q_place = compute_ik(place_pos, q_pre_place)
            
            # Move to place with grasp updates
            current_q = sim.data.qpos[:6].copy()
            for i in range(100):
                alpha = i/100
                q_des = current_q*(1-alpha) + q_place*alpha
                
                q   = sim.data.qpos[:6]
                qd  = sim.data.qvel[:6]
                error = q_des - q
                
                sim.data.ctrl[:6] = Kp*error - Kd*qd
                sim.step()
                
                # Update grasp constraint
                grasper.update_grasp()
                
                if i % 5 == 0:
                    save_frame()
            
            print("Releasing cube...")
            grasper.release_cube()
            
            print("Lifting after placement...")
            q_lifted_final = compute_ik(pre_place_pos, q_place)
            move_to(q_lifted_final)
            
            print("Returning to home position...")
            move_to(q_home)
            
            print("Pick-and-place completed successfully!")
        else:
            print("Failed to grasp cube")
            
    except Exception as e:
        traceback.print_exc()
        print(f"Error during execution: {e}")

    # Create video
    if frame_paths:
        video_path = os.path.join(output_dir, "pick_place_constraint.mp4")
        try:
            with imageio.get_writer(video_path, fps=30) as writer:
                for path in frame_paths:
                    writer.append_data(imageio.v2.imread(path))
            print(f"Video saved at {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")

if __name__ == "__main__":
    main()
"""

"""
------------------------------------------------------------------------------------------
import mujoco_py
import time
import os
import numpy as np
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import traceback

# Configuration
os.environ["MUJOCO_GL"] = "osmesa"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Load model with enhancements
model = mujoco_py.load_model_from_path("ur5e_mjcf.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjRenderContextOffscreen(sim)
camera_id = model.camera_name2id("fixed") if "fixed" in model.camera_names else -1

# Initialize controls
sim.data.ctrl[:] = np.zeros(model.nu)

# Body IDs
try:
    gripper_body_id = model.body_name2id("wrist_3_link")
    cube_body_id = model.body_name2id("cube")
    grasp_site_id = model.site_name2id("grasp_site")
except:
    print("Warning: Body/site names not found. Using default IDs.")
    gripper_body_id = 0
    cube_body_id = 1
    grasp_site_id = -1

# Frame saving
frame_count = 0
frame_paths = []

def save_frame():
    global frame_count
    try:
        viewer.render(1280, 960)
        img = viewer.read_pixels(1280, 960, depth=False)
        img = img[400:680, 400:680]
        img = np.flipud(img)
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        Image.fromarray(img).save(frame_path)
        frame_paths.append(frame_path)
        frame_count += 1
    except Exception as e:
        print(f"Error saving frame: {e}")

# Enhanced IK solver
def compute_ik(target_pos, initial_guess, max_iter=100):
    old_qpos = sim.data.qpos.copy()
    
    def objective(q):
        sim.data.qpos[:6] = q
        try:
            sim.forward()
            if grasp_site_id != -1:
                ee_pos = sim.data.site_xpos[grasp_site_id]
            else:
                ee_pos = sim.data.body_xpos[gripper_body_id]
            pos_err = np.linalg.norm(ee_pos - target_pos)
            
            # Joint limit penalty
            joint_penalty = sum(max(0, val - 2.8) + max(0, -2.8 - val) for val in q)
            
            return pos_err + 10 * joint_penalty
        except:
            return float('inf')
        finally:
            sim.data.qpos[:] = old_qpos
            sim.forward()
    
    bounds = [(-2.8, 2.8)] * 6
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', 
                      options={'maxiter': max_iter})
    
    # Apply solution and settle physics
    sim.data.qpos[:6] = result.x
    sim.forward()
    for _ in range(10):
        sim.step()
    
    return result.x if result.success else initial_guess

# Dynamic trajectory generation
def generate_trajectory(start_pos, end_pos, height=0.1, num_points=10):
    via_points = []
    
    # Lift up
    for i in range(num_points):
        alpha = i / (num_points - 1)
        z = start_pos[2] + height * alpha
        via_points.append([start_pos[0], start_pos[1], z])
    
    # Move horizontally
    for i in range(num_points):
        alpha = i / (num_points - 1)
        x = start_pos[0] * (1 - alpha) + end_pos[0] * alpha
        y = start_pos[1] * (1 - alpha) + end_pos[1] * alpha
        via_points.append([x, y, start_pos[2] + height])
    
    # Lower down
    for i in range(num_points):
        alpha = i / (num_points - 1)
        z = end_pos[2] + height * (1 - alpha)
        via_points.append([end_pos[0], end_pos[1], z])
    
    return np.array(via_points)

# Motion control
def move_through_path(path_points, current_q, steps_per_segment=50):
    for point in path_points:
        q_target = compute_ik(point, current_q)
        move_to(q_target, steps=steps_per_segment)
        current_q = q_target.copy()
    return current_q

def move_to(target_q, steps=100):
    current_q = sim.data.qpos[:6].copy()
    
    Kp, Kd = 100.0, 2.0
    for i in range(steps):
        alpha = i/steps
        q_des = current_q*(1-alpha) + target_q*alpha
        
        q   = sim.data.qpos[:6]
        qd  = sim.data.qvel[:6]
        error = q_des - q
        
        # PD control:
        sim.data.ctrl[:6] = Kp*error - Kd*qd
        
        sim.step()
        if i % 5 == 0:
            save_frame()

    
    # Settle
    for _ in range(10):
        sim.step()
        save_frame()


GRIP_OPEN  = np.array([0, 0, 0, 0, 0, 0])
GRIP_CLOSE = np.array([ 0.8, -0.8,  0.8,  0.8, -0.8, -0.8 ])  

def control_gripper(state):
    q_des = GRIP_CLOSE if state=="close" else GRIP_OPEN
    Kp, Kd = 200.0, 5.0
    
    for _ in range(30):
        q   = sim.data.qpos[6:12]
        qd  = sim.data.qvel[6:12]
        error = q_des - q
        
        # PD on fingers:
        sim.data.ctrl[6:12] = Kp*error - Kd*qd
        
        sim.step()
        if _ % 5 == 0:
            save_frame()

# Cube manipulation - FIXED VERSION
def grasp_cube():
    #Check contact and apply forces to hold cube
    gripper_geoms = []
    cube_geom = None
    
    # Identify gripper geoms
    for geom_id in range(model.ngeom):
        try:
            body_name = model.body_names[model.geom_bodyid[geom_id]]
            if "knuckle" in body_name or "finger" in body_name or "wrist_3" in body_name:
                gripper_geoms.append(geom_id)
            if "cube" in body_name:
                cube_geom = geom_id
        except IndexError:
            continue
    
    if cube_geom is None:
        print("Couldn't find cube geom")
        return False
    if not gripper_geoms:
        print("Couldn't find gripper geoms")
        return False
    
    print(f"Gripper geoms: {gripper_geoms}")
    print(f"Cube geom: {cube_geom}")
    
    for attempt in range(3):
        print(f"Attempt {attempt+1} to grasp cube...")
        control_gripper("close")
        
        # Let gripper settle
        for _ in range(20):
            sim.step()
            save_frame()
        
        contact_found = False
        for step in range(100):
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                if (g1 in gripper_geoms and g2 == cube_geom) or \
                (g2 in gripper_geoms and g1 == cube_geom):
                    print(f"Contact detected between geom{g1} and geom{g2}")
                    contact_found = True
                    break
            
            if contact_found:
                print("Contact found! Verifying if cube is lifted...")

                # Record initial cube position
                initial_cube_pos = sim.data.body_xpos[cube_body_id].copy()

                # Attempt a small lift
                q_lift = sim.data.qpos[:6].copy()
                q_lift[2] += 0.05  # Slight upward movement in Z joint
                move_to(q_lift, steps=30)

                # Check new cube position
                new_cube_pos = sim.data.body_xpos[cube_body_id].copy()
                lift_distance = np.linalg.norm(new_cube_pos - initial_cube_pos)
                print(f"Cube movement after lift attempt: {lift_distance:.4f} meters")

                # Optional: return cube to original Z position
                move_to(sim.data.qpos[:6], steps=30)

                if lift_distance > 0.015:
                    print("Cube grasped and lifted successfully!")
                    return True
                else:
                    print("Cube did not move – grasp unsuccessful.")

            
            sim.step()
            if step % 5 == 0:
                save_frame()
        
        # Retry with open
        control_gripper("open")
        for _ in range(30):
            sim.step()
            save_frame()

    
    print("Failed to grasp cube after 3 attempts")
    return False

def release_cube():
    control_gripper("open")
    return True

# Main sequence
def main():
    # Initialization
    q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    sim.data.qpos[:6] = q_home
    sim.forward()
    save_frame()
    
    # Get cube position
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    print(f"Cube position: {cube_pos}")
    
    # Placement location
    place_pos = np.array([0.0, 0.5, cube_pos[2]])
    
    # Approach positions
    approach_height = 0.1
    grasp_pos = cube_pos.copy() + np.array([0, 0, 0.015])  # Lower approach
    pre_grasp_pos = grasp_pos.copy() + np.array([0, 0, approach_height])
    pre_place_pos = place_pos.copy() + np.array([0, 0, approach_height])
    
    # Execute sequence
    try:
        print("Moving to pre-grasp position...")
        q_pre_grasp = compute_ik(pre_grasp_pos, q_home)
        move_to(q_pre_grasp)
        
        print("Moving to grasp position...")
        q_grasp = compute_ik(grasp_pos, q_pre_grasp)
        move_to(q_grasp)
        
        print("Attempting to grasp cube...")
        if grasp_cube():
            print("Lifting cube...")
            q_lifted = compute_ik(pre_grasp_pos, q_grasp)
            move_to(q_lifted)
            
            print("Moving to pre-place position...")
            trajectory = generate_trajectory(pre_grasp_pos, pre_place_pos)
            current_q = move_through_path(trajectory, q_lifted)
            
            print("Moving to place position...")
            q_place = compute_ik(place_pos, current_q)
            move_to(q_place)
            
            print("Releasing cube...")
            release_cube()
            
            print("Lifting after placement...")
            move_to(current_q)
            
            print("Returning to home position...")
            move_to(q_home)
            
            print("Pick-and-place completed successfully!")
        else:
            print("Failed to grasp cube")
            
    except Exception as e:
        traceback.print_exc()
        print(f"Error during execution: {e}")

    # Create video
    if frame_paths:
        video_path = os.path.join(output_dir, "pick_place_3s.mp4")
        try:
            with imageio.get_writer(video_path, fps=30) as writer:
                for path in frame_paths:
                    writer.append_data(imageio.v2.imread(path))  # Use v2 to avoid warning
            print(f"Video saved at {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")

if __name__ == "__main__":
    main()



"""





"""
# Get body IDs
try:
    gripper_body_id = model.body_name2id("wrist_3_link")
    cube_body_id = model.body_name2id("cube")
except:
    print("Warning: Could not find expected body names. Check your XML file.")
    gripper_body_id = 0
    cube_body_id = 1

# Initial pose - use more reasonable starting position
q_init = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])  # Common UR5 home position

# Reset simulation to initial state
sim.data.qpos[:6] = q_init
sim.forward()

# Targets
pick_pos = np.array([0.5, 0, 0.65])
place_pos = np.array([0.0, 0.5, 0.65])
quat_identity = np.array([1, 0, 0, 0])

# Save rendered image
frame_count = 0
frame_paths = []
def save_frame():
    global frame_count
    viewer.render(1280, 960)
    img = viewer.read_pixels(1280, 960, depth=False)
    img = img[400:680, 400:680]
    img = np.flipud(img)
    frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    Image.fromarray(img).save(frame_path)
    frame_paths.append(frame_path)
    frame_count += 1

# Move to joint config with proper control
def move_to(target_q, steps=500):
    current_q = sim.data.qpos[:6].copy()
    
    for i in range(steps):
        # Interpolate between current and target
        alpha = i / (steps - 1)
        interpolated_q = current_q * (1 - alpha) + target_q * alpha
        
        # Set target position (this should be control input, not direct position)
        sim.data.ctrl[:6] = interpolated_q
        
        # Step simulation
        try:
            sim.step()
            
            # Check for instability
            if np.any(np.isnan(sim.data.qpos)) or np.any(np.isinf(sim.data.qpos)):
                print("Simulation became unstable, stopping...")
                break
                
            # Save frame every few steps
            if i % 5 == 0:
                save_frame()
                
        except Exception as e:
            print(f"Error during simulation step: {e}")
            break
    
    # Final position
    sim.data.ctrl[:6] = target_q
    for _ in range(50):  # Let it settle
        sim.step()
        if _ % 10 == 0:
            save_frame()

# Constraint handling (simplified)
constraint_attached = False
def attach_cube():
    global constraint_attached
    constraint_attached = True
    print("Cube attached (simulated)")

def detach_cube():
    global constraint_attached
    constraint_attached = False
    print("Cube detached (simulated)")

# Main sequence
print("Starting pick-and-place sequence...")

try:
    # Move to initial position
    print("Moving to initial position...")
    move_to(q_init)
    
    # Compute IK for pick position
    print("Computing IK for pick position...")
    q1 = compute_ik(pick_pos, quat_identity, q_init)
    print(f"Pick joints: {q1}")
    
    # Move to pick position
    print("Moving to pick position...")
    move_to(q1)
    
    # Attach cube
    attach_cube()
    
    # Compute IK for place position
    print("Computing IK for place position...")
    q2 = compute_ik(place_pos, quat_identity, q1)
    print(f"Place joints: {q2}")
    
    # Move to place position
    print("Moving to place position...")
    move_to(q2)
    
    # Detach cube
    detach_cube()
    
    print("Pick-and-place sequence completed successfully!")
    
except Exception as e:
    print(f"Error during execution: {e}")

# Save video
if frame_paths:
    video_path = os.path.join(output_dir, "pick_place_demo.mp4")
    try:
        with imageio.get_writer(video_path, fps=10) as writer:
            for frame_file in frame_paths:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        print(f"Video saved at {video_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
else:
    print("No frames were saved.")

print("Done.")"""

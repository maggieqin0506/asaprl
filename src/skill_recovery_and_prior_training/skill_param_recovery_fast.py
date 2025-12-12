import matplotlib.pyplot as plt
import math, pdb
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import random
import os
import pickle
import copy
from scipy.ndimage import gaussian_filter1d
from asaprl.policy.planning_model import PathParam, SpeedParam, dynamic_constraint, dist_constraint, motion_skill_model

def cost_function(u, *args):
    current_v = args[0]
    current_a = args[1]
    horizon = args[2]
    reference_traj = args[3]

    lat1 = u[0]
    yaw1 = u[1]
    v1 = u[2]
    generate_traj, _, _, _  = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    # motion_skill_model returns horizon+1 points, skip the first point (initial position) to match reference_traj
    generate_traj = generate_traj[1:, :]
    
    # Ensure trajectories have the same length
    min_len = min(len(generate_traj), len(reference_traj))
    generate_traj = generate_traj[:min_len]
    reference_traj = reference_traj[:min_len]

    # Handle both 2D trajectory (x, y) from rule expert and 4D trajectory (x, y, speed, yaw) from RL expert
    if reference_traj.shape[1] == 2:
        # Rule expert data: only compare x, y positions
        cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - reference_traj[:,:2]), axis=1)))
    else:
        # RL expert data: compare x, y, speed, yaw
        cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - reference_traj[:,:2]), axis=1)))
        cost += np.sum(np.sqrt(np.square(generate_traj[:,2] - reference_traj[:,2])))
        cost += np.sum(np.sqrt(np.square(generate_traj[:,3] - reference_traj[:,3])))
    return cost 

def global_cost_function(u_flat, *args):
    """
    Global cost function for optimizing all segments simultaneously.
    u_flat: Flattened array of parameters [lat_0, yaw_0, v_0, lat_1, yaw_1, v_1, ...]
    """
    all_reference_trajs = args[0]
    all_current_vs = args[1]
    horizon = args[2]
    smoothness_weight = args[3]

    num_segments = len(all_reference_trajs)
    total_cost = 0
    
    # Reshape u_flat to (num_segments, 3)
    u = u_flat.reshape((num_segments, 3))
    
    # 1. Reconstruction Cost
    for i in range(num_segments):
        lat1 = u[i, 0]
        yaw1 = u[i, 1]
        v1 = u[i, 2]
        current_v = all_current_vs[i]
        current_a = 0 # Assuming 0 for now as in original code
        reference_traj = all_reference_trajs[i]
        
        generate_traj, _, _, _ = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)
        
        # motion_skill_model returns horizon+1 points, skip the first point (initial position) to match reference_traj
        generate_traj = generate_traj[1:, :]
        
        # Ensure trajectories have the same length (use copy to avoid modifying original)
        ref_traj = reference_traj.copy()
        min_len = min(len(generate_traj), len(ref_traj))
        generate_traj = generate_traj[:min_len]
        ref_traj = ref_traj[:min_len]
        
        # Trajectory difference
        if ref_traj.shape[1] == 2:
            cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - ref_traj[:,:2]), axis=1)))
        else:
            cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - ref_traj[:,:2]), axis=1)))
            cost += np.sum(np.sqrt(np.square(generate_traj[:,2] - ref_traj[:,2])))
            cost += np.sum(np.sqrt(np.square(generate_traj[:,3] - ref_traj[:,3])))
        total_cost += cost

    # 2. Smoothness Cost (Difference between adjacent parameters)
    if num_segments > 1:
        diffs = u[1:] - u[:-1]
        # Weighted smoothness: maybe penalize yaw changes more? For now, uniform.
        smoothness_cost = np.sum(np.square(diffs))
        total_cost += smoothness_weight * smoothness_cost

    return total_cost

def recover_parameter_global(all_reference_trajs, all_current_vs, horizon, lat_bound=10, smoothness_weight=1.0):
    num_segments = len(all_reference_trajs)
    
    # Initial guess: can be all zeros or some heuristic
    # Let's try to initialize with individual optimizations first (optional, but good for convergence)
    # Or just random/zeros. Let's use zeros for simplicity, or maybe the previous 'recover_parameter' logic for initialization?
    # Using individual recovery for initialization is safer.
    
    u_init_list = []
    print("Initializing with individual recovery...")
    for i in range(num_segments):
        # Simplified individual recovery for initialization
        # We can just use 0,0,5 as a rough guess to save time, or run the full search.
        # Let's run a quick single optimization per segment to get a good start point.
        current_v = all_current_vs[i]
        reference_traj = all_reference_trajs[i]
        
        # Quick local optimization
        bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
        u_local_init = np.array([0, 0, 5]) 
        # Clamp initial guess
        for j in range(3):
             lower, upper = bounds[j]
             u_local_init[j] = np.clip(u_local_init[j], lower + 1e-4, upper - 1e-4)

        res = minimize(cost_function, u_local_init, (current_v, 0, horizon, reference_traj),
                       method='L-BFGS-B', bounds=bounds, tol=1e-3)
        u_init_list.append(res.x)
        
    u_init_flat = np.array(u_init_list).flatten()
    
    # Global bounds
    bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]] * num_segments
    
    # Clamp u_init_flat to be strictly within bounds
    for i in range(len(u_init_flat)):
        lower, upper = bounds[i]
        u_init_flat[i] = np.clip(u_init_flat[i], lower + 1e-4, upper - 1e-4)
        if not (lower <= u_init_flat[i] <= upper):
             print(f"Bound violation at index {i}: val={u_init_flat[i]}, bounds=[{lower}, {upper}]")

    # Sliding Window Optimization
    # Instead of optimizing all segments at once (which scales poorly), we optimize in overlapping windows.
    window_size = 20
    overlap = 5
    
    u_current = u_init_flat.reshape((num_segments, 3))
    
    print(f"Starting sliding window optimization (Window: {window_size}, Overlap: {overlap})...")
    
    for start_idx in range(0, num_segments, window_size - overlap):
        end_idx = min(start_idx + window_size, num_segments)
        current_window_size = end_idx - start_idx
        
        if current_window_size < 2:
            break
            
        print(f"Optimizing window [{start_idx}:{end_idx}]...")
        
        # Extract data for this window
        window_refs = all_reference_trajs[start_idx:end_idx]
        window_vs = all_current_vs[start_idx:end_idx]
        
        # Initial guess for this window (from current best)
        u_window_init = u_current[start_idx:end_idx].flatten()
        
        # Bounds for this window
        window_bounds = bounds[start_idx*3 : end_idx*3]
        
        # Optimize window
        res = minimize(global_cost_function, u_window_init, 
                       (window_refs, window_vs, horizon, smoothness_weight),
                       method='L-BFGS-B', bounds=window_bounds, tol=1e-3, options={'maxiter': 20})
        
        # Update current solution
        u_window_opt = res.x.reshape((current_window_size, 3))
        
        # Blend overlapping regions (simple replacement for now, could be weighted average)
        u_current[start_idx:end_idx] = u_window_opt
        
    u_final = u_current
    return u_final

def recover_parameter_fast(all_reference_trajs, all_current_vs, horizon, lat_bound=10, sigma=2.0):
    """
    Fast recovery method: Local Optimization + Gaussian Smoothing.
    Extremely fast but may not respect dynamic constraints as strictly as global optimization.
    """
    num_segments = len(all_reference_trajs)
    u_list = []
    
    print(f"Running FAST recovery (Local Opt + Smoothing sigma={sigma})...")
    
    # 1. Local Optimization (Parallelizable in theory, but loop is fast enough here)
    for i in range(num_segments):
        current_v = all_current_vs[i]
        reference_traj = all_reference_trajs[i]
        
        # Initial guess
        u_local_init = np.array([0, 0, 5]) 
        bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
        
        # Clamp initial
        for j in range(3):
             lower, upper = bounds[j]
             u_local_init[j] = np.clip(u_local_init[j], lower + 1e-4, upper - 1e-4)

        # Fast local minimization
        # L-BFGS-B is good, but for speed we could even reduce tolerance or maxiter
        res = minimize(cost_function, u_local_init, (current_v, 0, horizon, reference_traj),
                       method='L-BFGS-B', bounds=bounds, tol=1e-2, options={'maxiter': 10})
        u_list.append(res.x)
        
    u_array = np.array(u_list)
    
    # 2. Gaussian Smoothing
    # Apply smoothing to each parameter dimension independently
    u_smoothed = np.zeros_like(u_array)
    for dim in range(3):
        u_smoothed[:, dim] = gaussian_filter1d(u_array[:, dim], sigma=sigma)
        
    return u_smoothed

def transform_planning_param_to_latentvar(lat1, yaw1, v1, lat_range=5):
    action0 = lat1 / lat_range
    action1 = yaw1 / 30
    action2 = v1 / 5 - 1
    return action0, action1, action2

# annotate raw demonstration and save annotated demonstration
class annotate_data():
    def __init__(self, scenario, skill_length = 10, max_files=0):
        self.scenario = scenario
        self.skill_length = skill_length
        self.max_files = max_files
        self.load_data_path = 'demonstration_rule_expert/{}/'.format(self.scenario)
        self.save_data_path = 'demonstration_RL_expert/{}_fast_annotated/'.format(self.scenario)
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        '''
        Fast recovery method: Local Optimization + Gaussian Smoothing
        '''
        all_file_lst = os.listdir(self.load_data_path)
        
        if self.max_files > 0:
            all_file_lst = all_file_lst[:self.max_files]
            
        for file_idx, one_file in enumerate(all_file_lst):
            ''' load demo '''
            one_file_full_path = self.load_data_path + one_file
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            ''' annotate data '''
            annotate_one_file_data = copy.deepcopy(one_file_data)
            annotate_one_file_data['fast_recovered_latent_var'] = []
            
            # Collect all data for fast optimization
            all_reference_trajs = []
            all_current_vs = []
            
            # Handle both RL expert data (with 'latent_var') and rule expert data (with 'action')
            if 'latent_var' in one_file_data:
                # RL expert data format
                num_samples = len(one_file_data['latent_var'])
                traj_list = one_file_data['rela_state']
                if 'current_spd' in one_file_data:
                    speed_list = [spd.item() if hasattr(spd, 'item') else spd for spd in one_file_data['current_spd']]
                else:
                    speed_list = [one_file_data.get('vehicle_start_speed', [5.0])[0]] * num_samples
            else:
                # Rule expert data format
                num_samples = len(one_file_data['rela_state'])
                traj_list = one_file_data['rela_state']
                if 'vehicle_start_speed' in one_file_data:
                    speed_list = one_file_data['vehicle_start_speed']
                else:
                    speed_list = [5.0] * num_samples
            
            for latent_var_idx in range(num_samples):
                one_traj = traj_list[latent_var_idx]
                one_spd = speed_list[latent_var_idx]
                
                # Handle speed format
                if hasattr(one_spd, 'item'):
                    one_spd = one_spd.item()
                elif isinstance(one_spd, (list, np.ndarray)) and len(one_spd) > 0:
                    one_spd = one_spd[0] if hasattr(one_spd[0], 'item') else one_spd[0]
                
                # Clip speed as in original code
                if one_traj.shape[1] >= 4:
                    current_v = np.clip(np.sqrt(np.sum(np.square(one_traj[0,2:]))), 0.1, 9.9)
                else:
                    current_v = one_spd if one_spd > 0 else 5.0
                    current_v = np.clip(current_v, 0.1, 9.9)
                
                all_reference_trajs.append(one_traj)
                all_current_vs.append(current_v)
            
            # Run fast recovery
            recovered_params = recover_parameter_fast(all_reference_trajs, all_current_vs, horizon=10, lat_bound=5, sigma=2.0)
            
            for i in range(num_samples):
                lat1, yaw1, v1 = recovered_params[i]
                recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = transform_planning_param_to_latentvar(lat1, yaw1, v1, lat_range=5)
                one_recovered_latent_var = np.array([recovered_latent_var0, recovered_latent_var1, recovered_latent_var2])
                annotate_one_file_data['fast_recovered_latent_var'].append(one_recovered_latent_var)

            # Preserve original file name
            output_file_path = os.path.join(self.save_data_path, one_file)
            with open(output_file_path, 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


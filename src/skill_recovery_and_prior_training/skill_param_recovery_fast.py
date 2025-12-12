"""
Fast Skill Parameter Recovery (Local Optimization + Gaussian Smoothing)

This module implements a FAST recovery method that trades some accuracy for speed.
It uses local optimization per segment followed by Gaussian smoothing for continuity.

Key Features:
- Local optimization per segment (very fast)
- Gaussian smoothing (sigma=2.0) for parameter continuity
- Point-wise cost function (like old method) but with smoothing
- Much faster than trajectory-level methods

Used by: main_skill_recovery_fast.py

Trade-off: Speed vs. Accuracy
- Very fast execution
- Good for prototyping and quick testing
- May sacrifice some accuracy compared to trajectory-level methods
"""

import os
import pickle
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from asaprl.policy.planning_model import motion_skill_model

# Constants
DEFAULT_HORIZON = 10
DEFAULT_LAT_BOUND = 5
DEFAULT_LAT_RANGE = 5
DEFAULT_SIGMA = 2.0
MIN_SPEED = 0.1
MAX_SPEED = 9.9
INITIAL_LAT = 0.0
INITIAL_YAW = 0.0
INITIAL_V = 5.0

def cost_function(u, *args):
    """
    Point-wise cost function for fast recovery method.
    Uses sum of point-wise differences (faster than trajectory-level).
    
    Args:
        u: Parameter vector [lat1, yaw1, v1]
        *args: (current_v, current_a, horizon, reference_traj)
    
    Returns:
        Cost value (float)
    """
    current_v, current_a, horizon, reference_traj = args

    lat1, yaw1, v1 = u[0], u[1], u[2]
    generate_traj, _, _, _ = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    # motion_skill_model returns horizon+1 points, skip the first point (initial position)
    generate_traj = generate_traj[1:, :]
    
    # Ensure trajectories have the same length
    min_len = min(len(generate_traj), len(reference_traj))
    generate_traj = generate_traj[:min_len]
    reference_traj = reference_traj[:min_len]

    # Handle both 2D trajectory (x, y) from rule expert and 4D trajectory (x, y, speed, yaw) from RL expert
    if reference_traj.shape[1] == 2:
        # Rule expert data: only compare x, y positions
        cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:, :2] - reference_traj[:, :2]), axis=1)))
    else:
        # RL expert data: compare x, y, speed, yaw
        cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:, :2] - reference_traj[:, :2]), axis=1)))
        cost += np.sum(np.sqrt(np.square(generate_traj[:, 2] - reference_traj[:, 2])))
        cost += np.sum(np.sqrt(np.square(generate_traj[:, 3] - reference_traj[:, 3])))
    
    return cost 

def recover_parameter_fast(all_reference_trajs, all_current_vs, horizon, lat_bound=10, sigma=2.0):
    """
    Fast recovery method: Local Optimization + Gaussian Smoothing.
    Extremely fast but may not respect dynamic constraints as strictly as trajectory-level optimization.
    
    Args:
        all_reference_trajs: List of reference trajectories
        all_current_vs: List of current velocities
        horizon: Optimization horizon
        lat_bound: Lateral bound for optimization
        sigma: Gaussian smoothing sigma parameter
    
    Returns:
        Array of recovered parameters (N_segments, 3)
    """
    num_segments = len(all_reference_trajs)
    u_list = []
    
    print(f"Running FAST recovery (Local Opt + Smoothing sigma={sigma})...")
    
    # 1. Local Optimization per segment
    bounds = [[-lat_bound + 0.1, lat_bound - 0.1], 
              [-30 + 0.1, 30 - 0.1], 
              [MIN_SPEED + 0.1, MAX_SPEED - 0.1]]
    
    for i in range(num_segments):
        current_v = all_current_vs[i]
        reference_traj = all_reference_trajs[i]
        
        # Initial guess
        u_local_init = np.array([INITIAL_LAT, INITIAL_YAW, INITIAL_V])
        
        # Clamp initial guess to bounds
        for j in range(3):
            lower, upper = bounds[j]
            u_local_init[j] = np.clip(u_local_init[j], lower + 1e-4, upper - 1e-4)

        # Fast local minimization (reduced tolerance and iterations for speed)
        res = minimize(
            cost_function, 
            u_local_init, 
            (current_v, 0, horizon, reference_traj),
            method='L-BFGS-B', 
            bounds=bounds, 
            tol=1e-2, 
            options={'maxiter': 10}
        )
        u_list.append(res.x)
        
    u_array = np.array(u_list)
    
    # 2. Gaussian Smoothing for continuity
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
        self.load_data_path = os.path.join('demonstration_rule_expert', self.scenario)
        self.save_data_path = os.path.join('demonstration_RL_expert', f'{self.scenario}_fast_annotated')
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
            one_file_full_path = os.path.join(self.load_data_path, one_file)
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
                    current_v = np.clip(np.sqrt(np.sum(np.square(one_traj[0, 2:]))), 
                                       MIN_SPEED, MAX_SPEED)
                else:
                    current_v = one_spd if one_spd > 0 else INITIAL_V
                    current_v = np.clip(current_v, MIN_SPEED, MAX_SPEED)
                
                all_reference_trajs.append(one_traj)
                all_current_vs.append(current_v)
            
            # Run fast recovery
            recovered_params = recover_parameter_fast(
                all_reference_trajs, 
                all_current_vs, 
                horizon=DEFAULT_HORIZON, 
                lat_bound=DEFAULT_LAT_BOUND, 
                sigma=DEFAULT_SIGMA
            )
            
            for i in range(num_samples):
                lat1, yaw1, v1 = recovered_params[i]
                recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = \
                    transform_planning_param_to_latentvar(
                        lat1, yaw1, v1, lat_range=DEFAULT_LAT_RANGE
                    )
                one_recovered_latent_var = np.array([
                    recovered_latent_var0, 
                    recovered_latent_var1, 
                    recovered_latent_var2
                ])
                annotate_one_file_data['fast_recovered_latent_var'].append(one_recovered_latent_var)

            # Preserve original file name
            output_file_path = os.path.join(self.save_data_path, one_file)
            with open(output_file_path, 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


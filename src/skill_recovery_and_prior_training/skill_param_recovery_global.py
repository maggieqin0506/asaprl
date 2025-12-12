"""
Global trajectory optimization approach for skill parameter recovery.
Optimizes all segments simultaneously to ensure smooth overall trajectories.
"""

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
from asaprl.policy.planning_model import PathParam, SpeedParam, dynamic_constraint, dist_constraint, motion_skill_model

# --- Trajectory-level metrics (from improved version) ---

def compute_trajectory_smoothness_penalty(traj):
    """
    Compute smoothness penalty for a trajectory to encourage smoother paths.
    Penalizes large changes in direction, curvature, and speed.
    """
    if len(traj) < 3:
        return 0.0
    
    # Penalize large changes in direction (yaw changes)
    yaw_changes = np.abs(np.diff(traj[:, 3]))
    yaw_smoothness = np.sum(yaw_changes ** 2)
    
    # Penalize large changes in curvature (second derivative of position)
    if len(traj) >= 3:
        pos_2nd_deriv = np.diff(traj[:, :2], n=2, axis=0)
        curvature_penalty = np.sum(np.linalg.norm(pos_2nd_deriv, axis=1) ** 2)
    else:
        curvature_penalty = 0.0
    
    # Penalize large changes in speed
    speed_changes = np.abs(np.diff(traj[:, 2]))
    speed_smoothness = np.sum(speed_changes ** 2)
    
    # Weighted combination
    smoothness_weight = 0.1  # Can be tuned
    return smoothness_weight * (yaw_smoothness + curvature_penalty + speed_smoothness)

def compute_trajectory_shape_distance(traj1, traj2):
    """
    Compute trajectory-level distance that considers overall shape rather than point-wise differences.
    """
    # Endpoint distance (weighted more heavily as it represents overall trajectory goal)
    endpoint_weight = 2.0
    endpoint_dist = np.linalg.norm(traj1[-1, :2] - traj2[-1, :2])
    
    # Average displacement across trajectory
    avg_displacement = np.mean(np.linalg.norm(traj1[:, :2] - traj2[:, :2], axis=1))
    
    # Path length difference (encourages similar trajectory scales)
    def compute_path_length(traj):
        if len(traj) < 2:
            return 0.0
        diffs = np.diff(traj[:, :2], axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    path_len1 = compute_path_length(traj1)
    path_len2 = compute_path_length(traj2)
    path_length_diff = abs(path_len1 - path_len2)
    
    # Speed profile difference (average speed difference)
    speed_diff = np.mean(np.abs(traj1[:, 2] - traj2[:, 2]))
    
    # Yaw difference (average yaw difference)
    yaw_diff = np.mean(np.abs(traj1[:, 3] - traj2[:, 3]))
    
    # Combined trajectory-level distance
    shape_distance = (endpoint_weight * endpoint_dist + 
                      avg_displacement + 
                      0.5 * path_length_diff + 
                      speed_diff + 
                      yaw_diff)
    
    return shape_distance

# --- Original Local Cost Function (for initialization) ---

def cost_function(u, *args):
    current_v = args[0]
    current_a = args[1]
    horizon = args[2]
    reference_traj = args[3]

    lat1 = u[0]
    yaw1 = u[1]
    v1 = u[2]
    generate_traj, _, _, _  = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    # Use trajectory-level distance for better optimization
    cost = compute_trajectory_shape_distance(generate_traj, reference_traj)
    cost += compute_trajectory_smoothness_penalty(generate_traj)
    
    return cost

# --- Improved Global Cost Function ---

def global_cost_function(u_flat, *args):
    """
    Calculates the total error for the entire trajectory sequence with:
    1. Trajectory-level distance metrics (not just point-wise)
    2. Smoothness penalties
    3. Parameter smoothness regularization across segments
    """
    initial_v = args[0]
    initial_a = args[1]
    horizon = args[2]
    all_reference_trajs = args[3]  # List of segments
    
    num_segments = len(all_reference_trajs)
    u = u_flat.reshape((num_segments, 3))  # Reshape back to (N, 3)
    
    total_cost = 0
    
    # Initialize state propagation
    current_v = initial_v
    current_a = initial_a 
    
    all_generated_trajs = []
    
    for i in range(num_segments):
        lat1, yaw1, v1 = u[i]
        reference_traj = all_reference_trajs[i]
        
        # Generate skill segment using propagated current_v/current_a
        generate_traj, rec_lat1, rec_yaw1, rec_v1 = motion_skill_model(
            lat1, yaw1, current_v, current_a, v1, horizon
        )
        all_generated_trajs.append(generate_traj)
        
        # Use trajectory-level distance instead of point-wise
        segment_cost = compute_trajectory_shape_distance(generate_traj, reference_traj)
        
        # Add smoothness penalty for this segment
        segment_cost += compute_trajectory_smoothness_penalty(generate_traj)
        
        total_cost += segment_cost
        
        # State propagation
        # motion_skill_model uses SpeedParam with acc1=0 (default), so final acc = 0
        current_v = rec_v1
        current_a = 0
    
    # Add parameter smoothness regularization across segments
    # This encourages smooth transitions in parameters between segments
    param_smoothness_weight = 0.05  # Can be tuned
    for i in range(num_segments - 1):
        # Penalize large changes in lateral, yaw, and speed parameters
        param_diff = np.abs(u[i+1] - u[i])
        total_cost += param_smoothness_weight * np.sum(param_diff ** 2)
    
    # Add trajectory continuity penalty (encourages smooth transitions)
    continuity_weight = 0.1  # Can be tuned
    for i in range(num_segments - 1):
        prev_traj_end = all_generated_trajs[i][-1, :]
        curr_traj_start = all_generated_trajs[i+1][0, :]
        
        # Position continuity
        pos_continuity = np.linalg.norm(prev_traj_end[:2] - curr_traj_start[:2])
        
        # Velocity continuity
        vel_continuity = abs(prev_traj_end[2] - curr_traj_start[2])
        
        # Yaw continuity (handle wrap-around)
        yaw_diff = abs(prev_traj_end[3] - curr_traj_start[3])
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        
        total_cost += continuity_weight * (pos_continuity + vel_continuity + yaw_diff)
    
    return total_cost

def recover_parameter(reference_traj, current_v, current_a, horizon, lat_bound=10):
    """
    Original local recovery (used for initialization).
    Uses trajectory-level optimization for better initialization.
    """
    bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
    
    # Heuristic search for best initialization
    best_res = None
    min_cost = float('inf')
    
    # Try a few seeds to avoid local minima
    search_space = [
        [0, 0, 5],      # Center
        [-2, -15, 2.5],
        [2, 15, 7.5],
        [-2, 15, 7.5],
        [2, -15, 2.5]
    ]
    
    for init_guess in search_space:
        u_init = np.array(init_guess)
        try:
            u_solution = minimize(
                cost_function, u_init, 
                (current_v, current_a, horizon, reference_traj),
                method='SLSQP',
                bounds=bounds,
                tol=1e-4
            )
            if u_solution.fun < min_cost:
                min_cost = u_solution.fun
                best_res = u_solution
        except:
            continue
    
    if best_res is None:
        # Fallback to default
        return 0.0, 0.0, current_v
    
    lat1 = best_res.x[0]
    yaw1 = best_res.x[1]
    v1 = best_res.x[2]
    
    # Apply constraints to get final realizable params
    recovered_lat1, recovered_yaw1, _, _, recovered_v1 = dynamic_constraint(
        lat1, yaw1, current_v, current_a, v1, horizon
    )
    return recovered_lat1, recovered_yaw1, recovered_v1

def recover_parameter_global(all_traj_segments, initial_v, initial_a, horizon, lat_bound=10):
    """
    Optimizes all segments simultaneously to ensure trajectory smoothness.
    This is the key improvement: optimizing overall trajectories instead of segments.
    """
    num_segments = len(all_traj_segments)
    
    if num_segments == 0:
        return []
    
    # 1. Initialization Step: Run local recovery to get a good initial guess
    print(f"Initializing global optimization with local greedy search ({num_segments} segments)...")
    u_init_list = []
    curr_v = initial_v
    curr_a = initial_a
    
    for i in range(num_segments):
        # Use the previous recovered v/a for better greedy estimation
        lat, yaw, v = recover_parameter(
            all_traj_segments[i], curr_v, curr_a, horizon, lat_bound
        )
        u_init_list.append([lat, yaw, v])
        # Propagate state
        curr_v = v
        curr_a = 0
    
    u_init_flat = np.array(u_init_list).flatten()
    
    # 2. Global Optimization Step
    print(f"Running global optimization on {num_segments} segments...")
    
    # Bounds for all segments
    single_bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
    all_bounds = single_bounds * num_segments
    
    # Minimize total trajectory error
    res = minimize(
        global_cost_function, 
        u_init_flat, 
        args=(initial_v, initial_a, horizon, all_traj_segments),
        method='SLSQP',
        bounds=all_bounds,
        tol=1e-5,
        options={'maxiter': 200}  # Increased iterations for better convergence
    )
    
    print(f"Global optimization finished. Final Cost: {res.fun:.4f}")
    
    # 3. Extract Results
    u_final = res.x.reshape((num_segments, 3))
    
    # Apply dynamic constraints one last time to ensure validity
    recovered_params = []
    curr_v = initial_v
    curr_a = initial_a
    
    for i in range(num_segments):
        lat1, yaw1, v1 = u_final[i]
        rec_lat, rec_yaw, _, _, rec_v = dynamic_constraint(
            lat1, yaw1, curr_v, curr_a, v1, horizon
        )
        recovered_params.append([rec_lat, rec_yaw, rec_v])
        
        curr_v = rec_v
        curr_a = 0
    
    return recovered_params

def transform_planning_param_to_latentvar(lat1, yaw1, v1, lat_range=5):
    action0 = lat1 / lat_range
    action1 = yaw1 / 30
    action2 = v1 / 5 - 1
    return action0, action1, action2

# --- Updated Data Annotation Class ---

class annotate_data():
    def __init__(self, scenario, skill_length=10, use_global=True):
        self.scenario = scenario
        self.skill_length = skill_length
        self.use_global = use_global
        self.load_data_path = './demonstration_RL_expert/{}/'.format(self.scenario)
        self.save_data_path = './demonstration_RL_expert/{}_annotated/'.format(self.scenario)
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        all_file_lst = os.listdir(self.load_data_path)
        for file_idx, one_file in enumerate(all_file_lst):
            one_file_full_path = self.load_data_path + one_file
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            
            annotate_one_file_data = copy.deepcopy(one_file_data)
            annotate_one_file_data['recovered_latent_var'] = []
            
            if self.use_global:
                # Global optimization approach
                all_segments = one_file_data['rela_state']
                initial_speed = one_file_data['current_spd'][0].item()
                initial_acc = 0
                
                print(f'Processing File {file_idx+1}/{len(all_file_lst)} with {len(all_segments)} segments (GLOBAL).')
                
                recovered_params_list = recover_parameter_global(
                    all_segments, 
                    initial_speed, 
                    initial_acc, 
                    horizon=10, 
                    lat_bound=5
                )
                
                # Transform and Store
                for params in recovered_params_list:
                    rec_lat, rec_yaw, rec_v = params
                    rec_latent0, rec_latent1, rec_latent2 = transform_planning_param_to_latentvar(
                        rec_lat, rec_yaw, rec_v, lat_range=5
                    )
                    one_recovered_latent_var = np.array([rec_latent0, rec_latent1, rec_latent2])
                    annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)
            else:
                # Original local approach (for comparison)
                for latent_var_idx, one_latent_var in enumerate(one_file_data['latent_var']):
                    print('file {} of {}, data {} of {}'.format(
                        file_idx+1, len(all_file_lst), latent_var_idx, len(one_file_data['latent_var'])
                    ))
                    one_traj = one_file_data['rela_state'][latent_var_idx]
                    one_spd = one_file_data['current_spd'][latent_var_idx].item()
                    recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter(
                        one_traj, one_spd, 0, 10, lat_bound=5
                    )
                    recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = transform_planning_param_to_latentvar(
                        recovered_lat1, recovered_yaw1, recovered_v1, lat_range=5
                    )
                    one_recovered_latent_var = np.array([recovered_latent_var0, recovered_latent_var1, recovered_latent_var2])
                    annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)

            with open(self.save_data_path + '{}_expert_data_{}.pickle'.format(self.scenario, file_idx+1), 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved annotated file: {self.scenario}_expert_data_{file_idx+1}.pickle")


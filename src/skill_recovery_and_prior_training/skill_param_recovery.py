"""
Trajectory-Level Skill Parameter Recovery

This module implements skill parameter recovery using TRAJECTORY-LEVEL optimization.
It processes segments one at a time (segment-by-segment) but uses trajectory shape
distance instead of point-wise differences for better matching.

Key Features:
- Trajectory-level cost function (considers overall trajectory shape)
- Smoothness penalties for smoother paths
- Segment-by-segment processing (not global all-segments optimization)
- Handles both RL expert and rule expert data formats

Used by: main_skill_recovery.py (called "Global Optimization" in comparisons)

Note: Despite being called "Global" in comparisons, this is NOT global optimization
of all segments simultaneously. The name refers to "trajectory-level" (global shape)
optimization, not "all-segments" optimization.
"""

import os
import pickle
import copy
import numpy as np
from scipy.optimize import minimize
from asaprl.policy.planning_model import dynamic_constraint, motion_skill_model

# Constants for smoothness penalty weights
YAW_WEIGHT = 0.1
CURVATURE_WEIGHT = 0.1
SPEED_WEIGHT = 0.5
JERK_WEIGHT = 0.3

# Constants for trajectory shape distance weights
ENDPOINT_WEIGHT = 2.0
SPEED_DIFF_WEIGHT = 2.0
PATH_LENGTH_WEIGHT = 0.5
SPEED_SMOOTHNESS_WEIGHT = 0.5

# Optimization constants
DEFAULT_HORIZON = 10
DEFAULT_LAT_BOUND = 5
DEFAULT_LAT_RANGE = 5
INITIAL_YAW_OPTIONS = [-15, 15]
INITIAL_V = 5.0
MIN_SPEED = 0.1
MAX_SPEED = 9.9

def compute_trajectory_smoothness_penalty(traj):
    """
    Compute smoothness penalty for a trajectory to encourage smoother paths.
    Penalizes large changes in direction, curvature, and speed.
    
    Args:
        traj: Trajectory array of shape (N, 2) or (N, 4) for (x, y) or (x, y, speed, yaw)
    
    Returns:
        Smoothness penalty value (float)
    """
    if len(traj) < 3:
        return 0.0
    
    # Penalize large changes in curvature (second derivative of position)
    pos_2nd_deriv = np.diff(traj[:, :2], n=2, axis=0)
    curvature_penalty = np.sum(np.linalg.norm(pos_2nd_deriv, axis=1) ** 2)
    
    # Handle both 2D (x, y) and 4D (x, y, speed, yaw) trajectories
    if traj.shape[1] >= 4:
        # Penalize large changes in direction (yaw changes)
        yaw_changes = np.abs(np.diff(traj[:, 3]))
        yaw_smoothness = np.sum(yaw_changes ** 2)
        
        # Penalize large changes in speed (increased weight for speed smoothness)
        speed_changes = np.abs(np.diff(traj[:, 2]))
        speed_smoothness = np.sum(speed_changes ** 2)
        
        # Also penalize jerk (rate of change of acceleration) for speed
        if len(traj) >= 4:
            speed_2nd_deriv = np.diff(traj[:, 2], n=2)
            speed_jerk = np.sum(speed_2nd_deriv ** 2)
        else:
            speed_jerk = 0.0
        
        # Weighted combination with higher weight on speed smoothness
        return (YAW_WEIGHT * yaw_smoothness + 
                CURVATURE_WEIGHT * curvature_penalty + 
                SPEED_WEIGHT * speed_smoothness +
                JERK_WEIGHT * speed_jerk)
    else:
        # For 2D trajectories, only use curvature penalty
        return CURVATURE_WEIGHT * curvature_penalty

def _compute_path_length(traj):
    """Helper function to compute path length of a trajectory."""
    if len(traj) < 2:
        return 0.0
    diffs = np.diff(traj[:, :2], axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def compute_trajectory_shape_distance(traj1, traj2):
    """
    Compute trajectory-level distance that considers overall shape rather than point-wise differences.
    Uses a combination of:
    1. Endpoint distance (important for overall trajectory matching)
    2. Average displacement (overall shape similarity)
    3. Path length difference (trajectory scale similarity)
    4. Speed and yaw differences (for 4D trajectories)
    
    Args:
        traj1: Generated trajectory array
        traj2: Reference trajectory array
    
    Returns:
        Shape distance value (float)
    """
    # Ensure trajectories have the same length (use copies to avoid modifying originals)
    min_len = min(len(traj1), len(traj2))
    if min_len == 0:
        return 1e6  # Large penalty for empty trajectories
    
    traj1 = traj1[:min_len].copy()
    traj2 = traj2[:min_len].copy()
    
    # Endpoint distance (weighted more heavily as it represents overall trajectory goal)
    endpoint_dist = np.linalg.norm(traj1[-1, :2] - traj2[-1, :2])
    
    # Average displacement across trajectory
    avg_displacement = np.mean(np.linalg.norm(traj1[:, :2] - traj2[:, :2], axis=1))
    
    # Path length difference (encourages similar trajectory scales)
    path_len1 = _compute_path_length(traj1)
    path_len2 = _compute_path_length(traj2)
    path_length_diff = abs(path_len1 - path_len2)
    
    # Handle both 2D (x, y) and 4D (x, y, speed, yaw) trajectories
    if traj1.shape[1] >= 4 and traj2.shape[1] >= 4:
        # Speed profile difference (average speed difference)
        speed_diff = np.mean(np.abs(traj1[:, 2] - traj2[:, 2]))
        
        # Speed profile smoothness difference (penalize if recovered speed is less smooth)
        if len(traj1) > 1 and len(traj2) > 1:
            speed_smoothness_ref = np.var(np.diff(traj2[:, 2]))
            speed_smoothness_gen = np.var(np.diff(traj1[:, 2]))
            speed_smoothness_penalty = abs(speed_smoothness_gen - speed_smoothness_ref)
        else:
            speed_smoothness_penalty = 0.0
        
        # Yaw difference (average yaw difference)
        yaw_diff = np.mean(np.abs(traj1[:, 3] - traj2[:, 3]))
        
        # Combined trajectory-level distance with better speed matching
        shape_distance = (ENDPOINT_WEIGHT * endpoint_dist + 
                          avg_displacement + 
                          PATH_LENGTH_WEIGHT * path_length_diff + 
                          SPEED_DIFF_WEIGHT * speed_diff +
                          SPEED_SMOOTHNESS_WEIGHT * speed_smoothness_penalty +
                          yaw_diff)
    else:
        # For 2D trajectories, only use position-based metrics
        shape_distance = (ENDPOINT_WEIGHT * endpoint_dist + 
                          avg_displacement + 
                          PATH_LENGTH_WEIGHT * path_length_diff)
    
    return shape_distance

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
    
    # Ensure trajectories have the same length (use copy for reference to avoid modifying original)
    ref_traj = reference_traj.copy()
    min_len = min(len(generate_traj), len(ref_traj))
    generate_traj = generate_traj[:min_len]
    ref_traj = ref_traj[:min_len]

    # Use trajectory-level distance instead of point-wise differences
    # This optimizes the overall trajectory shape rather than individual segments
    cost = compute_trajectory_shape_distance(generate_traj, ref_traj)
    
    # Add smoothness penalty to encourage smoother trajectories
    cost += compute_trajectory_smoothness_penalty(generate_traj)
    
    return cost 

def recover_parameter(reference_traj, current_v, current_a, horizon, lat_bound=10):
    """
    Recover skill parameters from a reference trajectory using trajectory-level optimization.
    
    Args:
        reference_traj: Reference trajectory array
        current_v: Current velocity
        current_a: Current acceleration
        horizon: Optimization horizon (number of timesteps)
        lat_bound: Lateral bound for optimization
    
    Returns:
        Tuple of (recovered_lat1, recovered_yaw1, recovered_v1)
    """
    bounds = [[-lat_bound + 0.1, lat_bound - 0.1], 
              [-30 + 0.1, 30 - 0.1], 
              [MIN_SPEED + 0.1, MAX_SPEED - 0.1]]  # lat, yaw1, v1
    
    # Extract and clip current velocity from trajectory if available
    if reference_traj.shape[1] >= 4:
        current_v = np.clip(np.sqrt(np.sum(np.square(reference_traj[0, 2:]))), 
                           MIN_SPEED, MAX_SPEED)
    
    recover_dict = {}
    
    # Try multiple initial yaw angles to find best solution
    for i_yaw1 in INITIAL_YAW_OPTIONS:
        u_init = np.array([0, i_yaw1, INITIAL_V])  # lat, yaw1, v1
        u_solution = minimize(
            cost_function, 
            u_init, 
            (current_v, current_a, horizon, reference_traj),
            method='SLSQP',
            bounds=bounds,
            tol=1e-5
        )
        
        lat1, yaw1, v1 = u_solution.x
        cost = u_solution.fun
        
        # Apply dynamic constraints
        recovered_lat1, recovered_yaw1, current_v1, current_a, recovered_v1 = \
            dynamic_constraint(lat1, yaw1, current_v, current_a, v1, horizon)
        
        recover_dict[len(recover_dict)] = {
            'error': cost, 
            'param': [recovered_lat1, recovered_yaw1, recovered_v1]
        }
    
    # Select best solution (lowest error)
    min_key = min(recover_dict, key=lambda x: recover_dict[x]['error'])
    recovered_lat1, recovered_yaw1, recovered_v1 = recover_dict[min_key]['param']
    
    print(f'recovered skill param: lat {recovered_lat1:.4f}, '
          f'yaw {recovered_yaw1:.4f}, speed {recovered_v1:.4f}')
    print(f'recovery trajectory error: {recover_dict[min_key]["error"]:.4f}')
    
    return recovered_lat1, recovered_yaw1, recovered_v1

def transform_planning_param_to_latentvar(lat1, yaw1, v1, lat_range=5):
    action0 = lat1 / lat_range
    action1 = yaw1 / 30
    action2 = v1 / 5 - 1
    return action0, action1, action2

def annotate(one_traj, one_latent_var, one_current_spd):
    """
    Annotate a single trajectory segment with recovered latent variables.
    
    Args:
        one_traj: Single trajectory segment
        one_latent_var: Original latent variable (unused, kept for compatibility)
        one_current_spd: Current speed
    
    Returns:
        Recovered latent variable array
    """
    current_v = one_current_spd
    current_a = 0
    horizon = DEFAULT_HORIZON
    
    recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter(
        one_traj, current_v, current_a, horizon, lat_bound=DEFAULT_LAT_BOUND
    )
    
    recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = \
        transform_planning_param_to_latentvar(
            recovered_lat1, recovered_yaw1, recovered_v1, lat_range=DEFAULT_LAT_RANGE
        )
    
    return np.array([recovered_latent_var0, recovered_latent_var1, recovered_latent_var2])

# annotate raw demonstration and save annotated demonstration
class annotate_data():
    def __init__(self, scenario, skill_length = 10, max_files=0):
        self.scenario = scenario
        self.skill_length = skill_length
        self.max_files = max_files
        # Use rule expert data for consistency with other methods
        self.load_data_path = os.path.join('demonstration_rule_expert', self.scenario)
        self.save_data_path = os.path.join('demonstration_RL_expert', f'{self.scenario}_annotated')
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        all_file_lst = os.listdir(self.load_data_path)
        
        if self.max_files > 0:
            all_file_lst = all_file_lst[:self.max_files]
            
        for file_idx, one_file in enumerate(all_file_lst):
            one_file_full_path = os.path.join(self.load_data_path, one_file)
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            annotate_one_file_data = copy.deepcopy(one_file_data)
            annotate_one_file_data['recovered_latent_var'] = []
            
            # Handle both RL expert data (with 'latent_var') and rule expert data (with 'action')
            if 'latent_var' in one_file_data:
                # RL expert data format
                latent_vars = one_file_data['latent_var']
                traj_list = one_file_data['rela_state']
                if 'current_spd' in one_file_data:
                    speeds = [spd.item() if hasattr(spd, 'item') else spd for spd in one_file_data['current_spd']]
                else:
                    speeds = [one_file_data.get('vehicle_start_speed', [5.0])[0]] * len(latent_vars)
            else:
                # Rule expert data format: use 'action' as latent_var and 'vehicle_start_speed' as current_spd
                traj_list = one_file_data['rela_state']
                if 'action' in one_file_data:
                    latent_vars = one_file_data['action']
                else:
                    latent_vars = [None] * len(traj_list)
                if 'vehicle_start_speed' in one_file_data:
                    speeds = one_file_data['vehicle_start_speed']
                else:
                    speeds = [5.0] * len(traj_list)
            
            for latent_var_idx, (one_traj, one_spd, one_latent_var) in enumerate(zip(traj_list, speeds, latent_vars)):
                print(f'file {file_idx+1} of {len(all_file_lst)}, '
                      f'data {latent_var_idx+1} of {len(traj_list)}')
                
                # Handle speed format
                if hasattr(one_spd, 'item'):
                    one_spd = one_spd.item()
                elif isinstance(one_spd, (list, np.ndarray)) and len(one_spd) > 0:
                    one_spd = one_spd[0] if hasattr(one_spd[0], 'item') else one_spd[0]
                
                one_recovered_latent_var = annotate(one_traj, one_latent_var, one_spd)
                annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)

            # Preserve original file name
            output_file_path = os.path.join(self.save_data_path, one_file)
            with open(output_file_path, 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

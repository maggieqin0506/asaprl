"""
Sliding window skill parameter recovery module with support for:
1. Rule expert data (2D trajectories: x, y)
2. Multiple cost function methods (point-wise, trajectory-level, etc.)
3. Sliding window optimization

This module contains sliding window methods without modifying the original author's work.
"""

import matplotlib.pyplot as plt
import math, pdb
import numpy as np
import time
from scipy.optimize import minimize
import random
import os
import pickle
import copy
from asaprl.policy.planning_model import PathParam, SpeedParam, dynamic_constraint, dist_constraint, motion_skill_model

# Re-export the original transform function (unchanged from original)
from skill_param_recovery import transform_planning_param_to_latentvar

# Simple wrapper functions for backward compatibility with sliding window method
def recover_parameter(reference_traj, current_v, current_a, horizon, lat_bound=10):
    """
    Wrapper for recover_parameter_sliding with method=1 (point-wise with 2D/4D support).
    This is the version used by the sliding window method.
    """
    return recover_parameter_sliding(reference_traj, current_v, current_a, horizon, lat_bound, method=1)

def compute_trajectory_smoothness_penalty(traj):
    """
    Compute smoothness penalty for a trajectory to encourage smoother paths.
    Penalizes large changes in direction, curvature, and speed.

    Used in Method 2: Trajectory-level optimization
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
    yaw_weight = 0.1
    curvature_weight = 0.1
    speed_weight = 0.5  # Increased from 0.1 to prioritize speed smoothness
    jerk_weight = 0.3    # Additional penalty for acceleration changes

    return (yaw_weight * yaw_smoothness +
            curvature_weight * curvature_penalty +
            speed_weight * speed_smoothness +
            jerk_weight * speed_jerk)

def compute_trajectory_shape_distance(traj1, traj2):
    """
    Compute trajectory-level distance that considers overall shape rather than point-wise differences.
    Uses a combination of:
    1. Endpoint distance (important for overall trajectory matching)
    2. Average displacement (overall shape similarity)
    3. Path length difference (trajectory scale similarity)

    Used in Method 2: Trajectory-level optimization
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

    # Speed profile difference (average speed difference) - increased weight
    speed_diff = np.mean(np.abs(traj1[:, 2] - traj2[:, 2]))
    speed_diff_weight = 2.0  # Increased from 1.0 to better match speed profiles

    # Speed profile smoothness difference (penalize if recovered speed is less smooth)
    if len(traj1) > 1 and len(traj2) > 1:
        speed_smoothness_ref = np.var(np.diff(traj2[:, 2]))  # Variance of speed changes in reference
        speed_smoothness_gen = np.var(np.diff(traj1[:, 2]))  # Variance of speed changes in generated
        speed_smoothness_penalty = abs(speed_smoothness_gen - speed_smoothness_ref)
    else:
        speed_smoothness_penalty = 0.0

    # Yaw difference (average yaw difference)
    yaw_diff = np.mean(np.abs(traj1[:, 3] - traj2[:, 3]))

    # Combined trajectory-level distance with better speed matching
    shape_distance = (endpoint_weight * endpoint_dist +
                      avg_displacement +
                      0.5 * path_length_diff +
                      speed_diff_weight * speed_diff +
                      0.5 * speed_smoothness_penalty +  # Penalize speed smoothness mismatch
                      yaw_diff)

    return shape_distance

def cost_function_sliding(u, *args, method=1):
    """
    Sliding window cost function with multiple method support.

    Args:
        u: parameter vector [lat1, yaw1, v1]
        args: (current_v, current_a, horizon, reference_traj)
        method: which cost calculation method to use
            1 - Point-wise with 2D/4D support (handles rule expert data)
            2 - Trajectory-level optimization (smoothness + shape)
            3 - Reserved for future methods

    Returns:
        cost: scalar cost value
    """
    current_v = args[0]
    current_a = args[1]
    horizon = args[2]
    reference_traj = args[3]

    lat1 = u[0]
    yaw1 = u[1]
    v1 = u[2]
    generate_traj, _, _, _  = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    if method == 1:
        # Method 1: Point-wise optimization with 2D/4D trajectory support
        # motion_skill_model returns horizon+1 points, skip the first point (initial position) to match reference_traj
        generate_traj = generate_traj[1:, :]

        # Handle both 2D trajectory (x, y) from rule expert and 4D trajectory (x, y, speed, yaw) from RL expert
        if reference_traj.shape[1] == 2:
            # Rule expert data: only compare x, y positions
            cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - reference_traj[:,:2]), axis=1)))
        else:
            # RL expert data: compare x, y, speed, yaw
            cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - reference_traj[:,:2]), axis=1)))
            cost += np.sum(np.sqrt(np.square(generate_traj[:,2] - reference_traj[:,2])))
            cost += np.sum(np.sqrt(np.square(generate_traj[:,3] - reference_traj[:,3])))

    elif method == 2:
        # Method 2: Trajectory-level optimization with smoothness
        # Use trajectory-level distance instead of point-wise differences
        # This optimizes the overall trajectory shape rather than individual segments
        cost = compute_trajectory_shape_distance(generate_traj, reference_traj)

        # Add smoothness penalty to encourage smoother trajectories
        cost += compute_trajectory_smoothness_penalty(generate_traj)

    elif method == 3:
        # Method 3: Reserved for future implementation
        # TODO: Implement third method cost calculation
        raise NotImplementedError("Method 3 not yet implemented")

    else:
        raise ValueError(f"Invalid method {method} selected for cost function.")

    return cost

def recover_parameter_sliding(reference_traj, current_v, current_a, horizon, lat_bound=10, method=1):
    """
    Sliding window parameter recovery with multiple cost function methods.

    Args:
        reference_traj: reference trajectory to match
        current_v: current velocity
        current_a: current acceleration
        horizon: optimization horizon
        lat_bound: lateral position bound
        method: which cost function method to use (1, 2, or 3)

    Returns:
        recovered_lat1, recovered_yaw1, recovered_v1: recovered parameters
    """
    bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]  # lat, yaw1, v1
    recover_dict = {}
    print("current_v: ", current_v)
    current_v = np.clip(np.sqrt(np.sum(np.square(reference_traj[0,2:]))), 0.1, 9.9)
    print("current_v: ", current_v)

    i_lat, i_yaw1, i_v1 = 0, 0, 5
    for i_yaw1 in [-15, 15]:
        u_init = np.array([i_lat, i_yaw1, i_v1]) # lat, yaw1, v1

        # Use sliding window cost function with specified method
        # Note: We use a lambda to pass the method parameter to the cost function
        u_solution = minimize(
            lambda u, *args: cost_function_sliding(u, *args, method=method),
            u_init,
            (current_v, current_a, horizon, reference_traj),
            method='SLSQP',
            bounds=bounds,
            tol=1e-5
        )

        lat1 = u_solution.x[0]
        yaw1 = u_solution.x[1]
        v1 = u_solution.x[2]
        cost = u_solution.fun
        recovered_lat1, recovered_yaw1, current_v1, current_a, recovered_v1 = dynamic_constraint(lat1, yaw1, current_v, current_a, v1, horizon)
        recover_dict[len(recover_dict)] = {'error': cost, 'param': [recovered_lat1, recovered_yaw1, recovered_v1]}

    min_key = min(recover_dict, key=lambda x: recover_dict[x]['error'])
    recovered_lat1, recovered_yaw1, recovered_v1 = recover_dict[min_key]['param']
    print('recovered skill param: lat {}, yaw {}, speed {}'.format(recovered_lat1, recovered_yaw1, recovered_v1))
    print('recovery trajectory error:', recover_dict[min_key]['error'])
    return recovered_lat1, recovered_yaw1, recovered_v1

# Annotate function for sliding window methods
def annotate_sliding(one_traj, one_latent_var, one_current_spd, method=1):
    """
    Sliding window annotate function with method support.

    Args:
        one_traj: trajectory segment
        one_latent_var: latent variable (can be None for rule expert)
        one_current_spd: current speed
        method: which recovery method to use

    Returns:
        one_recovered_latent_var: recovered latent variable
    """
    current_v = one_current_spd
    current_a = 0
    horizon = 10
    recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter_sliding(
        one_traj, current_v, current_a, horizon, lat_bound=5, method=method
    )
    recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = transform_planning_param_to_latentvar(
        recovered_lat1, recovered_yaw1, recovered_v1, lat_range=5
    )
    one_recovered_latent_var = np.array([recovered_latent_var0, recovered_latent_var1, recovered_latent_var2])
    return one_recovered_latent_var

# Sliding window annotate_data class with rule expert support
class annotate_data_sliding():
    """
    Sliding window data annotation class that supports:
    - Rule expert data (with 'action' and 'vehicle_start_speed')
    - RL expert data (with 'latent_var' and 'current_spd')
    - Multiple recovery methods
    - Max files limit
    """
    def __init__(self, scenario, skill_length=10, max_files=0, method=1, data_source='rule'):
        """
        Args:
            scenario: scenario name (highway, intersection, roundabout)
            skill_length: length of skill segments
            max_files: maximum number of files to process (0 = all)
            method: which recovery method to use (1, 2, or 3)
            data_source: 'rule' for rule expert, 'rl' for RL expert
        """
        self.scenario = scenario
        self.skill_length = skill_length
        self.max_files = max_files
        self.method = method
        self.data_source = data_source

        if data_source == 'rule':
            self.load_data_path = 'demonstration_rule_expert/{}/'.format(self.scenario)
        else:
            self.load_data_path = 'demonstration_RL_expert/{}/'.format(self.scenario)

        self.save_data_path = 'demonstration_RL_expert/{}_annotated_method{}/'.format(self.scenario, method)

        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        """Process all data files and annotate them with recovered parameters."""
        all_file_lst = [f for f in os.listdir(self.load_data_path) if f.endswith('.pickle')]

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
                speeds = one_file_data['current_spd']
            else:
                # Rule expert data format: use 'action' as latent_var and 'vehicle_start_speed' as current_spd
                latent_vars = one_file_data['action']
                speeds = one_file_data['vehicle_start_speed']

            for latent_var_idx, one_latent_var in enumerate(latent_vars):
                print('file {} of {}, data {} of {}'.format(
                    file_idx+1, len(all_file_lst), latent_var_idx, len(latent_vars)
                ))
                one_traj = one_file_data['rela_state'][latent_var_idx]
                one_spd = speeds[latent_var_idx].item() if hasattr(speeds[latent_var_idx], 'item') else speeds[latent_var_idx]
                one_recovered_latent_var = annotate_sliding(one_traj, one_latent_var, one_spd, method=self.method)
                annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)

            output_filename = '{}_expert_data_{}.pickle'.format(self.scenario, file_idx+1)
            with open(os.path.join(self.save_data_path, output_filename), 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

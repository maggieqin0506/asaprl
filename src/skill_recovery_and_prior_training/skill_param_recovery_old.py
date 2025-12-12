"""
Point-Wise Skill Parameter Recovery (Legacy/Backup)

This module implements the OLD point-wise optimization method for skill parameter recovery.
It uses point-by-point differences in the cost function, which was replaced by the
trajectory-level approach in skill_param_recovery.py.

Key Features:
- Point-wise cost function (sums differences at each point)
- Simple but less smooth results
- Legacy implementation for comparison purposes

Used by: compare_code_versions.py (for comparing old vs new implementations)

Note: This is NOT used in production. It's kept only for comparing the old
(point-wise) vs new (trajectory-level) optimization approaches.
"""

import os
import pickle
import copy
import numpy as np
from scipy.optimize import minimize
from asaprl.policy.planning_model import dynamic_constraint, motion_skill_model

# Constants
DEFAULT_HORIZON = 10
DEFAULT_LAT_BOUND = 5
DEFAULT_LAT_RANGE = 5
MIN_SPEED = 0.1
MAX_SPEED = 9.9
INITIAL_YAW_OPTIONS = [-15, 15]
INITIAL_V = 5.0

def cost_function(u, *args):
    """
    Point-wise cost function (legacy method).
    Uses sum of point-wise differences.
    
    Args:
        u: Parameter vector [lat1, yaw1, v1]
        *args: (current_v, current_a, horizon, reference_traj)
    
    Returns:
        Cost value (float)
    """
    current_v, current_a, horizon, reference_traj = args

    lat1, yaw1, v1 = u[0], u[1], u[2]
    generate_traj, _, _, _ = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:, :2] - reference_traj[:, :2]), axis=1)))
    cost += np.sum(np.sqrt(np.square(generate_traj[:, 2] - reference_traj[:, 2])))
    cost += np.sum(np.sqrt(np.square(generate_traj[:, 3] - reference_traj[:, 3])))
    return cost 

def recover_parameter(reference_traj, current_v, current_a, horizon, lat_bound=10):
    """
    Recover skill parameters using point-wise optimization (legacy method).
    
    Args:
        reference_traj: Reference trajectory array
        current_v: Current velocity
        current_a: Current acceleration
        horizon: Optimization horizon
        lat_bound: Lateral bound for optimization
    
    Returns:
        Tuple of (recovered_lat1, recovered_yaw1, recovered_v1)
    """
    bounds = [[-lat_bound + 0.1, lat_bound - 0.1], 
              [-30 + 0.1, 30 - 0.1], 
              [MIN_SPEED + 0.1, MAX_SPEED - 0.1]]
    
    # Extract and clip current velocity from trajectory
    current_v = np.clip(np.sqrt(np.sum(np.square(reference_traj[0, 2:]))), 
                       MIN_SPEED, MAX_SPEED)

    recover_dict = {}
    
    # Try multiple initial yaw angles
    for i_yaw1 in INITIAL_YAW_OPTIONS:
        u_init = np.array([0, i_yaw1, INITIAL_V])
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
        
        recovered_lat1, recovered_yaw1, current_v1, current_a, recovered_v1 = \
            dynamic_constraint(lat1, yaw1, current_v, current_a, v1, horizon)
        
        recover_dict[len(recover_dict)] = {
            'error': cost, 
            'param': [recovered_lat1, recovered_yaw1, recovered_v1]
        }
    
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

# annotate function
def annotate(one_traj, one_latent_var, one_current_spd):
    current_v = one_current_spd
    current_a = 0
    horizon = 10
    recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter(one_traj, current_v, current_a, horizon, lat_bound = 5)
    recovered_latent_var0, recovered_latent_var1, recovered_latent_var2 = transform_planning_param_to_latentvar(recovered_lat1, recovered_yaw1, recovered_v1, lat_range=5)
    one_recovered_latent_var = np.array([recovered_latent_var0, recovered_latent_var1, recovered_latent_var2])
    return one_recovered_latent_var

# annotate raw demonstration and save annotated demonstration
class annotate_data():
    def __init__(self, scenario, skill_length = 10):
        self.scenario = scenario
        self.skill_length = skill_length
        self.load_data_path = os.path.join('demonstration_RL_expert', self.scenario)
        self.save_data_path = os.path.join('demonstration_RL_expert', f'{self.scenario}_annotated')
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        all_file_lst = os.listdir(self.load_data_path)
        for file_idx, one_file in enumerate(all_file_lst):
            one_file_full_path = os.path.join(self.load_data_path, one_file)
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            annotate_one_file_data = copy.deepcopy(one_file_data)
            annotate_one_file_data['recovered_latent_var'] = []
            for latent_var_idx, one_latent_var in enumerate(one_file_data['latent_var']):
                print('file {} of {}, data {} of {}'.format(file_idx+1, len(all_file_lst), latent_var_idx, len(one_file_data['latent_var'])))
                one_traj = one_file_data['rela_state'][latent_var_idx]
                one_spd = one_file_data['current_spd'][latent_var_idx].item()
                one_recovered_latent_var = annotate(one_traj, one_latent_var, one_spd)
                annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)

            output_file = os.path.join(
                self.save_data_path, 
                f'{self.scenario}_expert_data_{file_idx+1}.pickle'
            )
            with open(output_file, 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

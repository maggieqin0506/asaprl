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

def cost_function(u, *args):
    current_v = args[0]
    current_a = args[1]
    horizon = args[2]
    reference_traj = args[3]

    lat1 = u[0]
    yaw1 = u[1]
    v1 = u[2]
    generate_traj, _, _, _  = motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

    cost = np.sum(np.sqrt(np.sum(np.square(generate_traj[:,:2] - reference_traj[:,:2]), axis=1)))
    cost += np.sum(np.sqrt(np.square(generate_traj[:,2] - reference_traj[:,2])))
    cost += np.sum(np.sqrt(np.square(generate_traj[:,3] - reference_traj[:,3])))
    return cost 

def recover_parameter(reference_traj, current_v, current_a, horizon, lat_bound = 10):
    bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]  # lat, yaw1, v1
    recover_dict = {}
    print("current_v: ", current_v)
    current_v = np.clip(np.sqrt(np.sum(np.square(reference_traj[0,2:]))), 0.1, 9.9)
    print("current_v: ", current_v)

    i_lat, i_yaw1, i_v1 = 0, 0, 5
    for i_yaw1 in [-15, 15]:
        u_init = np.array([i_lat, i_yaw1, i_v1]) # lat, yaw1, v1
        u_solution = minimize(cost_function, u_init, (current_v, current_a, horizon, reference_traj),
                                        method='SLSQP',
                                        bounds=bounds,
                                        tol = 1e-5)
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
            for latent_var_idx, one_latent_var in enumerate(one_file_data['latent_var']):
                print('file {} of {}, data {} of {}'.format(file_idx+1, len(all_file_lst), latent_var_idx, len(one_file_data['latent_var'])))
                one_traj = one_file_data['rela_state'][latent_var_idx]
                one_spd = one_file_data['current_spd'][latent_var_idx].item()
                one_recovered_latent_var = annotate(one_traj, one_latent_var, one_spd)
                annotate_one_file_data['recovered_latent_var'].append(one_recovered_latent_var)

            with open(self.save_data_path + '{}_expert_data_{}.pickle'.format(self.scenario, file_idx+1), 'wb') as handle:
                pickle.dump(annotate_one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
Helper script to compare old (point-wise) vs new (trajectory-level) optimization methods.
This script helps you backup, restore, and compare both methods.
"""

import os
import shutil
import subprocess
import sys
import argparse
import pickle

def backup_current_version():
    """Backup current skill_param_recovery.py"""
    src_file = 'skill_param_recovery.py'
    backup_file = 'skill_param_recovery_new_backup.py'
    
    if os.path.exists(src_file):
        shutil.copy(src_file, backup_file)
        print(f"✓ Backed up current version to {backup_file}")
        return True
    else:
        print(f"✗ Error: {src_file} not found!")
        return False

def create_old_version():
    """Create the old point-wise optimization version for comparison"""
    old_code = '''import matplotlib.pyplot as plt
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
'''
    
    with open('skill_param_recovery_old.py', 'w') as f:
        f.write(old_code)
    print("✓ Created old version backup: skill_param_recovery_old.py")

def restore_old_version():
    """Restore old version"""
    old_file = 'skill_param_recovery_old.py'
    src_file = 'skill_param_recovery.py'
    
    if os.path.exists(old_file):
        shutil.copy(old_file, src_file)
        print(f"✓ Restored old version to {src_file}")
        return True
    else:
        print(f"✗ Error: {old_file} not found! Run with --create-old first.")
        return False

def restore_new_version():
    """Restore new version"""
    backup_file = 'skill_param_recovery_new_backup.py'
    src_file = 'skill_param_recovery.py'
    
    if os.path.exists(backup_file):
        shutil.copy(backup_file, src_file)
        print(f"✓ Restored new version to {src_file}")
        return True
    else:
        print(f"✗ Error: {backup_file} not found!")
        return False

def run_evaluation(scenario, method_name):
    """Run evaluation script"""
    print(f"\n{'='*80}")
    print(f"Running evaluation for {method_name} method...")
    print(f"{'='*80}\n")
    
    result_file = f'evaluation_results_{method_name}_{scenario}.pkl'
    cmd = [
        sys.executable,
        'evaluate_recovery_improvement.py',
        '--scenario', scenario,
        '--output_file', result_file
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    success = result.returncode == 0
    
    # Check if result file was actually created
    if success and not os.path.exists(result_file):
        print(f"Warning: Evaluation completed but result file {result_file} was not created.")
        success = False
    
    return success, result_file

def compare_results(old_result_file, new_result_file):
    """Compare results from old and new methods"""
    with open(old_result_file, 'rb') as f:
        old_results = pickle.load(f)
    
    with open(new_result_file, 'rb') as f:
        new_results = pickle.load(f)
    
    print("\n" + "="*80)
    print("COMPARISON: OLD (Point-wise) vs NEW (Trajectory-level) METHOD")
    print("="*80)
    
    # Smoothness comparison
    if 'smoothness' in old_results and 'smoothness' in new_results:
        print("\n--- SMOOTHNESS METRICS (Lower is Better) ---")
        for key in old_results['smoothness'].keys():
            old_val = old_results['smoothness'][key]
            new_val = new_results['smoothness'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Matching comparison
    if 'matching' in old_results and 'matching' in new_results:
        print("\n--- TRAJECTORY MATCHING METRICS (Lower is Better) ---")
        for key in old_results['matching'].keys():
            old_val = old_results['matching'][key]
            new_val = new_results['matching'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Continuity comparison
    if 'continuity' in old_results and 'continuity' in new_results:
        print("\n--- SEGMENT CONTINUITY METRICS (Lower is Better) ---")
        for key in old_results['continuity'].keys():
            old_val = old_results['continuity'][key]
            new_val = new_results['continuity'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Parameter error comparison
    if 'param_error' in old_results and 'param_error' in new_results:
        print("\n--- PARAMETER RECOVERY ACCURACY (Lower is Better) ---")
        for key in old_results['param_error'].keys():
            old_val = old_results['param_error'][key]
            new_val = new_results['param_error'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Compare old vs new recovery methods')
    parser.add_argument('--scenario', type=str, default='highway',
                       choices=['highway', 'intersection', 'roundabout'],
                       help='Scenario to evaluate')
    parser.add_argument('--backup', action='store_true',
                       help='Backup current version')
    parser.add_argument('--create-old', action='store_true',
                       help='Create old version backup')
    parser.add_argument('--restore-old', action='store_true',
                       help='Restore old version')
    parser.add_argument('--restore-new', action='store_true',
                       help='Restore new version')
    parser.add_argument('--compare', action='store_true',
                       help='Run full comparison (evaluate both methods and compare)')
    
    args = parser.parse_args()
    
    if args.backup:
        backup_current_version()
    
    if args.create_old:
        create_old_version()
    
    if args.restore_old:
        restore_old_version()
    
    if args.restore_new:
        restore_new_version()
    
    if args.compare:
        # Backup current (new) version
        if not backup_current_version():
            return
        
        # Create old version
        create_old_version()
        
        # Evaluate old method
        restore_old_version()
        old_success, old_file = run_evaluation(args.scenario, 'old')
        
        if not old_success:
            print("\n" + "="*80)
            print("ERROR: Old method evaluation failed!")
            print("="*80)
            print("\nPossible reasons:")
            print("1. Demonstration data not found. Run data collection first:")
            print("   python src/data_collection/highway/rule_expert.py")
            print("2. Data path is incorrect. Check that ./demonstration_RL_expert/highway/ exists")
            print("="*80)
            restore_new_version()
            return
        
        # Evaluate new method
        restore_new_version()
        new_success, new_file = run_evaluation(args.scenario, 'new')
        
        if not new_success:
            print("\n" + "="*80)
            print("ERROR: New method evaluation failed!")
            print("="*80)
            print("\nPossible reasons:")
            print("1. Demonstration data not found. Run data collection first:")
            print("   python src/data_collection/highway/rule_expert.py")
            print("2. Data path is incorrect. Check that ./demonstration_RL_expert/highway/ exists")
            print("="*80)
            return
        
        # Compare results
        if not os.path.exists(old_file):
            print(f"\nError: Result file {old_file} not found!")
            return
        
        if not os.path.exists(new_file):
            print(f"\nError: Result file {new_file} not found!")
            return
        
        compare_results(old_file, new_file)
        
        print(f"\nResults saved to:")
        print(f"  - {old_file}")
        print(f"  - {new_file}")

if __name__ == '__main__':
    main()


"""
Comparison script to evaluate and compare Global Optimization vs Sliding Window methods.
This script loads pre-computed results from both methods and compares them using various metrics.
"""

import numpy as np
import pickle
import os
import sys
import argparse
from tqdm import tqdm

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add current directory to path for local imports
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from asaprl.policy.planning_model import motion_skill_model
from skill_param_recovery import transform_planning_param_to_latentvar

def compute_curvature(traj):
    """Compute curvature of a trajectory."""
    if len(traj) < 3:
        return np.array([])
    
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    ds = np.where(ds < 1e-6, 1e-6, ds)
    
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (ds[:-1]**3)
    return curvature

def compute_trajectory_smoothness_metrics(traj):
    """Compute comprehensive smoothness metrics for a trajectory."""
    metrics = {}
    
    # Curvature metrics
    curvature = compute_curvature(traj)
    if len(curvature) > 0:
        metrics['mean_curvature'] = np.mean(curvature)
        metrics['max_curvature'] = np.max(curvature)
        metrics['curvature_variance'] = np.var(curvature)
    else:
        metrics['mean_curvature'] = 0.0
        metrics['max_curvature'] = 0.0
        metrics['curvature_variance'] = 0.0
    
    # Speed change metrics
    if len(traj) > 1:
        speed_changes = np.abs(np.diff(traj[:, 2]))
        metrics['mean_speed_change'] = np.mean(speed_changes)
        metrics['max_speed_change'] = np.max(speed_changes)
    else:
        metrics['mean_speed_change'] = 0.0
        metrics['max_speed_change'] = 0.0
    
    # Yaw rate metrics
    if len(traj) > 1:
        yaw = traj[:, 3]
        yaw_rate = np.abs(np.diff(yaw))
        metrics['mean_yaw_rate'] = np.mean(yaw_rate)
        metrics['max_yaw_rate'] = np.max(yaw_rate)
    else:
        metrics['mean_yaw_rate'] = 0.0
        metrics['max_yaw_rate'] = 0.0
    
    return metrics

def compute_trajectory_matching_metrics(recovered_traj, reference_traj):
    """Compute metrics for how well recovered trajectory matches reference."""
    metrics = {}
    metrics['endpoint_error'] = np.linalg.norm(recovered_traj[-1, :2] - reference_traj[-1, :2])
    metrics['avg_displacement'] = np.mean(np.linalg.norm(recovered_traj[:, :2] - reference_traj[:, :2], axis=1))
    metrics['max_displacement'] = np.max(np.linalg.norm(recovered_traj[:, :2] - reference_traj[:, :2], axis=1))
    metrics['speed_rmse'] = np.sqrt(np.mean((recovered_traj[:, 2] - reference_traj[:, 2])**2))
    metrics['yaw_rmse'] = np.sqrt(np.mean((recovered_traj[:, 3] - reference_traj[:, 3])**2))
    return metrics

def compute_segment_continuity(prev_traj, curr_traj):
    """Compute continuity metrics between consecutive segments."""
    if prev_traj is None or len(prev_traj) == 0 or len(curr_traj) == 0:
        return {}
    
    metrics = {}
    prev_end = prev_traj[-1, :2]
    curr_start = curr_traj[0, :2]
    metrics['position_discontinuity'] = np.linalg.norm(prev_end - curr_start)
    
    prev_end_vel = prev_traj[-1, 2]
    curr_start_vel = curr_traj[0, 2]
    metrics['velocity_discontinuity'] = abs(prev_end_vel - curr_start_vel)
    
    prev_end_yaw = prev_traj[-1, 3]
    curr_start_yaw = curr_traj[0, 3]
    yaw_diff = abs(prev_end_yaw - curr_start_yaw)
    if yaw_diff > 180:
        yaw_diff = 360 - yaw_diff
    metrics['yaw_discontinuity'] = yaw_diff
    
    return metrics

def transform_latentvar_to_planning_param(latent_var, lat_range=5):
    """Transform latent variables back to planning parameters."""
    lat1 = latent_var[0] * lat_range
    yaw1 = latent_var[1] * 30
    v1 = (latent_var[2] + 1) * 5
    return lat1, yaw1, v1

def evaluate_method_results(scenario, data_path, method_key='recovered_latent_var', use_ground_truth=False):
    """
    Evaluate a method's results from pre-computed latent variables.
    
    Args:
        scenario: Scenario name
        data_path: Path to data directory
        method_key: Key in data dict containing recovered parameters ('recovered_latent_var' or 'sliding_recovered_latent_var')
        use_ground_truth: Whether to compute parameter error (requires 'latent_var' in data)
    """
    all_file_lst = [f for f in os.listdir(data_path) if f.endswith('.pickle')]
    
    all_smoothness_metrics = []
    all_matching_metrics = []
    all_continuity_metrics = []
    all_param_errors = []
    
    prev_recovered_traj = None
    
    for file_idx, one_file in enumerate(tqdm(all_file_lst, desc=f"Evaluating {method_key}")):
        one_file_full_path = os.path.join(data_path, one_file)
        with open(one_file_full_path, 'rb') as handle:
            one_file_data = pickle.load(handle)
        
        # Get trajectories and speeds
        traj_list = one_file_data['rela_state']
        
        if 'current_spd' in one_file_data:
            speed_list = [spd.item() if hasattr(spd, 'item') else spd for spd in one_file_data['current_spd']]
        elif 'vehicle_start_speed' in one_file_data:
            speed_list = one_file_data['vehicle_start_speed']
        else:
            speed_list = [5.0] * len(traj_list)
        
        # Get recovered parameters
        if method_key not in one_file_data:
            print(f"Warning: {method_key} not found in {one_file}, skipping")
            continue
        
        recovered_latent_vars = one_file_data[method_key]
        
        # Get ground truth if available
        if use_ground_truth and 'latent_var' in one_file_data:
            gt_latent_vars = one_file_data['latent_var']
        else:
            gt_latent_vars = [None] * len(traj_list)
        
        for idx, (one_traj, one_spd, recovered_latent_var, gt_latent_var) in enumerate(
            zip(traj_list, speed_list, recovered_latent_vars, gt_latent_vars)
        ):
            # Handle speed format
            if hasattr(one_spd, 'item'):
                one_spd = one_spd.item()
            elif isinstance(one_spd, (list, np.ndarray)) and len(one_spd) > 0:
                one_spd = one_spd[0] if hasattr(one_spd[0], 'item') else one_spd[0]
            
            # Convert trajectory to expected format [x, y, speed, yaw]
            if one_traj.shape[1] == 2:
                horizon = one_traj.shape[0] - 1
                full_traj = np.zeros((horizon + 1, 4))
                full_traj[:, :2] = one_traj
                
                if horizon > 0:
                    pos_diffs = np.diff(one_traj, axis=0)
                    distances = np.linalg.norm(pos_diffs, axis=1)
                    speeds = distances / 0.1
                    full_traj[:-1, 2] = speeds
                    full_traj[-1, 2] = speeds[-1] if len(speeds) > 0 else one_spd
                    
                    yaws = np.arctan2(pos_diffs[:, 1], pos_diffs[:, 0]) * 180 / np.pi
                    full_traj[:-1, 3] = yaws
                    full_traj[-1, 3] = yaws[-1] if len(yaws) > 0 else 0.0
                else:
                    full_traj[:, 2] = one_spd if one_spd > 0 else 5.0
                    full_traj[:, 3] = 0.0
                
                one_traj = full_traj
            elif one_traj.shape[1] == 4:
                horizon = one_traj.shape[0] - 1
            else:
                continue
            
            # Convert recovered latent var to planning parameters
            recovered_latent_var = np.array(recovered_latent_var)
            lat1, yaw1, v1 = transform_latentvar_to_planning_param(recovered_latent_var, lat_range=5)
            
            # Generate recovered trajectory
            current_v = one_spd if one_spd > 0 else one_traj[0, 2] if one_traj[0, 2] > 0 else 5.0
            current_a = 0
            
            try:
                recovered_traj, _, _, _ = motion_skill_model(
                    lat1, yaw1, current_v, current_a, v1, horizon
                )
            except Exception as e:
                print(f"Warning: Skipping trajectory {idx} in file {one_file} due to error: {e}")
                continue
            
            # Compute metrics
            smoothness = compute_trajectory_smoothness_metrics(recovered_traj)
            all_smoothness_metrics.append(smoothness)
            
            matching = compute_trajectory_matching_metrics(recovered_traj, one_traj)
            all_matching_metrics.append(matching)
            
            if prev_recovered_traj is not None:
                continuity = compute_segment_continuity(prev_recovered_traj, recovered_traj)
                all_continuity_metrics.append(continuity)
            prev_recovered_traj = recovered_traj
            
            # Compute parameter error if ground truth available
            if use_ground_truth and gt_latent_var is not None:
                gt_latent_var = np.array(gt_latent_var)
                param_error = {
                    'latent_var_error': np.linalg.norm(recovered_latent_var - gt_latent_var),
                    'lat_error': abs(recovered_latent_var[0] - gt_latent_var[0]) * 5,
                    'yaw_error': abs(recovered_latent_var[1] - gt_latent_var[1]) * 30,
                    'v_error': abs(recovered_latent_var[2] - gt_latent_var[2]) * 5
                }
                all_param_errors.append(param_error)
    
    # Aggregate metrics
    results = {}
    
    if all_smoothness_metrics:
        results['smoothness'] = {
            key: np.mean([m[key] for m in all_smoothness_metrics])
            for key in all_smoothness_metrics[0].keys()
        }
    
    if all_matching_metrics:
        results['matching'] = {
            key: np.mean([m[key] for m in all_matching_metrics])
            for key in all_matching_metrics[0].keys()
        }
    
    if all_continuity_metrics:
        results['continuity'] = {
            key: np.mean([m[key] for m in all_continuity_metrics])
            for key in all_continuity_metrics[0].keys()
        }
    
    if all_param_errors:
        results['param_error'] = {
            key: np.mean([e[key] for e in all_param_errors])
            for key in all_param_errors[0].keys()
        }
    
    return results

def print_comparison(global_results, sliding_results):
    """Print comparison between global and sliding window methods."""
    print("\n" + "="*80)
    print("COMPARISON: GLOBAL OPTIMIZATION vs SLIDING WINDOW METHOD")
    print("="*80)
    
    # Smoothness comparison
    if 'smoothness' in global_results and 'smoothness' in sliding_results:
        print("\n--- SMOOTHNESS METRICS (Lower is Better) ---")
        for key in global_results['smoothness'].keys():
            global_val = global_results['smoothness'][key]
            sliding_val = sliding_results['smoothness'][key]
            improvement = ((global_val - sliding_val) / global_val * 100) if global_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: GLOBAL={global_val:10.6f}  SLIDING={sliding_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Matching comparison
    if 'matching' in global_results and 'matching' in sliding_results:
        print("\n--- TRAJECTORY MATCHING METRICS (Lower is Better) ---")
        for key in global_results['matching'].keys():
            global_val = global_results['matching'][key]
            sliding_val = sliding_results['matching'][key]
            improvement = ((global_val - sliding_val) / global_val * 100) if global_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: GLOBAL={global_val:10.6f}  SLIDING={sliding_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Continuity comparison
    if 'continuity' in global_results and 'continuity' in sliding_results:
        print("\n--- SEGMENT CONTINUITY METRICS (Lower is Better) ---")
        for key in global_results['continuity'].keys():
            global_val = global_results['continuity'][key]
            sliding_val = sliding_results['continuity'][key]
            improvement = ((global_val - sliding_val) / global_val * 100) if global_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: GLOBAL={global_val:10.6f}  SLIDING={sliding_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Parameter error comparison
    if 'param_error' in global_results and 'param_error' in sliding_results:
        print("\n--- PARAMETER RECOVERY ACCURACY (Lower is Better) ---")
        for key in global_results['param_error'].keys():
            global_val = global_results['param_error'][key]
            sliding_val = sliding_results['param_error'][key]
            improvement = ((global_val - sliding_val) / global_val * 100) if global_val > 0 else 0
            symbol = "✓" if improvement > 0 else "✗"
            print(f"{symbol} {key:25s}: GLOBAL={global_val:10.6f}  SLIDING={sliding_val:10.6f}  Improvement={improvement:6.2f}%")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Compare Global Optimization vs Sliding Window methods')
    parser.add_argument('--scenario', type=str, default='highway',
                       choices=['highway', 'intersection', 'roundabout'],
                       help='Scenario to evaluate')
    parser.add_argument('--global_data_path', type=str, default=None,
                       help='Path to global optimization results (default: ./demonstration_RL_expert/{scenario}_annotated/)')
    parser.add_argument('--sliding_data_path', type=str, default=None,
                       help='Path to sliding window results (default: ./demonstration_RL_expert/{scenario}_sliding_annotated/)')
    parser.add_argument('--use_ground_truth', action='store_true',
                       help='Compute parameter error (requires ground truth in data)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Save comparison results to file')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Set default paths
    if args.global_data_path is None:
        args.global_data_path = os.path.join(project_root, f'demonstration_RL_expert/{args.scenario}_annotated/')
    elif not os.path.isabs(args.global_data_path):
        args.global_data_path = os.path.join(project_root, args.global_data_path)
    
    if args.sliding_data_path is None:
        args.sliding_data_path = os.path.join(project_root, f'demonstration_RL_expert/{args.scenario}_sliding_annotated/')
    elif not os.path.isabs(args.sliding_data_path):
        args.sliding_data_path = os.path.join(project_root, args.sliding_data_path)
    
    # Check if paths exist
    if not os.path.exists(args.global_data_path):
        print(f"\n{'='*80}")
        print(f"ERROR: Global optimization data path {args.global_data_path} does not exist!")
        print(f"{'='*80}")
        print("\nTo fix this, run the global optimization method first:")
        print(f"  python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario {args.scenario}")
        print("="*80)
        return
    
    if not os.path.exists(args.sliding_data_path):
        print(f"\n{'='*80}")
        print(f"ERROR: Sliding window data path {args.sliding_data_path} does not exist!")
        print(f"{'='*80}")
        print("\nTo fix this, run the sliding window method first:")
        print(f"  python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario {args.scenario}")
        print("="*80)
        return
    
    print(f"Comparing methods on scenario: {args.scenario}")
    print(f"Global optimization data: {args.global_data_path}")
    print(f"Sliding window data: {args.sliding_data_path}")
    
    # Evaluate both methods
    print("\nEvaluating Global Optimization method...")
    global_results = evaluate_method_results(
        args.scenario, args.global_data_path, 
        method_key='recovered_latent_var',
        use_ground_truth=args.use_ground_truth
    )
    
    print("\nEvaluating Sliding Window method...")
    sliding_results = evaluate_method_results(
        args.scenario, args.sliding_data_path,
        method_key='sliding_recovered_latent_var',
        use_ground_truth=args.use_ground_truth
    )
    
    # Print comparison
    print_comparison(global_results, sliding_results)
    
    # Save results
    if args.output_file:
        comparison_results = {
            'global': global_results,
            'sliding': sliding_results
        }
        with open(args.output_file, 'wb') as f:
            pickle.dump(comparison_results, f)
        print(f"\nComparison results saved to {args.output_file}")

if __name__ == '__main__':
    main()


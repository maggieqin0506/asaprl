"""
Evaluation script to measure improvement from trajectory-level optimization.
This script compares the old point-wise optimization with the new trajectory-level optimization.

Metrics computed:
1. Trajectory smoothness (curvature, jerk, yaw rate changes)
2. Parameter recovery accuracy (if ground truth available)
3. Trajectory matching quality (endpoint error, average displacement)
4. Cross-segment continuity (smoothness between consecutive segments)
"""

import numpy as np
import pickle
import os
import sys
import argparse

# Add project root to path for imports (must be before importing asaprl)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add current directory to path for local imports
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from asaprl.policy.planning_model import motion_skill_model
from skill_param_recovery import recover_parameter, transform_planning_param_to_latentvar

def compute_curvature(traj):
    """Compute curvature of a trajectory."""
    if len(traj) < 3:
        return np.array([])
    
    # Compute first and second derivatives
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    ds = np.where(ds < 1e-6, 1e-6, ds)
    
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    
    # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (ds[:-1]**3)
    
    return curvature

def compute_jerk(traj, dt=0.1):
    """Compute jerk (rate of change of acceleration) for a trajectory."""
    if len(traj) < 4:
        return np.array([])
    
    # Compute velocity
    vel = np.diff(traj[:, :2], axis=0) / dt
    
    # Compute acceleration
    acc = np.diff(vel, axis=0) / dt
    
    # Compute jerk
    jerk = np.diff(acc, axis=0) / dt
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    
    return jerk_magnitude

def compute_yaw_rate(traj, dt=0.1):
    """Compute yaw rate (rate of change of heading)."""
    if len(traj) < 2:
        return np.array([])
    
    yaw = traj[:, 3]
    yaw_rate = np.abs(np.diff(yaw)) / dt
    
    return yaw_rate

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
    
    # Jerk metrics
    jerk = compute_jerk(traj)
    if len(jerk) > 0:
        metrics['mean_jerk'] = np.mean(jerk)
        metrics['max_jerk'] = np.max(jerk)
        metrics['jerk_variance'] = np.var(jerk)
    else:
        metrics['mean_jerk'] = 0.0
        metrics['max_jerk'] = 0.0
        metrics['jerk_variance'] = 0.0
    
    # Yaw rate metrics
    yaw_rate = compute_yaw_rate(traj)
    if len(yaw_rate) > 0:
        metrics['mean_yaw_rate'] = np.mean(yaw_rate)
        metrics['max_yaw_rate'] = np.max(yaw_rate)
        metrics['yaw_rate_variance'] = np.var(yaw_rate)
    else:
        metrics['mean_yaw_rate'] = 0.0
        metrics['max_yaw_rate'] = 0.0
        metrics['yaw_rate_variance'] = 0.0
    
    # Speed change metrics
    if len(traj) > 1:
        speed_changes = np.abs(np.diff(traj[:, 2]))
        metrics['mean_speed_change'] = np.mean(speed_changes)
        metrics['max_speed_change'] = np.max(speed_changes)
    else:
        metrics['mean_speed_change'] = 0.0
        metrics['max_speed_change'] = 0.0
    
    return metrics

def compute_trajectory_matching_metrics(recovered_traj, reference_traj):
    """Compute metrics for how well recovered trajectory matches reference."""
    metrics = {}
    
    # Endpoint error
    metrics['endpoint_error'] = np.linalg.norm(recovered_traj[-1, :2] - reference_traj[-1, :2])
    
    # Average displacement
    metrics['avg_displacement'] = np.mean(np.linalg.norm(recovered_traj[:, :2] - reference_traj[:, :2], axis=1))
    
    # Maximum displacement
    metrics['max_displacement'] = np.max(np.linalg.norm(recovered_traj[:, :2] - reference_traj[:, :2], axis=1))
    
    # Speed profile error
    metrics['speed_rmse'] = np.sqrt(np.mean((recovered_traj[:, 2] - reference_traj[:, 2])**2))
    
    # Yaw error
    metrics['yaw_rmse'] = np.sqrt(np.mean((recovered_traj[:, 3] - reference_traj[:, 3])**2))
    
    # Path length difference
    def path_length(traj):
        if len(traj) < 2:
            return 0.0
        diffs = np.diff(traj[:, :2], axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    metrics['path_length_diff'] = abs(path_length(recovered_traj) - path_length(reference_traj))
    
    return metrics

def compute_segment_continuity(prev_traj, curr_traj):
    """Compute continuity metrics between consecutive segments."""
    if prev_traj is None or len(prev_traj) == 0 or len(curr_traj) == 0:
        return {}
    
    metrics = {}
    
    # Position continuity (how well segments connect)
    prev_end = prev_traj[-1, :2]
    curr_start = curr_traj[0, :2]
    metrics['position_discontinuity'] = np.linalg.norm(prev_end - curr_start)
    
    # Velocity continuity
    prev_end_vel = prev_traj[-1, 2]
    curr_start_vel = curr_traj[0, 2]
    metrics['velocity_discontinuity'] = abs(prev_end_vel - curr_start_vel)
    
    # Yaw continuity
    prev_end_yaw = prev_traj[-1, 3]
    curr_start_yaw = curr_traj[0, 3]
    yaw_diff = abs(prev_end_yaw - curr_start_yaw)
    # Handle wrap-around
    if yaw_diff > 180:
        yaw_diff = 360 - yaw_diff
    metrics['yaw_discontinuity'] = yaw_diff
    
    return metrics

def evaluate_recovery_method(scenario, data_path, use_ground_truth=False):
    """Evaluate parameter recovery on a dataset."""
    all_file_lst = os.listdir(data_path)
    
    all_smoothness_metrics = []
    all_matching_metrics = []
    all_continuity_metrics = []
    all_param_errors = []
    
    prev_recovered_traj = None
    
    for file_idx, one_file in enumerate(all_file_lst):
        one_file_full_path = os.path.join(data_path, one_file)
        with open(one_file_full_path, 'rb') as handle:
            one_file_data = pickle.load(handle)
        
        # Handle different data formats
        # RL expert data has: 'latent_var', 'rela_state', 'current_spd'
        # Rule expert data has: 'rela_state', 'vehicle_start_speed', 'action'
        if 'latent_var' in one_file_data:
            # RL expert format
            traj_list = one_file_data['rela_state']
            if 'current_spd' in one_file_data:
                speed_list = [spd.item() if hasattr(spd, 'item') else spd for spd in one_file_data['current_spd']]
            else:
                speed_list = [one_file_data.get('vehicle_start_speed', [5.0])[0]] * len(traj_list)
            latent_var_list = one_file_data['latent_var']
        else:
            # Rule expert format - rela_state is a list of trajectories
            traj_list = one_file_data['rela_state']
            if 'vehicle_start_speed' in one_file_data:
                speed_list = one_file_data['vehicle_start_speed']
            else:
                speed_list = [5.0] * len(traj_list)  # Default speed
            latent_var_list = [None] * len(traj_list)  # No ground truth for rule expert
        
        for latent_var_idx, (one_traj, one_spd, one_latent_var) in enumerate(zip(traj_list, speed_list, latent_var_list)):
            # Handle speed format
            if hasattr(one_spd, 'item'):
                one_spd = one_spd.item()
            elif isinstance(one_spd, (list, np.ndarray)) and len(one_spd) > 0:
                one_spd = one_spd[0] if hasattr(one_spd[0], 'item') else one_spd[0]
            
            # Convert trajectory to expected format [x, y, speed, yaw]
            # Rule expert data may only have [x, y], so we need to construct full format
            if one_traj.shape[1] == 2:
                # Only position data, need to add speed and yaw
                horizon = one_traj.shape[0] - 1
                full_traj = np.zeros((horizon + 1, 4))
                full_traj[:, :2] = one_traj
                
                # Compute speed from position differences
                if horizon > 0:
                    pos_diffs = np.diff(one_traj, axis=0)
                    distances = np.linalg.norm(pos_diffs, axis=1)
                    # Assume dt = 0.1 (typical timestep)
                    speeds = distances / 0.1
                    full_traj[:-1, 2] = speeds
                    full_traj[-1, 2] = speeds[-1] if len(speeds) > 0 else one_spd
                else:
                    full_traj[:, 2] = one_spd if one_spd > 0 else 5.0
                
                # Compute yaw from position differences
                if horizon > 0:
                    yaws = np.arctan2(pos_diffs[:, 1], pos_diffs[:, 0]) * 180 / np.pi
                    full_traj[:-1, 3] = yaws
                    full_traj[-1, 3] = yaws[-1] if len(yaws) > 0 else 0.0
                else:
                    full_traj[:, 3] = 0.0
                
                one_traj = full_traj
            elif one_traj.shape[1] == 4:
                # Already in correct format [x, y, speed, yaw]
                horizon = one_traj.shape[0] - 1
            else:
                print(f"Warning: Unexpected trajectory shape {one_traj.shape}, skipping")
                continue
            
            # Recover parameters
            current_v = one_spd if one_spd > 0 else one_traj[0, 2] if one_traj[0, 2] > 0 else 5.0
            current_a = 0
            
            try:
                recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter(
                    one_traj, current_v, current_a, horizon, lat_bound=5
                )
            except (ValueError, Exception) as e:
                # Skip trajectories that cause optimization errors
                print(f"Warning: Skipping trajectory {latent_var_idx} in file {one_file} due to error: {e}")
                continue
            
            # Generate recovered trajectory
            try:
                recovered_traj, _, _, _ = motion_skill_model(
                    recovered_lat1, recovered_yaw1, current_v, current_a, recovered_v1, horizon
                )
            except Exception as e:
                print(f"Warning: Skipping trajectory {latent_var_idx} in file {one_file} due to trajectory generation error: {e}")
                continue
            
            # Compute smoothness metrics
            smoothness = compute_trajectory_smoothness_metrics(recovered_traj)
            all_smoothness_metrics.append(smoothness)
            
            # Compute matching metrics
            matching = compute_trajectory_matching_metrics(recovered_traj, one_traj)
            all_matching_metrics.append(matching)
            
            # Compute continuity with previous segment
            if prev_recovered_traj is not None:
                continuity = compute_segment_continuity(prev_recovered_traj, recovered_traj)
                all_continuity_metrics.append(continuity)
            prev_recovered_traj = recovered_traj
            
            # Compute parameter error if ground truth available
            if use_ground_truth and one_latent_var is not None:
                try:
                    recovered_latent_var = transform_planning_param_to_latentvar(
                        recovered_lat1, recovered_yaw1, recovered_v1, lat_range=5
                    )
                    recovered_latent_var = np.array(recovered_latent_var)
                    gt_latent_var = np.array(one_latent_var)
                    param_error = {
                        'latent_var_error': np.linalg.norm(recovered_latent_var - gt_latent_var),
                        'lat_error': abs(recovered_latent_var[0] - gt_latent_var[0]) * 5,
                        'yaw_error': abs(recovered_latent_var[1] - gt_latent_var[1]) * 30,
                        'v_error': abs(recovered_latent_var[2] - gt_latent_var[2]) * 5
                    }
                    all_param_errors.append(param_error)
                except Exception as e:
                    print(f"Warning: Could not compute parameter error: {e}")
    
    # Aggregate metrics
    results = {}
    
    # Average smoothness metrics
    if all_smoothness_metrics:
        results['smoothness'] = {
            key: np.mean([m[key] for m in all_smoothness_metrics])
            for key in all_smoothness_metrics[0].keys()
        }
    
    # Average matching metrics
    if all_matching_metrics:
        results['matching'] = {
            key: np.mean([m[key] for m in all_matching_metrics])
            for key in all_matching_metrics[0].keys()
        }
    
    # Average continuity metrics
    if all_continuity_metrics:
        results['continuity'] = {
            key: np.mean([m[key] for m in all_continuity_metrics])
            for key in all_continuity_metrics[0].keys()
        }
    
    # Average parameter errors
    if all_param_errors:
        results['param_error'] = {
            key: np.mean([e[key] for e in all_param_errors])
            for key in all_param_errors[0].keys()
        }
    
    return results

def print_comparison(old_results, new_results):
    """Print comparison between old and new results."""
    print("\n" + "="*80)
    print("COMPARISON: OLD vs NEW METHOD")
    print("="*80)
    
    # Smoothness comparison
    if 'smoothness' in old_results and 'smoothness' in new_results:
        print("\n--- SMOOTHNESS METRICS (Lower is Better) ---")
        for key in old_results['smoothness'].keys():
            old_val = old_results['smoothness'][key]
            new_val = new_results['smoothness'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            print(f"{key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Matching comparison
    if 'matching' in old_results and 'matching' in new_results:
        print("\n--- TRAJECTORY MATCHING METRICS (Lower is Better) ---")
        for key in old_results['matching'].keys():
            old_val = old_results['matching'][key]
            new_val = new_results['matching'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            print(f"{key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Continuity comparison
    if 'continuity' in old_results and 'continuity' in new_results:
        print("\n--- SEGMENT CONTINUITY METRICS (Lower is Better) ---")
        for key in old_results['continuity'].keys():
            old_val = old_results['continuity'][key]
            new_val = new_results['continuity'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            print(f"{key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    # Parameter error comparison
    if 'param_error' in old_results and 'param_error' in new_results:
        print("\n--- PARAMETER RECOVERY ACCURACY (Lower is Better) ---")
        for key in old_results['param_error'].keys():
            old_val = old_results['param_error'][key]
            new_val = new_results['param_error'][key]
            improvement = ((old_val - new_val) / old_val * 100) if old_val > 0 else 0
            print(f"{key:25s}: OLD={old_val:10.6f}  NEW={new_val:10.6f}  Improvement={improvement:6.2f}%")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate skill parameter recovery improvement')
    parser.add_argument('--scenario', type=str, default='highway', 
                       choices=['highway', 'intersection', 'roundabout'],
                       help='Scenario to evaluate')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to demonstration data (default: ./demonstration_RL_expert/{scenario}/)')
    parser.add_argument('--compare_old', action='store_true',
                       help='Compare with old method (requires backup of old code)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Get project root (go up from skill_recovery_and_prior_training to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if args.data_path is None:
        args.data_path = os.path.join(project_root, f'demonstration_RL_expert/{args.scenario}/')
    elif not os.path.isabs(args.data_path):
        # If relative path, make it relative to project root
        args.data_path = os.path.join(project_root, args.data_path)
    
    if not os.path.exists(args.data_path):
        print(f"\n{'='*80}")
        print(f"ERROR: Data path {args.data_path} does not exist!")
        print(f"{'='*80}")
        print("\nTo fix this, you need to collect demonstration data first:")
        print("\nOption 1: Rule-based expert (simplest, no trained model needed):")
        print(f"  python src/data_collection/{args.scenario}/rule_expert.py")
        print("\nOption 2: Trained RL agent (requires pre-trained model):")
        print("  See launch_command/data_collection_commands/rl_agent")
        print(f"\nThis will create the directory: {args.data_path}")
        print("="*80)
        return
    
    print(f"Evaluating recovery method on scenario: {args.scenario}")
    print(f"Data path: {args.data_path}")
    
    # Evaluate new method
    print("\nEvaluating NEW method (trajectory-level optimization)...")
    new_results = evaluate_recovery_method(args.scenario, args.data_path, use_ground_truth=True)
    
    print("\n--- NEW METHOD RESULTS ---")
    for category, metrics in new_results.items():
        print(f"\n{category.upper()}:")
        for key, value in metrics.items():
            print(f"  {key:25s}: {value:10.6f}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'wb') as f:
            pickle.dump(new_results, f)
        print(f"\nResults saved to {args.output_file}")
    
    # Compare with old method if requested
    if args.compare_old:
        print("\n" + "="*80)
        print("NOTE: To compare with old method, you need to:")
        print("1. Backup current skill_param_recovery.py")
        print("2. Restore old version (point-wise optimization)")
        print("3. Run this script again with --compare_old")
        print("="*80)

if __name__ == '__main__':
    main()


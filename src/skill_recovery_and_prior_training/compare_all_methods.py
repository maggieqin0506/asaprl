"""
Comparison script to evaluate and compare all three recovery methods:
1. Global Optimization (segment-by-segment)
2. Sliding Window (overlapping windows with averaging)
3. Fast Method (local optimization + Gaussian smoothing)

This script loads pre-computed results from all three methods and compares them.
"""

import numpy as np
import pickle
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass  # seaborn is optional
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

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

def evaluate_method_results(scenario, data_path, method_key='recovered_latent_var', use_ground_truth=False, max_files=0):
    """
    Evaluate a method's results from pre-computed latent variables.
    
    Args:
        scenario: Scenario name
        data_path: Path to data directory
        method_key: Key in data dict containing recovered parameters
        use_ground_truth: Whether to compute parameter error (requires 'latent_var' in data)
        max_files: Maximum number of files to process (0 = all files)
    """
    all_file_lst = [f for f in os.listdir(data_path) if f.endswith('.pickle')]
    all_file_lst.sort()  # Sort for consistent ordering
    
    # Limit number of files if specified
    if max_files > 0:
        all_file_lst = all_file_lst[:max_files]
        print(f"Processing {len(all_file_lst)} files (limited from {len(os.listdir(data_path))} total)")
    
    all_smoothness_metrics = []
    all_matching_metrics = []
    all_continuity_metrics = []
    all_param_errors = []
    all_recovered_trajectories = []
    all_reference_trajectories = []
    
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
            
            # Store trajectories for visualization
            all_recovered_trajectories.append(recovered_traj.copy())
            all_reference_trajectories.append(one_traj.copy())
            
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
    
    # Store trajectories for visualization
    results['recovered_trajectories'] = all_recovered_trajectories
    results['reference_trajectories'] = all_reference_trajectories
    
    return results

def plot_trajectory_comparison(global_results, sliding_results, fast_results, output_dir, max_trajectories=10):
    """Plot spatial trajectory comparisons for all methods."""
    # Get trajectory lists
    ref_trajs = global_results.get('reference_trajectories', [])
    global_trajs = global_results.get('recovered_trajectories', [])
    sliding_trajs = sliding_results.get('recovered_trajectories', [])
    fast_trajs = fast_results.get('recovered_trajectories', [])
    
    # Ensure all methods have trajectories
    min_len = min(len(ref_trajs), len(global_trajs), len(sliding_trajs), len(fast_trajs))
    if min_len == 0:
        print("Warning: No trajectories to plot")
        return
    
    # Sample trajectories for visualization (to avoid overcrowding)
    n_traj = min(max_trajectories, min_len)
    indices = np.linspace(0, min_len - 1, n_traj, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for plot_idx, traj_idx in enumerate(indices[:4]):  # Plot first 4 trajectories
        ax = axes[plot_idx]
        
        # Get trajectories
        ref_traj = ref_trajs[traj_idx]
        global_traj = global_trajs[traj_idx]
        sliding_traj = sliding_trajs[traj_idx]
        fast_traj = fast_trajs[traj_idx]
        
        # Plot trajectories
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'k-', linewidth=3, label='Original Path', alpha=0.8, zorder=4)
        ax.plot(global_traj[:, 0], global_traj[:, 1], 'b-', linewidth=2, label='Global Optimization', alpha=0.7, zorder=3)
        ax.plot(sliding_traj[:, 0], sliding_traj[:, 1], 'r-', linewidth=2, label='Sliding Window', alpha=0.7, zorder=2)
        ax.plot(fast_traj[:, 0], fast_traj[:, 1], 'g-', linewidth=2, label='Fast Method', alpha=0.7, zorder=1)
        
        # Mark start and end points
        ax.plot(ref_traj[0, 0], ref_traj[0, 1], 'ko', markersize=10, zorder=5)
        ax.plot(ref_traj[-1, 0], ref_traj[-1, 1], 'k*', markersize=15, zorder=5)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Trajectory Comparison {traj_idx + 1}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_spatial_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory spatial comparison to {output_dir}/trajectory_spatial_comparison.png")

def plot_metric_comparison_bar(global_results, sliding_results, fast_results, output_dir):
    """Create bar charts comparing metrics across methods."""
    metrics_to_plot = {
        'smoothness': ['mean_curvature', 'max_curvature', 'mean_speed_change', 'mean_yaw_rate'],
        'matching': ['endpoint_error', 'avg_displacement', 'max_displacement', 'speed_rmse', 'yaw_rmse'],
        'continuity': ['position_discontinuity', 'velocity_discontinuity', 'yaw_discontinuity']
    }
    
    for category, metric_keys in metrics_to_plot.items():
        if category not in global_results or category not in sliding_results or category not in fast_results:
            continue
        
        n_metrics = len(metric_keys)
        if n_metrics == 0:
            continue
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric_key in enumerate(metric_keys):
            if metric_key not in global_results[category]:
                continue
            
            ax = axes[idx]
            methods = ['Original\n(Reference)', 'Global', 'Sliding', 'Fast']
            values = [
                0,  # Original path has no recovery error, so 0 for comparison
                global_results[category][metric_key],
                sliding_results[category][metric_key],
                fast_results[category][metric_key]
            ]
            
            colors = ['gray', 'blue', 'red', 'green']
            bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title(f'{metric_key.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
        
        plt.suptitle(f'{category.replace("_", " ").title()} Metrics Comparison', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{category}_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {category} metrics comparison to {output_dir}/{category}_metrics_comparison.png")

def plot_metric_comparison_radar(global_results, sliding_results, fast_results, output_dir):
    """Create radar charts for comprehensive method comparison."""
    import math
    
    # Prepare metrics for radar chart (normalize to 0-1 scale for better visualization)
    categories = {
        'Smoothness': ['mean_curvature', 'mean_speed_change', 'mean_yaw_rate'],
        'Matching': ['endpoint_error', 'avg_displacement', 'speed_rmse'],
        'Continuity': ['position_discontinuity', 'velocity_discontinuity', 'yaw_discontinuity']
    }
    
    fig, axes = plt.subplots(1, len(categories), figsize=(6*len(categories), 6), subplot_kw=dict(projection='polar'))
    if len(categories) == 1:
        axes = [axes]
    
    for cat_idx, (cat_name, metric_keys) in enumerate(categories.items()):
        ax = axes[cat_idx]
        
        # Filter available metrics
        available_metrics = []
        for mk in metric_keys:
            for cat in ['smoothness', 'matching', 'continuity']:
                if cat in global_results and mk in global_results[cat]:
                    available_metrics.append(mk)
                    break
        
        if len(available_metrics) == 0:
            continue
        
        # Get values
        angles = [n / float(len(available_metrics)) * 2 * math.pi for n in range(len(available_metrics))]
        angles += angles[:1]  # Complete the circle
        
        global_vals = []
        sliding_vals = []
        fast_vals = []
        
        for mk in available_metrics:
            for cat in ['smoothness', 'matching', 'continuity']:
                if cat in global_results and mk in global_results[cat]:
                    global_vals.append(global_results[cat][mk])
                    sliding_vals.append(sliding_results[cat][mk])
                    fast_vals.append(fast_results[cat][mk])
                    break
        
        # Normalize values (0-1 scale)
        all_vals = global_vals + sliding_vals + fast_vals
        max_val = max(all_vals) if all_vals else 1.0
        min_val = min(all_vals) if all_vals else 0.0
        range_val = max_val - min_val if max_val != min_val else 1.0
        
        global_vals_norm = [(v - min_val) / range_val for v in global_vals]
        sliding_vals_norm = [(v - min_val) / range_val for v in sliding_vals]
        fast_vals_norm = [(v - min_val) / range_val for v in fast_vals]
        
        global_vals_norm += global_vals_norm[:1]
        sliding_vals_norm += sliding_vals_norm[:1]
        fast_vals_norm += fast_vals_norm[:1]
        
        # Plot
        ax.plot(angles, global_vals_norm, 'o-', linewidth=2, label='Global', color='blue', alpha=0.7)
        ax.fill(angles, global_vals_norm, alpha=0.25, color='blue')
        ax.plot(angles, sliding_vals_norm, 'o-', linewidth=2, label='Sliding', color='red', alpha=0.7)
        ax.fill(angles, sliding_vals_norm, alpha=0.25, color='red')
        ax.plot(angles, fast_vals_norm, 'o-', linewidth=2, label='Fast', color='green', alpha=0.7)
        ax.fill(angles, fast_vals_norm, alpha=0.25, color='green')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([mk.replace('_', ' ').title() for mk in available_metrics], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(cat_name, fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved radar chart comparison to {output_dir}/radar_chart_comparison.png")

def plot_trajectory_metrics_over_time(global_results, sliding_results, fast_results, output_dir):
    """Plot how metrics vary across trajectory segments."""
    # Compute per-trajectory metrics
    def compute_per_traj_metrics(traj_list, ref_traj_list):
        metrics_list = []
        for traj, ref_traj in zip(traj_list, ref_traj_list):
            matching = compute_trajectory_matching_metrics(traj, ref_traj)
            smoothness = compute_trajectory_smoothness_metrics(traj)
            metrics_list.append({**matching, **smoothness})
        return metrics_list
    
    global_traj_list = global_results.get('recovered_trajectories', [])
    sliding_traj_list = sliding_results.get('recovered_trajectories', [])
    fast_traj_list = fast_results.get('recovered_trajectories', [])
    ref_traj_list = global_results.get('reference_trajectories', [])
    
    if len(global_traj_list) == 0:
        return
    
    global_metrics = compute_per_traj_metrics(global_traj_list, ref_traj_list)
    sliding_metrics = compute_per_traj_metrics(sliding_traj_list, ref_traj_list)
    fast_metrics = compute_per_traj_metrics(fast_traj_list, ref_traj_list)
    
    # Find minimum length to ensure all methods have the same number of trajectories
    min_len = min(len(global_metrics), len(sliding_metrics), len(fast_metrics))
    if min_len == 0:
        print("Warning: No metrics to plot over time")
        return
    
    # Warn if lengths differ
    if len(global_metrics) != len(sliding_metrics) or len(global_metrics) != len(fast_metrics):
        print(f"Note: Different methods have different numbers of trajectories. "
              f"Plotting first {min_len} trajectories for comparison.")
        print(f"  Global: {len(global_metrics)}, Sliding: {len(sliding_metrics)}, Fast: {len(fast_metrics)}")
    
    # Truncate to minimum length
    global_metrics = global_metrics[:min_len]
    sliding_metrics = sliding_metrics[:min_len]
    fast_metrics = fast_metrics[:min_len]
    
    # Plot key metrics over trajectory index
    key_metrics = ['endpoint_error', 'avg_displacement', 'mean_curvature', 'mean_speed_change']
    
    n_metrics = len([m for m in key_metrics if m in global_metrics[0]])
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    traj_indices = np.arange(min_len)
    
    for idx, metric_key in enumerate(key_metrics[:4]):
        if metric_key not in global_metrics[0]:
            continue
        
        ax = axes[idx]
        
        global_vals = [m[metric_key] for m in global_metrics]
        sliding_vals = [m[metric_key] for m in sliding_metrics]
        fast_vals = [m[metric_key] for m in fast_metrics]
        
        ax.plot(traj_indices, global_vals, 'b-o', linewidth=2, markersize=4, label='Global', alpha=0.7)
        ax.plot(traj_indices, sliding_vals, 'r-s', linewidth=2, markersize=4, label='Sliding', alpha=0.7)
        ax.plot(traj_indices, fast_vals, 'g-^', linewidth=2, markersize=4, label='Fast', alpha=0.7)
        
        ax.set_xlabel('Trajectory Index', fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=11)
        ax.set_title(f'{metric_key.replace("_", " ").title()} Over Trajectories', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_over_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics over trajectories to {output_dir}/metrics_over_trajectories.png")

def plot_summary_comparison(global_results, sliding_results, fast_results, output_dir):
    """Create a summary comparison plot with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall matching error comparison
    ax1 = axes[0, 0]
    if 'matching' in global_results:
        metrics = ['endpoint_error', 'avg_displacement', 'speed_rmse', 'yaw_rmse']
        methods = ['Global', 'Sliding', 'Fast']
        x = np.arange(len(metrics))
        width = 0.25
        
        global_vals = [global_results['matching'].get(m, 0) for m in metrics]
        sliding_vals = [sliding_results['matching'].get(m, 0) for m in metrics]
        fast_vals = [fast_results['matching'].get(m, 0) for m in metrics]
        
        ax1.bar(x - width, global_vals, width, label='Global', color='blue', alpha=0.7)
        ax1.bar(x, sliding_vals, width, label='Sliding', color='red', alpha=0.7)
        ax1.bar(x + width, fast_vals, width, label='Fast', color='green', alpha=0.7)
        
        ax1.set_xlabel('Metrics', fontsize=11)
        ax1.set_ylabel('Error Value', fontsize=11)
        ax1.set_title('Trajectory Matching Error Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Smoothness comparison
    ax2 = axes[0, 1]
    if 'smoothness' in global_results:
        metrics = ['mean_curvature', 'mean_speed_change', 'mean_yaw_rate']
        x = np.arange(len(metrics))
        width = 0.25
        
        global_vals = [global_results['smoothness'].get(m, 0) for m in metrics]
        sliding_vals = [sliding_results['smoothness'].get(m, 0) for m in metrics]
        fast_vals = [fast_results['smoothness'].get(m, 0) for m in metrics]
        
        ax2.bar(x - width, global_vals, width, label='Global', color='blue', alpha=0.7)
        ax2.bar(x, sliding_vals, width, label='Sliding', color='red', alpha=0.7)
        ax2.bar(x + width, fast_vals, width, label='Fast', color='green', alpha=0.7)
        
        ax2.set_xlabel('Metrics', fontsize=11)
        ax2.set_ylabel('Smoothness Value', fontsize=11)
        ax2.set_title('Trajectory Smoothness Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Continuity comparison
    ax3 = axes[1, 0]
    if 'continuity' in global_results:
        metrics = ['position_discontinuity', 'velocity_discontinuity', 'yaw_discontinuity']
        x = np.arange(len(metrics))
        width = 0.25
        
        global_vals = [global_results['continuity'].get(m, 0) for m in metrics]
        sliding_vals = [sliding_results['continuity'].get(m, 0) for m in metrics]
        fast_vals = [fast_results['continuity'].get(m, 0) for m in metrics]
        
        ax3.bar(x - width, global_vals, width, label='Global', color='blue', alpha=0.7)
        ax3.bar(x, sliding_vals, width, label='Sliding', color='red', alpha=0.7)
        ax3.bar(x + width, fast_vals, width, label='Fast', color='green', alpha=0.7)
        
        ax3.set_xlabel('Metrics', fontsize=11)
        ax3.set_ylabel('Discontinuity Value', fontsize=11)
        ax3.set_title('Segment Continuity Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Overall performance score (composite metric)
    ax4 = axes[1, 1]
    if 'matching' in global_results and 'smoothness' in global_results:
        # Create a composite score (lower is better for all metrics)
        def compute_score(results):
            score = 0
            if 'matching' in results:
                score += results['matching'].get('endpoint_error', 0) * 0.3
                score += results['matching'].get('avg_displacement', 0) * 0.2
            if 'smoothness' in results:
                score += results['smoothness'].get('mean_curvature', 0) * 0.2
                score += results['smoothness'].get('mean_speed_change', 0) * 0.15
            if 'continuity' in results:
                score += results['continuity'].get('position_discontinuity', 0) * 0.15
            return score
        
        scores = {
            'Global': compute_score(global_results),
            'Sliding': compute_score(sliding_results),
            'Fast': compute_score(fast_results)
        }
        
        colors = ['blue', 'red', 'green']
        bars = ax4.bar(scores.keys(), scores.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, (method, score) in zip(bars, scores.items()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_ylabel('Composite Score (Lower is Better)', fontsize=11)
        ax4.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary comparison to {output_dir}/summary_comparison.png")

def generate_all_graphs(global_results, sliding_results, fast_results, output_dir):
    """Generate all comparison graphs."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Generating comparison graphs...")
    print("="*80)
    
    # Generate all plots
    plot_trajectory_comparison(global_results, sliding_results, fast_results, output_dir)
    plot_metric_comparison_bar(global_results, sliding_results, fast_results, output_dir)
    plot_metric_comparison_radar(global_results, sliding_results, fast_results, output_dir)
    plot_trajectory_metrics_over_time(global_results, sliding_results, fast_results, output_dir)
    plot_summary_comparison(global_results, sliding_results, fast_results, output_dir)
    
    print("\n" + "="*80)
    print(f"All graphs saved to: {output_dir}")
    print("="*80)

def print_comparison(global_results, sliding_results, fast_results):
    """Print comparison between all three methods."""
    print("\n" + "="*100)
    print("COMPARISON: GLOBAL OPTIMIZATION vs SLIDING WINDOW vs FAST METHOD")
    print("="*100)
    
    # Helper function to find best method
    def find_best(values_dict):
        """Find the method with the lowest value (best)."""
        best_val = min(values_dict.values())
        best_method = [k for k, v in values_dict.items() if v == best_val][0]
        return best_method, best_val
    
    # Smoothness comparison
    if 'smoothness' in global_results and 'smoothness' in sliding_results and 'smoothness' in fast_results:
        print("\n--- SMOOTHNESS METRICS (Lower is Better) ---")
        for key in global_results['smoothness'].keys():
            global_val = global_results['smoothness'][key]
            sliding_val = sliding_results['smoothness'][key]
            fast_val = fast_results['smoothness'][key]
            
            values = {'GLOBAL': global_val, 'SLIDING': sliding_val, 'FAST': fast_val}
            best_method, best_val = find_best(values)
            
            print(f"\n{key}:")
            for method, val in values.items():
                marker = "★" if method == best_method else " "
                print(f"  {marker} {method:8s}: {val:10.6f}")
    
    # Matching comparison
    if 'matching' in global_results and 'matching' in sliding_results and 'matching' in fast_results:
        print("\n--- TRAJECTORY MATCHING METRICS (Lower is Better) ---")
        for key in global_results['matching'].keys():
            global_val = global_results['matching'][key]
            sliding_val = sliding_results['matching'][key]
            fast_val = fast_results['matching'][key]
            
            values = {'GLOBAL': global_val, 'SLIDING': sliding_val, 'FAST': fast_val}
            best_method, best_val = find_best(values)
            
            print(f"\n{key}:")
            for method, val in values.items():
                marker = "★" if method == best_method else " "
                print(f"  {marker} {method:8s}: {val:10.6f}")
    
    # Continuity comparison
    if 'continuity' in global_results and 'continuity' in sliding_results and 'continuity' in fast_results:
        print("\n--- SEGMENT CONTINUITY METRICS (Lower is Better) ---")
        for key in global_results['continuity'].keys():
            global_val = global_results['continuity'][key]
            sliding_val = sliding_results['continuity'][key]
            fast_val = fast_results['continuity'][key]
            
            values = {'GLOBAL': global_val, 'SLIDING': sliding_val, 'FAST': fast_val}
            best_method, best_val = find_best(values)
            
            print(f"\n{key}:")
            for method, val in values.items():
                marker = "★" if method == best_method else " "
                print(f"  {marker} {method:8s}: {val:10.6f}")
    
    # Parameter error comparison
    if 'param_error' in global_results and 'param_error' in sliding_results and 'param_error' in fast_results:
        print("\n--- PARAMETER RECOVERY ACCURACY (Lower is Better) ---")
        for key in global_results['param_error'].keys():
            global_val = global_results['param_error'][key]
            sliding_val = sliding_results['param_error'][key]
            fast_val = fast_results['param_error'][key]
            
            values = {'GLOBAL': global_val, 'SLIDING': sliding_val, 'FAST': fast_val}
            best_method, best_val = find_best(values)
            
            print(f"\n{key}:")
            for method, val in values.items():
                marker = "★" if method == best_method else " "
                print(f"  {marker} {method:8s}: {val:10.6f}")
    
    print("\n" + "="*100)
    print("★ = Best method for each metric")

def main():
    parser = argparse.ArgumentParser(description='Compare all three recovery methods')
    parser.add_argument('--scenario', type=str, default='highway',
                       choices=['highway', 'intersection', 'roundabout'],
                       help='Scenario to evaluate')
    parser.add_argument('--global_data_path', type=str, default=None,
                       help='Path to global optimization results')
    parser.add_argument('--sliding_data_path', type=str, default=None,
                       help='Path to sliding window results')
    parser.add_argument('--fast_data_path', type=str, default=None,
                       help='Path to fast method results')
    parser.add_argument('--use_ground_truth', action='store_true',
                       help='Compute parameter error (requires ground truth in data)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Save comparison results to file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save comparison graphs')
    parser.add_argument('--no_graphs', action='store_true',
                       help='Skip generating comparison graphs')
    parser.add_argument('--max_files', type=int, default=0,
                       help='Maximum number of files to process per method (0 = all files). Useful for testing.')
    
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
    
    if args.fast_data_path is None:
        args.fast_data_path = os.path.join(project_root, f'demonstration_RL_expert/{args.scenario}_fast_annotated/')
    elif not os.path.isabs(args.fast_data_path):
        args.fast_data_path = os.path.join(project_root, args.fast_data_path)
    
    # Check if paths exist
    missing_paths = []
    if not os.path.exists(args.global_data_path):
        missing_paths.append(('Global', args.global_data_path, 'main_skill_recovery.py'))
    if not os.path.exists(args.sliding_data_path):
        missing_paths.append(('Sliding Window', args.sliding_data_path, 'main_skill_recovery_sliding.py'))
    if not os.path.exists(args.fast_data_path):
        missing_paths.append(('Fast', args.fast_data_path, 'main_skill_recovery_fast.py'))
    
    if missing_paths:
        print(f"\n{'='*80}")
        print("ERROR: Some data paths do not exist!")
        print(f"{'='*80}")
        for method_name, path, script_name in missing_paths:
            print(f"\n{method_name} method data path missing: {path}")
            print(f"  Run: python src/skill_recovery_and_prior_training/{script_name} --scenario {args.scenario}")
        print("="*80)
        return
    
    print(f"Comparing all three methods on scenario: {args.scenario}")
    print(f"Global optimization data: {args.global_data_path}")
    print(f"Sliding window data: {args.sliding_data_path}")
    print(f"Fast method data: {args.fast_data_path}")
    
    # Evaluate all three methods
    print("\nEvaluating Global Optimization method...")
    global_results = evaluate_method_results(
        args.scenario, args.global_data_path, 
        method_key='recovered_latent_var',
        use_ground_truth=args.use_ground_truth,
        max_files=args.max_files
    )
    
    print("\nEvaluating Sliding Window method...")
    sliding_results = evaluate_method_results(
        args.scenario, args.sliding_data_path,
        method_key='sliding_recovered_latent_var',
        use_ground_truth=args.use_ground_truth,
        max_files=args.max_files
    )
    
    print("\nEvaluating Fast method...")
    fast_results = evaluate_method_results(
        args.scenario, args.fast_data_path,
        method_key='fast_recovered_latent_var',
        use_ground_truth=args.use_ground_truth,
        max_files=args.max_files
    )
    
    # Print comparison
    print_comparison(global_results, sliding_results, fast_results)
    
    # Generate graphs (by default)
    if not args.no_graphs:
        if args.output_dir is None:
            # Default output directory
            output_dir = os.path.join(project_root, f'trajectory_comparisons/{args.scenario}')
        else:
            if not os.path.isabs(args.output_dir):
                output_dir = os.path.join(project_root, args.output_dir)
            else:
                output_dir = args.output_dir
        
        generate_all_graphs(global_results, sliding_results, fast_results, output_dir)
    
    # Save results
    if args.output_file:
        comparison_results = {
            'global': global_results,
            'sliding': sliding_results,
            'fast': fast_results
        }
        with open(args.output_file, 'wb') as f:
            pickle.dump(comparison_results, f)
        print(f"\nComparison results saved to {args.output_file}")

if __name__ == '__main__':
    main()


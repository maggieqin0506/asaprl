"""
Sliding Window Skill Parameter Recovery

This module implements skill parameter recovery using SLIDING WINDOW optimization.
It uses overlapping windows and averages parameters from multiple windows for better continuity.

Key Features:
- Overlapping windows with configurable step size
- Parameter averaging across windows for smoothness
- Uses trajectory-level optimization from skill_param_recovery.py
- Better continuity than segment-by-segment methods

Used by: main_skill_recovery_sliding.py
"""

import pickle
import numpy as np
import os
from tqdm import tqdm
from skill_param_recovery import recover_parameter, transform_planning_param_to_latentvar

def sliding_window_recovery(data, window_len=10, step_size=2, lat_range=5, action_shape=3):
    """
    Applies sliding window optimization by reconstructing the full trajectory, 
    running optimization on overlapping segments, and averaging the results.
    
    Args:
        data: Dictionary containing 'rela_state' and speed information
        window_len: Length of the optimization horizon (T)
        step_size: Stride/Step size (in timesteps). Smaller = smoother.
        lat_range: Lateral range bound
        action_shape: Dimension of skill parameters (default: 3)
    
    Returns:
        Array of recovered parameters (N_segments, 3)
    """
    
    if not data.get('rela_state', []):
        return np.array([])
        
    # T is the length of one parameter's validity (optimization horizon)
    OPTIMIZATION_HORIZON = window_len 
    
    # 1. Reconstruct the full trajectory array (T points per original segment)
    # The optimization requires T+1 points for horizon T.
    # We use the full segment length (assumed to be T+1) for the optimization reference trajectory.
    full_trajectory_states = np.concatenate([s for s in data['rela_state']], axis=0)

    # 2. Setup sliding window parameters
    total_len = full_trajectory_states.shape[0]
    window_len = OPTIMIZATION_HORIZON  # The number of points in each segment (rule expert has T points)
    param_dim = action_shape
    
    # The parameter sequence will be total_len / window_len * window_len (approx)
    # We accumulate results per segment step (T=10)
    accumulated_params_per_step = np.zeros((total_len, param_dim))
    counts_per_step = np.zeros((total_len, 1))
    
    print(f"  -> Total points: {total_len}, Window size (T+1): {window_len}, Stride: {step_size}")

    # Sliding Window Loop
    # We stop when the window can no longer fit a full segment
    for start_idx in tqdm(range(0, total_len - OPTIMIZATION_HORIZON, step_size), desc="Sliding window"):
        end_idx = start_idx + OPTIMIZATION_HORIZON

        # 1. Extract segment trajectory (T points, not T+1, since rule expert data has T points per segment)
        segment_traj = full_trajectory_states[start_idx:end_idx]
        
        # 2. Extract segment start speed
        # Handle both RL expert data (with 'current_spd') and rule expert data (with 'vehicle_start_speed')
        current_v = None
        if 'current_spd' in data and len(data['current_spd']) > 0:
            # RL expert data: We approximate the speed using the speed from the closest original segment start.
            original_segment_idx = start_idx // OPTIMIZATION_HORIZON

            # Ensure index is within bounds (edge case for very large stride or end of data)
            if original_segment_idx < len(data['current_spd']):
                current_v = data['current_spd'][original_segment_idx].item()
        elif 'vehicle_start_speed' in data and len(data['vehicle_start_speed']) > 0:
            # Rule expert data: use vehicle_start_speed instead
            original_segment_idx = start_idx // OPTIMIZATION_HORIZON

            if original_segment_idx < len(data['vehicle_start_speed']):
                speed_val = data['vehicle_start_speed'][original_segment_idx]
                current_v = speed_val.item() if hasattr(speed_val, 'item') else speed_val

        # FALLBACK: If speed data is missing or failed, use a default value
        # Note: For rule expert data with 2D trajectories (x,y only), we can't extract speed from trajectory
        if current_v is None:
            if segment_traj.shape[1] >= 4:
                # If trajectory has speed data (4 columns: x, y, speed, yaw)
                current_v = np.sqrt(np.sum(np.square(segment_traj[0, 2:])))
                current_v = np.clip(current_v, 0.1, 9.9)
            else:
                # For 2D trajectories, use a reasonable default speed
                # This will be adjusted by recover_parameter's internal correction
                current_v = 0.1
            
        current_a = 0 # As assumed in the user's code
        
        # 3. Run core parameter recovery
        recovered_lat1, recovered_yaw1, recovered_v1 = recover_parameter(
            reference_traj=segment_traj, 
            current_v=current_v, 
            current_a=current_a, 
            horizon=OPTIMIZATION_HORIZON, 
            lat_bound=lat_range
        )
        
        # 4. Transform and accumulate (a single parameter is recovered for the full segment)
        recovered_latent_var = np.array(transform_planning_param_to_latentvar(
            recovered_lat1, recovered_yaw1, recovered_v1, lat_range=lat_range
        ))
        
        # The single recovered parameter is valid for the T steps of the current window.
        param_start_idx = start_idx
        param_end_idx = start_idx + OPTIMIZATION_HORIZON
        
        # Accumulate the single recovered parameter across the segment steps
        accumulated_params_per_step[param_start_idx:param_end_idx] += recovered_latent_var
        counts_per_step[param_start_idx:param_end_idx] += 1
    
    # 5. Compute Average (Smoothed Parameters per Timestep)
    counts_per_step[counts_per_step == 0] = 1 
    final_params_per_step = accumulated_params_per_step / counts_per_step
    
    # 6. Reshape output to match original segment structure (N_segments, 3)
    # The user's original file stores one parameter set per original segment. 
    # We average the per-timestep smoothed parameters over the 10 steps of each original segment.
    N_segments = len(data['rela_state'])
    final_segmented_params = []
    
    for i in range(N_segments):
        start = i * OPTIMIZATION_HORIZON
        end = (i + 1) * OPTIMIZATION_HORIZON
        
        # Average the smoothed parameters over the original segment's duration
        avg_param = np.mean(final_params_per_step[start:end], axis=0) 
        final_segmented_params.append(avg_param)
    
    return np.array(final_segmented_params)

# annotate raw demonstration and save annotated demonstration
class annotate_data():
    def __init__(self, scenario, skill_length=10, max_files=0, window_len=10, step_size=2, lat_range=5, action_shape=3):
        self.scenario = scenario
        self.skill_length = skill_length
        self.max_files = max_files
        self.window_len = window_len
        self.step_size = step_size
        self.lat_range = lat_range
        self.action_shape = action_shape
        # Use rule expert data for consistency with other methods
        self.load_data_path = 'demonstration_rule_expert/{}/'.format(self.scenario)
        self.save_data_path = 'demonstration_RL_expert/{}_sliding_annotated/'.format(self.scenario)
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        self.annotate_all_data()

    def annotate_all_data(self):
        all_file_lst = os.listdir(self.load_data_path)
        
        if self.max_files > 0:
            all_file_lst = all_file_lst[:self.max_files]
            
        for file_idx, one_file in enumerate(tqdm(all_file_lst, desc=f"Processing {self.scenario}")):
            one_file_full_path = os.path.join(self.load_data_path, one_file)
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            
            # Run sliding window recovery
            smooth_params = sliding_window_recovery(
                one_file_data,
                window_len=self.window_len,
                step_size=self.step_size,
                lat_range=self.lat_range,
                action_shape=self.action_shape
            )
            
            # Save results: Adding a new key to the file to save the smoothed result
            one_file_data['sliding_recovered_latent_var'] = smooth_params.tolist()
            
            # Preserve original file name
            output_file_path = os.path.join(self.save_data_path, one_file)
            with open(output_file_path, 'wb') as handle:
                pickle.dump(one_file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Success! Sliding window parameters saved to: {self.save_data_path}")

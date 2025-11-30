import argparse
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path for external packages/modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- CORRECTED IMPORT ---
# Import the core optimization and transformation functions from your existing file.
try:
    from skill_param_recovery import recover_parameter, transform_planning_param_to_latentvar
except ImportError:
    try:
        from .skill_param_recovery import recover_parameter, transform_planning_param_to_latentvar
    except ImportError:
        # Final fallback check
        try:
            from src.skill_recovery_and_prior_training.skill_param_recovery import recover_parameter, transform_planning_param_to_latentvar
        except ImportError:
            print("--- CRITICAL IMPORT ERROR ---")
            print("Cannot find 'recover_parameter' or 'transform_planning_param_to_latentvar' in 'skill_param_recovery.py'.")
            sys.exit(1)
# --- END CORRECTED IMPORT ---

def sliding_window_recovery(data, args):
    """
    Applies sliding window optimization by reconstructing the full trajectory, 
    running optimization on overlapping segments, and averaging the results.
    """
    
    if not data.get('rela_state', []):
        return np.array([])
        
    # T is the length of one parameter's validity (optimization horizon)
    OPTIMIZATION_HORIZON = args.window_len 
    
    # 1. Reconstruct the full trajectory array (T points per original segment)
    # The optimization requires T+1 points for horizon T.
    # We use the full segment length (assumed to be T+1) for the optimization reference trajectory.
    full_trajectory_states = np.concatenate([s for s in data['rela_state']], axis=0)

    # 2. Setup sliding window parameters
    total_len = full_trajectory_states.shape[0]
    window_len = OPTIMIZATION_HORIZON  # The number of points in each segment (rule expert has T points)
    step_size = args.step_size
    param_dim = args.action_shape
    
    # The parameter sequence will be total_len / window_len * window_len (approx)
    # We accumulate results per segment step (T=10)
    accumulated_params_per_step = np.zeros((total_len, param_dim))
    counts_per_step = np.zeros((total_len, 1))
    
    print(f"  -> Total points: {total_len}, Window size (T+1): {window_len}, Stride: {step_size}")

    # Sliding Window Loop
    # We stop when the window can no longer fit a full segment
    for start_idx in tqdm(range(0, total_len - OPTIMIZATION_HORIZON, step_size)):
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
            lat_bound=args.lat_range
        )
        
        # 4. Transform and accumulate (a single parameter is recovered for the full segment)
        recovered_latent_var = np.array(transform_planning_param_to_latentvar(
            recovered_lat1, recovered_yaw1, recovered_v1, lat_range=args.lat_range
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

def main():
    parser = argparse.ArgumentParser(description="Sliding Window Skill Recovery")
    
    # Standard arguments matching your existing setup
    parser.add_argument('--scenario', type=str, default='highway', help='Scenario name')
    parser.add_argument('--action_shape', type=int, default=3, help='Dimension of skill parameters (default: 3 latent vars)')
    parser.add_argument('--reward_version', type=int, default=27)
    parser.add_argument('--lat_range', type=float, default=5)
    
    # New Sliding Window arguments
    parser.add_argument('--window_len', type=int, default=10, help='Length of the optimization horizon (T). Must match original setup.')
    parser.add_argument('--step_size', type=int, default=2, help='Stride/Step size (in timesteps). Smaller = smoother.')
    parser.add_argument('--max_files', type=int, default=0, help='Maximum number of .pickle files to process. 0 means all files.')
    args = parser.parse_args()

    # Define paths
    input_dir = f"demonstration_rule_expert/{args.scenario}/" 
    output_dir = f"demonstration_RL_expert/{args.scenario}_sliding_annotated/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found. Check data collection step.")
        return

    data_files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    if not data_files:
        print(f"No .pickle files found in {input_dir}.")
        return

    if args.max_files > 0:
        data_files = data_files[:args.max_files]

    print(f"Found {len(data_files)} trajectories. Starting Sliding Window Recovery...")

    for file_name in tqdm(data_files):
        file_path = os.path.join(input_dir, file_name)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Run the sliding window logic
        smooth_params = sliding_window_recovery(data, args)
        
        # Save results: Adding a new key to the file to save the smoothed result
        data['sliding_recovered_latent_var'] = smooth_params.tolist()
        
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(f"Success! Smoothed parameters saved to: {output_dir}")

if __name__ == "__main__":
    main()
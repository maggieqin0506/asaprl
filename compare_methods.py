
import numpy as np
import time
import pickle
import os
from scipy.optimize import minimize
from asaprl.policy.planning_model import PathParam, SpeedParam, motion_skill_model

# --- RE-IMPLEMENT ORIGINAL LOCAL OPTIMIZATION ---
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

def recover_local(all_reference_trajs, all_current_vs, horizon, lat_bound=10):
    num_segments = len(all_reference_trajs)
    u_list = []
    
    start_time = time.time()
    for i in range(num_segments):
        current_v = all_current_vs[i]
        reference_traj = all_reference_trajs[i]
        
        bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
        u_init = np.array([0, 0, 5])
        
        # Original method used SLSQP or L-BFGS-B per segment
        res = minimize(cost_function, u_init, (current_v, 0, horizon, reference_traj),
                       method='L-BFGS-B', bounds=bounds, tol=1e-3)
        u_list.append(res.x)
        
    end_time = time.time()
    return np.array(u_list), end_time - start_time

def recover_local_warm_start(all_reference_trajs, all_current_vs, horizon, lat_bound=10):
    """
    Local optimization but initializes with the solution of the previous segment.
    This acts as a primitive form of smoothing/continuity.
    """
    num_segments = len(all_reference_trajs)
    u_list = []
    
    # Initial guess for first segment
    u_prev = np.array([0, 0, 5])
    
    start_time = time.time()
    for i in range(num_segments):
        current_v = all_current_vs[i]
        reference_traj = all_reference_trajs[i]
        
        bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]
        
        # Clamp u_prev to bounds
        u_init = np.copy(u_prev)
        for j in range(3):
             lower, upper = bounds[j]
             u_init[j] = np.clip(u_init[j], lower + 1e-4, upper - 1e-4)
        
        res = minimize(cost_function, u_init, (current_v, 0, horizon, reference_traj),
                       method='L-BFGS-B', bounds=bounds, tol=1e-3)
        u_list.append(res.x)
        u_prev = res.x # Warm start for next
        
    end_time = time.time()
    return np.array(u_list), end_time - start_time

# --- IMPORT NEW SLIDING WINDOW METHOD ---
# We will import the functions from the existing file to ensure we test exactly what's running
from src.skill_recovery_and_prior_training.skill_param_recovery import recover_parameter_global

# --- METRICS CALCULATION ---
def calculate_detailed_metrics(u_reshaped, all_reference_trajs, all_current_vs, horizon):
    num_segments = len(u_reshaped)
    
    # Lists to collect point-wise or segment-wise data
    all_curvatures = []
    all_jerks = []
    all_yaw_rates = []
    all_speed_changes = [] # Acceleration
    
    # Trajectory matching errors
    endpoint_errors = []
    avg_displacements = []
    max_displacements = []
    speed_rmses = []
    yaw_rmses = []
    path_length_diffs = []
    
    # Continuity errors
    pos_discontinuities = []
    vel_discontinuities = []
    yaw_discontinuities = []
    
    generated_trajs = []
    
    for i in range(num_segments):
        u = u_reshaped[i]
        ref_traj = all_reference_trajs[i]
        v0 = all_current_vs[i]
        
        # Generate trajectory
        traj, _, _, _ = motion_skill_model(u[0], u[1], v0, 0, u[2], horizon)
        generated_trajs.append(traj)
        
        # --- Trajectory Matching Metrics ---
        # Endpoint error
        endpoint_errors.append(np.linalg.norm(traj[-1, :2] - ref_traj[-1, :2]))
        
        # Displacement (Euclidean dist per point)
        dists = np.linalg.norm(traj[:, :2] - ref_traj[:, :2], axis=1)
        avg_displacements.append(np.mean(dists))
        max_displacements.append(np.max(dists))
        
        # Speed RMSE
        speed_rmses.append(np.sqrt(np.mean((traj[:, 2] - ref_traj[:, 2])**2)))
        
        # Yaw RMSE
        yaw_rmses.append(np.sqrt(np.mean((traj[:, 3] - ref_traj[:, 3])**2)))
        
        # Path length diff
        len_gen = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
        len_ref = np.sum(np.linalg.norm(np.diff(ref_traj[:, :2], axis=0), axis=1))
        path_length_diffs.append(abs(len_gen - len_ref))
        
        # --- Smoothness Metrics (Per Trajectory) ---
        # Curvature
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        # Need at least 2 points for diff, 3 for curvature
        if len(dx) > 1:
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            # Pad to match length if needed or just take valid
            # k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
            # Use dx[:-1] to match size of ddx
            k = np.abs(dx[:-1]*ddy - dy[:-1]*ddx) / (dx[:-1]**2 + dy[:-1]**2 + 1e-6)**1.5
            all_curvatures.extend(k)
            
        # Yaw Rate (diff yaw)
        yaw_rate = np.abs(np.diff(traj[:, 3]))
        all_yaw_rates.extend(yaw_rate)
        
        # Speed Change (Acceleration)
        speed_change = np.abs(np.diff(traj[:, 2]))
        all_speed_changes.extend(speed_change)
        
        # Jerk (diff accel)
        if len(speed_change) > 1:
            jerk = np.abs(np.diff(speed_change)) # This is actually diff of abs accel, maybe diff of accel is better?
            # Re-calculate signed accel for jerk
            accel = np.diff(traj[:, 2])
            jerk = np.abs(np.diff(accel))
            all_jerks.extend(jerk)

    # --- Continuity Metrics (Between Segments) ---
    for i in range(num_segments - 1):
        # End of current vs Start of next (Start of next is usually fixed by data, but let's compare generated end vs generated start? 
        # No, usually we compare generated end vs next segment's start (which is the data point).
        # Actually, if we are recovering parameters, the "start" of the next segment is determined by the data (current_v, current_pos).
        # So we compare generated_traj[i][-1] with generated_traj[i+1][0]? 
        # generated_traj[i+1][0] is (0,0) in relative coords? No, motion_skill_model returns relative trajectory.
        # We need to be careful. The segments are relative. 
        # Continuity usually implies: does the end of seg i match the start of seg i+1 in GLOBAL frame?
        # Since we don't have global frame reconstruction here easily, we can check if the *state* at end of i matches the *initial condition* of i+1.
        # Initial condition of i+1 is (0, 0, 0, v_i+1). 
        # End of i is (x_end, y_end, yaw_end, v_end).
        # This is hard to compare without stitching.
        
        # Let's stick to Parameter Continuity or simple Velocity/Yaw continuity at boundaries if possible.
        # The user's image has "position_discontinuity".
        # Let's assume we compare:
        # v_end_i vs v_start_i+1 (which is all_current_vs[i+1])
        v_disc = np.abs(generated_trajs[i][-1, 2] - all_current_vs[i+1])
        vel_discontinuities.append(v_disc)
        
        # Yaw discontinuity: yaw_end_i vs 0? No, relative yaw.
        # This is tricky with relative coordinates.
        # Let's use the metric from the previous code: yaw parameter continuity?
        # The image says "yaw_discontinuity". 
        # Let's use the difference in Yaw Parameter as a proxy for now, or skip if unsure.
        # Actually, let's use the parameter difference as "Yaw Continuity" in the previous step.
        # But here let's try to match the image.
        pass

    # Parameter continuity (proxy for yaw/pos discontinuity in latent space)
    yaw_param_diff = np.abs(np.diff(u_reshaped[:, 1]))
    
    metrics = {
        # Smoothness
        'mean_curvature': np.mean(all_curvatures) if all_curvatures else 0,
        'max_curvature': np.max(all_curvatures) if all_curvatures else 0,
        'curvature_variance': np.var(all_curvatures) if all_curvatures else 0,
        'mean_jerk': np.mean(all_jerks) if all_jerks else 0,
        'max_jerk': np.max(all_jerks) if all_jerks else 0,
        'jerk_variance': np.var(all_jerks) if all_jerks else 0,
        'mean_yaw_rate': np.mean(all_yaw_rates) if all_yaw_rates else 0,
        'max_yaw_rate': np.max(all_yaw_rates) if all_yaw_rates else 0,
        'yaw_rate_variance': np.var(all_yaw_rates) if all_yaw_rates else 0,
        'mean_speed_change': np.mean(all_speed_changes) if all_speed_changes else 0,
        'max_speed_change': np.max(all_speed_changes) if all_speed_changes else 0,
        
        # Matching
        'endpoint_error': np.mean(endpoint_errors),
        'avg_displacement': np.mean(avg_displacements),
        'max_displacement': np.mean(max_displacements), # Mean of max displacements per traj
        'speed_rmse': np.mean(speed_rmses),
        'yaw_rmse': np.mean(yaw_rmses),
        'path_length_diff': np.mean(path_length_diffs),
        
        # Continuity
        'position_discontinuity': 0, # Hard to calc in relative
        'velocity_discontinuity': np.mean(vel_discontinuities) if vel_discontinuities else 0,
        'yaw_discontinuity': np.mean(yaw_param_diff) if len(yaw_param_diff) > 0 else 0
    }
    return metrics

# --- MAIN COMPARISON ---
def run_comparison():
    # Load a small chunk of real data
    data_path = './demonstration_RL_expert/highway/highway_expert_data_1.pickle'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # Take first 100 segments
    num_test = 100
    print(f"Running comparison on first {num_test} segments...")
    
    all_refs = []
    all_vs = []
    for i in range(num_test):
        traj = data['rela_state'][i]
        v = np.clip(np.sqrt(np.sum(np.square(traj[0,2:]))), 0.1, 9.9)
        all_refs.append(traj)
        all_vs.append(v)
        
    # 1. Run OLD (Local with Warm Start)
    print("\nRunning OLD Method (Local + Warm Start)...")
    u_old, _ = recover_local_warm_start(all_refs, all_vs, horizon=10, lat_bound=5)
    m_old = calculate_detailed_metrics(u_old, all_refs, all_vs, 10)
    
    # 2. Run NEW (Fast Smoothing)
    from src.skill_recovery_and_prior_training.skill_param_recovery import recover_parameter_fast
    print("Running NEW Method (Local + Smoothing)...")
    u_new = recover_parameter_fast(all_refs, all_vs, horizon=10, lat_bound=5, sigma=2.0)
    m_new = calculate_detailed_metrics(u_new, all_refs, all_vs, 10)
    
    # 3. Print Comparison Table
    print("\n" + "="*90)
    print("COMPARISON: OLD (Point-wise) vs NEW (Trajectory-level) METHOD")
    print("="*90)
    
    def print_section(title, keys, lower_is_better=True):
        print(f"\n--- {title} (Lower is Better) ---")
        for key in keys:
            old_val = m_old[key]
            new_val = m_new[key]
            
            if old_val == 0:
                imp = 0.0
            else:
                # Improvement: how much lower is NEW than OLD?
                # (OLD - NEW) / OLD * 100
                imp = (old_val - new_val) / old_val * 100
            
            # Symbol: Check if improved (positive imp), Cross if worsened (negative imp)
            # For "Lower is Better", positive improvement means NEW < OLD.
            symbol = "✓" if imp > 0 else "×"
            
            print(f"{symbol} {key:<25} : OLD= {old_val:8.6f}  NEW= {new_val:8.6f}  Improvement= {imp:6.2f}%")

    smoothness_keys = [
        'mean_curvature', 'max_curvature', 'curvature_variance',
        'mean_jerk', 'max_jerk', 'jerk_variance',
        'mean_yaw_rate', 'max_yaw_rate', 'yaw_rate_variance',
        'mean_speed_change', 'max_speed_change'
    ]
    print_section("SMOOTHNESS METRICS", smoothness_keys)
    
    matching_keys = [
        'endpoint_error', 'avg_displacement', 'max_displacement',
        'speed_rmse', 'yaw_rmse', 'path_length_diff'
    ]
    print_section("TRAJECTORY MATCHING METRICS", matching_keys)
    
    continuity_keys = [
        'velocity_discontinuity', 'yaw_discontinuity'
    ]
    print_section("SEGMENT CONTINUITY METRICS", continuity_keys)
    print("="*90)

if __name__ == "__main__":
    run_comparison()

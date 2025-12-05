import os
import pickle
import numpy as np
import time
from scipy.optimize import minimize
from asaprl.policy.planning_model import motion_skill_model, dynamic_constraint
from skill_param_recovery_fast import recover_parameter_fast as recover_new

# --- OLD METHOD (Inlined for standalone execution) ---
def cost_function_old(u, *args):
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

def recover_old(reference_traj, current_v, current_a, horizon, lat_bound = 10):
    bounds = [[-lat_bound+0.1, lat_bound-0.1], [-30+0.1, 30-0.1], [0+0.1, 10-0.1]]  # lat, yaw1, v1
    recover_dict = {}
    # print("current_v: ", current_v)
    current_v = np.clip(np.sqrt(np.sum(np.square(reference_traj[0,2:]))), 0.1, 9.9)
    # print("current_v: ", current_v)

    i_lat, i_yaw1, i_v1 = 0, 0, 5
    for i_yaw1 in [-15, 15]:
        u_init = np.array([i_lat, i_yaw1, i_v1]) # lat, yaw1, v1
        u_solution = minimize(cost_function_old, u_init, (current_v, current_a, horizon, reference_traj),
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
    return recovered_lat1, recovered_yaw1, recovered_v1
# -----------------------------------------------------

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
    vel_discontinuities = []
    
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
        if len(dx) > 1:
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            # k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
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
            accel = np.diff(traj[:, 2])
            jerk = np.abs(np.diff(accel))
            all_jerks.extend(jerk)

    # --- Continuity Metrics (Between Segments) ---
    for i in range(num_segments - 1):
        v_disc = np.abs(generated_trajs[i][-1, 2] - all_current_vs[i+1])
        vel_discontinuities.append(v_disc)

    # Parameter continuity
    yaw_param_diff = np.abs(np.diff(u_reshaped[:, 1]))
    
    metrics = {
        # Smoothness
        'mean_curvature': np.mean(all_curvatures) if all_curvatures else 0,
        'curvature_variance': np.var(all_curvatures) if all_curvatures else 0,
        'mean_jerk': np.mean(all_jerks) if all_jerks else 0,
        'mean_yaw_rate': np.mean(all_yaw_rates) if all_yaw_rates else 0,
        
        # Matching
        'endpoint_error': np.mean(endpoint_errors),
        'speed_rmse': np.mean(speed_rmses),
        
        # Continuity
        'yaw_discontinuity': np.mean(yaw_param_diff) if len(yaw_param_diff) > 0 else 0
    }
    return metrics

def run_comparison():
    # Load a small chunk of real data
    data_path = './demonstration_RL_expert/highway/highway_expert_data_1.pickle'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # Take first 50 segments for speed
    num_test = 50
    print(f"Running comparison on first {num_test} segments...")
    
    all_refs = []
    all_vs = []
    for i in range(num_test):
        traj = data['rela_state'][i]
        v = np.clip(np.sqrt(np.sum(np.square(traj[0,2:]))), 0.1, 9.9)
        all_refs.append(traj)
        all_vs.append(v)
        
    # 1. Run OLD (Local)
    print("\nRunning OLD Method (Local Point-wise)...")
    start = time.time()
    u_old_list = []
    for i in range(num_test):
        lat, yaw, v = recover_old(all_refs[i], all_vs[i], 0, 10, lat_bound=5)
        u_old_list.append([lat, yaw, v])
    u_old = np.array(u_old_list)
    time_old = time.time() - start
    m_old = calculate_detailed_metrics(u_old, all_refs, all_vs, 10)
    
    # 2. Run NEW (Fast Smoothing)
    print("\nRunning NEW Method (Local + Smoothing)...")
    start = time.time()
    u_new = recover_new(all_refs, all_vs, horizon=10, lat_bound=5, sigma=2.0)
    time_new = time.time() - start
    m_new = calculate_detailed_metrics(u_new, all_refs, all_vs, 10)
    
    # 3. Print Comparison Table
    print("\n" + "="*90)
    print("COMPARISON: OLD (Local) vs NEW (Fast Smoothing)")
    print("="*90)
    print(f"Time (50 segments): OLD={time_old:.2f}s  NEW={time_new:.2f}s  Speedup={time_old/time_new:.1f}x")
    
    def print_section(title, keys):
        print(f"\n--- {title} (Lower is Better) ---")
        for key in keys:
            old_val = m_old[key]
            new_val = m_new[key]
            if old_val == 0: imp = 0.0
            else: imp = (old_val - new_val) / old_val * 100
            
            symbol = "✓" if imp > 0 else "✗"
            print(f"{symbol} {key:<25} : OLD= {old_val:8.6f}  NEW= {new_val:8.6f}  Improvement= {imp:6.2f}%")

    smoothness_keys = ['mean_yaw_rate', 'curvature_variance', 'mean_jerk', 'yaw_discontinuity']
    print_section("SMOOTHNESS & CONTINUITY", smoothness_keys)
    
    matching_keys = ['endpoint_error', 'speed_rmse']
    print_section("MATCHING ACCURACY (Trade-off)", matching_keys)
    print("="*90)

if __name__ == "__main__":
    run_comparison()

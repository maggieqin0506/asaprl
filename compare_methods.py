
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

# --- IMPORT NEW SLIDING WINDOW METHOD ---
# We will import the functions from the existing file to ensure we test exactly what's running
from src.skill_recovery_and_prior_training.skill_param_recovery import recover_parameter_global

# --- METRICS CALCULATION ---
def calculate_metrics(u_reshaped, all_reference_trajs, all_current_vs, horizon, smoothness_weight=10.0):
    num_segments = len(u_reshaped)
    
    # 1. Smoothness Cost
    smoothness = 0
    for i in range(1, num_segments):
        diff = u_reshaped[i] - u_reshaped[i-1]
        smoothness += np.sum(diff**2)
        
    # 2. Reconstruction Cost
    recon_error = 0
    for i in range(num_segments):
        u = u_reshaped[i]
        ref_traj = all_reference_trajs[i]
        v0 = all_current_vs[i]
        
        traj, _, _, _ = motion_skill_model(u[0], u[1], v0, 0, u[2], horizon)
        
        diff = traj - ref_traj
        recon_error += np.sum(diff**2)
        
    total_cost = recon_error + smoothness_weight * smoothness
    return smoothness, recon_error, total_cost

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
        
    # Take first 50 segments
    num_test = 50
    print(f"Running comparison on first {num_test} segments...")
    
    all_refs = []
    all_vs = []
    for i in range(num_test):
        traj = data['rela_state'][i]
        v = np.clip(np.sqrt(np.sum(np.square(traj[0,2:]))), 0.1, 9.9)
        all_refs.append(traj)
        all_vs.append(v)
        
    # 1. Run Original (Local)
    print("\nRunning Original Method (Local Optimization)...")
    u_local, time_local = recover_local(all_refs, all_vs, horizon=10, lat_bound=5)
    smooth_local, recon_local, total_local = calculate_metrics(u_local, all_refs, all_vs, 10)
    
    # 2. Run New (Sliding Window)
    print("Running New Method (Sliding Window Global Optimization)...")
    start_time = time.time()
    u_global = recover_parameter_global(all_refs, all_vs, horizon=10, lat_bound=5, smoothness_weight=10.0)
    time_global = time.time() - start_time
    smooth_global, recon_global, total_global = calculate_metrics(u_global, all_refs, all_vs, 10)
    
    # 3. Print Comparison
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'Original (Local)':<15} | {'New (Sliding Window)':<15}")
    print("-" * 60)
    print(f"{'Smoothness Cost':<25} | {smooth_local:<15.4f} | {smooth_global:<15.4f}")
    print(f"{'Reconstruction Error':<25} | {recon_local:<15.4f} | {recon_global:<15.4f}")
    print(f"{'Total Cost (w=10)':<25} | {total_local:<15.4f} | {total_global:<15.4f}")
    print(f"{'Execution Time (s)':<25} | {time_local:<15.4f} | {time_global:<15.4f}")
    print("="*60)
    
    print("\nAnalysis:")
    print(f"Smoothness Improvement: {(1 - smooth_global/smooth_local)*100:.2f}%")
    print(f"Total Cost Reduction: {(1 - total_global/total_local)*100:.2f}%")

if __name__ == "__main__":
    run_comparison()

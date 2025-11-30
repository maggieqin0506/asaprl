"""
Comparison script for Original vs Sliding Window skill recovery methods
"""
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(data_dir):
    """Load all pickle files from a directory"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pickle')])
    all_data = []
    valid_files = []

    for file in all_files:
        try:
            with open(os.path.join(data_dir, file), 'rb') as f:
                data = pickle.load(f)
                all_data.append(data)
                valid_files.append(file)
        except Exception as e:
            print(f"  Warning: Skipping corrupted file {file}: {str(e)}")
            continue

    return all_data, valid_files

def calculate_parameter_differences(original_params, recovered_params):
    """Calculate differences between original and recovered parameters"""
    if len(original_params) == 0 or len(recovered_params) == 0:
        return None

    # Convert to numpy arrays
    orig = np.array(original_params)
    recov = np.array(recovered_params)

    # Handle length mismatch by using the minimum length
    min_len = min(len(orig), len(recov))
    if min_len == 0:
        return None

    orig = orig[:min_len]
    recov = recov[:min_len]

    # Calculate absolute differences
    diffs = np.abs(orig - recov)

    return {
        'mean_diff': np.mean(diffs, axis=0),
        'std_diff': np.std(diffs, axis=0),
        'max_diff': np.max(diffs, axis=0),
        'mae': np.mean(diffs),  # Mean Absolute Error
        'rmse': np.sqrt(np.mean(diffs**2))  # Root Mean Square Error
    }

def calculate_smoothness(params):
    """Calculate smoothness of parameter trajectory (lower is smoother)"""
    if len(params) < 2:
        return None

    params = np.array(params)
    # Calculate differences between consecutive parameters
    diffs = np.diff(params, axis=0)

    # Smoothness metrics
    smoothness = {
        'mean_variation': np.mean(np.abs(diffs), axis=0),
        'std_variation': np.std(diffs, axis=0),
        'total_variation': np.sum(np.abs(diffs), axis=0)
    }

    return smoothness

def compare_methods(original_dir, sliding_dir, output_dir='comparison_results'):
    """Compare original and sliding window methods"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("SKILL RECOVERY METHOD COMPARISON")
    print("="*60)

    # Load data
    print("\nLoading data...")
    original_data, orig_files = load_data(original_dir)
    sliding_data, slide_files = load_data(sliding_dir)

    print(f"Original method: {len(original_data)} files")
    print(f"Sliding window method: {len(sliding_data)} files")

    # Compare common files
    common_files = set(orig_files) & set(slide_files)
    print(f"\nComparing {len(common_files)} common files...")

    results = {
        'original': {
            'params': [],
            'smoothness': []
        },
        'sliding': {
            'params': [],
            'smoothness': []
        },
        'differences': []
    }

    for file_name in sorted(common_files):
        # Load file data
        orig_idx = orig_files.index(file_name)
        slide_idx = slide_files.index(file_name)

        orig_data = original_data[orig_idx]
        slide_data = sliding_data[slide_idx]

        # Extract recovered parameters
        orig_params = orig_data.get('recovered_latent_var', [])
        slide_params = slide_data.get('sliding_recovered_latent_var',
                                      slide_data.get('recovered_latent_var', []))

        if len(orig_params) > 0 and len(slide_params) > 0:
            results['original']['params'].extend(orig_params)
            results['sliding']['params'].extend(slide_params)

            # Calculate smoothness for this file
            orig_smooth = calculate_smoothness(orig_params)
            slide_smooth = calculate_smoothness(slide_params)

            if orig_smooth:
                results['original']['smoothness'].append(orig_smooth)
            if slide_smooth:
                results['sliding']['smoothness'].append(slide_smooth)

            # Calculate differences
            diff = calculate_parameter_differences(orig_params, slide_params)
            if diff:
                results['differences'].append(diff)

    # Aggregate results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # 1. Parameter Recovery Accuracy (comparing the two methods)
    print("\n1. PARAMETER RECOVERY DIFFERENCES")
    print("-" * 40)
    if results['differences']:
        avg_mae = np.mean([d['mae'] for d in results['differences']])
        avg_rmse = np.mean([d['rmse'] for d in results['differences']])
        print(f"Mean Absolute Error (MAE): {avg_mae:.6f}")
        print(f"Root Mean Square Error (RMSE): {avg_rmse:.6f}")

        # Per-dimension differences
        mean_diffs = np.mean([d['mean_diff'] for d in results['differences']], axis=0)
        print(f"\nMean differences per dimension:")
        print(f"  Lateral:  {mean_diffs[0]:.6f}")
        print(f"  Yaw:      {mean_diffs[1]:.6f}")
        print(f"  Velocity: {mean_diffs[2]:.6f}")

    # 2. Smoothness Comparison
    print("\n2. PARAMETER SMOOTHNESS (lower is smoother)")
    print("-" * 40)

    if results['original']['smoothness'] and results['sliding']['smoothness']:
        # Original method smoothness
        orig_mean_var = np.mean([s['mean_variation'] for s in results['original']['smoothness']], axis=0)
        slide_mean_var = np.mean([s['mean_variation'] for s in results['sliding']['smoothness']], axis=0)

        print("\nOriginal Method:")
        print(f"  Mean variation - Lateral:  {orig_mean_var[0]:.6f}")
        print(f"  Mean variation - Yaw:      {orig_mean_var[1]:.6f}")
        print(f"  Mean variation - Velocity: {orig_mean_var[2]:.6f}")
        print(f"  Overall smoothness: {np.mean(orig_mean_var):.6f}")

        print("\nSliding Window Method:")
        print(f"  Mean variation - Lateral:  {slide_mean_var[0]:.6f}")
        print(f"  Mean variation - Yaw:      {slide_mean_var[1]:.6f}")
        print(f"  Mean variation - Velocity: {slide_mean_var[2]:.6f}")
        print(f"  Overall smoothness: {np.mean(slide_mean_var):.6f}")

        improvement = ((orig_mean_var - slide_mean_var) / orig_mean_var) * 100
        print(f"\nImprovement (%):")
        print(f"  Lateral:  {improvement[0]:.2f}%")
        print(f"  Yaw:      {improvement[1]:.2f}%")
        print(f"  Velocity: {improvement[2]:.2f}%")
        print(f"  Overall:  {np.mean(improvement):.2f}%")

    # 3. Statistical Summary
    print("\n3. PARAMETER STATISTICS")
    print("-" * 40)

    if results['original']['params'] and results['sliding']['params']:
        orig_params_arr = np.array(results['original']['params'])
        slide_params_arr = np.array(results['sliding']['params'])

        print("\nOriginal Method:")
        print(f"  Mean: {np.mean(orig_params_arr, axis=0)}")
        print(f"  Std:  {np.std(orig_params_arr, axis=0)}")

        print("\nSliding Window Method:")
        print(f"  Mean: {np.mean(slide_params_arr, axis=0)}")
        print(f"  Std:  {np.std(slide_params_arr, axis=0)}")

    # 4. Create visualizations
    print("\n4. GENERATING VISUALIZATIONS...")
    print("-" * 40)

    if results['original']['params'] and results['sliding']['params']:
        create_comparison_plots(results, output_dir)
        print(f"Plots saved to {output_dir}/")

    # Save numerical results
    results_file = os.path.join(output_dir, 'comparison_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("SKILL RECOVERY METHOD COMPARISON\n")
        f.write("="*60 + "\n\n")

        if results['differences']:
            f.write("1. PARAMETER RECOVERY DIFFERENCES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Absolute Error (MAE): {avg_mae:.6f}\n")
            f.write(f"Root Mean Square Error (RMSE): {avg_rmse:.6f}\n")
            f.write(f"\nMean differences per dimension:\n")
            f.write(f"  Lateral:  {mean_diffs[0]:.6f}\n")
            f.write(f"  Yaw:      {mean_diffs[1]:.6f}\n")
            f.write(f"  Velocity: {mean_diffs[2]:.6f}\n\n")

        if results['original']['smoothness'] and results['sliding']['smoothness']:
            f.write("2. PARAMETER SMOOTHNESS\n")
            f.write("-" * 40 + "\n")
            f.write(f"\nOriginal: {np.mean(orig_mean_var):.6f}\n")
            f.write(f"Sliding:  {np.mean(slide_mean_var):.6f}\n")
            f.write(f"Improvement: {np.mean(improvement):.2f}%\n")

    print(f"\nResults saved to {results_file}")
    print("\n" + "="*60)

    return results

def create_comparison_plots(results, output_dir):
    """Create visualization plots comparing the methods"""

    orig_params = np.array(results['original']['params'])
    slide_params = np.array(results['sliding']['params'])

    # Limit to first 1000 points for visualization
    max_points = min(1000, len(orig_params), len(slide_params))
    orig_params = orig_params[:max_points]
    slide_params = slide_params[:max_points]

    param_names = ['Lateral', 'Yaw', 'Velocity']

    # Plot 1: Parameter trajectories
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(orig_params[:, i], label='Original', alpha=0.7, linewidth=1)
        ax.plot(slide_params[:, i], label='Sliding Window', alpha=0.7, linewidth=1)
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Step')
    axes[0].set_title('Parameter Trajectories Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_trajectories.png'), dpi=150)
    plt.close()

    # Plot 2: Parameter distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(orig_params[:, i], bins=50, alpha=0.5, label='Original', density=True)
        ax.hist(slide_params[:, i], bins=50, alpha=0.5, label='Sliding Window', density=True)
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Parameter Distributions Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'), dpi=150)
    plt.close()

    # Plot 3: Smoothness (variation over time)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        orig_diff = np.abs(np.diff(orig_params[:, i]))
        slide_diff = np.abs(np.diff(slide_params[:, i]))

        ax.plot(orig_diff, label='Original', alpha=0.7, linewidth=1)
        ax.plot(slide_diff, label='Sliding Window', alpha=0.7, linewidth=1)
        ax.set_ylabel(f'{name} Variation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Step')
    axes[0].set_title('Parameter Variation (Smoothness) Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_smoothness.png'), dpi=150)
    plt.close()

    print("  - parameter_trajectories.png")
    print("  - parameter_distributions.png")
    print("  - parameter_smoothness.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare skill recovery methods')
    parser.add_argument('--original_dir', type=str,
                       default='demonstration_RL_expert/highway_annotated',
                       help='Directory with original method results')
    parser.add_argument('--sliding_dir', type=str,
                       default='demonstration_RL_expert/highway_sliding_annotated',
                       help='Directory with sliding window method results')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_results',
                       help='Output directory for comparison results')

    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(args.original_dir):
        print(f"Error: Original directory not found: {args.original_dir}")
        exit(1)

    if not os.path.exists(args.sliding_dir):
        print(f"Warning: Sliding window directory not found yet: {args.sliding_dir}")
        print("Make sure the sliding window script has completed.")
        exit(1)

    # Run comparison
    results = compare_methods(args.original_dir, args.sliding_dir, args.output_dir)

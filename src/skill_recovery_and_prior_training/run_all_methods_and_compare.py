"""
Unified script to run all three recovery methods and compare them.

This script:
1. Runs Global Optimization method
2. Runs Sliding Window method  
3. Runs Fast method
4. Compares all three methods

Usage:
    python run_all_methods_and_compare.py --scenario highway
"""

import argparse
import os
import sys
import subprocess

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"\n{'='*80}")
        print(f"ERROR: {description} failed with return code {result.returncode}")
        print(f"{'='*80}\n")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run all three recovery methods and compare')
    parser.add_argument('--scenario', type=str, default='highway',
                       choices=['highway', 'intersection', 'roundabout'],
                       help='Scenario to process')
    parser.add_argument('--skip_global', action='store_true',
                       help='Skip global optimization method')
    parser.add_argument('--skip_sliding', action='store_true',
                       help='Skip sliding window method')
    parser.add_argument('--skip_fast', action='store_true',
                       help='Skip fast method')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip comparison (only run methods)')
    parser.add_argument('--use_ground_truth', action='store_true',
                       help='Use ground truth for parameter error computation')
    parser.add_argument('--max_files', type=int, default=0,
                       help='Maximum number of files to process per method (0 = all)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RUNNING ALL THREE RECOVERY METHODS AND COMPARING")
    print("="*80)
    print(f"Scenario: {args.scenario}")
    print(f"Max files per method: {args.max_files if args.max_files > 0 else 'All'}")
    print("="*80)
    
    # Check if input data exists
    input_dir = os.path.join(project_root, f"demonstration_rule_expert/{args.scenario}/")
    if not os.path.exists(input_dir):
        print(f"\n{'='*80}")
        print(f"ERROR: Input data directory not found: {input_dir}")
        print(f"{'='*80}")
        print("\nPlease collect demonstration data first:")
        print(f"  python src/data_collection/{args.scenario}/rule_expert.py")
        print("="*80)
        return
    
    # 1. Run Global Optimization method
    if not args.skip_global:
        cmd = [
            sys.executable,
            'src/skill_recovery_and_prior_training/main_skill_recovery.py',
            '--scenario', args.scenario
        ]
        if args.max_files > 0:
            cmd.extend(['--max_files', str(args.max_files)])
        if not run_command(cmd, "Global Optimization Method"):
            print("Warning: Global optimization failed, but continuing...")
    else:
        print("\nSkipping Global Optimization method (--skip_global)")
    
    # 2. Run Sliding Window method
    if not args.skip_sliding:
        cmd = [
            sys.executable,
            'src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py',
            '--scenario', args.scenario,
            '--window_len', '10',
            '--step_size', '2'
        ]
        if args.max_files > 0:
            cmd.extend(['--max_files', str(args.max_files)])
        if not run_command(cmd, "Sliding Window Method"):
            print("Warning: Sliding window method failed, but continuing...")
    else:
        print("\nSkipping Sliding Window method (--skip_sliding)")
    
    # 3. Run Fast method
    if not args.skip_fast:
        cmd = [
            sys.executable,
            'src/skill_recovery_and_prior_training/main_skill_recovery_fast.py',
            '--scenario', args.scenario
        ]
        if args.max_files > 0:
            cmd.extend(['--max_files', str(args.max_files)])
        if not run_command(cmd, "Fast Method"):
            print("Warning: Fast method failed, but continuing...")
    else:
        print("\nSkipping Fast method (--skip_fast)")
    
    # 4. Compare all methods
    if not args.skip_comparison:
        print("\n" + "="*80)
        print("COMPARING ALL THREE METHODS")
        print("="*80)
        
        cmd = [
            sys.executable,
            'src/skill_recovery_and_prior_training/compare_recovery_methods.py',
            '--scenario', args.scenario
        ]
        if args.use_ground_truth:
            cmd.append('--use_ground_truth')
        if args.max_files > 0:
            cmd.extend(['--max_files', str(args.max_files)])
        
        output_file = os.path.join(project_root, f'comparison_results_{args.scenario}.pkl')
        cmd.extend(['--output_file', output_file])
        
        if not run_command(cmd, "Comparison"):
            print("Warning: Comparison failed")
        else:
            print(f"\nComparison results saved to: {output_file}")
    else:
        print("\nSkipping comparison (--skip_comparison)")
    
    print("\n" + "="*80)
    print("ALL METHODS COMPLETED")
    print("="*80)
    print("\nResults saved to:")
    print(f"  - Global: demonstration_RL_expert/{args.scenario}_annotated/")
    print(f"  - Sliding: demonstration_RL_expert/{args.scenario}_sliding_annotated/")
    print(f"  - Fast: demonstration_RL_expert/{args.scenario}_fast_annotated/")
    if not args.skip_comparison:
        print(f"  - Comparison: comparison_results_{args.scenario}.pkl")
    print("="*80)

if __name__ == '__main__':
    main()


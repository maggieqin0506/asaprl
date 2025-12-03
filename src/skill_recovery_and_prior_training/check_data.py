"""
Quick script to check if demonstration data exists and provide instructions.
"""

import os
import sys

def check_data(scenario='highway'):
    """Check if demonstration data exists for a scenario."""
    # Get project root (go up from skill_recovery_and_prior_training to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Check both possible locations (rule_expert vs RL_expert)
    data_path_rl = os.path.join(project_root, f'demonstration_RL_expert/{scenario}/')
    data_path_rule = os.path.join(project_root, f'demonstration_rule_expert/{scenario}/')
    
    print(f"\n{'='*80}")
    print(f"Checking for demonstration data: {scenario}")
    print(f"{'='*80}\n")
    
    # Check RL_expert location (expected by skill recovery)
    if os.path.exists(data_path_rl):
        files = [f for f in os.listdir(data_path_rl) if f.endswith('.pickle')]
        print(f"✓ Data directory exists: {data_path_rl}")
        print(f"✓ Found {len(files)} data file(s)")
        if len(files) > 0:
            print(f"  Example files: {files[:3]}")
        return True
    
    # Check rule_expert location (where rule expert saves)
    if os.path.exists(data_path_rule):
        files = [f for f in os.listdir(data_path_rule) if f.endswith('.pickle')]
        print(f"⚠ Data found in: {data_path_rule}")
        print(f"  Found {len(files)} data file(s)")
        print(f"\n⚠ NOTE: Rule expert saves to 'demonstration_rule_expert' but")
        print(f"   skill recovery expects 'demonstration_RL_expert'")
        print(f"\nTo fix, copy the data (from project root):")
        print(f"  mkdir -p ./demonstration_RL_expert/{scenario}/")
        print(f"  cp -r ./demonstration_rule_expert/{scenario}/* ./demonstration_RL_expert/{scenario}/")
        return False
    
    # No data found
    print(f"✗ Data directory does not exist in either location:")
    print(f"  - {data_path_rl}")
    print(f"  - {data_path_rule}")
    print("\nTo collect data, run one of these commands:")
    print(f"\nOption 1: Rule-based expert (simplest):")
    print(f"  python src/data_collection/{scenario}/rule_expert.py")
    print(f"  Note: This saves to 'demonstration_rule_expert', you may need to copy it")
    print(f"\nOption 2: Trained RL agent (requires model):")
    print(f"  See launch_command/data_collection_commands/rl_agent")
    return False

if __name__ == '__main__':
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'highway'
    exists = check_data(scenario)
    sys.exit(0 if exists else 1)


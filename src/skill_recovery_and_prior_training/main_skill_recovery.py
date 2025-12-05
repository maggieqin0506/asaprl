import argparse
from skill_param_recovery import annotate_data

parser = argparse.ArgumentParser(description="Original Author's Skill Recovery")
parser.add_argument('--scenario', type=str, default='highway', help='Scenario name')
parser.add_argument('--skill_length', type=int, default=10, help='Skill length')
args = parser.parse_args()

# Use original author's annotate_data (RL expert data only, no max_files parameter)
annotate_data(args.scenario, args.skill_length)
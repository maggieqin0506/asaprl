import argparse
from skill_param_recovery import annotate_data

parser = argparse.ArgumentParser(description="Global Optimization Skill Recovery")
parser.add_argument('--scenario', type=str, default='highway', help='Scenario name')
parser.add_argument('--skill_length', type=int, default=10, help='Skill length')
parser.add_argument('--max_files', type=int, default=0, help='Maximum number of files to process. 0 means all files.')
args = parser.parse_args()

annotate_data(args.scenario, args.skill_length, args.max_files)
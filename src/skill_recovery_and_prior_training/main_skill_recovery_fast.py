import argparse
import os
import sys

# Add project root to path for external packages/modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add current directory to path for local imports
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from skill_param_recovery_fast import annotate_data

parser = argparse.ArgumentParser(description="Fast Skill Recovery (Local Opt + Gaussian Smoothing)")
parser.add_argument('--scenario', type=str, default='highway', help='Scenario name')
parser.add_argument('--skill_length', type=int, default=10, help='Skill length')
parser.add_argument('--max_files', type=int, default=0, help='Maximum number of files to process. 0 means all files.')
args = parser.parse_args()

annotate_data(args.scenario, args.skill_length, args.max_files)


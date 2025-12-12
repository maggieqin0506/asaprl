import argparse
from skill_param_recovery_sliding import annotate_data

parser = argparse.ArgumentParser(description="Sliding Window Skill Recovery")
parser.add_argument('--scenario', type=str, default='highway', help='Scenario name')
parser.add_argument('--skill_length', type=int, default=10, help='Skill length')
parser.add_argument('--action_shape', type=int, default=3, help='Dimension of skill parameters (default: 3 latent vars)')
parser.add_argument('--reward_version', type=int, default=27)
parser.add_argument('--lat_range', type=float, default=5)
parser.add_argument('--window_len', type=int, default=10, help='Length of the optimization horizon (T). Must match original setup.')
parser.add_argument('--step_size', type=int, default=2, help='Stride/Step size (in timesteps). Smaller = smoother.')
parser.add_argument('--max_files', type=int, default=0, help='Maximum number of files to process. 0 means all files.')
args = parser.parse_args()

annotate_data(
    args.scenario, 
    args.skill_length, 
    args.max_files,
    args.window_len,
    args.step_size,
    args.lat_range,
    args.action_shape
)


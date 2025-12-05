# Data Collection Guide: What Data is Collected?

## Overview

When you run `python src/data_collection/highway/rule_expert.py`, the script collects **demonstration data** from a rule-based expert driving in the MetaDrive simulation environment.

## What Data is Collected?

Each successful episode saves a pickle file containing the following data structure:

```python
{
    'obs': [],                    # Observations (visual/state information)
    'abs_state': [],              # Absolute state (world coordinates)
    'rela_state': [],             # Relative state (relative to vehicle)
    'action': [],                 # Actions taken by the expert
    'reward': [],                 # Step rewards received
    'vehicle_start_speed': []     # Vehicle starting speed for each skill
}
```

### Detailed Breakdown

#### 1. **`obs`** - Observations
- **Type**: List of observation arrays
- **Shape**: `[5, 200, 200]` (5-channel image-like representation)
- **Content**: Visual/state observations from the environment
- **Frequency**: Collected at each timestep within each skill execution
- **Purpose**: Used for training policies that learn from visual input

#### 2. **`abs_state`** - Absolute State
- **Type**: List of state arrays
- **Content**: Absolute world coordinates and vehicle state
- **Frequency**: Collected at each timestep within each skill execution
- **Purpose**: Provides global position and orientation information

#### 3. **`rela_state`** - Relative State (Most Important for Recovery)
- **Type**: List of trajectory arrays
- **Shape**: `[N, 2]` or `[N, 4]` where N is trajectory length
  - `[N, 2]`: `[x, y]` positions
  - `[N, 4]`: `[x, y, speed, yaw]` full trajectory
- **Content**: Relative trajectory states (relative to the vehicle's frame)
- **Frequency**: One trajectory per skill execution
- **Purpose**: **This is the key data used for skill parameter recovery!**
  - These are the reference trajectories that recovery methods try to match

#### 4. **`action`** - Actions
- **Type**: List of action arrays
- **Shape**: `[action_shape]` (typically 2D: lateral offset, yaw change)
- **Content**: Actions taken by the rule-based expert
- **Frequency**: One action per skill execution
- **Purpose**: Shows what actions the expert chose for each skill

#### 5. **`reward`** - Rewards
- **Type**: List of scalar rewards
- **Content**: Step rewards received from the environment
- **Frequency**: One reward per skill execution
- **Purpose**: Indicates how good each action was

#### 6. **`vehicle_start_speed`** - Starting Speed
- **Type**: List of scalar speeds
- **Content**: Vehicle's starting speed at the beginning of each skill
- **Frequency**: One speed per skill execution
- **Purpose**: Needed to generate trajectories with the motion skill model

## Data Collection Process

### Step-by-Step:

1. **Episode Start**: A new driving episode begins
2. **Skill Execution**: The rule-based expert executes skills (typically 10 timesteps each)
3. **Data Collection**: For each skill:
   - Observations are collected at each timestep
   - Absolute states are collected at each timestep
   - Relative trajectory is collected (one per skill)
   - Action is recorded (one per skill)
   - Reward is recorded (one per skill)
   - Starting speed is recorded (one per skill)
4. **Episode End**: When episode completes
5. **Save Condition**: **Only successful episodes are saved** (episode must reach destination)
6. **File Saved**: `highway_expert_data_{N}.pickle` in `demonstration_rule_expert/highway/`

### Important Notes:

- **Only successful episodes are saved**: If the vehicle crashes, goes off-road, or times out, that episode's data is discarded
- **Default: 20 episodes**: The script runs 20 episodes by default (`n_evaluator_episode=20`)
- **Each file = one successful episode**: Each pickle file contains data from one complete successful driving episode
- **Multiple skills per episode**: Each episode contains many skill executions (typically 10-50+ skills depending on route length)

## What This Data is Used For

### Primary Use: Skill Parameter Recovery

The main purpose of this data is to:
1. **Recover skill parameters** from the relative trajectories (`rela_state`)
2. **Learn what actions** the expert took in different situations
3. **Train policies** to imitate the expert behavior

### Recovery Process:

1. **Input**: `rela_state` trajectories (reference trajectories)
2. **Process**: Recovery methods (Global, Sliding, Fast) try to find skill parameters that generate trajectories matching `rela_state`
3. **Output**: Recovered skill parameters (latent variables) that can reproduce similar trajectories

## Example Data Structure

Here's what a typical data file looks like:

```python
{
    'obs': [
        array([...]),  # Observation at timestep 0 of skill 0
        array([...]),  # Observation at timestep 1 of skill 0
        ...
        array([...]),  # Observation at timestep 0 of skill 1
        ...
    ],
    'abs_state': [
        array([...]),  # Absolute state at timestep 0 of skill 0
        array([...]),  # Absolute state at timestep 1 of skill 0
        ...
    ],
    'rela_state': [
        array([[x1, y1], [x2, y2], ...]),  # Trajectory for skill 0
        array([[x1, y1], [x2, y2], ...]),  # Trajectory for skill 1
        array([[x1, y1], [x2, y2], ...]),  # Trajectory for skill 2
        ...  # One trajectory per skill in the episode
    ],
    'action': [
        array([lat_offset, yaw_change]),  # Action for skill 0
        array([lat_offset, yaw_change]),  # Action for skill 1
        ...
    ],
    'reward': [
        0.5,   # Reward for skill 0
        0.3,   # Reward for skill 1
        ...
    ],
    'vehicle_start_speed': [
        5.0,   # Starting speed for skill 0
        5.2,   # Starting speed for skill 1
        ...
    ]
}
```

## File Locations

- **Collected data**: `demonstration_rule_expert/highway/`
- **File naming**: `highway_expert_data_{N}.pickle` where N is the episode number
- **After recovery**: `demonstration_RL_expert/highway_annotated/` (with recovered parameters added)

## Checking Your Data

You can inspect the collected data:

```python
import pickle

# Load a data file
with open('demonstration_rule_expert/highway/highway_expert_data_1.pickle', 'rb') as f:
    data = pickle.load(f)

# Check what's inside
print("Keys:", data.keys())
print("Number of skills:", len(data['rela_state']))
print("First trajectory shape:", data['rela_state'][0].shape)
print("Number of observations:", len(data['obs']))
```

## Summary

**What you're collecting:**
- Expert driving demonstrations from a rule-based controller
- Trajectories, observations, actions, and rewards
- Only from successful episodes (reached destination)

**Why you need it:**
- To recover skill parameters that can reproduce expert trajectories
- To train policies that learn from expert demonstrations
- To evaluate different recovery methods

**Key data for recovery:**
- **`rela_state`**: The reference trajectories that recovery methods try to match
- **`vehicle_start_speed`**: Needed to generate trajectories with correct initial conditions


# Skill Recovery Evaluation Guide

This guide explains how to run the skill parameter recovery program and measure improvements from the trajectory-level optimization changes.

## Overview

The skill recovery process has been improved to use **trajectory-level optimization** instead of **point-wise optimization**. This change:
- Optimizes the overall trajectory shape rather than individual points
- Encourages smoother trajectories through smoothness penalties
- Produces more consistent parameter recovery across segments

## Quick Start

### 0. Collect Demonstration Data First (Required!)

Before running skill recovery, you need to collect demonstration data. You have two options:

#### Option A: Rule-Based Expert (Simplest - No trained model needed)

```bash
# Collect data from rule-based expert
python src/data_collection/highway/rule_expert.py
```

This will create `./demonstration_RL_expert/highway/` with demonstration data.

For other scenarios:
```bash
python src/data_collection/intersection/rule_expert.py
python src/data_collection/roundabout/rule_expert.py
```

#### Option B: Trained RL Agent (Requires trained model)

If you have a trained model, you can collect data from it:

```bash
python src/data_collection/highway/skill.py \
    --exp_name 'saved_model/ASPA_NoPrior_fixed10_RV27_lat5_highway_seed1' \
    --ckpt_file ckpt/iteration_50000.pth.tar \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

### 1. Run Skill Recovery (New Method)

After collecting data, run the skill recovery with the new trajectory-level optimization:

```bash
cd /path/to/asaprl
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
```

For other scenarios:
```bash
# Intersection
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario intersection

# Roundabout
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario roundabout
```

This will:
- Read demonstration data from `./demonstration_RL_expert/{scenario}/`
- Recover skill parameters using the new trajectory-level optimization
- Save annotated data to `./demonstration_RL_expert/{scenario}_annotated/`

### 2. Evaluate the New Method

To evaluate the new method and see metrics:

```bash
cd src/skill_recovery_and_prior_training
python evaluate_recovery_improvement.py --scenario highway
```

This will compute and display:
- **Smoothness metrics**: curvature, jerk, yaw rate changes
- **Trajectory matching**: endpoint error, average displacement, path length difference
- **Segment continuity**: smoothness between consecutive segments
- **Parameter accuracy**: error compared to ground truth (if available)

## Comparing Old vs New Method

To compare the old (point-wise) and new (trajectory-level) methods:

### Option 1: Automated Comparison Script

```bash
cd src/skill_recovery_and_prior_training
python compare_methods.py --scenario highway --compare
```

This script will:
1. Backup the current (new) version
2. Create and restore the old version
3. Run evaluation on old method
4. Restore new version
5. Run evaluation on new method
6. Compare and display results

### Option 2: Manual Comparison

If you want more control:

```bash
cd src/skill_recovery_and_prior_training

# Step 1: Backup current version
python compare_methods.py --backup

# Step 2: Create old version backup
python compare_methods.py --create-old

# Step 3: Restore old version and evaluate
python compare_methods.py --restore-old
python evaluate_recovery_improvement.py --scenario highway --output_file results_old.pkl

# Step 4: Restore new version and evaluate
python compare_methods.py --restore-new
python evaluate_recovery_improvement.py --scenario highway --output_file results_new.pkl

# Step 5: Compare results (you can write a simple script to load and compare the pickle files)
```

## Metrics Explained

### Smoothness Metrics

- **Mean/Max Curvature**: Measures how much the trajectory curves (lower = smoother)
- **Mean/Max Jerk**: Measures rate of change of acceleration (lower = smoother)
- **Mean/Max Yaw Rate**: Measures rate of change of heading (lower = smoother)
- **Mean/Max Speed Change**: Measures speed variations (lower = smoother)

### Trajectory Matching Metrics

- **Endpoint Error**: Distance between recovered and reference trajectory endpoints
- **Average Displacement**: Average point-wise distance between trajectories
- **Max Displacement**: Maximum point-wise distance
- **Speed/Yaw RMSE**: Root mean square error for speed and yaw profiles
- **Path Length Difference**: Difference in total path lengths

### Segment Continuity Metrics

- **Position Discontinuity**: Gap between consecutive segments
- **Velocity Discontinuity**: Speed change between segments
- **Yaw Discontinuity**: Heading change between segments

### Parameter Recovery Accuracy

- **Latent Variable Error**: Overall parameter recovery error
- **Lat/Yaw/V Error**: Individual parameter errors (lateral, yaw, velocity)

## Expected Improvements

With the new trajectory-level optimization, you should see:

1. **Improved Smoothness**: 
   - Lower curvature variance
   - Lower jerk values
   - More consistent yaw rates

2. **Better Trajectory Matching**:
   - Similar or better endpoint accuracy
   - Better overall shape matching
   - More consistent path lengths

3. **Better Segment Continuity**:
   - Smaller gaps between segments
   - Smoother transitions in velocity and yaw

4. **Maintained or Improved Accuracy**:
   - Parameter recovery accuracy should be similar or better
   - Overall trajectory quality should improve

## Troubleshooting

### Data Not Found

If you get an error about missing data:
```bash
# Make sure you have collected demonstration data first
# See launch_command/data_collection_commands/ for data collection scripts
```

### Import Errors

If you get import errors:
```bash
# Make sure you're in the correct directory
cd src/skill_recovery_and_prior_training

# Or install the package
cd /path/to/asaprl
python -m pip install -e .
```

### Comparison Script Issues

If the comparison script fails:
- Make sure you have write permissions in the directory
- Check that `skill_param_recovery.py` exists
- Verify the data path is correct

## Advanced Usage

### Custom Data Path

```bash
python evaluate_recovery_improvement.py \
    --scenario highway \
    --data_path /custom/path/to/data/
```

### Save Results for Later Analysis

```bash
python evaluate_recovery_improvement.py \
    --scenario highway \
    --output_file my_results.pkl
```

Then load and analyze:
```python
import pickle
with open('my_results.pkl', 'rb') as f:
    results = pickle.load(f)
    print(results)
```

## Files Created

- `skill_param_recovery_new_backup.py`: Backup of new version
- `skill_param_recovery_old.py`: Backup of old version
- `evaluation_results_old_{scenario}.pkl`: Old method results
- `evaluation_results_new_{scenario}.pkl`: New method results

## Next Steps

After evaluating:
1. Review the metrics to understand improvements
2. Visualize trajectories if needed (can extend evaluation script)
3. Tune smoothness weights if needed (in `skill_param_recovery.py`)
4. Use the improved recovery for downstream tasks (actor pretraining)


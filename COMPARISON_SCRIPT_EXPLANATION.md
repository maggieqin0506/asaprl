# What Does `compare_all_methods.py` Do?

## Overview

The `compare_all_methods.py` script **evaluates and compares three different skill parameter recovery methods** to see which one performs best. It's like running a benchmark test on different algorithms.

## What It Does (Step by Step)

### Step 1: Load Pre-computed Results

The script assumes you've already run the three recovery methods and they've saved their results. It looks for data in:

- **Global Optimization**: `demonstration_RL_expert/highway_annotated/`
- **Sliding Window**: `demonstration_RL_expert/highway_sliding_annotated/`
- **Fast Method**: `demonstration_RL_expert/highway_fast_annotated/`

Each directory should contain pickle files with recovered skill parameters.

### Step 2: Evaluate Each Method

For each method, it:

1. **Loads the recovered parameters** from the pickle files
2. **Generates trajectories** using those recovered parameters
3. **Compares generated trajectories** to the original reference trajectories
4. **Computes metrics** to measure performance

### Step 3: Compute Performance Metrics

For each method, it calculates:

#### **Smoothness Metrics** (Lower is Better)
- `mean_curvature`: How much the trajectory curves
- `max_curvature`: Maximum curvature
- `mean_speed_change`: How much speed varies
- `mean_yaw_rate`: How much heading changes

#### **Matching Metrics** (Lower is Better)
- `endpoint_error`: Distance between trajectory endpoints
- `avg_displacement`: Average distance between trajectories
- `max_displacement`: Maximum distance
- `speed_rmse`: Speed matching error
- `yaw_rmse`: Heading matching error

#### **Continuity Metrics** (Lower is Better)
- `position_discontinuity`: Gaps between consecutive segments
- `velocity_discontinuity`: Speed jumps between segments
- `yaw_discontinuity`: Heading jumps between segments

#### **Parameter Error** (If ground truth available)
- `latent_var_error`: Overall parameter recovery error
- `lat_error`, `yaw_error`, `v_error`: Individual parameter errors

### Step 4: Aggregate Results

It averages all metrics across:
- All files in the directory
- All trajectory segments within each file

This gives you **overall performance statistics** for each method.

### Step 5: Print Comparison

It prints a detailed comparison table showing:
- All metrics for all three methods
- Which method is best (marked with ★) for each metric

Example output:
```
--- SMOOTHNESS METRICS (Lower is Better) ---

mean_curvature:
  ★ GLOBAL  : 0.123456
    SLIDING  : 0.145678
    FAST     : 0.134567
```

### Step 6: Generate Comparison Graphs

By default, it creates 7 types of comparison graphs:

1. **Trajectory Spatial Comparison**: Shows all 4 trajectories (Original + 3 methods) overlaid on the same plot
2. **Smoothness Metrics Comparison**: Bar charts comparing smoothness
3. **Matching Metrics Comparison**: Bar charts comparing trajectory matching
4. **Continuity Metrics Comparison**: Bar charts comparing segment continuity
5. **Radar Chart Comparison**: Multi-metric radar charts
6. **Metrics Over Trajectories**: Line plots showing how metrics vary
7. **Summary Comparison**: Overall performance summary

All graphs saved to: `trajectory_comparisons/highway/`

### Step 7: Save Results (Optional)

If you specify `--output_file`, it saves the numerical results to a pickle file for later analysis.

## What It Does NOT Do

- ❌ **Does NOT run the recovery methods** - you must run those first
- ❌ **Does NOT collect demonstration data** - that's done separately
- ❌ **Does NOT train any models** - it only evaluates existing results

## Prerequisites

Before running this script, you need:

1. **Demonstration data** collected (from `rule_expert.py`)
2. **All three recovery methods run**:
   - `main_skill_recovery.py` (Global)
   - `main_skill_recovery_sliding.py` (Sliding)
   - `main_skill_recovery_fast.py` (Fast)

## Example Workflow

```bash
# Step 1: Collect data (if not done)
python src/data_collection/highway/rule_expert.py

# Step 2: Run all three recovery methods
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario highway
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario highway

# Step 3: Compare all methods
python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario highway
```

## What You Get

After running, you'll have:

1. **Console output**: Detailed comparison tables showing which method is best
2. **Graph files**: 7 PNG files with visual comparisons
3. **Optional pickle file**: Numerical results for further analysis

## Key Insight

This script answers the question: **"Which recovery method works best?"**

It does this by:
- Taking the recovered parameters from each method
- Generating trajectories from those parameters
- Comparing those trajectories to the original reference trajectories
- Measuring how well each method matches the original

The method with the **lowest errors** and **best smoothness** is the winner!

## Command Options

```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \              # Which scenario to compare
    --max_files 10 \                  # Limit to 10 files (for testing)
    --no_graphs \                     # Skip graph generation
    --output_file results.pkl \       # Save numerical results
    --use_ground_truth                 # Compute parameter error (if GT available)
```

## Summary

**In one sentence**: This script evaluates and compares three skill parameter recovery methods by measuring how well each method's recovered parameters can reproduce the original expert trajectories.


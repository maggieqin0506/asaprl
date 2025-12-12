# Full Demo Guide: Comparing All Three Recovery Methods

This guide walks you through running a complete demonstration that:
1. Collects demonstration data (if needed)
2. Runs all three recovery methods (Global, Sliding Window, Fast)
3. Generates comprehensive comparison graphs

## Quick Start (Easiest Method)

If you already have demonstration data, you can run everything with one command:

**Highway Scenario:**
```bash
# From the project root directory
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario highway
```

**Intersection Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario intersection
```

**Roundabout Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario roundabout
```

This will:
- Run all three recovery methods
- Generate comparison graphs automatically
- Save results to `trajectory_comparisons/{scenario}/`

## Step-by-Step Guide

### Step 0: Collect Demonstration Data (If Needed)

First, make sure you have demonstration data. You have two options:

#### Option A: Rule-Based Expert (Simplest - No trained model needed)

**Highway Scenario:**
```bash
# From project root
python src/data_collection/highway/rule_expert.py
```
This creates `demonstration_rule_expert/highway/` with demonstration data.

**Intersection Scenario:**
```bash
python src/data_collection/intersection/rule_expert.py
```
This creates `demonstration_rule_expert/intersection/` with demonstration data.

**Roundabout Scenario:**
```bash
python src/data_collection/roundabout/rule_expert.py
```
This creates `demonstration_rule_expert/roundabout/` with demonstration data.

#### Option B: Trained RL Agent (Requires trained model)

If you have a trained model:

**Highway Scenario:**
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

**Intersection Scenario:**
```bash
python src/data_collection/intersection/skill.py \
    --exp_name 'saved_model/ASPA_NoPrior_fixed10_RV27_lat5_intersection_seed1' \
    --ckpt_file ckpt/iteration_50000.pth.tar \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

**Roundabout Scenario:**
```bash
python src/data_collection/roundabout/skill.py \
    --exp_name 'saved_model/ASPA_NoPrior_fixed10_RV27_lat5_roundabout_seed1' \
    --ckpt_file ckpt/iteration_50000.pth.tar \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

### Step 1: Run All Three Recovery Methods

You can run all three methods individually or use the unified script:

#### Method 1: Unified Script (Recommended)

**Highway Scenario:**
```bash
# From project root
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario highway
```

**Intersection Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario intersection
```

**Roundabout Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario roundabout
```

#### Method 2: Run Individually

If you prefer to run each method separately:

**Highway Scenario:**
```bash
# 1. Global Optimization Method
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway

# 2. Sliding Window Method
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py \
    --scenario highway \
    --window_len 10 \
    --step_size 2

# 3. Fast Method
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario highway
```

**Intersection Scenario:**
```bash
# 1. Global Optimization Method
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario intersection

# 2. Sliding Window Method
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py \
    --scenario intersection \
    --window_len 10 \
    --step_size 2

# 3. Fast Method
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario intersection
```

**Roundabout Scenario:**
```bash
# 1. Global Optimization Method
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario roundabout

# 2. Sliding Window Method
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py \
    --scenario roundabout \
    --window_len 10 \
    --step_size 2

# 3. Fast Method
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario roundabout
```

**Output directories:**
- Global: `demonstration_RL_expert/{scenario}_annotated/`
- Sliding: `demonstration_RL_expert/{scenario}_sliding_annotated/`
- Fast: `demonstration_RL_expert/{scenario}_fast_annotated/`

### Step 2: Generate Comparison Graphs

If you used the unified script, graphs are generated automatically. Otherwise, run:

**Highway Scenario:**
```bash
# From project root
python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario highway
```

**Intersection Scenario:**
```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario intersection
```

**Roundabout Scenario:**
```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario roundabout
```

**Graphs will be saved to:** `trajectory_comparisons/{scenario}/`

The script generates:
1. **trajectory_spatial_comparison.png** - Spatial plots showing all 4 trajectories (Original + 3 methods) overlaid
2. **smoothness_metrics_comparison.png** - Bar charts comparing smoothness metrics
3. **matching_metrics_comparison.png** - Bar charts comparing trajectory matching metrics
4. **continuity_metrics_comparison.png** - Bar charts comparing segment continuity metrics
5. **radar_chart_comparison.png** - Radar charts for comprehensive multi-metric comparison
6. **metrics_over_trajectories.png** - Line plots showing how metrics vary across trajectory segments
7. **summary_comparison.png** - Summary plots with key metrics and overall performance scores

### Step 3: View Results

All graphs are saved as high-resolution PNG files (300 DPI) in:
```
trajectory_comparisons/{scenario}/
```

You can also save the numerical comparison results:

**Highway Scenario:**
```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --output_file comparison_results_highway.pkl
```

**Intersection Scenario:**
```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario intersection \
    --output_file comparison_results_intersection.pkl
```

**Roundabout Scenario:**
```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario roundabout \
    --output_file comparison_results_roundabout.pkl
```

## Command Options

### Unified Script Options

```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario {highway|intersection|roundabout} \
    --skip_global          # Skip global optimization
    --skip_sliding         # Skip sliding window
    --skip_fast            # Skip fast method
    --skip_comparison      # Skip comparison (only run methods)
    --use_ground_truth     # Use ground truth for parameter error
    --max_files 10         # Limit number of files to process (for testing)
```

### Comparison Script Options

```bash
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario {highway|intersection|roundabout} \
    --global_data_path /custom/path/    # Custom path for global method
    --sliding_data_path /custom/path/   # Custom path for sliding method
    --fast_data_path /custom/path/      # Custom path for fast method
    --output_dir /custom/output/        # Custom output directory for graphs
    --output_file results.pkl           # Save numerical results
    --use_ground_truth                  # Compute parameter error
    --no_graphs                         # Skip graph generation
```

## Example: Full Demo for All Scenarios

### Highway Scenario

```bash
# From project root directory

# Step 0: Collect data (if needed)
python src/data_collection/highway/rule_expert.py

# Step 1 & 2: Run all methods and compare (one command)
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario highway \
    --output_file comparison_results_highway.pkl

# Results will be in:
# - trajectory_comparisons/highway/ (graphs)
# - comparison_results_highway.pkl (numerical results)
```

### Intersection Scenario

```bash
# From project root directory

# Step 0: Collect data (if needed)
python src/data_collection/intersection/rule_expert.py

# Step 1 & 2: Run all methods and compare (one command)
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario intersection \
    --output_file comparison_results_intersection.pkl

# Results will be in:
# - trajectory_comparisons/intersection/ (graphs)
# - comparison_results_intersection.pkl (numerical results)
```

### Roundabout Scenario

```bash
# From project root directory

# Step 0: Collect data (if needed)
python src/data_collection/roundabout/rule_expert.py

# Step 1 & 2: Run all methods and compare (one command)
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario roundabout \
    --output_file comparison_results_roundabout.pkl

# Results will be in:
# - trajectory_comparisons/roundabout/ (graphs)
# - comparison_results_roundabout.pkl (numerical results)
```

## Example: Quick Test (Limited Files)

For quick testing, limit the number of files processed:

**Highway Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario highway \
    --max_files 5  # Only process 5 files per method
```

**Intersection Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario intersection \
    --max_files 5  # Only process 5 files per method
```

**Roundabout Scenario:**
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario roundabout \
    --max_files 5  # Only process 5 files per method
```

## Troubleshooting

### Missing Data Paths

If you get errors about missing data paths:

1. **Check input data exists:**
   ```bash
   # For highway
   ls demonstration_rule_expert/highway/
   ls demonstration_RL_expert/highway/
   
   # For intersection
   ls demonstration_rule_expert/intersection/
   ls demonstration_RL_expert/intersection/
   
   # For roundabout
   ls demonstration_rule_expert/roundabout/
   ls demonstration_RL_expert/roundabout/
   ```

2. **Collect data if missing:**
   ```bash
   # Highway
   python src/data_collection/highway/rule_expert.py
   
   # Intersection
   python src/data_collection/intersection/rule_expert.py
   
   # Roundabout
   python src/data_collection/roundabout/rule_expert.py
   ```

### Missing Annotated Data

If comparison fails because annotated data is missing:

1. **Run recovery methods first:**

   **Highway:**
   ```bash
   python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
   python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario highway
   python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario highway
   ```

   **Intersection:**
   ```bash
   python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario intersection
   python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario intersection
   python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario intersection
   ```

   **Roundabout:**
   ```bash
   python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario roundabout
   python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario roundabout
   python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario roundabout
   ```

2. **Then run comparison:**
   ```bash
   # Highway
   python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario highway
   
   # Intersection
   python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario intersection
   
   # Roundabout
   python src/skill_recovery_and_prior_training/compare_all_methods.py --scenario roundabout
   ```

### Different Number of Trajectories

If you see warnings about different numbers of trajectories:
- This is normal if methods processed different numbers of files
- The comparison will use the minimum number across all methods
- Check that all three methods completed successfully

## Output Files Summary

After running a full demo, you'll have:

### Data Directories:
- `demonstration_RL_expert/{scenario}_annotated/` - Global method results
- `demonstration_RL_expert/{scenario}_sliding_annotated/` - Sliding method results
- `demonstration_RL_expert/{scenario}_fast_annotated/` - Fast method results

Where `{scenario}` can be `highway`, `intersection`, or `roundabout`.

### Comparison Results:
- `trajectory_comparisons/{scenario}/` - Directory with all comparison graphs
- `comparison_results_{scenario}.pkl` - Numerical results (if --output_file specified)

### Graph Files:
1. `trajectory_spatial_comparison.png` - Spatial trajectory overlays
2. `smoothness_metrics_comparison.png` - Smoothness metric bars
3. `matching_metrics_comparison.png` - Matching metric bars
4. `continuity_metrics_comparison.png` - Continuity metric bars
5. `radar_chart_comparison.png` - Radar charts
6. `metrics_over_trajectories.png` - Metrics over time
7. `summary_comparison.png` - Summary comparison

## Next Steps

After viewing the graphs:
1. Analyze which method performs best for your use case
2. Check smoothness vs. accuracy trade-offs
3. Review trajectory visualizations to see qualitative differences
4. Use the best method for downstream tasks (e.g., actor pretraining)


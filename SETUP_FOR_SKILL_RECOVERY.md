# Setup Guide for Skill Recovery

## Step 1: Install Dependencies

Make sure you have all required packages installed:

```bash
conda activate asaprl
pip install -r requirements.txt
```

Or if you haven't installed the package yet:
```bash
conda activate asaprl
python -m pip install .
```

## Step 2: Collect Demonstration Data

You need demonstration data before running skill recovery. Choose one option:

### Option A: Rule-Based Expert (Simplest - Recommended for Testing)

This doesn't require a trained model:

```bash
# For highway scenario
python src/data_collection/highway/rule_expert.py

# For other scenarios
python src/data_collection/intersection/rule_expert.py
python src/data_collection/roundabout/rule_expert.py
```

This will create `./demonstration_RL_expert/{scenario}/` directories with demonstration data.

**Note**: The rule_expert.py script collects data without visualization. If you want to see what's happening, use:
```bash
python src/data_collection/highway/rule_expert_visualization.py
```

### Option B: Trained RL Agent (Requires Pre-trained Model)

If you have a trained model, you can collect higher-quality data:

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

## Step 3: Verify Data Collection

Check that data was collected:

```bash
ls ./demonstration_RL_expert/highway/
```

You should see `.pickle` files in this directory.

## Step 4: Run Skill Recovery

Now you can run skill recovery:

```bash
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
```

This will:
- Read data from `./demonstration_RL_expert/highway/`
- Recover skill parameters using the new trajectory-level optimization
- Save annotated data to `./demonstration_RL_expert/highway_annotated/`

## Troubleshooting

### ModuleNotFoundError: No module named 'metadrive'

Install metadrive:
```bash
pip install metadrive-simulator==0.2.4
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### FileNotFoundError: No such file or directory: './demonstration_RL_expert/highway/'

You need to collect demonstration data first (Step 2).

### Other Import Errors

Make sure the package is installed:
```bash
python -m pip install -e .
```


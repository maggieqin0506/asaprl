# Quick Evaluation Guide

## Step 1: Collect Demonstration Data (Required First!)

You need demonstration data before running skill recovery. Simplest option:

```bash
# Collect from rule-based expert (no trained model needed)
python src/data_collection/highway/rule_expert.py
```

This creates `./demonstration_RL_expert/highway/` directory.

## Step 2: Run Skill Recovery (New Method)

```bash
# From project root
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
```

## Evaluate New Method Only

```bash
cd src/skill_recovery_and_prior_training
python evaluate_recovery_improvement.py --scenario highway
```

## Compare Old vs New Methods

```bash
cd src/skill_recovery_and_prior_training
python compare_methods.py --scenario highway --compare
```

This automatically:
- Backs up current version
- Tests old method
- Tests new method  
- Shows comparison

## What Gets Measured

### Smoothness (Lower = Better)
- Curvature (how much trajectory curves)
- Jerk (acceleration changes)
- Yaw rate (heading changes)
- Speed changes

### Trajectory Matching (Lower = Better)
- Endpoint error
- Average displacement
- Path length difference

### Segment Continuity (Lower = Better)
- Position gaps between segments
- Velocity changes between segments
- Yaw changes between segments

### Parameter Accuracy (Lower = Better)
- Recovery error vs ground truth

## Expected Results

The new method should show:
- ✓ Lower smoothness metrics (smoother trajectories)
- ✓ Better or similar matching metrics
- ✓ Better segment continuity (smoother transitions)
- ✓ Similar or better parameter accuracy

See `SKILL_RECOVERY_EVALUATION_GUIDE.md` for detailed documentation.


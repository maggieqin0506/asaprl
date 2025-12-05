# Quick Start Guide - Step by Step

## Prerequisites
- Conda environment `asaprl` is created and activated
- All dependencies are installed

## Step 1: Activate Environment and Navigate to Project

```bash
conda activate asaprl
cd /Users/maggie/Documents/Projects/asaprl

# Create necessary directories (if they don't exist)
mkdir -p saved_model log
```

## Step 2: Verify Installation

```bash
# Check Python version (should be 3.7.x)
python --version

# Verify package is installed
python -c "import asaprl; print('ASAP-RL installed successfully')"
```

## Step 3: Choose What to Run

### Option A: Train ASAP-RL (No Prior - Simplest)

This trains ASAP-RL without requiring pretrained models:

```bash
python src/training/highway/skill.py \
    --exp_name 'saved_model/my_first_training' \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

**What this does:**
- Trains ASAP-RL on highway scenario
- Saves model to `saved_model/my_first_training/`
- Uses reduced memory settings (replay buffer: 500, batch size: 64)

### Option B: Train SAC Baseline (Even Simpler)

```bash
python src/training/highway/sac.py \
    --exp_name 'saved_model/my_sac_training'
```

**What this does:**
- Trains standard SAC algorithm
- Saves model to `saved_model/my_sac_training/`
- Uses reduced memory settings (replay buffer: 500, batch size: 32)

### Option C: Train PPO Baseline

```bash
python src/training/highway/ppo.py \
    --exp_name 'saved_model/my_ppo_training'
```

### Option D: Train on Different Scenarios

**Intersection:**
```bash
python src/training/intersection/skill.py \
    --exp_name 'saved_model/my_intersection_training' \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

**Roundabout:**
```bash
python src/training/roundabout/skill.py \
    --exp_name 'saved_model/my_roundabout_training' \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

## Step 4: Monitor Training

While training, you can monitor:

**Training logs:**
```bash
# View latest log
tail -f log/<exp_name>/train.log

# Or check tensorboard (if installed)
tensorboard --logdir log/
```

**Check saved checkpoints:**
```bash
ls -lh saved_model/<exp_name>/ckpt/
```

## Step 5: Evaluate/Visualize Trained Model

After training (or if you have a pretrained model), you can visualize it:

```bash
python src/evaluation/highway/skill.py \
    --exp_name 'saved_model/my_first_training' \
    --ckpt_file ckpt/iteration_50000.pth.tar \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

**Note:** Replace `iteration_50000.pth.tar` with the actual checkpoint file name from `saved_model/<exp_name>/ckpt/`

## Step 6: Collect Expert Data (Optional)

If you want to collect expert demonstrations:

**Rule-based expert:**
```bash
python src/data_collection/highway/rule_expert.py
```

**Visualize rule-based expert:**
```bash
python src/data_collection/highway/rule_expert_visualization.py
```

## Common Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'asaprl'"
**Solution:** Install the package:
```bash
python -m pip install .
```

### Issue: Out of Memory
**Solution:** The memory settings are already reduced. If still having issues:
1. Close other applications
2. Reduce `replay_buffer_size` further in the training script (e.g., to 250)
3. Reduce `batch_size` further (e.g., to 16)

### Issue: Training is slow
**Solution:** This is expected with reduced memory settings. Training may take hours/days depending on your hardware.

### Issue: CUDA/GPU errors
**Solution:** The code uses CUDA by default. If you don't have a GPU, you may need to modify the training scripts to set `cuda=False` in the policy config.

## Quick Reference: File Locations

- **Training scripts:** `src/training/<scenario>/<method>.py`
- **Evaluation scripts:** `src/evaluation/<scenario>/<method>.py`
- **Data collection:** `src/data_collection/<scenario>/`
- **Training logs:** `log/<exp_name>/`
- **Saved models:** `saved_model/<exp_name>/ckpt/`
- **Example commands:** `launch_command/training_commands/`

## Next Steps

1. Start with **Option B (SAC)** - it's the simplest and doesn't require any special parameters
2. Once that works, try **Option A (ASAP-RL)** 
3. Check the logs to see training progress
4. After some training iterations, try evaluating the model

## Example: Complete Training Session

```bash
# 1. Activate environment
conda activate asaprl
cd /Users/maggie/Documents/Projects/asaprl

# 2. Start training (this will run until you stop it with Ctrl+C)
python src/training/highway/sac.py \
    --exp_name 'saved_model/test_run_1'

# 3. In another terminal, monitor the logs
tail -f log/test_run_1/train.log

# 4. After training for a while, stop with Ctrl+C
# 5. Check what checkpoints were saved
ls saved_model/test_run_1/ckpt/

# 6. Evaluate the model (replace with actual checkpoint name)
python src/evaluation/highway/sac.py \
    --exp_name 'saved_model/test_run_1' \
    --ckpt_file ckpt/iteration_1000.pth.tar
```


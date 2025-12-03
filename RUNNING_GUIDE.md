# Running Guide for ASAP-RL

## Quick Start

### 1. Activate the Environment
```bash
conda activate asaprl
cd /path/to/asaprl
```

### 2. Install the Package (if not already done)
```bash
python -m pip install .
```

## How to Run Different Components

### Training

Training scripts are in `src/training/` organized by scenario (highway, intersection, roundabout) and method (skill.py, sac.py, ppo.py, const_sac.py).

**Example: Train ASAP-RL on highway scenario**
```bash
python src/training/highway/skill.py \
    --exp_name 'saved_model/ASPA_NoPrior_fixed10_RV27_lat5_highway_seed1' \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

**Example: Train SAC baseline**
```bash
python src/training/highway/sac.py \
    --exp_name 'saved_model/sac_highway_RV27_sparse_seed1'
```

See `launch_command/training_commands/` for more examples.

### Evaluation/Visualization

Evaluation scripts are in `src/evaluation/` and turn on rendering by default.

**Example: Evaluate a trained model**
```bash
python src/evaluation/highway/skill.py \
    --exp_name 'saved_model/ASPA_NoPrior_fixed10_RV27_lat5_highway_seed1' \
    --ckpt_file ckpt/iteration_50000.pth.tar \
    --traj_mode 'planningfixed' \
    --action_shape 3 \
    --SEQ_TRAJ_LEN 10 \
    --reward_version 27 \
    --lat_range 5
```

### Data Collection

**Collect expert demonstrations:**
```bash
# Rule-based expert
python src/data_collection/highway/rule_expert.py
python src/data_collection/intersection/rule_expert.py
python src/data_collection/roundabout/rule_expert.py

# Visualize rule-based expert
python src/data_collection/highway/rule_expert_visualization.py
```

**Collect data from trained RL agent:**
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

## Reducing Memory Requirements

The paper experiments used **~150GB memory** with these settings:
- 12 parallel collector environments
- 2 parallel evaluator environments  
- Replay buffer size: 100,000
- Samples per collection: 5,000

The code **already has reduced defaults** for limited resources, but you can reduce them further:

### Memory-Related Parameters

These parameters are in the training scripts (e.g., `src/training/highway/skill.py`):

1. **`collector_env_num`** (line ~73)
   - Current: `1` (paper: `12`)
   - **Memory impact**: Each environment uses ~1-2GB
   - **To reduce**: Keep at `1` (minimum)

2. **`evaluator_env_num`** (line ~74)
   - Current: `1` (paper: `2`)
   - **Memory impact**: Each environment uses ~1-2GB
   - **To reduce**: Keep at `1` (minimum)

3. **`replay_buffer_size`** (line ~113)
   - Current: `1000` (paper: `100000`)
   - **Memory impact**: Major - this is the biggest memory consumer
   - **To reduce further**: Try `500` or `250` (may affect learning quality)

4. **`n_sample`** (line ~104)
   - Current: `200` (paper: `5000`)
   - **Memory impact**: Moderate - affects temporary memory during collection
   - **To reduce further**: Try `100` or `50` (will slow down training)

5. **`batch_size`** (line ~94)
   - Current: `128` (for skill.py) or `64` (for sac.py)
   - **Memory impact**: Moderate - affects GPU memory
   - **To reduce**: Try `32` or `16` if you run out of GPU memory

6. **`n_evaluator_episode`** (line ~71)
   - Current: `1` (paper: `10`)
   - **Memory impact**: Low - only during evaluation
   - **To reduce**: Keep at `1` (minimum)

### Example: Ultra-Low Memory Configuration

If you have very limited memory (e.g., <8GB), modify the config in your training script:

```python
collector_env_num=1,        # Keep at minimum
evaluator_env_num=1,        # Keep at minimum
n_evaluator_episode=1,      # Keep at minimum
n_sample=50,                # Reduced from 200
replay_buffer_size=250,     # Reduced from 1000
batch_size=32,              # Reduced from 128
```

### Trade-offs

- **Lower replay buffer size**: May reduce learning stability and sample efficiency
- **Lower n_sample**: Will slow down training (fewer samples collected per iteration)
- **Lower batch_size**: May slow down learning and reduce stability
- **Fewer environments**: Will slow down data collection significantly

### Monitoring Memory Usage

While training, monitor memory usage:
```bash
# On Linux/Mac
watch -n 1 free -h
# or
top

# Check GPU memory (if using CUDA)
nvidia-smi -l 1
```

## Output Locations

- **Training logs**: `log/{exp_name}/`
- **Model checkpoints**: `saved_model/{exp_name}/ckpt/`
- **Expert demonstrations**: `demonstration_RL_expert/{scenario}/`
- **Annotated demonstrations**: `demonstration_RL_expert/{scenario}_annotated/`

## Troubleshooting

### Out of Memory Errors

1. Reduce `replay_buffer_size` first (biggest impact)
2. Reduce `batch_size` if using GPU
3. Reduce `n_sample`
4. Ensure only 1 collector and 1 evaluator environment

### Slow Training

- The reduced memory settings will make training slower
- Consider using a machine with more memory if possible
- Training may take days/weeks with minimal resources

### Import Errors

Make sure you've installed the package:
```bash
conda activate asaprl
python -m pip install .
```


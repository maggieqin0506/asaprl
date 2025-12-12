# Recovery Methods Overview

This folder contains three skill-parameter recovery pipelines. Use this guide to locate the code for each method and its entrypoints.

## 1) Global Optimization Method
- **Entrypoint script:** `main_skill_recovery.py`
- **Core logic:** `skill_param_recovery.py`
  - `recover_parameter(...)` performs SLSQP-based recovery with bounds.
  - `annotate_data` class loads trajectories, runs recovery, and writes annotated pickle files to `demonstration_RL_expert/<scenario>_annotated/`.

## 2) Sliding Window Method
- **Entrypoint script:** `main_skill_recovery_sliding.py`
- **Core logic:** also reuses `recover_parameter(...)` from `skill_param_recovery.py`.
- **Sliding window orchestration:** `sliding_window_recovery(...)` inside `main_skill_recovery_sliding.py` stitches overlapping windows and averages recovered parameters.
- **Output:** annotated pickle files in `demonstration_RL_expert/<scenario>_sliding_annotated/`.

## 3) Fast Method (Local Opt + Smoothing)
- **Entrypoint script:** `main_skill_recovery_fast.py`
- **Core logic:** `skill_param_recovery_fast.py`
  - `recover_parameter_fast(...)` runs lightweight L-BFGS-B per segment, then Gaussian-smooths parameters.
- **Output:** annotated pickle files in `demonstration_RL_expert/<scenario>_fast_annotated/`.

## Comparison Runner
- **Script:** `run_all_methods_and_compare.py`
- Runs the three methods sequentially for a scenario and reports status/errors.

## Supporting Models
- **Motion model and constraints:** `asaprl/policy/planning_model.py`
- Used by all methods for trajectory generation and dynamic constraints.

## Typical invocation examples
```bash
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py --scenario roundabout
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario intersection --window_len 10 --step_size 2
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario roundabout
```


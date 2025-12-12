# Skill Parameter Recovery Methods - File Guide

This directory contains multiple implementations of skill parameter recovery methods. This guide explains what each file does and when to use them.

## Active Recovery Methods

### 1. `skill_param_recovery.py` - Global Optimization (Segment-by-Segment)
**Used by:** `main_skill_recovery.py`

**What it does:**
- Recovers skill parameters segment-by-segment using trajectory-level optimization
- Uses trajectory shape distance (not point-wise) for better matching
- Includes smoothness penalties for smoother trajectories
- Called "Global Optimization" in comparisons, but actually processes segments sequentially

**Output:** `demonstration_RL_expert/{scenario}_annotated/`

**Key features:**
- Trajectory-level cost function (considers overall shape)
- Smoothness penalties
- Handles both RL expert and rule expert data formats

---

### 2. `skill_param_recovery_fast.py` - Fast Method
**Used by:** `main_skill_recovery_fast.py`

**What it does:**
- Fast recovery using local optimization + Gaussian smoothing
- Optimizes each segment independently (very fast)
- Applies Gaussian smoothing to parameters for continuity
- Trade-off: Speed vs. accuracy

**Output:** `demonstration_RL_expert/{scenario}_fast_annotated/`

**Key features:**
- Local optimization per segment
- Gaussian smoothing (sigma=2.0) for parameter continuity
- Much faster than other methods
- May sacrifice some accuracy for speed

---

### 3. `skill_param_recovery_sliding.py` - Sliding Window Method
**Used by:** `main_skill_recovery_sliding.py`

**What it does:**
- Uses sliding windows with overlapping segments
- Averages parameters from multiple windows for smoothness
- Better continuity than segment-by-segment methods
- Uses trajectory-level optimization from `skill_param_recovery.py` for each window

**Output:** `demonstration_RL_expert/{scenario}_sliding_annotated/`

**Key features:**
- Overlapping windows with configurable step size
- Parameter averaging across windows
- Reuses `recover_parameter` from `skill_param_recovery.py` for core optimization
- Better segment continuity than global method

---

## Backup/Utility Files

### 4. `skill_param_recovery_old.py` - Old Point-wise Method
**Used by:** `compare_code_versions.py` (for comparison only)

**What it does:**
- Old implementation using point-wise optimization
- Used for comparing old vs new code implementations
- Not used in production

**Purpose:** Allows comparison between old (point-wise) and new (trajectory-level) optimization approaches

---

### 5. `skill_param_recovery_new_backup.py` - Backup of New Version
**Created by:** `compare_code_versions.py`

**What it does:**
- Automatic backup created when comparing code versions
- Temporary file for version comparison workflow
- Can be safely deleted if not actively comparing versions

---

## Naming Confusion Explained

### Why is it called "Global Optimization"?
The method in `skill_param_recovery.py` is called "Global Optimization" in comparisons, but it actually processes segments **one at a time** (segment-by-segment). The name refers to:
- **Global trajectory-level optimization** (optimizing entire trajectory shape, not individual points)
- Not "global" in the sense of optimizing all segments simultaneously

### What about `skill_param_recovery_global.py`?
This file was a true global optimization (all segments at once) but was **never used** and has been removed. It was redundant.

---

## Quick Reference: Which Method to Use?

| Method | Speed | Accuracy | Smoothness | Use Case |
|--------|-------|----------|------------|----------|
| **Global** (`skill_param_recovery.py`) | Medium | High | High | Best overall quality |
| **Sliding Window** | Medium | High | Very High | Best continuity |
| **Fast** (`skill_param_recovery_fast.py`) | Very Fast | Medium | Medium | Quick testing/prototyping |

---

## File Locations: Where Each Method Lives

### 1. Global Optimization Method

**Core Implementation:**
- `skill_param_recovery.py` - Contains the `annotate_data` class with trajectory-level optimization

**Entry Point:**
- `main_skill_recovery.py` - Imports and calls `annotate_data` from `skill_param_recovery.py`

**How to run:**
```bash
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
```

---

### 2. Sliding Window Method

**Core Implementation:**
- `skill_param_recovery_sliding.py` - Contains the `annotate_data` class and `sliding_window_recovery()` function
- Uses `recover_parameter` and `transform_planning_param_to_latentvar` from `skill_param_recovery.py` for core optimization

**Entry Point:**
- `main_skill_recovery_sliding.py` - Imports and calls `annotate_data` from `skill_param_recovery_sliding.py`

**How to run:**
```bash
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py \
    --scenario highway \
    --window_len 10 \
    --step_size 2
```

---

### 3. Fast Method

**Core Implementation:**
- `skill_param_recovery_fast.py` - Contains the `annotate_data` class with `recover_parameter_fast()` (local optimization + Gaussian smoothing)

**Entry Point:**
- `main_skill_recovery_fast.py` - Imports and calls `annotate_data` from `skill_param_recovery_fast.py`

**How to run:**
```bash
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario highway
```

---

## File Usage Summary

```
main_skill_recovery.py
  └─> skill_param_recovery.py (Global method - trajectory-level optimization)

main_skill_recovery_sliding.py
  └─> skill_param_recovery_sliding.py (Sliding window method)
      └─> skill_param_recovery.py (uses recover_parameter function)

main_skill_recovery_fast.py
  └─> skill_param_recovery_fast.py (Fast method - local opt + smoothing)

compare_code_versions.py
  ├─> skill_param_recovery_old.py (old point-wise version)
  └─> skill_param_recovery_new_backup.py (backup of new version)
```

## Quick Reference Table

| Method | Core File | Entry Script | Key Function/Class |
|--------|----------|--------------|-------------------|
| **Global** | `skill_param_recovery.py` | `main_skill_recovery.py` | `annotate_data` class |
| **Sliding Window** | `skill_param_recovery_sliding.py` | `main_skill_recovery_sliding.py` | `annotate_data` class |
| **Fast** | `skill_param_recovery_fast.py` | `main_skill_recovery_fast.py` | `annotate_data` class |

---

## Recommendations

1. **For production use:** Use `skill_param_recovery.py` (Global) or Sliding Window method
2. **For quick testing:** Use `skill_param_recovery_fast.py`
3. **For comparing methods:** Use `run_all_methods_and_compare.py` which runs all three
4. **Backup files:** Can be ignored unless actively comparing code versions

---

## Future Improvements

Consider renaming for clarity:
- `skill_param_recovery.py` → `skill_param_recovery_trajectory.py` (more descriptive)
- Or keep current name but document that "Global" refers to trajectory-level, not all-segments

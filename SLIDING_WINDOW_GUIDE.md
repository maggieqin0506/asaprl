# Sliding Window Skill Recovery Guide

This guide explains how to use the **Sliding Window** script to generate smoother skill parameters without modifying the core optimization logic in `skill_param_recovery.py`.

## Overview

The standard recovery method optimizes segments independently, which can cause discontinuities in the skill parameters at segment boundaries.

The **Sliding Window** method solves this by:
1. Reconstructing the continuous trajectory from the segmented data.
2. Running your existing optimization function (`recover_parameter`) on **overlapping** segments.
3. Averaging the recovered parameters in the regions of overlap.

**Benefit:** Significantly smoother skill parameter transitions between segments.

## Usage

### 1. Run Sliding Window Recovery

Run the new script from the `skill_recovery_and_prior_training` directory.

```bash
cd src/skill_recovery_and_prior_training

# Run with default settings (Window=10 steps, Stride=2 steps)
python main_skill_recovery_sliding.py --scenario highway --window_len 10 --step_size 2
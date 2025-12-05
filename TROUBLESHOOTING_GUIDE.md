# Troubleshooting Guide: Testing and Handling Freezes

## Testing with One File

### Option 1: Use `--max_files` Flag (Recommended)

The easiest way to test with just one file:

```bash
# Test comparison with just 1 file
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 1
```

Or if using the unified script:

```bash
# Test entire pipeline with 1 file
python src/skill_recovery_and_prior_training/run_all_methods_and_compare.py \
    --scenario highway \
    --max_files 1
```

### Option 2: Temporarily Move Files

If you want to test with specific files:

```bash
# Create a test directory
mkdir -p demonstration_RL_expert/highway_annotated_test

# Copy just one file
cp demonstration_RL_expert/highway_annotated/highway_expert_data_1.pickle \
   demonstration_RL_expert/highway_annotated_test/

# Run comparison on test directory
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --global_data_path demonstration_RL_expert/highway_annotated_test
```

### Option 3: Test Individual Recovery Methods First

Test each recovery method with one file before running comparison:

```bash
# Test Global method with 1 file
python src/skill_recovery_and_prior_training/main_skill_recovery.py \
    --scenario highway \
    --max_files 1

# Test Sliding method with 1 file
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py \
    --scenario highway \
    --max_files 1

# Test Fast method with 1 file
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py \
    --scenario highway \
    --max_files 1

# Then compare (will use the 1 file from each method)
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 1
```

## What to Do If It Freezes

### Step 1: Check What's Happening

1. **Check if it's actually frozen or just slow:**
   - Look at the terminal output - is there a progress bar?
   - Check if CPU/GPU usage is high (it might be working)
   - Wait a few minutes - processing many files can take time

2. **Check which stage it's at:**
   - "Evaluating Global Optimization method..." - Processing files
   - "Generating comparison graphs..." - Creating plots
   - If stuck on a specific file, note the filename

### Step 2: Interrupt and Debug

If it's truly frozen:

1. **Interrupt the process:**
   ```bash
   # Press Ctrl+C to stop
   ```

2. **Check for problematic files:**
   ```bash
   # Check file sizes - very large files might cause issues
   ls -lh demonstration_RL_expert/highway_annotated/*.pickle
   
   # Check if any files are corrupted
   python -c "import pickle; pickle.load(open('demonstration_RL_expert/highway_annotated/highway_expert_data_1.pickle', 'rb'))"
   ```

3. **Test with fewer files:**
   ```bash
   # Start with 1 file
   python src/skill_recovery_and_prior_training/compare_all_methods.py \
       --scenario highway \
       --max_files 1
   
   # If that works, try 5 files
   python src/skill_recovery_and_prior_training/compare_all_methods.py \
       --scenario highway \
       --max_files 5
   
   # Gradually increase
   ```

### Step 3: Common Causes and Solutions

#### Cause 1: Memory Issues (Large Files)

**Symptoms:**
- Process uses a lot of RAM
- System becomes slow
- Eventually crashes or freezes

**Solution:**
```bash
# Process files in smaller batches
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 10  # Process 10 at a time
```

#### Cause 2: Corrupted File

**Symptoms:**
- Stuck on a specific file
- Error message about pickle file

**Solution:**
```bash
# Find and remove problematic file
# Test each file individually
for file in demonstration_RL_expert/highway_annotated/*.pickle; do
    echo "Testing $file"
    python -c "import pickle; pickle.load(open('$file', 'rb'))" || echo "ERROR: $file"
done
```

#### Cause 3: Graph Generation Issues

**Symptoms:**
- Stuck at "Generating comparison graphs..."
- Matplotlib/display issues

**Solution:**
```bash
# Skip graphs to test if that's the issue
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --no_graphs

# Or check matplotlib backend
export MPLBACKEND=Agg  # Use non-interactive backend
```

#### Cause 4: Different Number of Files Across Methods

**Symptoms:**
- Warning about different trajectory counts
- Script hangs during trajectory alignment

**Solution:**
```bash
# Check file counts
ls demonstration_RL_expert/highway_annotated/*.pickle | wc -l
ls demonstration_RL_expert/highway_sliding_annotated/*.pickle | wc -l
ls demonstration_RL_expert/highway_fast_annotated/*.pickle | wc -l

# Re-run methods to ensure same number of files
python src/skill_recovery_and_prior_training/main_skill_recovery.py --scenario highway
python src/skill_recovery_and_prior_training/main_skill_recovery_sliding.py --scenario highway
python src/skill_recovery_and_prior_training/main_skill_recovery_fast.py --scenario highway
```

### Step 4: Resume from Where You Left Off

If you've already processed some files and want to continue:

1. **Check what's already done:**
   ```bash
   # See which files have been processed
   ls demonstration_RL_expert/highway_annotated/*.pickle
   ```

2. **The script processes files in sorted order**, so if it stopped at file 20, files 1-19 are already processed.

3. **You can't resume mid-comparison**, but you can:
   - Process remaining files separately
   - Or just re-run (it will re-process everything, but that's usually fine)

## Quick Test Workflow

For quick testing, use this workflow:

```bash
# 1. Test with 1 file first
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 1

# 2. If successful, test with 5 files
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 5

# 3. If that works, run full comparison
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway
```

## Monitoring Progress

To see progress in real-time:

```bash
# Run with verbose output (if available)
python src/skill_recovery_and_prior_training/compare_all_methods.py \
    --scenario highway \
    --max_files 10

# Or monitor system resources
# In another terminal:
top  # or htop if available
```

## Expected Processing Times

Approximate times per file:
- **Recovery method**: 1-5 seconds per file
- **Comparison evaluation**: 0.5-2 seconds per file
- **Graph generation**: 5-30 seconds total (regardless of file count)

So for 38 files:
- Recovery: ~1-3 minutes per method
- Comparison: ~20-80 seconds
- Graphs: ~5-30 seconds
- **Total: ~5-10 minutes** (if all methods already run)

If recovery methods haven't run yet, add ~3-9 minutes for those.

## If Nothing Works

1. **Check Python environment:**
   ```bash
   python --version
   which python
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check disk space:**
   ```bash
   df -h
   ```

4. **Check for error logs:**
   ```bash
   # Look for any error messages in terminal
   # Check log files if they exist
   ls log/
   ```

5. **Try minimal test:**
   ```bash
   # Test with absolute minimum
   python src/skill_recovery_and_prior_training/compare_all_methods.py \
       --scenario highway \
       --max_files 1 \
       --no_graphs
   ```



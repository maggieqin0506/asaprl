#!/bin/bash
# Script to create a test subset of demonstration data for faster evaluation

SCENARIO=${1:-highway}
NUM_FILES=${2:-3}

SOURCE_DIR="./demonstration_RL_expert/${SCENARIO}/"
TEST_DIR="./demonstration_RL_expert/${SCENARIO}_test/"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist!"
    exit 1
fi

# Create test directory
mkdir -p "$TEST_DIR"

# Copy first N files
echo "Creating test subset with $NUM_FILES files from $SCENARIO scenario..."
ls -1 "$SOURCE_DIR"/*.pickle 2>/dev/null | head -$NUM_FILES | while read file; do
    filename=$(basename "$file")
    cp "$file" "$TEST_DIR/$filename"
    echo "  Copied: $filename"
done

echo ""
echo "Test subset created in: $TEST_DIR"
echo "Files copied: $(ls -1 "$TEST_DIR"/*.pickle 2>/dev/null | wc -l)"
echo ""
echo "To use this subset, run:"
echo "  python src/skill_recovery_and_prior_training/evaluate_recovery_improvement.py --scenario $SCENARIO --data_path $TEST_DIR"


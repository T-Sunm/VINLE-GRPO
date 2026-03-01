#!/bin/bash
# Evaluate GRPO: outputs/inference/grpo/
set -e

# Edit filename here:
FILES="v2-20251229-175222.json"

# Use args if provided
[ $# -gt 0 ] && FILENAMES="$@" || FILENAMES="$FILES"

if [ -z "$FILENAMES" ]; then
    echo "Evaluating all GRPO results..."
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/grpo --device cuda:0
else
    echo "Evaluating GRPO: $FILENAMES"
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/grpo --filenames $FILENAMES --device cuda:0
fi


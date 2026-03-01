#!/bin/bash
# Evaluate ZEROSHOT: outputs/inference/zeroshot/
set -e

# Edit filename here:
FILES="v2-20251229-175414.json"

# Use args if provided
[ $# -gt 0 ] && FILENAMES="$@" || FILENAMES="$FILES"

if [ -z "$FILENAMES" ]; then
    echo "Evaluating all ZEROSHOT results..."
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/zeroshot --device cuda:0
else
    echo "Evaluating ZEROSHOT: $FILENAMES"
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/zeroshot --filenames $FILENAMES --device cuda:0
fi


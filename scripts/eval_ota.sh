#!/bin/bash
# Evaluate OTA: outputs/inference/ota/
set -e

# Edit filename here:
FILES="v2-20251229-175311.json"

# Use args if provided
[ $# -gt 0 ] && FILENAMES="$@" || FILENAMES="$FILES"

if [ -z "$FILENAMES" ]; then
    echo "Evaluating all OTA results..."
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/ota --device cuda:0
else
    echo "Evaluating OTA: $FILENAMES"
    python3 -m src.evaluation.calculate_scores --input-dir outputs/inference/ota --filenames $FILENAMES --device cuda:0
fi


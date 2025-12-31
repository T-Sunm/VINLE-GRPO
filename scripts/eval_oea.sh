#!/bin/bash
# Evaluate OEA: outputs/inference/oea/
set -e

# Edit filename here:
FILES="v2-20251229-175350.json"

# Use args if provided
[ $# -gt 0 ] && FILENAMES="$@" || FILENAMES="$FILES"

if [ -z "$FILENAMES" ]; then
    echo "Evaluating all OEA results..."
    python -m src.evaluation.calculate_scores --input-dir outputs/inference/oea --device cuda:0
else
    echo "Evaluating OEA: $FILENAMES"
    python -m src.evaluation.calculate_scores --input-dir outputs/inference/oea --filenames $FILENAMES --device cuda:0
fi

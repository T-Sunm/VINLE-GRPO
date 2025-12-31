#!/bin/bash
# Evaluate all formats recursively
set -e

# Edit output filename here:
OUTPUT_FILE="${1:-all_formats_results.csv}"

echo "Evaluating all formats → $OUTPUT_FILE"
python3 -m src.evaluation.calculate_scores \
    --input-dir outputs/inference \
    --recursive \
    --output-file "$OUTPUT_FILE" \
    --device cuda:0


#!/bin/bash
# Run Zero-shot Inference
# Usage: bash scripts/inference/run_zeroshot.sh [NUM_SAMPLES|all]

set -e
cd "$(dirname "$0")/../.."

ARG1=${1:-3}
if [ "$ARG1" == "all" ] || [ "$ARG1" == "full" ]; then
    NUM_SAMPLES=0
else
    NUM_SAMPLES=$ARG1
fi

MODEL="5CD-AI/Vintern-3B-R-beta"
OUTPUT_DIR="outputs/inference/zeroshot"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUTPUT_DIR"
MAX_VER=$(find "$OUTPUT_DIR" -name "v*.json" 2>/dev/null | grep -o 'v[0-9]*' | sed 's/v//' | sort -n | tail -1)
VERSION=$((${MAX_VER:-0} + 1))

if [ "$NUM_SAMPLES" -eq 0 ]; then
    SAMPLE_STR="all"
else
    SAMPLE_STR="$NUM_SAMPLES"
fi
OUTPUT_FILE="v${VERSION}-${TIMESTAMP}-samples${SAMPLE_STR}"

echo "Running Zero-shot Inference"
echo "  Model:   $MODEL"
echo "  Output:  $OUTPUT_DIR/$OUTPUT_FILE"
echo "  Samples: ${SAMPLE_STR}"

# Use project-relative paths (fixed in zeroshot.py)
python -m src.inference.internvl_based.zero_shot \
    --model "$MODEL" \
    --limit "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_FILE"

echo "✅ Done: $OUTPUT_DIR/$OUTPUT_FILE"

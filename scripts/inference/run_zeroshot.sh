#!/bin/bash
# Run Zero-shot Inference
# Usage: bash scripts/inference/run_zeroshot.sh [NUM_SAMPLES]

set -e
cd "$(dirname "$0")/../.."

NUM_SAMPLES=${1:-3}
MODEL="OpenGVLab/InternVL3_5-2B"
OUTPUT_DIR="outputs/inference/zeroshot"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUTPUT_DIR"
MAX_VER=$(find "$OUTPUT_DIR" -name "v*.json" 2>/dev/null | sed 's/.*v\([0-9]*\)-.*/\1/' | sort -n | tail -1)
VERSION=$((${MAX_VER:-0} + 1))
OUTPUT_FILE="v${VERSION}-${TIMESTAMP}"

echo "Running Zero-shot Inference"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR/$OUTPUT_FILE"
echo "  Samples: $NUM_SAMPLES"

python -m src.inference.internvl_based.zero_shot \
    --model "$MODEL" \
    --limit "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_FILE"

echo "✅ Done: $OUTPUT_DIR/$OUTPUT_FILE"

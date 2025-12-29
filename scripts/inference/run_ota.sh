#!/bin/bash
# Run OTA Ablation Inference
# Usage: bash scripts/inference/run_ota.sh [NUM_SAMPLES]

set -e
cd "$(dirname "$0")/../.."

NUM_SAMPLES=${1:-3}
MODEL="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/only_think_answer/v2-20251220-212353/checkpoint-1000-merged"
OUTPUT_DIR="outputs/inference/ota"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUTPUT_DIR"
MAX_VER=$(find "$OUTPUT_DIR" -name "v*.json" 2>/dev/null | sed 's/.*v\([0-9]*\)-.*/\1/' | sort -n | tail -1)
VERSION=$((${MAX_VER:-0} + 1))
OUTPUT_FILE="v${VERSION}-${TIMESTAMP}"

echo "Running OTA Inference"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR/$OUTPUT_FILE"
echo "  Samples: $NUM_SAMPLES"

python -m src.inference.internvl_based.ota \
    --model "$MODEL" \
    --limit "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_FILE"

echo "✅ Done: $OUTPUT_DIR/$OUTPUT_FILE"

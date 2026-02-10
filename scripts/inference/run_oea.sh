#!/bin/bash
# Run OEA Ablation Inference
# Usage: bash scripts/inference/run_oea.sh [NUM_SAMPLES]

set -e
cd "$(dirname "$0")/../.."

NUM_SAMPLES=${1:-3}
# TODO: Update MODEL to your trained & merged checkpoint path
MODEL="${OEA_MODEL_PATH:-outputs/training/grpo/ablation_oea/checkpoint-merged}"
OUTPUT_DIR="outputs/inference/oea"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUTPUT_DIR"
MAX_VER=$(find "$OUTPUT_DIR" -name "v*.json" 2>/dev/null | sed 's/.*v\([0-9]*\)-.*/\1/' | sort -n | tail -1)
VERSION=$((${MAX_VER:-0} + 1))
OUTPUT_FILE="v${VERSION}-${TIMESTAMP}"

echo "Running OEA Inference"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR/$OUTPUT_FILE"
echo "  Samples: $NUM_SAMPLES"

python -m src.inference.internvl_based.oea \
    --model "$MODEL" \
    --limit "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_FILE"

echo "✅ Done: $OUTPUT_DIR/$OUTPUT_FILE"

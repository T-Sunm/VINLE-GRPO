#!/bin/bash
# Merge SFT LoRA Weights Script
# Usage: bash merge_lora.sh <checkpoint_path> [output_dir]

set -e

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${VINLE_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_PATH="${1:-}"
OUTPUT_DIR="${2:-}"

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Available SFT checkpoints:"
    find "$PROJECT_ROOT/outputs/training/sft" -name "checkpoint-*" -type d 2>/dev/null || echo "  (none found)"
    echo ""
    echo "ERROR: Please provide checkpoint path."
    echo "Usage: bash merge_lora.sh <checkpoint_path> [output_dir]"
    exit 1
fi

CMD="swift export \
    --use_hf true \
    --model_type \"internvl3\" \
    --model \"OpenGVLab/InternVL3_5-2B\" \
    --ckpt_dir \"$CHECKPOINT_PATH\" \
    --merge_lora true"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

echo "Merging LoRA from: $CHECKPOINT_PATH"
eval $CMD

echo "Hoàn thành merge LoRA"

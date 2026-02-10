#!/bin/bash
# Merge LoRA Weights Script
# 
# Usage: bash merge_lora.sh [checkpoint_path]
# Example: bash merge_lora.sh /path/to/outputs/training/grpo/vinle_full/v0-xxx/checkpoint-200

set -e

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
PROJECT_ROOT="${VINLE_PROJECT_ROOT:-$AUTO_PROJECT_ROOT}"
export PROJECT_ROOT

echo "Project Root: $PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1

# === Configuration ===
BASE_MODEL="OpenGVLab/InternVL3_5-2B"
MODEL_TYPE="internvl3"

# Get checkpoint path from argument or prompt
CHECKPOINT_PATH="${1:-}"
if [ -z "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "Available checkpoints:"
    find "$PROJECT_ROOT/outputs/training" -name "checkpoint-*" -type d 2>/dev/null || echo "  (none found)"
    echo ""
    echo "ERROR: Please provide checkpoint path as argument."
    echo "Usage: bash merge_lora.sh <checkpoint_path>"
    exit 1
fi

echo "Merging LoRA from: $CHECKPOINT_PATH"
echo ""

swift export \
    --use_hf true \
    --model_type "$MODEL_TYPE" \
    --model "$BASE_MODEL" \
    --ckpt_dir "$CHECKPOINT_PATH" \
    --merge_lora true
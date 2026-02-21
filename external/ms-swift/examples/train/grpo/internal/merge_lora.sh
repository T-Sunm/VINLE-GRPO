#!/bin/bash
# Merge LoRA Weights Script
# 
# Usage: bash merge_lora.sh [checkpoint_path]
# Example: bash merge_lora.sh outputs/training/grpo/vinle_full/v0-xxx/checkpoint-4000

set -e

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
PROJECT_ROOT="${VINLE_PROJECT_ROOT:-$AUTO_PROJECT_ROOT}"
export PROJECT_ROOT

echo "Project Root: $PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1

# === Get checkpoint path from argument ===
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

# Convert to absolute path if relative
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="$PROJECT_ROOT/$CHECKPOINT_PATH"
fi

echo "Merging LoRA from: $CHECKPOINT_PATH"
echo ""

# === Auto-detect BASE_MODEL from adapter_config.json ===
ADAPTER_CONFIG="$CHECKPOINT_PATH/adapter_config.json"
if [ ! -f "$ADAPTER_CONFIG" ]; then
    echo "ERROR: adapter_config.json not found in $CHECKPOINT_PATH"
    exit 1
fi

BASE_MODEL=$(python3 -c "import json; print(json.load(open('$ADAPTER_CONFIG'))['base_model_name_or_path'])")
echo "Auto-detected base model: $BASE_MODEL"

# Auto-detect MODEL_TYPE based on base model name
if echo "$BASE_MODEL" | grep -qi "vintern\|InternVL"; then
    MODEL_TYPE="internvl3"
else
    MODEL_TYPE="internvl3"  # default fallback
fi
echo "Model type: $MODEL_TYPE"
echo ""

swift export \
    --use_hf true \
    --model_type "$MODEL_TYPE" \
    --model "$BASE_MODEL" \
    --ckpt_dir "$CHECKPOINT_PATH" \
    --merge_lora true
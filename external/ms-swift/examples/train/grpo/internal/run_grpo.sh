#!/bin/bash
# GRPO Training Script - Reads from YAML config
# 
# Usage (run from project root):
#   bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/vinle_full.yaml
#   bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/ablation_think_answer.yaml

set -e

# ============================================================
# Auto-detect PROJECT_ROOT
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
PROJECT_ROOT="${VINLE_PROJECT_ROOT:-$AUTO_PROJECT_ROOT}"
export PROJECT_ROOT

if [ ! -f "$PROJECT_ROOT/README.md" ] || [ ! -d "$PROJECT_ROOT/configs" ]; then
    echo "ERROR: Could not detect project root. Set VINLE_PROJECT_ROOT env var."
    echo "  Detected: $PROJECT_ROOT"
    exit 1
fi
echo "Project Root: $PROJECT_ROOT"

# ============================================================
# Config file (required)
# ============================================================
CONFIG_FILE="${1:-}"
if [ -z "$CONFIG_FILE" ]; then
    echo ""
    echo "ERROR: Config file is required."
    echo "Usage: bash run_grpo.sh <config.yaml>"
    echo ""
    echo "Available configs:"
    find "$PROJECT_ROOT/configs/grpo" -name "*.yaml" 2>/dev/null | sort
    exit 1
fi

echo "Loading config: $CONFIG_FILE"
echo ""

# ============================================================
# Parse YAML config
# ============================================================
eval $(python3 - <<EOF
import yaml, os, sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        content = f.read()
    content = content.replace('\${PROJECT_ROOT}', os.environ.get('PROJECT_ROOT', ''))
    config = yaml.safe_load(content)
    
    def v(val, default):
        """Return default if val is None."""
        return default if val is None else val

    # Environment
    env = config.get('environment', {})
    print(f"export CUDA_VISIBLE_DEVICES='{v(env.get('cuda_visible_devices'), '0')}'")
    print(f"export PYTORCH_CUDA_ALLOC_CONF='{v(env.get('pytorch_cuda_alloc_conf'), 'expandable_segments:True')}'")
    print(f"export HF_ENDPOINT='{v(env.get('hf_endpoint'), 'https://huggingface.co')}'")

    # Model
    model = config.get('model', {})
    print(f"MODEL_TYPE='{v(model.get('type'), 'internvl3')}'")
    print(f"MODEL_PATH='{v(model.get('id_or_path'), 'OpenGVLab/InternVL3_5-2B')}'")

    # Data
    data = config.get('data', {})
    print(f"TRAIN_DATASET='{v(data.get('train_dataset'), '')}'")
    print(f"MAX_LENGTH={v(data.get('max_length'), 4096)}")
    print(f"MAX_COMPLETION_LENGTH={v(data.get('max_completion_length'), 1024)}")

    # Rewards
    rewards = config.get('rewards', {})
    print(f"PLUGIN_PATH='{v(rewards.get('plugin_path'), '')}'")
    print(f"REWARD_FUNCS='{' '.join(rewards.get('functions', []))}'")

    # GRPO
    grpo = config.get('grpo', {})
    print(f"NUM_GENERATIONS={v(grpo.get('num_generations'), 4)}")
    print(f"TEMPERATURE={v(grpo.get('temperature'), 0.9)}")
    print(f"TOP_P={v(grpo.get('top_p'), 0.9)}")
    print(f"TOP_K={v(grpo.get('top_k'), 50)}")
    print(f"BETA={v(grpo.get('beta'), 0.04)}")

    # Training
    train = config.get('training', {})
    print(f"TRAIN_TYPE='{v(train.get('train_type'), 'lora')}'")
    print(f"LORA_RANK={v(train.get('lora_rank'), 32)}")
    print(f"LORA_ALPHA={v(train.get('lora_alpha'), 64)}")
    print(f"TARGET_MODULES='{v(train.get('target_modules'), 'all-linear')}'")
    print(f"FREEZE_VIT={v(train.get('freeze_vit'), True)}")
    print(f"NUM_EPOCHS={v(train.get('num_train_epochs'), 2)}")
    max_steps = train.get('max_steps')
    print(f"MAX_STEPS={max_steps if max_steps is not None else ''}")
    print(f"BATCH_SIZE={v(train.get('per_device_train_batch_size'), 2)}")
    print(f"EVAL_BATCH_SIZE={v(train.get('per_device_eval_batch_size'), 4)}")
    print(f"GRAD_ACCUM={v(train.get('gradient_accumulation_steps'), 4)}")
    print(f"LR={v(train.get('learning_rate'), 1e-5)}")
    print(f"WARMUP_RATIO={v(train.get('warmup_ratio'), 0.05)}")
    print(f"SAVE_STEPS={v(train.get('save_steps'), 50)}")
    print(f"LOGGING_STEPS={v(train.get('logging_steps'), 1)}")
    print(f"EVAL_STEPS={v(train.get('eval_steps'), 1)}")
    print(f"SAVE_LIMIT={v(train.get('save_total_limit'), 2)}")
    print(f"TORCH_DTYPE='{v(train.get('torch_dtype'), 'bfloat16')}'")
    print(f"ATTN_IMPL='{v(train.get('attn_impl'), 'flash_attention_2')}'")
    print(f"WORKERS={v(train.get('dataloader_num_workers'), 16)}")
    print(f"DATASET_PROC={v(train.get('dataset_num_proc'), 16)}")
    print(f"QUANT_METHOD='{v(train.get('quant_method'), 'bnb')}'")
    print(f"QUANT_BITS={v(train.get('quant_bits'), 4)}")

    # Output
    output = config.get('output', {})
    print(f"OUTPUT_DIR='{v(output.get('dir'), '')}'")
    resume = output.get('resume_from_checkpoint')
    print(f"RESUME_CHECKPOINT='{resume if resume else ''}'")

except Exception as e:
    print(f"echo 'Error parsing YAML: {e}'", file=sys.stderr)
    sys.exit(1)
EOF
)

# ============================================================
# Print summary
# ============================================================
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Model:    $MODEL_TYPE - $MODEL_PATH"
echo "Dataset:  $TRAIN_DATASET"
echo "Output:   $OUTPUT_DIR"
echo "Rewards:  $REWARD_FUNCS"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Steps:    ${MAX_STEPS:-all epochs ($NUM_EPOCHS)}"
echo "=========================================="
echo ""

# ============================================================
# Build and run swift command
# ============================================================
CMD="swift rlhf \
    --rlhf_type grpo \
    --model_type $MODEL_TYPE \
    --model $MODEL_PATH \
    --use_vllm false \
    --attn_impl $ATTN_IMPL \
    --use_hf true \
    --dataset $TRAIN_DATASET \
    --external_plugins $PLUGIN_PATH \
    --reward_funcs $REWARD_FUNCS \
    --train_type $TRAIN_TYPE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --target_modules $TARGET_MODULES \
    --freeze_vit $FREEZE_VIT \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --beta $BETA \
    --log_completions true \
    --torch_dtype $TORCH_DTYPE \
    --save_only_model false \
    --save_total_limit $SAVE_LIMIT \
    --warmup_ratio $WARMUP_RATIO \
    --dataloader_num_workers $WORKERS \
    --dataset_num_proc $DATASET_PROC \
    --report_to wandb \
    --quant_method $QUANT_METHOD \
    --quant_bits $QUANT_BITS \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true"

# Optional args
[ -n "$MAX_STEPS" ] && CMD="$CMD --max_steps $MAX_STEPS"
[ -n "$RESUME_CHECKPOINT" ] && CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"

echo "Starting GRPO training..."
echo ""
eval $CMD

#!/bin/bash
# GRPO Training Script - Reads from YAML config
# 
# Usage:
#   bash run_grpo.sh [config_file]
#   bash run_grpo.sh ../../../../configs/grpo/vinle_full.yaml
#   bash run_grpo.sh ../../../../configs/grpo/ablation_think_answer.yaml
#
# If no config specified, uses default paths below

set -e

# Get config file from argument or use inline defaults
CONFIG_FILE="${1:-}"

if [ -n "$CONFIG_FILE" ]; then
    echo "=========================================="
    echo "Loading config from: $CONFIG_FILE"
    echo "=========================================="
    
    # Parse YAML using Python (simpler than yq)
    eval $(python3 - <<EOF
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment
    env = config.get('environment', {})
    print(f"export CUDA_VISIBLE_DEVICES='{env.get('cuda_visible_devices', '2')}'")
    print(f"export PYTORCH_CUDA_ALLOC_CONF='{env.get('pytorch_cuda_alloc_conf', 'expandable_segments:True')}'")
    print(f"export HF_ENDPOINT='{env.get('hf_endpoint', 'https://huggingface.co')}'")
    
    # Model
    model = config.get('model', {})
    print(f"MODEL_TYPE='{model.get('type', 'internvl3')}'")
    print(f"MODEL_PATH='{model.get('id_or_path', 'OpenGVLab/InternVL3_5-2B')}'")
    
    # Data
    data = config.get('data', {})
    print(f"TRAIN_DATASET='{data.get('train_dataset', '')}'")
    print(f"MAX_LENGTH={data.get('max_length', 4096)}")
    print(f"MAX_COMPLETION_LENGTH={data.get('max_completion_length', 1024)}")
    
    # Rewards
    rewards = config.get('rewards', {})
    print(f"PLUGIN_PATH='{rewards.get('plugin_path', '')}'")
    reward_funcs = ' '.join(rewards.get('functions', []))
    print(f"REWARD_FUNCS='{reward_funcs}'")
    
    # GRPO
    grpo = config.get('grpo', {})
    print(f"NUM_GENERATIONS={grpo.get('num_generations', 4)}")
    print(f"TEMPERATURE={grpo.get('temperature', 0.9)}")
    print(f"TOP_P={grpo.get('top_p', 0.9)}")
    print(f"TOP_K={grpo.get('top_k', 50)}")
    print(f"BETA={grpo.get('beta', 0.04)}")
    
    # Training
    train = config.get('training', {})
    print(f"TRAIN_TYPE='{train.get('train_type', 'lora')}'")
    print(f"LORA_RANK={train.get('lora_rank', 32)}")
    print(f"LORA_ALPHA={train.get('lora_alpha', 64)}")
    print(f"TARGET_MODULES='{train.get('target_modules', 'all-linear')}'")
    print(f"FREEZE_VIT={train.get('freeze_vit', True)}")
    print(f"NUM_EPOCHS={train.get('num_train_epochs', 2)}")
    print(f"MAX_STEPS={train.get('max_steps', 2000)}")
    print(f"BATCH_SIZE={train.get('per_device_train_batch_size', 2)}")
    print(f"EVAL_BATCH_SIZE={train.get('per_device_eval_batch_size', 4)}")
    print(f"GRAD_ACCUM={train.get('gradient_accumulation_steps', 4)}")
    print(f"LR={train.get('learning_rate', 1e-5)}")
    print(f"WARMUP_RATIO={train.get('warmup_ratio', 0.05)}")
    print(f"SAVE_STEPS={train.get('save_steps', 50)}")
    print(f"LOGGING_STEPS={train.get('logging_steps', 1)}")
    print(f"EVAL_STEPS={train.get('eval_steps', 1)}")
    print(f"SAVE_LIMIT={train.get('save_total_limit', 2)}")
    print(f"TORCH_DTYPE='{train.get('torch_dtype', 'bfloat16')}'")
    print(f"ATTN_IMPL='{train.get('attn_impl', 'flash_attention_2')}'")
    print(f"WORKERS={train.get('dataloader_num_workers', 16)}")
    print(f"DATASET_PROC={train.get('dataset_num_proc', 16)}")
    print(f"QUANT_METHOD='{train.get('quant_method', 'bnb')}'")
    print(f"QUANT_BITS={train.get('quant_bits', 4)}")
    
    # Output
    output = config.get('output', {})
    print(f"OUTPUT_DIR='{output.get('dir', '')}'")
    resume = output.get('resume_from_checkpoint', '')
    print(f"RESUME_CHECKPOINT='{resume if resume else ''}'")
    
except Exception as e:
    print(f"echo 'Error parsing YAML: {e}'", file=sys.stderr)
    sys.exit(1)
EOF
)
    
    echo "Config loaded successfully"
    echo ""
else
    echo "=========================================="
    echo "No config file specified - using defaults"
    echo "=========================================="
    echo ""
    
    # Default configuration (same as old grpo_our.sh)
    export HF_ENDPOINT="https://huggingface.co"
    export CUDA_VISIBLE_DEVICES=2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    MODEL_TYPE="internvl3"
    MODEL_PATH="OpenGVLab/InternVL3_5-2B"
    TRAIN_DATASET="/path/to/ViVQA-X_train_grpo.jsonl"
    PLUGIN_PATH="../../../../external/ms-swift/examples/train/grpo/plugin/plugin.py"
    OUTPUT_DIR="/path/to/output/grpo"
    
    REWARD_FUNCS="vinle_format_our vinle_accuracy vinle_explanation"
    
    MAX_LENGTH=4096
    MAX_COMPLETION_LENGTH=1024
    NUM_GENERATIONS=4
    TEMPERATURE=0.9
    TOP_P=0.9
    TOP_K=50
    BETA=0.04
    
    TRAIN_TYPE="lora"
    LORA_RANK=32
    LORA_ALPHA=64
    TARGET_MODULES="all-linear"
    FREEZE_VIT=True
    
    NUM_EPOCHS=2
    MAX_STEPS=2000
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=4
    GRAD_ACCUM=4
    LR=1e-5
    WARMUP_RATIO=0.05
    
    SAVE_STEPS=50
    LOGGING_STEPS=1
    EVAL_STEPS=1
    SAVE_LIMIT=2
    
    TORCH_DTYPE="bfloat16"
    ATTN_IMPL="flash_attention_2"
    WORKERS=16
    DATASET_PROC=16
    
    QUANT_METHOD="bnb"
    QUANT_BITS=4
    
    RESUME_CHECKPOINT=""
fi

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Model: $MODEL_TYPE - $MODEL_PATH"
echo "Dataset: $TRAIN_DATASET"
echo "Output: $OUTPUT_DIR"
echo "Rewards: $REWARD_FUNCS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Build swift command
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
    --max_steps $MAX_STEPS \
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

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# Execute
echo "Starting GRPO training..."
echo ""
eval $CMD

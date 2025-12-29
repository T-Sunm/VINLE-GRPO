#!/bin/bash
# SFT Training Script - Reads from YAML config
# 
# Usage:
#   bash run_sft.sh [config_file]
#   bash run_sft.sh ../../../../configs/sft/baseline.yaml
#
# If no config specified, uses default paths below

set -e

# Get config file from argument or use inline defaults
CONFIG_FILE="${1:-}"

if [ -n "$CONFIG_FILE" ]; then
    echo "=========================================="
    echo "Loading SFT config from: $CONFIG_FILE"
    echo "=========================================="
    
    # Parse YAML using Python
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
    
    # Training
    train = config.get('training', {})
    print(f"TRAIN_TYPE='{train.get('train_type', 'lora')}'")
    print(f"LORA_RANK={train.get('lora_rank', 32)}")
    print(f"LORA_ALPHA={train.get('lora_alpha', 64)}")
    print(f"TARGET_MODULES='{train.get('target_modules', 'all-linear')}'")
    print(f"FREEZE_VIT={train.get('freeze_vit', True)}")
    print(f"NUM_EPOCHS={train.get('num_train_epochs', 3)}")
    print(f"BATCH_SIZE={train.get('per_device_train_batch_size', 4)}")
    print(f"GRAD_ACCUM={train.get('gradient_accumulation_steps', 2)}")
    print(f"LR={train.get('learning_rate', 5e-5)}")
    print(f"WARMUP_RATIO={train.get('warmup_ratio', 0.03)}")
    print(f"SAVE_STEPS={train.get('save_steps', 100)}")
    print(f"LOGGING_STEPS={train.get('logging_steps', 10)}")
    print(f"EVAL_STEPS={train.get('eval_steps', 100)}")
    print(f"SAVE_LIMIT={train.get('save_total_limit', 3)}")
    print(f"TORCH_DTYPE='{train.get('torch_dtype', 'bfloat16')}'")
    print(f"ATTN_IMPL='{train.get('attn_impl', 'flash_attention_2')}'")
    print(f"WORKERS={train.get('dataloader_num_workers', 8)}")
    print(f"DATASET_PROC={train.get('dataset_num_proc', 8)}")
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
    echo "No config file - using defaults"
    echo "=========================================="
    echo ""
    
    # Default configuration
    export HF_ENDPOINT="https://huggingface.co"
    export CUDA_VISIBLE_DEVICES=2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    MODEL_TYPE="internvl3"
    MODEL_PATH="OpenGVLab/InternVL3_5-2B"
    TRAIN_DATASET="/path/to/ViVQA-X_train_sft.jsonl"
    OUTPUT_DIR="/path/to/output/sft"
    
    MAX_LENGTH=4096
    TRAIN_TYPE="lora"
    LORA_RANK=32
    LORA_ALPHA=64
    TARGET_MODULES="all-linear"
    FREEZE_VIT=True
    
    NUM_EPOCHS=3
    BATCH_SIZE=4
    GRAD_ACCUM=2
    LR=5e-5
    WARMUP_RATIO=0.03
    
    SAVE_STEPS=100
    LOGGING_STEPS=10
    EVAL_STEPS=100
    SAVE_LIMIT=3
    
    TORCH_DTYPE="bfloat16"
    ATTN_IMPL="flash_attention_2"
    WORKERS=8
    DATASET_PROC=8
    
    QUANT_METHOD="bnb"
    QUANT_BITS=4
    
    RESUME_CHECKPOINT=""
fi

# Print configuration
echo "=========================================="
echo "SFT Training Configuration:"
echo "=========================================="
echo "Model: $MODEL_TYPE - $MODEL_PATH"
echo "Dataset: $TRAIN_DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Build swift sft command
CMD="swift sft \
    --use_hf true \
    --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --train_type $TRAIN_TYPE \
    --dataset $TRAIN_DATASET \
    --torch_dtype $TORCH_DTYPE \
    --attn_impl $ATTN_IMPL \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --target_modules $TARGET_MODULES \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --save_steps $SAVE_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --save_total_limit $SAVE_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio $WARMUP_RATIO \
    --dataloader_num_workers $WORKERS \
    --quant_method $QUANT_METHOD \
    --quant_bits $QUANT_BITS \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --gradient_checkpointing true \
    --report_to wandb"

# Add freeze_vit if needed
if [ "$FREEZE_VIT" = "True" ] || [ "$FREEZE_VIT" = "true" ]; then
    CMD="$CMD --freeze_vit true"
fi

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# Execute
echo "Starting SFT training..."
echo ""
eval $CMD

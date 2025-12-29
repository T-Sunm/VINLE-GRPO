#!/bin/bash
export HF_ENDPOINT="https://huggingface.co"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_ID_OR_PATH="5CD-AI/Vintern-3B-R-beta"
MODEL_TYPE="internvl3"

# Data Configuration
TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed_EN/VQA-X_train_grpo.jsonl"
PLUGIN_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/plugin/plugin.py"

# Output Configuration
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/vqax"
FAILED_PROMPTS_LOG="failed_question_ids_vqax.json"

# GRPO Training Parameters
NUM_GENERATIONS=4
TEMPERATURE=1.0
BATCH_SIZE_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-7

# Length Limits
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=1024

# Training Schedule
EPOCHS=1
MAX_STEPS=1000
SAVE_STEPS=50
LOGGING_STEPS=1
EVAL_STEPS=1

swift rlhf \
    --use_hf true \
    --rlhf_type grpo \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --dataset "$TRAIN_DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs custom_format_reward_ver3 custom_accuracy_reward custom_explaination_reward \
    --reward_weights 1 1 1 \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit True \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size $NUM_GENERATIONS \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --top_p 0.9 \
    --beta 0.001 \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --log_completions true \
    --torch_dtype bfloat16 \
    --save_only_model false \
    --save_total_limit 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 1 \
    --report_to wandb \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    --failed_prompts_log $FAILED_PROMPTS_LOG

    # --enable_gfpo true \
# dự kiến có thể sử dụng 8 bit sau này
#        --resume_from_checkpoint /home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/dat-vinternvl3B/v11-20251116-185104/checkpoint-550 \
echo "Training completed. Failed prompts logged to: $OUTPUT_DIR/$FAILED_PROMPTS_LOG"


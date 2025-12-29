#!/bin/bash
export HF_ENDPOINT="https://huggingface.co"
# --- Cấu hình ---
export CUDA_VISIBLE_DEVICES=2  # Chỉ định GPU muốn sử dụng
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID_OR_PATH="OpenGVLab/InternVL3_5-2B"
MODEL_TYPE="internvl3"
TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/standard_vivqax/ViVQA-X_train_msswift.jsonl"
PLUGIN_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/plugin/plugin.py"
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/standard"

# --- Tham số GRPO & Huấn luyện ---
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=1024  
NUM_GENERATIONS=4
TEMPERATURE=0.9
EPOCHS=2
BATCH_SIZE_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
MAX_STEPS=2000
LEARNING_RATE=1e-5

SAVE_STEPS=50
LOGGING_STEPS=1
EVAL_STEPS=1

swift rlhf \
    --rlhf_type grpo \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --use_vllm false \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --dataset "$TRAIN_DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs basic_format_reward basic_accuracy_reward \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit True \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size $NUM_GENERATIONS \
    --max_length $MAX_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --top_p 0.9 \
    --top_k 50 \
    --beta 0.04 \
    --log_completions true \
    --torch_dtype bfloat16 \
    --save_only_model false \
    --save_total_limit 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --report_to wandb \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    --resume_from_checkpoint "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/standard/v1-20251223-184215/checkpoint-1850"
    

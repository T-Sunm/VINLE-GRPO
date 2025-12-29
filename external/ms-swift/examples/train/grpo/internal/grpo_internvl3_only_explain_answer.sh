#!/bin/bash
export HF_ENDPOINT="https://huggingface.co"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID_OR_PATH="5CD-AI/Vintern-3B-R-beta"
MODEL_TYPE="internvl3"
TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/only_explain_answer/ViVQA-X_train_msswift.jsonl"
PLUGIN_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/plugin/plugin.py"
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/only_explain_answer"

# Tham số GRPO
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=1024
export NUM_GENERATIONS=4
TEMPERATURE=1.0
EPOCHS=1
BATCH_SIZE_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
MAX_STEPS=1000
LEARNING_RATE=1e-7

SAVE_STEPS=150
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
    --reward_funcs custom_format_reward_ViVQA_X_Only_Explain_Answer custom_accuracy_reward custom_explaination_reward \
    --reward_weights 1 1 1 \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit True \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size $NUM_GENERATIONS \
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
    --gradient_checkpointing true\
    --resume_from_checkpoint /home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/only_explain_answer/v4-20251218-160814/checkpoint-750
# dự kiến có thể sử dụng 8 bit sau này
    # --enable_gfpo true \
#        --resume_from_checkpoint /home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/dat-vinternvl3B/v11-20251116-185104/checkpoint-550 \
echo "Complete!"

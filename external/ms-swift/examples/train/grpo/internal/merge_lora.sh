#!/bin/bash
# Merge LoRA Weights Script
# 
# IMPORTANT: Edit the paths below before running!

export CUDA_VISIBLE_DEVICES=1

# === EDIT THESE PATHS ===
BASE_MODEL="OpenGVLab/InternVL3_5-2B"
MODEL_TYPE="internvl3"
CHECKPOINT_PATH="/home/vlai-vqa-nle/minhtq/VINLE-GRPO/output/grpo/vinle_full/v0-20251229-135812/checkpoint-2"
# ========================

swift export \
    --use_hf true \
    --model_type "$MODEL_TYPE" \
    --model "$BASE_MODEL" \
    --ckpt_dir "$CHECKPOINT_PATH" \
    --merge_lora true
export HF_ENDPOINT="https://huggingface.co"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_ID_OR_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_nouns/merged/stage3_250_curr_noun_ver3_2"
MODEL_TYPE="internvl3"

MAX_LENGTH=4096
TEMPERATURE=1.0
# EPOCHS=1
BATCH_SIZE_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-7
MAX_STEPS=250

SAVE_STEPS=50
LOGGING_STEPS=1
EVAL_STEPS=1

TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/curriculum_reasoning_failed/ViVQA-X_train_failed.jsonl"
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/sft/output/intern2B_failed_dataset_stage1"


CUDA_VISIBLE_DEVICES=2 \
swift sft \
    --use_hf=1 \
    --model "$MODEL_ID_OR_PATH" \
    --model_type "$MODEL_TYPE" \
    --train_type lora \
    --dataset "$TRAIN_DATASET_PATH" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --per_device_eval_batch_size 1 \
    --learning_rate $LEARNING_RATE \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --save_total_limit 2 \
    --logging_steps $LOGGING_STEPS \
    --max_length $MAX_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    --report_to wandb \
    # --resume_from_checkpoint "$OUTPUT_DIR/v1-20251124-140140/checkpoint-1450" \
    #    --num_train_epochs $EPOCHS \
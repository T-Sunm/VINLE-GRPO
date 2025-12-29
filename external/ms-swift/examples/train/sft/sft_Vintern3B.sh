export HF_ENDPOINT="https://huggingface.co"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# "OpenGVLab/InternVL3_5-2B", "5CD-AI/Vintern-3B-R-beta"
MODEL_ID_OR_PATH="OpenGVLab/InternVL3_5-2B"
MODEL_TYPE="internvl3"

MAX_LENGTH=4096
TEMPERATURE=0.8
EPOCHS=1
BATCH_SIZE_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-7
MAX_STEPS=8000

SAVE_STEPS=50
LOGGING_STEPS=1

TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/sft/ViVQA-X_train_msswift.jsonl"
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/sft/output/"


CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --use_hf=1 \
    --model "$MODEL_ID_OR_PATH" \
    --model_type "$MODEL_TYPE" \
    --attn_impl flash_attention_2 \
    --train_type lora \
    --dataset "$TRAIN_DATASET_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --learning_rate $LEARNING_RATE \
    --lora_rank 32 \
    --lora_alpha 64 \
    --freeze_vit True \
    --target_modules all-linear \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
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
export CUDA_VISIBLE_DEVICES=0

swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "5CD-AI/Vintern-3B-R-beta" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/sft/output/dat-vinternvl3B/v2-20251125-115404/checkpoint-2000" \
    --merge_lora true \
    --output_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/sft/output/dat-vinternvl3B/merged/" \

echo "Hoàn thành merge LoRA"

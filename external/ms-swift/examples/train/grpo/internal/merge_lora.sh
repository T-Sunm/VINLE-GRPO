export CUDA_VISIBLE_DEVICES=1
# "OpenGVLab/InternVL3_5-2B"
# "5CD-AI/Vintern-3B-R-beta"
swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "OpenGVLab/InternVL3_5-2B" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/our/intern_2000/checkpoint-2000" \
    --merge_lora true \

echo "Hoàn thành merge LoRA"
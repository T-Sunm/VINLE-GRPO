# Training Guide

Comprehensive guide for training VINLE-GRPO and baselines.

## Structure

```
external/ms-swift/examples/train/
├── grpo/
│   └── internal/
│       ├── run_grpo.sh         # Main GRPO training script
│       └── merge_lora.sh       # Script to merge LoRA weights
└── sft/
    └── sft_Vintern3B.sh        # SFT Baseline training
```

## 1. Environment Setup

Ensure you have installed the environment:

```bash
conda create -n vqa-nle python=3.10 -y
conda activate vqa-nle
bash scripts/setup/install_env.sh
```

## 2. Prepare Data

Generate the GRPO training dataset from ViVQA-X:

```bash
# Configure paths in .env first
pythom -m src.data.dataset_loader --mode grpo --split train
```

This will create JSONL files in `data/processed/`.

## 3. GRPO Training (Our Method)

We use **Group Relative Policy Optimization (GRPO)** to train the model to generate reasoning, explanation, and answer.

**Command:**

```bash
# Run from project root
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/vinle_full.yaml
```

**Configuration:**
- Edit `configs/grpo/vinle_full.yaml` to change hyperparameters (batch size, learning rate, beta, etc.).

**Output:**
- Checkpoints saved in: `outputs/training/grpo/vinle_full/`

## 4. SFT Training (Baseline)

Standard Supervised Fine-Tuning baseline.

**Command:**

```bash
bash external/ms-swift/examples/train/sft/sft_Vintern3B.sh
```

## 5. Merge LoRA Weights

After training (GRPO or SFT), you **must merge LoRA weights** before inference if you want optimal performance and easier loading.

**Command:**

```bash
# Run merge
bash external/ms-swift/examples/train/grpo/internal/merge_lora.sh
```

**Result:**
- A generic folder (without `checkpoint-` prefix) containing the full merged model.

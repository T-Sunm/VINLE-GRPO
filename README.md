# Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO

[![Paper](https://img.shields.io/badge/Paper-ICISN2026-blue)](./docs/paper/ICISN2026_GRPO_VQA-NLE.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation** of the paper "Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO" (ICISN 2026).

## 💡 Overview

We introduce a novel approach to improve Vietnamese Visual Question Answering (VQA) and Natural Language Explanations (NLE) using **Group Relative Policy Optimization (GRPO)**. 

By decoupling **reasoning (thinking)** from **explanation**, we achieve state-of-the-art performance on the ViVQA-X benchmark, enabling models to provide accurate answers with high-quality, interpretable rationales.

## 🔑 Key Features

- **Composite Reward System**:
  - **Format Reward**: Ensures structured output (`<think>`, `<answer>`, `<explain>`).
  - **Accuracy Reward**: Hybrid metric (BERTScore + ROUGE) handling Vietnamese synonyms.
  - **Explanation Reward**: Optimizes semantic alignment for rationales.
- **SOTA Performance**: Achieves **62.65%** accuracy on ViVQA-X (Vintern-3B), outperforming SFT and standard baselines.

## � Project Structure

```
vqa-nle/
├── external/                       # External dependencies (with modifications)
│   ├── ms-swift/                  # GRPO training framework
│   └── smile/                     # SMILE evaluation metric
│
├── src/                            # Research code
│   ├── data/                      # Data preparation for ViVQA-X
│   ├── rewards/                   # Custom reward functions
│   ├── evaluation/                # Evaluation pipeline
│   └── inference/                 # Inference scripts
│
├── scripts/                        # Executable scripts
│   ├── train/                     # Training wrappers
│   ├── eval/                      # Evaluation scripts
│   └── data/                      # Data preprocessing
│
├── configs/                        # Configuration files
│   ├── experiments/               # Per-experiment configs
│   └── models/                    # Model-specific configs
│
├── experiments/                    # Experiment tracking
│   ├── exp001_grpo_baseline/
│   ├── exp002_grpo_ours/         # Main paper results
│   └── exp003_ablation_study/
│
├── data/                          # Datasets
│   ├── raw/                       # ViVQA-X (symlink)
│   └── processed/                 # GRPO-formatted JSONL
│
├── docs/                          # Documentation
│   └── paper/                     # Paper materials
│
└── notebooks/                     # Analysis notebooks
```

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/yourusername/vqa-nle.git
cd vqa-nle
git submodule update --init --recursive  # Clone ms-swift and smile
```

### 2. Create Environment

```bash
# Create conda environment
conda create -n vqa-nle python=3.10 -y
conda activate vqa-nle

# Install dependencies
bash scripts/setup/install_env.sh
```

### 3. Setup External Repositories

```bash
# Setup ms-swift (with our modifications)
cd external/ms-swift
pip install -e .

# Setup SMILE metric
cd ../smile
pip install -e .
cd ../..
```

### 3. Evaluation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoModel; print('OK')"
```

## ⚡ Quick Start

### Reproduce Paper Results (Vintern-3B)

```bash
# 1. Prepare data
python -m src.data.dataset_loader

# 2. Train with GRPO (our method)
bash scripts/train/run_grpo_vintern.sh --exp_name exp002_grpo_ours

# 3. Inference
python -m src.inference.run_inference_grpo \
    --model experiments/exp002_grpo_ours/checkpoints/final \
    --output experiments/exp002_grpo_ours/results/predictions.jsonl

# 4. Evaluate
python -m src.evaluation.calculate_scores \
    --input experiments/exp002_grpo_ours/results/predictions.jsonl \
    --output experiments/exp002_grpo_ours/results/scores.json
```

**Expected Results** (ViVQA-X test set):
- Accuracy: **62.65%**
- SMILE: **60.42**
- BERTScore: **52.81**

## 📊 Data Preparation

### Dataset: ViVQA-X

Download or link the ViVQA-X dataset:

```bash
# Create symlink to ViVQA-X data
ln -s /mnt/VLAI_data/ViVQA-X data/raw/ViVQA-X
ln -s /mnt/VLAI_data/COCO_Images data/raw/COCO_Images
```

### Convert to GRPO Format

```bash
python -m src.data.dataset_loader
```

**Output format** (`data/processed/grpo/ViVQA-X_train_grpo.jsonl`):

```json
{
  "id": 1,
  "image": "COCO_train2014_000000139.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>You are a Visual Question Answering system...\nQuestion: {question}"
    },
    {
      "from": "gpt",
      "value": "<answer>{answer}</answer><explain>{explanation}</explain>"
    }
  ]
}
```

## 🏋️ Training

### GRPO Training (Our Method)

```bash
# Vintern-3B backbone
bash scripts/train/run_grpo_vintern.sh \
    --exp_name exp002_grpo_ours \
    --num_steps 1000 \
    --reward_funcs "accuracy format explanation"

# InternVL3.5 backbone
bash scripts/train/run_grpo_internvl.sh \
    --exp_name exp002_grpo_ours_internvl \
    --num_steps 1000 \
    --reward_funcs "accuracy format explanation"
```

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--learning_rate` | 1e-5 | Learning rate |
| `--max_steps` | 1000 | Training budget |
| `--lora_rank` | 32 | QLoRA rank |
| `--lora_alpha` | 64 | QLoRA alpha |
| `--num_generations` | 4 | GRPO group size (G) |
| `--beta` | 0.04 | KL penalty coefficient |
| `--temperature` | 0.9 | Sampling temperature |

### Reward Functions

Activate/deactivate rewards via `--reward_funcs`:

```bash
# Format only
--reward_funcs "format"

# Accuracy + Format
--reward_funcs "accuracy format"

# Full (our method)
--reward_funcs "accuracy format explanation"
```

**Implementation**: See `src/rewards/` for custom reward functions:
- `format_reward.py`: Tag structure validation
- `accuracy_reward.py`: Vietnamese hybrid matching (BERTScore + ROUGE)
- `explanation_reward.py`: Semantic alignment for rationales

## 🔍 Inference & Evaluation

### Inference

```bash
python -m src.inference.run_inference_grpo \
    --model experiments/exp002_grpo_ours/checkpoints/final \
    --dataset data/processed/grpo/ViVQA-X_test_grpo.jsonl \
    --output experiments/exp002_grpo_ours/results/predictions.jsonl \
    --batch_size 8
```

### Evaluation

```bash
python -m src.evaluation.calculate_scores \
    --input experiments/exp002_grpo_ours/results/predictions.jsonl \
    --ground_truth data/raw/ViVQA-X/ViVQA-X_test.json \
    --output experiments/exp002_grpo_ours/results/scores.json \
    --metrics accuracy smile bertscore
```

**Output** (`scores.json`):

```json
{
  "accuracy": 62.65,
  "smile": 60.42,
  "bertscore": 52.81,
  "breakdown": {
    "yes/no": {"accuracy": 78.3, "smile": 65.2},
    "number": {"accuracy": 52.1, "smile": 56.8},
    "other": {"accuracy": 58.9, "smile": 59.1}
  }
}
```

## 📈 Results

### Main Results (Table 1 from Paper)

| Method | Backbone | Acc ↑ | SMILE ↑ | BS ↑ |
|--------|----------|-------|---------|------|
| Base (Zero-shot) | Vintern-3B | 54.83 | 56.00 | 51.90 |
| SFT | Vintern-3B | 46.60 | 51.45 | 53.69 |
| GRPO (DeepSeek) | Vintern-3B | 56.15 | 57.07 | 52.20 |
| **GRPO (Ours)** | **Vintern-3B** | **62.65** | **60.42** | **52.81** |
| | | | | |
| Base (Zero-shot) | InternVL3.5 | 55.28 | 69.45 | 52.10 |
| SFT | InternVL3.5 | 56.20 | 69.00 | 52.20 |
| GRPO (DeepSeek) | InternVL3.5 | 54.98 | 69.14 | 52.14 |
| **GRPO (Ours)** | **InternVL3.5** | **61.23** | **65.47** | **52.24** |

### Ablation Study (Table 2 from Paper)

| Method | Acc ↑ | SMILE ↑ | BS ↑ |
|--------|-------|---------|------|
| Base (Direct) | 46.2 | 51.3 | 52.5 |
| Base (CoT) | 54.8 | 56.0 | 51.9 |
| GRPO w/o Reasoning | 42.8 | 54.7 | 53.9 |
| GRPO w/o Explanation | 47.4 | 56.7 | 50.7 |
| **GRPO (Full)** | **62.7** | **60.4** | **52.8** |

**Key Findings**:
- Reasoning improves accuracy by **+8.6%** (54.8% vs 46.2%)
- Decoupling reasoning from explanation: **+15.3%** (62.7% vs 47.4%)

## 📝 Citation

If you use this code or our methodology, please cite:

```bibtex
@inproceedings{icisn2026_grpo_vqa_nle,
  title={Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO},
  author={Quang-Minh Tran and Phat-Dat To and others},
  booktitle={ICISN 2026},
  year={2026}
}
```

## 📧 Contact

For questions, please contact **Quang-Minh Tran** or **Phat-Dat To**.
# Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO

[![Paper](https://img.shields.io/badge/Paper-ICISN2026-blue)](./docs/paper/ICISN2026_GRPO_VQA-NLE.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation** of the paper "Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO" (ICISN 2026).

## 💡 Overview

We introduce a novel approach to improve Vietnamese Visual Question Answering (VQA) and Natural Language Explanations (NLE) using **Group Relative Policy Optimization (GRPO)**. 

<p align="center">
  <img src="assets/main_figure.jpg" width="100%" alt="Main Figure">
</p>

By decoupling **reasoning (thinking)** from **explanation**, we achieve state-of-the-art performance on the ViVQA-X benchmark, enabling models to provide accurate answers with high-quality, interpretable rationales.

## 🔑 Key Features

- **Composite Reward System**:
  - **Format Reward**: Ensures structured output (`<REASONING>`, `<CONCLUSION>`, `<EXPLANATION>`).
  - **Accuracy Reward**: Hybrid metric (BERTScore + ROUGE) handling Vietnamese synonyms.
  - **Explanation Reward**: Optimizes semantic alignment for rationales.
- **SOTA Performance**: Achieves **62.65%** accuracy on ViVQA-X, outperforming SFT and standard baselines.

## 🗂 Project Structure

```
VINLE-GRPO/
├── configs/                        # ✨ Configuration files
│   ├── grpo/                      # GRPO experiments
│   │   ├── vinle_full.yaml       # Full method (R+C+E)
│   │   └── ablation_think_answer.yaml # Ablation (R+C)
│   └── sft/                       # SFT experiments
│       └── baseline.yaml          # SFT baseline (C+E)
│
├── external/                       # External dependencies 
│   ├── ms-swift/                  # Modified ms-swift framework
│   │   └── examples/train/       
│   │       ├── grpo/internal/run_grpo.sh  # GRPO runner script
│   │       └── sft/run_sft.sh             # SFT runner script
│   └── smile/                     # SMILE evaluation metric
│
├── src/                            # Research code
│   ├── data/                      # Data processing
│   ├── rewards/                   # Reward logic
│   └── ...
│
└── output/                         # Training outputs & checkpoints
```

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/T-Sunm/VINLE-GRPO.git
cd VINLE-GRPO
git submodule update --init --recursive
```

### 2. Environment Setup

```bash
# Create conda environment
conda create -n vqa-nle python=3.10 -y
conda activate vqa-nle

# Install dependencies (adjust based on your setup)
bash scripts/setup/install_env.sh
```

## ⚡ Workflow

### 1. Data Preparation

Generate data for different training modes:

```bash
# 1. Full GRPO (Reasoning + Conclusion + Explanation)
python -m src.data.dataset_loader --mode grpo --split train

# 2. Ablation (Reasoning + Conclusion only)
python -m src.data.dataset_loader --mode think_answer --split train

# 3. SFT Baseline (Conclusion + Explanation only)
python -m src.data.dataset_loader --mode sft --split train
```

Data will be saved to `data/processed/`.

### 2. Training with YAML Configs

We support three main training modes, configured via YAML files.

#### A. Full GRPO (Our Method)
Uses all 3 tags and all rewards (Accuracy + Format + Explanation).

```bash
# Edit config if needed
vim configs/grpo/vinle_full.yaml

# Run training
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh \
    configs/grpo/vinle_full.yaml
```

#### B. Ablation Study (No Explanation Reward)
Uses only Reasoning + Conclusion tags. No explanation reward.

```bash
# Run training
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh \
    configs/grpo/ablation_think_answer.yaml
```

#### C. SFT Baseline
Standard supervised fine-tuning. Uses Conclusion + Explanation tags.

```bash
# Run training
bash external/ms-swift/examples/train/sft/run_sft.sh \
    configs/sft/baseline.yaml
```

### 3. Monitoring

Training logs are reported to WandB (if enabled in config) and saved in `output/`.

```bash
# Check training progress
tail -f output/grpo/vinle_full/runs.log
```

## 📝 Configuration Rules

To create new experiments, simply copy a YAML config and modify it:

1.  **Duplicate config**: `cp configs/grpo/vinle_full.yaml configs/grpo/my_experiment.yaml`
2.  **Edit parameters**: Change `learning_rate`, `max_steps`, etc.
3.  **Run**: `bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/my_experiment.yaml`

**Important**: Always update `output.dir` in your new config to avoid overwriting previous results.

## � Results

### Main Results

| Method | Backbone | Acc ↑ | SMILE ↑ | BS ↑ |
|--------|----------|-------|---------|------|
| Base (Zero-shot) | Vintern-3B | 54.83 | 56.00 | 51.90 |
| SFT | Vintern-3B | 46.60 | 51.45 | 53.69 |
| GRPO (DeepSeek) | Vintern-3B | 56.15 | 57.07 | 52.20 |
| **GRPO (Ours)** | **Vintern-3B** | **62.65** | **60.42** | **52.81** |

### Ablation Study

| Method | Acc ↑ | SMILE ↑ | BS ↑ |
|--------|-------|---------|------|
| GRPO (Full) | **62.7** | **60.4** | **52.8** |
| w/o Reasoning | 42.8 | 54.7 | 53.9 |
| w/o Explanation | 47.4 | 56.7 | 50.7 |

## 📧 Contact

For questions, please contact **Quang-Minh Tran** or **Phat-Dat To**.
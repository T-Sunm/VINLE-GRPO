# VINLE-GRPO: Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO

[![Paper](https://img.shields.io/badge/Paper-ICISN2026-blue)](./docs/paper/ICISN2026_GRPO_VQA-NLE.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation** of "Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO" (ICISN 2026).

---

## Overview

This study investigates an adaptation of Group Relative Policy Optimization (GRPO) for the Vietnamese Visual Question Answering with Natural Language Explanations (VQA-NLE) task. We propose a composite reward mechanism to decouple the generation process into distinct stages: Reasoning (Thinking), Conclusion, and Explanation. This structured approach aims to mitigate hallucinations by enforcing explicit reasoning steps before response synthesis. Experiments on the ViVQA-X benchmark indicate that this method improves performance compared to supervised fine-tuning baselines, specifically on the Vintern-3B backbone.

<p align="center">
  <img src="assets/main_figure.jpg" width="100%" alt="Main Figure">
</p>

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/T-Sunm/VINLE-GRPO.git
cd VINLE-GRPO
git submodule update --init --recursive

# Create environment (Python 3.11 recommended)
conda create -n vqa-nle-swift python=3.11 -y
conda activate vqa-nle-swift

# Install dependencies + ms-swift
bash install_env.sh

# Setup Weights & Biases (for training logging)
export WANDB_ENTITY=<team-name>    # Create team at https://wandb.ai → Teams → Create
export WANDB_PROJECT=vinle_grpo
```

> **Note:** All paths are auto-detected from the project root. No manual path configuration needed.

### 2. Prepare Data

```bash
# Generate GRPO training data
python -m src.data.dataset_loader --mode grpo --split train
```

### 3. Train (GRPO)

All training parameters are configured via YAML files in `configs/grpo/`. Edit the YAML file to change GPU, steps, batch size, etc.

```bash
# Full GRPO method (our method)
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh \
    configs/grpo/vinle_full.yaml

# Ablation: Reasoning + Conclusion only (no Explanation)
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh \
    configs/grpo/ablation_think_answer.yaml
```

**Key config options** (in `configs/grpo/vinle_full.yaml`):
```yaml
training:
  max_steps: 4000           # Total training steps
  per_device_train_batch_size: 2
  learning_rate: 1.0e-5
environment:
  cuda_visible_devices: "0"  # Which GPU to use
```

### 4. Merge LoRA Weights

After training, merge LoRA adapter into base model:

```bash
bash external/ms-swift/examples/train/grpo/internal/merge_lora.sh \
    <path_to_checkpoint>

# Example:
bash external/ms-swift/examples/train/grpo/internal/merge_lora.sh \
    outputs/training/grpo/vinle_full/v2-20260211-010253/checkpoint-4000
```

### 5. Inference

```bash
# Run GRPO inference
python -m src.inference.internvl_based.grpo \
    --model outputs/training/grpo/vinle_full/.../checkpoint-XXX-merged \
    --output_dir outputs/inference/grpo
```

> **For all inference modes (GRPO, OTA, OEA, SFT, Zero-shot)**, see [Inference Documentation](src/inference/)

### 6. Evaluation

```bash
# Quick evaluation
bash scripts/eval_grpo.sh 

# Manual method
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0
```

> **For detailed evaluation**, see [Evaluation Documentation](src/evaluation/). 
> 
> *Note: Our pipeline includes **Automatic Round-Trip Sanitization** to protect GPUs from CUDA asserts during BERTScore/PhoBERT computation.*

---

## Project Structure

```
VINLE-GRPO/
├── configs/               # ⭐ YAML configurations (edit these!)
│   ├── grpo/
│   │   ├── vinle_full.yaml              # Full method config
│   │   └── ablation_think_answer.yaml   # Ablation config
│   └── sft/
│       └── baseline.yaml                # SFT baseline config
│
├── src/                    # Research code
│   ├── data/              # Dataset preparation
│   ├── rewards/           # Custom reward functions
│   ├── inference/         # Inference scripts
│   ├── evaluation/        # Evaluation pipeline
│   └── utils/             # Shared utilities
│
├── external/              # External dependencies
│   ├── ms-swift/         # GRPO training framework
│   └── smile-metric/     # SMILE evaluation metric
│
├── scripts/              # Helper scripts
│   ├── inference/        # run_grpo.sh, run_ota.sh, run_oea.sh
│   └── setup/            # Environment setup
│
├── outputs/              # Results (gitignored)
│   ├── training/        # Model checkpoints
│   └── inference/       # Inference results
│
├── install_env.sh        # Environment setup script
└── requirements.txt      # Python dependencies
```

---

## Inference Modes

We provide **5 inference modes** for systematic evaluation:

<div align="center">

| Mode | Tags | Description | Script |
|:---:|:---:|:---:|:---:|
| **GRPO** | R+C+E | Full GRPO method (our contribution) | `src.inference.internvl_based.grpo` |
| **OTA** | R+C | Ablation: Only Thinking + Answer | `src.inference.internvl_based.ota` |
| **OEA** | C+E | Ablation: Only Explanation + Answer | `src.inference.internvl_based.oea` |
| **SFT** | C+E | Supervised fine-tuning baseline | `src.inference.internvl_based.sft` |
| **Zero-shot** | R+C+E | Base model baseline | `src.inference.internvl_based.zero_shot` |

</div>

*(R=REASONING, C=CONCLUSION, E=EXPLANATION)*

> **See full inference documentation**: [src/inference/README.md](src/inference/)

---

## Main Results

### ViVQA-X Test Set
<div align="center">

| Method | Backbone | Acc ↑ | SMILE ↑ | BS ↑ |
|:---:|:---:|:---:|:---:|:---:|
| Base (Zero-shot) | Vintern-3B | 54.83 | 56.00 | 51.90 |
| SFT | Vintern-3B | 46.60 | 51.45 | 53.69 |
| GRPO (DeepSeek) | Vintern-3B | 56.15 | 57.07 | 52.20 |
| **GRPO (Ours)** | **Vintern-3B** | **62.65** | **60.42** | **52.81** |

</div>

### Ablation Study

<div align="center">

| Method | Acc ↑ | SMILE ↑ | BS ↑ |
|:---:|:---:|:---:|:---:|
| GRPO (Full) | **62.7** | **60.4** | **52.8** |
| w/o Reasoning | 42.8 | 54.7 | 53.9 |
| w/o Explanation | 47.4 | 56.7 | 50.7 |

</div>


<!-- 
## Citation

```bibtex
@inproceedings{vinle-grpo-2026,
  title={Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO},
  author={Tran, Quang-Minh and To, Phat-Dat},
  booktitle={Proceedings of ICISN 2026},
  year={2026}
}
```
-->

## Contact

For questions or collaborations, please contact **Quang-Minh Tran** or **Phat-Dat To**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<br>

**Last Updated**: 2026-02-11  
**Version**: 2.1
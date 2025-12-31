# VINLE-GRPO: Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO

[![Paper](https://img.shields.io/badge/Paper-ICISN2026-blue)](./docs/paper/ICISN2026_GRPO_VQA-NLE.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation** of "Enhancing Vietnamese VQA-NLE via Learning to Explain with GRPO" (ICISN 2026).

---

## Overview

We introduce a novel approach to improve Vietnamese Visual Question Answering with Natural Language Explanations (VQA-NLE) using **Group Relative Policy Optimization (GRPO)**. By decoupling **reasoning (thinking)** from **explanation**, we achieve state-of-the-art performance on the ViVQA-X benchmark.

<p align="center">
  <img src="assets/main_figure.jpg" width="100%" alt="Main Figure">
</p>

### Key Results

<div align="center">

| Method | Backbone | Acc ↑ | SMILE ↑ | BS ↑ |
|:---:|:---:|:---:|:---:|:---:|
| Base (Zero-shot) | Vintern-3B | 54.83 | 56.00 | 51.90 |
| SFT | Vintern-3B | 46.60 | 51.45 | 53.69 |
| **GRPO (Ours)** | **Vintern-3B** | **62.65** | **60.42** | **52.81** |

</div>

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/T-Sunm/VINLE-GRPO.git
cd VINLE-GRPO
git submodule update --init --recursive

# Create environment
conda create -n vqa-nle python=3.10 -y
conda activate vqa-nle
bash scripts/setup/install_env.sh
```

### 2. Prepare Data

```bash
# Configure paths in .env
cp .env.example .env

# Generate GRPO training data
python -m src.data.dataset_loader --mode grpo --split train
```

### 3. Train

```bash
# Full GRPO method (our method)
bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/vinle_full.yaml

# After training, merge LoRA weights
bash external/ms-swift/examples/train/grpo/internal/merge_lora.sh
```

> **For detailed training guide**, see [Training Documentation](docs/TRAINING.md)

### 4. Inference

```bash
# Run GRPO inference
python -m src.inference.internvl_based.grpo \
    --model outputs/training/grpo/vinle_full/.../checkpoint-XXX-merged \
    --output_dir outputs/inference/grpo
```

> **For all inference modes (GRPO, OTA, OEA, SFT, Zero-shot)**, see [Inference Documentation](src/inference/)

### 5. Evaluation

```bash
# Evaluate results
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0
```

> **For detailed evaluation options**, see [Evaluation Documentation](src/evaluation/)

Or use convenient scripts:

```bash
# Quick evaluation
bash scripts/eval_grpo.sh

# Evaluate all formats
bash scripts/eval_all.sh
```

> **For evaluation scripts guide**, see [Scripts Documentation](scripts/)

---

## Project Structure

```
VINLE-GRPO/
├── src/                    # Research code
│   ├── data/              # Dataset preparation → See README in folder
│   ├── rewards/           # Custom reward functions
│   ├── inference/         # Inference scripts → See README in folder
│   ├── evaluation/        # Evaluation pipeline → See README in folder
│   └── utils/             # Shared utilities
│
├── external/              # External dependencies (isolated)
│   ├── ms-swift/         # GRPO training framework → See docs/README.md
│   └── smile-metric/     # SMILE evaluation metric
│
├── configs/              # YAML configurations
│   ├── grpo/            # GRPO experiments
│   └── sft/             # SFT baseline
│
├── scripts/              # Executable scripts → See README in folder
│   ├── data/            # Data preparation
│   ├── eval/            # Evaluation helpers
│   └── setup/           # Environment setup
│
├── outputs/             # Results (gitignored)
│   ├── training/       # Model checkpoints
│   └── inference/      # Inference results
│
└── notebooks/          # Analysis notebooks
```

---

## Detailed Documentation
<div align="center">

| Component | Quick Example | Full Documentation |
|:---:|:---:|:---:|
| **Training** | `bash external/ms-swift/examples/train/grpo/internal/run_grpo.sh configs/grpo/vinle_full.yaml` | [Training Guide](docs/TRAINING.md) |
| **Inference** | `python -m src.inference.internvl_based.grpo --model <path>` | [Inference Guide](src/inference/) |
| **Evaluation** | `python -m src.evaluation.calculate_scores --input-dir <path>` | [Evaluation Guide](src/evaluation/) |
| **Scripts** | `bash scripts/eval_grpo.sh` | [Scripts Guide](scripts/) |

</div>

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

---

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

---

## Contact

For questions or collaborations, please contact **Quang-Minh Tran** or **Phat-Dat To**.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: 2025-12-31  
**Version**: 2.0 (Streamlined)
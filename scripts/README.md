# VINLE-GRPO Scripts

Utility scripts for data preparation, inference, evaluation, and setup.

## Directory Structure

```
scripts/
├── data/           # Data preparation scripts
├── inference/      # Inference testing scripts
├── eval/           # Evaluation & metrics scripts
└── setup/          # Environment setup scripts
```

## Quick Start

### 1. Setup Environment

```bash
bash scripts/setup/install_env.sh
bash scripts/setup/setup_external.sh
```

### 2. Prepare Data

```bash
python scripts/data/prepare_vivqa_sft.py
python scripts/data/prepare_vivqa_grpo.py
```

### 3. Run Inference

```bash
# Run all inference modes (GRPO, OTA, OEA, Zero-shot)
bash scripts/inference/run_all.sh 10

# Or run individual modes
bash scripts/inference/run_grpo.sh 10
bash scripts/inference/run_ota.sh 10
bash scripts/inference/run_oea.sh 10
bash scripts/inference/run_zeroshot.sh 10
```

### 4. Run Evaluation

```bash
bash scripts/eval/run_evaluation.sh
```

## Detailed Documentation

### inference/

Individual scripts for each inference mode with auto-versioning:

- `run_grpo.sh` - Full GRPO inference (R+C+E)
- `run_ota.sh` - OTA ablation (R+C)
- `run_oea.sh` - OEA ablation (C+E)
- `run_zeroshot.sh` - Zero-shot baseline
- `run_all.sh` - Run all modes sequentially

**Output**: `outputs/inference/{mode}/v{N}-{timestamp}.json`

**Usage**: `bash scripts/inference/run_grpo.sh [NUM_SAMPLES]`

### eval/

- `run_evaluation.sh` - Run evaluation metrics on inference results

### data/

- `prepare_vivqa_sft.py` - Prepare data for SFT training
- `prepare_vivqa_grpo.py` - Prepare data for GRPO training  
- `link_datasets.sh` - Create symlinks to datasets

### setup/

- `install_env.sh` - Install main environment
- `install_env_eval.sh` - Install evaluation environment
- `setup_external.sh` - Setup external dependencies

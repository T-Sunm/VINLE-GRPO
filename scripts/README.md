# Scripts Toolkit

This directory contains various utility scripts for setting up the environment, preparing data, running inference, and evaluating the models for the **VINLE-GRPO** project. While the [Main README](../README.md) showcases the fastest way to run the pipeline using the wrapper scripts, this document details **what each script does** and **how to run the underlying raw Python commands** for deeper customization.

## Directory Structure

```text
scripts/
├── data/              # Scripts for downloading and processing datasets
│   ├── link_datasets.sh
│   ├── prepare_vivqa_grpo.py
│   └── prepare_vivqa_sft.py
├── eval/              # Evaluation execution scripts
│   ├── eval_grpo.sh       # Quick wrapper to evaluate GRPO outputs (NLG metrics)
│   ├── eval_smile_grpo.sh # Quick wrapper to evaluate GRPO outputs (SMILE metric)
│   ├── eval_oea.sh        # Quick wrapper to evaluate OEA (Explanation+Answer) outputs
│   ├── eval_ota.sh        # Quick wrapper to evaluate OTA (Thinking+Answer) Outputs
│   ├── eval_zeroshot.sh   # Quick wrapper to evaluate Zero-shot baseline outputs
│   └── run_evaluation.sh
├── inference/         # Model inference execution scripts
│   ├── run_all.sh
│   ├── run_grpo.sh
│   ├── run_oea.sh
│   ├── run_ota.sh
│   └── run_zeroshot.sh
└── setup/             # Environment and dependency setup scripts
    ├── download_vivqax.py
    ├── install_env.sh
    ├── install_env_eval.sh
    └── setup_external.sh
```

---

## 1. Top-Level Evaluation Wrappers

These scripts are shortcuts to evaluate inference output files located in `outputs/inference/<mode>/`. 

| Script | Target Directory | Description |
|:---:|:---:|:---:|
| `eval/eval_grpo.sh` | `outputs/inference/grpo/` | Evaluates Full GRPO method outputs (Accuracy, BERTScore, BLEU). |
| `eval/eval_smile_grpo.sh` | `outputs/inference/grpo/` | Evaluates Full GRPO method outputs (SMILE logic & synthetic generated answers). |
| `eval/eval_ota.sh` | `outputs/inference/ota/` | Evaluates OTA (Reasoning + Conclusion) ablation outputs. |
| `eval/eval_oea.sh` | `outputs/inference/oea/` | Evaluates OEA (Conclusion + Explanation) ablation outputs. |
| `eval/eval_zeroshot.sh` | `outputs/inference/zeroshot/` | Evaluates Zero-shot baseline outputs. |

**Usage:**
```fish
# Evaluate all files in the respective format's folder (Default)
bash scripts/eval/eval_grpo.sh

# Evaluate a specific file
bash scripts/eval/eval_grpo.sh v2-20251229-175222.json
```

*Note: For deep-dive instructions on evaluation logic, check out [Evaluation Component Readme](../src/evaluation/README.md).*

---

## 2. Setup Subdirectory (`scripts/setup/`)

These scripts initialize the workspace, including downloading the raw datasets and setting up Conda environments for training and evaluation.

- **`download_vivqax.py`**: Downloads the raw ViVQA-X dataset from Hugging Face into `data/raw/vivqa-x/annotations/`. (Usage: `python scripts/setup/download_vivqax.py`)
- **`install_env.sh`**: Installs core training, inference, and SMILE framework requirements via pip (`vqa-nle-swift` env).
- **`install_env_eval.sh`**: Installs specific requirements needed for the separate evaluation environment targeting standard metrics (`vqa-nle-eval` env).
- **`setup_external.sh`**: Custom logic to initialize git submodules like `ms-swift` or metrics repositories.

---

## 3. Data Subdirectory (`scripts/data/`)

Scripts to transform downloaded raw datasets into the appropriate JSONL structures expected by the models.

- **`prepare_vivqa_grpo.py`**: Formats the reasoning data structurally for GRPO (`<think>...</think><answer>...</answer><explanation>...</explanation>`).
- **`prepare_vivqa_sft.py`**: Formats data for basic Supervised Fine-Tuning.
- **`link_datasets.sh`**: Safely soft-links external image folders (e.g., COCO images) into the project's data directory.

*(Check out `src.data.dataset_loader` for the main python invocation recommended in the [Main README](../README.md))*

---

## 4. Inference Subdirectory (`scripts/inference/`)

Easily kick off inference for various model types and ablations.

- **`run_all.sh [NUM_SAMPLES]`**: Consecutively runs GRPO, OTA, OEA, and Zero-shot inference for a small sample size (defaults to 3). Perfect for debugging.
  ```fish
  bash scripts/inference/run_all.sh 5
  ```
- **`run_grpo.sh`, `run_ota.sh`, `run_oea.sh`, `run_zeroshot.sh`**: Shell scripts to wrap Python calls to their respective module counterparts inside `src/inference/internvl_based/`.

> **Deep-Dive / Manual Override:**
> If you wish to bypass the bash wrappers, you can run the raw Python inference commands. For example, to run GRPO manually:
> ```fish
> python -m src.inference.internvl_based.grpo \
>     --model outputs/training/grpo/vinle_full/<timestamp>/checkpoint-4000-merged \
>     --output_dir outputs/inference/grpo
> ```

---

## 5. Eval Subdirectory (`scripts/eval/`)

These scripts are shortcuts to evaluate inference output files located in `outputs/inference/<mode>/`.  

- **`eval_grpo.sh`**, **`eval_smile_grpo.sh`**, **`eval_ota.sh`**, **`eval_oea.sh`**, **`eval_zeroshot.sh`**: Helper bash scripts that trigger evaluation.
- **`run_evaluation.sh`**: A comprehensive script for executing the full calculation suite (Accuracy, BERTScore, SMILE) continuously or against specific targets without using the fast top-level wrappers.

> **Deep-Dive / Manual Override:**
> For total control over metrics (e.g. changing CUDA bindings or testing CPU fallback for BERTScore), run the raw source modules. **Be deeply aware of your active conda environment!**
> 
> ```fish
> # 1. Standard NLG evaluations (Accuracy, BERTScore, CLIPScore) -> Uses install_env_eval.sh
> conda activate vqa-nle-eval
> python -m src.evaluation.calculate_scores \
>     --input-dir outputs/inference/grpo \
>     --device cuda:0
> 
> # 2. SMILE evaluation -> Uses install_env.sh
> conda activate vqa-nle-swift
> python -m src.evaluation.calculate_smile_scores_expl \
>     --input-dir outputs/inference/grpo \
>     --device cuda:0
> ```

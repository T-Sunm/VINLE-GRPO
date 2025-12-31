# VINLE-GRPO Inference

Inference scripts for VINLE-GRPO trained models and baselines.

## Structure

```
src/inference/
├── common/                      # Shared utilities
│   ├── prompts.py              # All prompt templates
│   ├── parsers.py              # Tag parsing functions
│   └── processors.py           # Data processing utilities
│
├── internvl_based/             # Our Methods (Paper Contributions)
│   ├── model.py                # VinternModel class
│   ├── grpo.py                 # Full GRPO (R+C+E)
│   ├── ota.py                  # OTA Ablation (R+C)
│   ├── oea.py                  # OEA Ablation (C+E)
│   ├── sft.py                  # SFT Baseline (C+E)
│   └── zero_shot.py            # Zero-shot (R+C+E)
│
└── other_models/               # Other VLM Baselines
    ├── zero_shot.py           # Unified runner
    ├── internvl.py, qwenvl.py, molmo.py
    ├── phi.py, ovis.py, minicpm.py
    └── videollama.py, vintern1b.py
```

## Inference Types (Our Methods)

| Mode | Tags | Description | Use Case |
|:---:|:---:|:---:|:---:|
| `grpo` | R+C+E | Full GRPO method | **Our main contribution** |
| `ota` | R+C | Only Thinking + Answer | Ablation: No explanation reward |
| `oea` | C+E | Only Explanation + Answer | Ablation: No reasoning reward |
| `sft` | C+E | Supervised fine-tuning | Baseline comparison |
| `zero_shot` | R+C+E | Base model | Baseline comparison |

*(R=REASONING, C=CONCLUSION, E=EXPLANATION)*

## Usage

### Our Methods (InternVL-based)

```bash
# 1. Full GRPO (Our Main Method)
python -m src.inference.internvl_based.grpo \
    --model output/grpo/vinle_full/.../checkpoint-XXX-merged

# 2. OTA Ablation (R+C only)
python -m src.inference.internvl_based.ota \
    --model output/grpo/ablation_think_answer/.../checkpoint-XXX-merged

# 3. OEA Ablation (C+E only)
python -m src.inference.internvl_based.oea \
    --model output/grpo/ablation_explain_answer/.../checkpoint-XXX-merged

# 4. SFT Baseline
python -m src.inference.internvl_based.sft \
    --model output/sft/baseline/.../checkpoint-XXX-merged

# 5. Zero-shot Baseline
python -m src.inference.internvl_based.zero_shot \
    --model OpenGVLab/InternVL3_5-2B
```

### Other VLM Baselines (Zero-shot only)

All other VLMs use **unified runner** with GRPO prompt:

```bash
# Available: qwenvl, molmo, phi, ovis, minicpm, videollama, internvl, vintern1b

# QwenVL
python -m src.inference.other_models.zero_shot qwenvl --limit 100

# Molmo
python -m src.inference.other_models.zero_shot molmo --limit 100

# Phi3.5-Vision
python -m src.inference.other_models.zero_shot phi

# Ovis
python -m src.inference.other_models.zero_shot ovis

# MiniCPM
python -m src.inference.other_models.zero_shot minicpm

# VideoLLaMA
python -m src.inference.other_models.zero_shot videollama
```

**Note**: All models use **GRPO prompt** (3 tags: R+C+E) for fair comparison.

## Common Arguments

- `--model` - Model path (HF ID or local merged checkpoint)
- `--data_path` - Test data JSON (default: ViVQA-X test)
- `--image_folder` - Image directory (default: COCO val2014)
- `--output_dir` - Output directory for results
- `--output_name` - Custom output filename  
- `--limit 100` - Limit number of samples (for testing)
- `--device cuda` - Device (cuda or cpu)

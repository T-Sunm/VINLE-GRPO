# Evaluation

Evaluate inference results using `calculate_scores.py`.

> **Tip**: For easier execution using shell scripts (recommended), see [Scripts Documentation](../../scripts/README.md).

## Structure

```
src/evaluation/
├── calculate_scores.py       # Main CLI script
├── core/
│   ├── format_detector.py    # Auto-detects GRPO/OTA/OEA formats
│   ├── shared_models.py      # Loads BERTScore & SMILE models
│   └── text_preprocessing.py # Normalizes Vietnamese text
└── metrics/
    ├── vqa_accuracy.py       # Accuracy with flexible matching
    ├── nlg_metrics.py        # BLEU, METEOR, ROUGE, CIDEr
    └── smile_metrics.py      # SMILE metric wrapper
```

## Usage

### 1. Basic Evaluation

Evaluate all JSON files in a directory:

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0
```

### 2. Specific Files

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --filenames model1.json model2.json
```

## Arguments

<div align="center">

| Argument | Default | Description |
|:---:|:---:|:---:|
| `--input-dir` | `outputs/inference` | Directory containing JSON files |
| `--filenames` | `[]` | List of specific filenames to evaluate |
| `--output-file` | auto | Output CSV filename |
| `--device` | `cuda:0` | GPU device for models |
| `--cuda-device` | `0` | CUDA_VISIBLE_DEVICES ID |

</div>

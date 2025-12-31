# Evaluation Module

Unified evaluation system with **automatic format detection** for all inference outputs.

---

## Quick Start

### Evaluate Single Format

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0
```

### Evaluate All Formats

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference \
    --recursive \
    --output-file results/all_formats.csv
```

### Using Scripts (Easier)

```bash
# Quick evaluation
bash scripts/eval_grpo.sh

# All formats
bash scripts/eval_all.sh
```

---

## Supported Formats

| Format | `thinking` | `pred_explanation` | Auto-detected ✅ |
|--------|-----------|-------------------|-----------------|
| **GRPO** | ✅ | ✅ | Yes |
| **OTA** | ✅ | ❌ | Yes |
| **OEA** | ❌ | ✅ | Yes |
| **Zero-shot** | ✅ | ✅ | Yes |

The evaluator automatically detects which fields are present and computes applicable metrics.

---

## Metrics Computed

| Metric Type | Metrics | Applied To |
|-------------|---------|-----------|
| **Accuracy** | Exact match + fuzzy matching | `pred_answer` |
| **NLG** | BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, BERTScore | `thinking`, `pred_explanation` |
| **SMILE** | SMILE_avg, SMILE_hm | `pred_answer` |

**Output**: CSV with breakdown by answer type (Overall, yes/no, number, other)

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `outputs/inference` | Input directory |
| `--filenames` | `[]` | Specific files to evaluate |
| `--recursive` | `False` | Search subdirectories |
| `--output-file` | auto | Output CSV path (auto: timestamped) |
| `--device` | `cuda:0` | Device for models |

---

## Examples

```bash
# Specific files
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --filenames model1.json model2.json

# Custom GPU
CUDA_VISIBLE_DEVICES=2 python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/ota \
    --device cuda:0
```

---

## Module Structure

```
src/evaluation/
├── calculate_scores.py       # Main CLI
├── core/
│   ├── text_preprocessing.py  # Text normalization
│   ├── shared_models.py       # BERTScore/SMILE models
│   └── format_detector.py     # Auto-detect format
└── metrics/
    ├── vqa_accuracy.py        # Accuracy evaluation
    ├── nlg_metrics.py         # BLEU/METEOR/ROUGE/CIDEr/BERTScore
    └── smile_metrics.py       # SMILE wrapper
```



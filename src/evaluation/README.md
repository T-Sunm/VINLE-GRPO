# VQA Evaluation Module

Unified evaluation system for VINLE-GRPO that supports all inference output formats with automatic format detection.

## Supported Formats

| Format | `thinking` | `pred_explanation` | Description |
|--------|-----------|-------------------|-------------|
| **GRPO** | ✅ | ✅ | Full GRPO output with thinking + explanation |
| **OEA** | ❌ | ✅ | Only-Explain-Answer format |
| **OTA** | ✅ | ❌ | Only-Think-Answer format |
| **ZEROSHOT** | ✅ | ✅ | Baseline model output (same as GRPO) |

## Structure

```
src/evaluation/
├── calculate_scores.py              # Main CLI entry point
├── __init__.py                      # Module exports
│
├── core/                            # Core utilities
│   ├── __init__.py
│   ├── text_preprocessing.py        # Text cleaning & normalization
│   ├── shared_models.py             # Singleton BERTScore/SMILE/SynGen
│   └── format_detector.py           # Auto-detect format
│
└── metrics/                         # Metric implementations
    ├── __init__.py
    ├── vqa_accuracy.py              # VQA accuracy with fuzzy matching
    ├── nlg_metrics.py               # BLEU/METEOR/ROUGE/CIDEr/BERTScore
    └── smile_metrics.py             # SMILE wrapper
```

## Usage

### Basic Usage

Evaluate a single file:

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0
```

### Batch Evaluation

Evaluate all formats in subdirectories:

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference \
    --recursive \
    --output-file results/all_formats.csv
```

### Specific Files

Evaluate specific JSON files:

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --filenames model1.json model2.json \
    --output-file results/grpo_results.csv
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | `outputs/inference` | Input directory with JSON files |
| `--filenames` | list | `[]` | Specific filenames to evaluate |
| `--recursive` | flag | `False` | Recursively find JSON in subdirs |
| `--output-file` | str | auto | Output CSV path (auto: timestamped) |
| `--device` | str | `cuda:0` | Device for model computation |
| `--cuda-device` | str | `0` | CUDA_VISIBLE_DEVICES value |

## Metrics Computed

### VQA Accuracy
- Exact match after normalization
- Yes/no answer variants (Vietnamese + English)
- Fuzzy unsorted substring matching

### NLG Metrics (for thinking/explanation)
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap
- **METEOR**: Unigram matching with synonyms
- **ROUGE-L**: Longest common subsequence
- **CIDEr**: Consensus-based metric
- **BERTScore F1**: PhoBERT semantic similarity

### SMILE Metrics (for answers)
- **SMILE_avg**: Average SMILE score
- **SMILE_hm**: Harmonic mean SMILE score

## Auto-Detection Logic

The evaluator automatically detects format and computes applicable metrics:

```python
# Format detection
format_info = detect_format(data)

# Conditional evaluation
if format_info['has_thinking']:
    thinking_scores = evaluate_thinking_field(...)

if format_info['has_pred_explanation']:
    explanation_scores = evaluate_explanation_field(...)

# Always evaluate
answer_scores = evaluate_answer_quality(...)
```

## Output Format

Results are saved as CSV with the following structure:

| model | answer_type | total | correct | accuracy | thinking_* | explanation_* | SMILE_* |
|-------|------------|-------|---------|----------|-----------|---------------|---------|
| model1 | Overall | 100 | 85 | 85.0 | ... | ... | ... |
| model1 | yes/no | 40 | 38 | 95.0 | ... | ... | ... |
| model1 | number | 20 | 15 | 75.0 | ... | ... | ... |
| model1 | other | 40 | 32 | 80.0 | ... | ... | ... |

## Dependencies

Required packages:
- `torch`, `transformers` - Deep learning models
- `bert-score` - BERTScore computation
- `underthesea` - Vietnamese text processing
- `pycocoevalcap` - Traditional NLG metrics
- `pandas` - Results formatting
- Custom: `SMILE` (from `external/smile-metric`)

## Examples

### Example 1: Evaluate GRPO output

```bash
cd /home/vlai-vqa-nle/minhtq/VINLE-GRPO

python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/grpo \
    --device cuda:0 \
    --output-file results/grpo_eval.csv
```

### Example 2: Evaluate all formats

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference \
    --recursive \
    --device cuda:0 \
    --output-file results/all_formats_eval.csv
```

### Example 3: Custom CUDA device

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/ota \
    --cuda-device 2 \
    --device cuda:0
```

## Notes

- **Format auto-detection**: Based on presence of `thinking` and `pred_explanation` fields
- **Shared models**: BERTScore and SMILE models are loaded once and reused
- **Synthetic answers**: Generated automatically for improved SMILE evaluation
- **Error handling**: Robust handling of CUDA errors and format inconsistencies
- **Progress tracking**: Real-time progress bars for long-running operations

---

**Version**: 1.0  
**Last Updated**: 2025-12-31  
**Author**: VINLE-GRPO Team

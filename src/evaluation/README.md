# Evaluation

Evaluate inference results using two separate scripts:
- **`calculate_scores.py`**: NLG metrics (BLEU, METEOR, ROUGE-L, BERTScore, CLIPScore)
- **`calculate_smile_scores.py`**: SMILE metrics (requires LLM for synthetic answer generation)

> **Tip**: For easier execution using shell scripts (recommended), see [Scripts Documentation](../../scripts/README.md).

## Structure

```
src/evaluation/
├── calculate_scores.py          # NLG metrics (vqa-nle-eval env)
├── calculate_smile_scores.py    # SMILE metrics (vqa-nle-smile env)
├── core/
│   ├── format_detector.py       # Auto-detects GRPO/OTA/OEA formats
│   ├── shared_models.py         # Singleton models (BERTScore, SMILE, CLIP, LLM)
│   └── text_preprocessing.py    # Normalizes Vietnamese text
└── metrics/
    ├── vqa_accuracy.py          # Accuracy with flexible matching
    ├── nlg_metrics.py           # BLEU, METEOR, ROUGE, BERTScore
    ├── clip_metrics.py          # CLIPScore (image-text similarity)
    └── smile_metrics.py         # SMILE metric wrapper
```

## Usage

### 1. NLG Metrics (Accuracy, BLEU, METEOR, ROUGE-L, BERTScore, CLIPScore)

**Environment**: `vqa-nle-eval`

```bash
conda activate vqa-nle-eval

# Recommended: BERTScore on CPU (avoids CUDA asserts), CLIPScore on GPU
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --bert-device cpu

# Alternative: All on GPU (may crash with CUDA asserts)
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --device cuda:0

# Alternative: All on CPU (slower but safe)
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --device cpu
```

**Output**: `nlg_results_YYYYMMDD_HHMMSS.csv`

### 2. SMILE Metrics (Synthetic Answer Generation + SMILE)

**Environment**: `vqa-nle-smile` (requires flash-attention for Qwen LLM)

```bash
conda activate vqa-nle-smile

# Default: Qwen LLM on GPU, PhoBERT on CPU
python -m src.evaluation.calculate_smile_scores \
    --input-dir outputs/inference/zeroshot

# Alternative: PhoBERT on GPU (may crash with CUDA asserts)
python -m src.evaluation.calculate_smile_scores \
    --input-dir outputs/inference/zeroshot \
    --bert-device cuda:0
```

**Output**: `smile_results_YYYYMMDD_HHMMSS.csv`

### 3. Evaluate Specific Files

```bash
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --filenames InternVL3_5-2B-full.json Vintern-3B-R-beta-full.json \
    --bert-device cpu
```

## Arguments

### `calculate_scores.py` (NLG Metrics)

| Argument | Default | Description |
|:---:|:---:|:---:|
| `--input-dir` | `outputs/inference` | Directory containing JSON files |
| `--filenames` | `[]` | List of specific filenames to evaluate |
| `--output-file` | auto | Output CSV filename |
| `--device` | `cuda:0` | GPU device for CLIPScore |
| `--bert-device` | `None` | Device for BERTScore (default: same as `--device`). **Use `cpu` to avoid CUDA asserts** |
| `--cuda-device` | `0` | CUDA_VISIBLE_DEVICES ID |
| `--image-dir` | `/mnt/VLAI_data/COCO_Images/val2014` | COCO images directory |

### `calculate_smile_scores.py` (SMILE Metrics)

| Argument | Default | Description |
|:---:|:---:|:---:|
| `--input-dir` | `outputs/inference` | Directory containing JSON files |
| `--filenames` | `[]` | List of specific filenames to evaluate |
| `--output-file` | auto | Output CSV filename |
| `--device` | `cuda:0` | GPU device for Qwen LLM |
| `--bert-device` | `cpu` | Device for PhoBERT in SMILE. **Default `cpu` to avoid CUDA asserts** |
| `--cuda-device` | `0` | CUDA_VISIBLE_DEVICES ID |

## Important Notes

### CUDA Asserts with PhoBERT

**Problem**: PhoBERT tokenization on GPU can trigger `CUDA error: device-side assert triggered` when encountering edge-case tokens (special characters, empty strings, etc.). This corrupts GPU state and causes all subsequent operations to fail.

**Solution**: Use `--bert-device cpu` to run BERTScore/SMILE's PhoBERT on CPU:

```bash
# NLG metrics
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --bert-device cpu

# SMILE metrics (already defaults to cpu)
python -m src.evaluation.calculate_smile_scores \
    --input-dir outputs/inference/zeroshot
```

**Performance**: PhoBERT-base is small (~135M params), so CPU performance is acceptable. CLIPScore and Qwen LLM remain on GPU for speed.

### Dependencies

- **Java**: Required for METEOR metric. If not installed, METEOR will be skipped automatically.
- **SMILE package**: Only needed for `calculate_smile_scores.py`. Install in `vqa-nle-smile` env:
  ```bash
  pip install underthesea
  ```

### Environment Separation

| Script | Environment | Key Dependencies |
|:---:|:---:|:---:|
| `calculate_scores.py` | `vqa-nle-eval` | `bert-score`, `torchmetrics`, `transformers` |
| `calculate_smile_scores.py` | `vqa-nle-smile` | `smile-metric`, `underthesea`, `flash-attn` |

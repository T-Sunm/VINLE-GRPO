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

# Default: All on GPU (Safeguarded by automatic sanitization)
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --device cuda:0

# Alternative: All on CPU (Slower but safe)
python -m src.evaluation.calculate_scores \
    --input-dir outputs/inference/zeroshot \
    --device cpu
```

**Output**: `nlg_results_YYYYMMDD_HHMMSS.csv`

### 2. SMILE Metrics (Synthetic Answer Generation + SMILE)

**Environment**: `vqa-nle-smile` (requires flash-attention for Qwen LLM)

```bash
conda activate vqa-nle-smile

# Default: Qwen LLM on GPU, PhoBERT on GPU (Sanitized)
python -m src.evaluation.calculate_smile_scores \
    --input-dir outputs/inference/zeroshot \
    --bert-device cuda:0

# Alternative: PhoBERT on CPU (Extra safe)
python -m src.evaluation.calculate_smile_scores \
    --input-dir outputs/inference/zeroshot \
    --bert-device cpu
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
| `--bert-device` | `cuda:0` | Device for PhoBERT. **Now safe on GPU via round-trip sanitization** |
| `--cuda-device` | `0` | CUDA_VISIBLE_DEVICES ID |

## Important Notes

### Robust GPU Tokenization (PhoBERT Sanitization)

**Mechanism**: To prevent `CUDA error: device-side assert triggered` common in PhoBERT models when encountering edge-case tokens (special characters, empty strings, etc.), the evaluation and reward systems now use **Automatic Round-Trip Sanitization**.

1.  **Text Cleaning**: Removes control characters and non-BMP symbols.
2.  **Round-Trip Tokenize**: Validates and clamps token IDs within the vocabulary range on CPU before sending tensors to the GPU.
3.  **Stability**: This ensures that all inputs entering the GPU are safe, allowing BERTScore and PhoBERT to run on CUDA without crashing the system.

**Performance**: Running on GPU is significantly faster than the previously recommended CPU workaround. CLIPScore and Qwen LLM also remain on GPU for maximum efficiency.

### Dependencies

- **Java**: Required for METEOR metric. If not installed, METEOR will be skipped automatically.
- **Key Packages**: Install in both environments to support device mapping and Vietnamese processing:
  ```bash
  pip install accelerate underthesea
  ```

### Environment Separation

| Script | Environment | Key Dependencies |
|:---:|:---:|:---:|
| `calculate_scores.py` | `vqa-nle-eval` | `bert-score`, `transformers`, `torchmetrics`, `accelerate` |
| `calculate_smile_scores.py` | `vqa-nle-smile` | `smile-metric`, `underthesea`, `flash-attn`, `accelerate` |

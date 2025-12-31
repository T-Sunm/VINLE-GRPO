# Evaluation Scripts

Quick shell scripts for evaluating inference results.

---

## Quick Start

### Evaluate Single Format

```bash
# Edit FILES variable in script (recommended)
nano scripts/eval_grpo.sh
# Set: FILES="v2-20251229-175222.json"

# Then run
bash scripts/eval_grpo.sh
```

Or pass filenames directly:

```bash
bash scripts/eval_grpo.sh v2-20251229-175222.json
```

### Evaluate All Formats

```bash
bash scripts/eval_all.sh
```

---

## Available Scripts

<div align="center">

| Script | Input Folder | Example |
|:---:|:---:|:---:|
| `eval_grpo.sh` | `outputs/inference/grpo/` | `bash scripts/eval_grpo.sh file.json` |
| `eval_ota.sh` | `outputs/inference/ota/` | `bash scripts/eval_ota.sh file.json` |
| `eval_oea.sh` | `outputs/inference/oea/` | `bash scripts/eval_oea.sh file.json` |
| `eval_zeroshot.sh` | `outputs/inference/zeroshot/` | `bash scripts/eval_zeroshot.sh file.json` |
| `eval_all.sh` | All formats (recursive) | `bash scripts/eval_all.sh output.csv` |

</div>

---

## Usage Methods

### Method 1: Edit Configuration Variable (Recommended)

```bash
# 1. Edit script
nano scripts/eval_grpo.sh

# 2. Uncomment and set FILES
FILES="v2-20251229-175222.json"

# 3. Run
bash scripts/eval_grpo.sh
```

### Method 2: Command Line Arguments

```bash
# Single file
bash scripts/eval_grpo.sh file1.json

# Multiple files
bash scripts/eval_grpo.sh file1.json file2.json file3.json

# All files (default)
bash scripts/eval_grpo.sh
```

---

## Configuration Examples

### Single File
```bash
FILES="v2-20251229-175222.json"
```

### Multiple Files
```bash
FILES="checkpoint1.json checkpoint2.json checkpoint3.json"
```

### All Files (Default)
```bash
FILES=""  # Leave empty
```

---

## Output

Results saved as CSV with metrics:
- **Accuracy** (overall + by answer type: yes/no, number, other)
- **NLG Metrics** (BLEU, METEOR, ROUGE, CIDEr, BERTScore)
- **SMILE** (avg + harmonic mean)

---

## Tips

- **Easy editing**: Edit `FILES` variable in script
- **Quick testing**: Pass filenames as arguments
- **Batch evaluation**: Use `eval_all.sh` for all formats
- **GPU selection**: Edit `--device cuda:0` in script

---

## Troubleshooting

**Script not executable?**
```bash
chmod +x scripts/*.sh
```

**File not found?**
```bash
# Make sure you're in repo root
cd /home/vlai-vqa-nle/minhtq/VINLE-GRPO
```

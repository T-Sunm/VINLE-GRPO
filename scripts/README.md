# VQA Evaluation Scripts

Simple shell scripts for evaluating inference results.

## Quick Start

### Method 1: Edit Configuration Variable (Recommended)

1. Open the script file
2. Edit the `FILES` variable at the top
3. Run the script

**Example:**
```bash
# 1. Edit eval_grpo.sh
nano scripts/eval_grpo.sh

# 2. Uncomment and edit FILES variable:
FILES="v2-20251229-175222.json"

# 3. Run the script
./scripts/eval_grpo.sh
```

### Method 2: Command Line Arguments

```bash
./scripts/eval_grpo.sh v2-20251229-175222.json
./scripts/eval_oea.sh file1.json file2.json
```

### Method 3: Evaluate All Files (Default)

```bash
./scripts/eval_grpo.sh  # Evaluates all files in outputs/inference/grpo/
```

## Available Scripts

| Script | Format | Configuration Variable |
|--------|--------|----------------------|
| `eval_grpo.sh` | GRPO | `FILES="yourfile.json"` |
| `eval_oea.sh` | OEA | `FILES="yourfile.json"` |
| `eval_ota.sh` | OTA | `FILES="yourfile.json"` |
| `eval_zeroshot.sh` | ZEROSHOT | `FILES="yourfile.json"` |
| `eval_all.sh` | All formats | `OUTPUT_FILE="results.csv"` |

## Configuration Examples

### Single File
Open `eval_grpo.sh` and edit:
```bash
# Uncomment this line:
FILES="v2-20251229-175222.json"
```

### Multiple Files
```bash
# Uncomment and edit:
FILES="checkpoint1.json checkpoint2.json checkpoint3.json"
```

### All Files
```bash
# Leave FILES empty (default):
FILES=""
```

### Custom Output (eval_all.sh)
```bash
# Uncomment and edit:
OUTPUT_FILE="my_comprehensive_results.csv"
```

## Full Example Workflow

```bash
# 1. Navigate to repo
cd /home/vlai-vqa-nle/minhtq/VINLE-GRPO

# 2. Edit the script
nano scripts/eval_grpo.sh

# In the file, uncomment and edit:
# FILES="v2-20251229-175222.json"

# 3. Save and run
./scripts/eval_grpo.sh
```

## Alternative: Direct Command Line

If you prefer not to edit files:

```bash
# Single file
./scripts/eval_grpo.sh v2-20251229-175222.json

# Multiple files
./scripts/eval_oea.sh file1.json file2.json file3.json

# All files
./scripts/eval_ota.sh

# All formats with custom output
./scripts/eval_all.sh my_results.csv
```

## Setup (First Time)

```bash
chmod +x scripts/*.sh
```

## What Gets Evaluated

Each script automatically knows which folder to use:

- `eval_grpo.sh` → `outputs/inference/grpo/`
- `eval_oea.sh` → `outputs/inference/oea/`
- `eval_ota.sh` → `outputs/inference/ota/`
- `eval_zeroshot.sh` → `outputs/inference/zeroshot/`
- `eval_all.sh` → All of the above (recursive)

## Output

Results are saved as CSV with metrics:
- Accuracy (overall and by answer type)
- Explanation metrics (BLEU, METEOR, ROUGE, CIDEr, BERTScore)
- Answer quality metrics (SMILE)

## Tips

1. **Easy editing**: Most users prefer Method 1 (edit FILES variable)
2. **Quick testing**: Use Method 2 (command line) for one-off evaluations
3. **Batch evaluation**: Use `eval_all.sh` to evaluate all formats at once
4. **Compare checkpoints**: Set `FILES` to multiple filenames separated by spaces

## Troubleshooting

**Script not running?**
```bash
chmod +x scripts/eval_grpo.sh
```

**Wrong GPU?**
Edit the script and change:
```bash
--device cuda:0  →  --device cuda:1
```

**File not found?**
Make sure you're in the repo root:
```bash
cd /home/vlai-vqa-nle/minhtq/VINLE-GRPO
```

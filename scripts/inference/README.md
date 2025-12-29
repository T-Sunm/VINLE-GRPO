# Inference Scripts

Individual inference scripts for each mode with auto-versioning.

## Available Scripts

| Script | Mode | Tags | Description |
|--------|------|------|-------------|
| `run_grpo.sh` | GRPO | R+C+E | Full GRPO method (our main contribution) |
| `run_ota.sh` | OTA | R+C | Ablation: No explanation reward |
| `run_oea.sh` | OEA | C+E | Ablation: No reasoning reward |
| `run_zeroshot.sh` | Zero-shot | R+C+E | Base model baseline |
| `run_all.sh` | All | - | Run all modes sequentially |

## Usage

### Run Individual Mode

```bash
# GRPO (with 10 samples)
bash scripts/inference/run_grpo.sh 10

# OTA
bash scripts/inference/run_ota.sh 10

# OEA
bash scripts/inference/run_oea.sh 10

# Zero-shot
bash scripts/inference/run_zeroshot.sh 10
```

### Run All Modes

```bash
# Test all modes with 10 samples each
bash scripts/inference/run_all.sh 10

# Default: 3 samples
bash scripts/inference/run_all.sh
```

## Output Structure

Results are saved with auto-versioning:

```
outputs/inference/
├── grpo/
│   ├── v1-20251229-171205.json
│   ├── v2-20251229-172030.json
│   └── v3-20251229-173145.json
├── ota/
│   └── v1-20251229-171205.json
├── oea/
│   └── v1-20251229-171205.json
└── zeroshot/
    └── v1-20251229-171205.json
```

## Configuration

To use different models, edit the `MODEL` variable in each script:

```bash
# Example: Edit run_grpo.sh
MODEL="/path/to/your/checkpoint"
```

Current default checkpoints:
- **GRPO**: `/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/.../vintern_2000_our_vivqax`
- **OTA**: `/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/.../checkpoint-1000-merged`
- **OEA**: `/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/.../checkpoint-1000-merged`
- **Zero-shot**: `OpenGVLab/InternVL3_5-2B`

## View Results

```bash
# View latest GRPO results
cat outputs/inference/grpo/v*.json | tail -1 | jq '.[] | {question, predict, thinking}'

# Compare all modes for first sample
for mode in grpo ota oea zeroshot; do
  echo "=== $mode ==="
  jq '.[0].predict' outputs/inference/$mode/v*.json | tail -1
done
```

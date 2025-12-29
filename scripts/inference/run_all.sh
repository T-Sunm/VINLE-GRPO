#!/bin/bash
# Run All Inference Tests
# Usage: bash scripts/inference/run_all.sh [NUM_SAMPLES]

set -e
cd "$(dirname "$0")"

NUM_SAMPLES=${1:-3}

echo "╔═══════════════════════════════════════════════╗"
echo "║   VINLE-GRPO Inference Testing (All Modes)    ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "Samples: $NUM_SAMPLES"
echo ""

# Run each mode
bash run_grpo.sh "$NUM_SAMPLES"
echo ""

bash run_ota.sh "$NUM_SAMPLES"
echo ""

bash run_oea.sh "$NUM_SAMPLES"
echo ""

bash run_zeroshot.sh "$NUM_SAMPLES"
echo ""

echo "╔═══════════════════════════════════════════════╗"
echo "║   All Tests Completed!                        ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "Results in: outputs/inference/"
tree ../../outputs/inference/ -L 2 2>/dev/null || find ../../outputs/inference/ -name "*.json"

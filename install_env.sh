#!/bin/bash
# ============================================================
# VINLE-GRPO Environment Installation Script
# ============================================================
# Usage: bash install_env.sh
# Can be run from any directory - project root is auto-detected.

set -e

# Auto-detect project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${VINLE_PROJECT_ROOT:-$SCRIPT_DIR}"
echo "Project Root: $PROJECT_ROOT"
echo ""

# --- Bước 1: Cài "Core" (Nặng nhất & Quan trọng nhất) trước ---
echo ">>> Installing PyTorch 2.6.0 (CUDA 12.4)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# --- Bước 2: Cài Flash Attention (Phụ thuộc vào Torch đã cài ở B1) ---
echo ">>> Installing Flash Attention..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# --- Bước 3: Cài các thư viện phụ trợ từ requirements.txt ---
echo ">>> Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# --- Bước 4: Cài ms-swift (Editable Mode) ---
echo ">>> Installing ms-swift in Editable Mode..."
cd "$PROJECT_ROOT/external/ms-swift"
# Dùng --no-build-isolation để nó nhận diện ngay Torch/FlashAttn đã cài
pip install -e . --no-build-isolation

echo ">>> Installation COMPLETED!"
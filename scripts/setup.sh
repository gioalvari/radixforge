#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== RadixForge Setup ==="

# Fetch the pinned llama.cpp submodule
echo "[1/3] Initializing llama.cpp submodule (pinned commit)..."
git -C "$PROJECT_DIR" submodule update --init --recursive vendor/llama.cpp

# Create build directory
echo "[2/3] Configuring CMake (Release + Metal)..."
cmake -B "$PROJECT_DIR/build" -S "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRADIXFORGE_METAL=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64

# Build
echo "[3/3] Building..."
cmake --build "$PROJECT_DIR/build" --config Release -j$(sysctl -n hw.ncpu)

echo ""
echo "=== Build complete! ==="
echo "Binary: $PROJECT_DIR/build/radixforge"
echo ""
echo "Usage:"
echo "  ./build/radixforge -m /path/to/model.gguf"
echo ""

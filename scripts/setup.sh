#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== RadixForge Setup ==="

# Clone llama.cpp if not present
if [ ! -d "$PROJECT_DIR/vendor/llama.cpp" ]; then
    echo "[1/3] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$PROJECT_DIR/vendor/llama.cpp"
else
    echo "[1/3] llama.cpp already present, pulling latest..."
    cd "$PROJECT_DIR/vendor/llama.cpp" && git pull --ff-only
fi

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

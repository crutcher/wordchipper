#!/usr/bin/env bash
# Set up everything needed for the Node.js demo in one directory.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
PKG="$REPO_ROOT/bindings/wasm/pkg"

# 1. Build WASM if needed.
if [ ! -f "$PKG/wordchipper_wasm.js" ]; then
  echo "Building WASM package..."
  wasm-pack build "$REPO_ROOT/bindings/wasm" --target web
fi

# 2. Copy WASM artifacts into this directory.
echo "Copying WASM files..."
cp "$PKG/wordchipper_wasm.js" "$DIR/"
cp "$PKG/wordchipper_wasm_bg.wasm" "$DIR/"

# 3. Download vocab file.
VOCAB="$DIR/o200k_base.tiktoken"
if [ -f "$VOCAB" ]; then
  echo "Already exists: o200k_base.tiktoken"
else
  echo "Downloading o200k_base vocab..."
  curl -fSL -o "$VOCAB" "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
fi

echo ""
echo "Ready! Run:"
echo "  cd $(basename "$DIR") && node index.mjs"

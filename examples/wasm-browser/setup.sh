#!/usr/bin/env bash
# Set up everything needed for the browser demo in one directory.
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
cp "$PKG/wordchipper_wasm.d.ts" "$DIR/"

# 3. Download vocab files.
VOCAB_DIR="$DIR/vocab"
mkdir -p "$VOCAB_DIR"
BASE="https://openaipublic.blob.core.windows.net/encodings"

for name in r50k_base p50k_base cl100k_base o200k_base; do
  dest="$VOCAB_DIR/${name}.tiktoken"
  if [ -f "$dest" ]; then
    echo "Already exists: vocab/${name}.tiktoken"
  else
    echo "Downloading ${name}..."
    curl -fSL -o "$dest" "${BASE}/${name}.tiktoken"
  fi
done

echo ""
echo "Ready! Run:"
echo "  cd $(basename "$DIR") && python3 -m http.server 8080"
echo "  open http://localhost:8080"

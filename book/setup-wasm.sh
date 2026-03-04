#!/usr/bin/env bash
# Build WASM and download vocab files for the mdBook interactive demo.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
PKG="$REPO_ROOT/bindings/wasm/pkg"
DEST="$DIR/src/wasm"

# 1. Build WASM (cargo/wasm-pack will skip if unchanged).
echo "Building WASM package..."
wasm-pack build "$REPO_ROOT/bindings/wasm" --target web

# 2. Copy WASM artifacts.
mkdir -p "$DEST"
echo "Copying WASM files..."
cp "$PKG/wordchipper_wasm.js" "$DEST/"
cp "$PKG/wordchipper_wasm_bg.wasm" "$DEST/"

# 3. Download vocab files.
VOCAB_DIR="$DEST/vocab"
mkdir -p "$VOCAB_DIR"
BASE="https://openaipublic.blob.core.windows.net/encodings"

for name in r50k_base p50k_base cl100k_base o200k_base; do
  dest="$VOCAB_DIR/${name}.tiktoken"
  if [ -f "$dest" ]; then
    echo "Already exists: ${name}.tiktoken"
  else
    echo "Downloading ${name}..."
    curl -fSL -o "$dest" "${BASE}/${name}.tiktoken"
  fi
done

echo ""
echo "Done. Run: mdbook serve"

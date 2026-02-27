# wordchipper WASM - Node.js Example

Demonstrates using the wordchipper WASM bindings from Node.js.

## Setup and Run

```bash
cargo x wasm-node
cd examples/wasm-node
node index.mjs
```

The setup task builds the WASM package (if needed), copies it into this directory,
and downloads the o200k_base vocab file.

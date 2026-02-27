# wordchipper WASM - Browser Example

Interactive browser demo for the wordchipper WASM bindings.

## Setup and Run

```bash
cargo x wasm-browser
cd examples/wasm-browser
python3 -m http.server 8080
```

Open http://localhost:8080

The setup task builds the WASM package (if needed), copies it into this directory,
and downloads the vocab files.

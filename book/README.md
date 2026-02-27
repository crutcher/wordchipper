# Wordchipper Book

Requires [mdbook](https://rust-lang.github.io/mdBook/guide/installation.html):

```bash
cargo install mdbook
```

## WASM interactive demo

The book includes an interactive tokenizer demo that runs in the browser via WebAssembly.
Before building or serving the book, run the setup task to build the WASM module and
download vocab files:

```bash
cargo x book-demo-setup
```

This creates `src/wasm/` (gitignored) with the WASM binary and vocab data.

## Serve locally (with live reload)

```bash
mdbook serve
```

Opens at [http://localhost:3000](http://localhost:3000).

## Build static HTML

```bash
mdbook build
```

Output is in `book/book/`.

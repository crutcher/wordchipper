#!/bin/bash

# Fix local clippy issues (stable):
cargo clippy --fix --allow-dirty --allow-staged

# I'm using (nightly) features for format:
cargo +nightly fmt $*


# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/crutcher/wordchipper/actions/workflows/ci.yml)
[![license](https://shields.io/badge/license-MIT-blue)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/crutcher/wordchipper)

I am usually available as `@crutcher` on the Burn Discord:

* Burn
  Discord: [![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

The primary documentation is for the [wordchipper crate](crates/wordchipper).

## Components

### Published Crates

- [wordchipper](crates/wordchipper)
- [wordchipper-disk-cache](crates/wordchipper-disk-cache)

### Unpublished Crates

- [wordchipper-data](crates/wordchipper-data)

### Tools

- [token-cli](examples/token-cli) - dev side-by-side.
- [tokenizer_trainer](examples/tokenizer_trainer) - dev training example.

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy) and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust tokenizer space.


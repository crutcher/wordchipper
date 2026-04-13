#!/usr/bin/env python3
"""Convert a HuggingFace tokenizers.json (ByteLevel BPE) to tiktoken format.

Tiktoken format is one line per token:
    {BASE64_ENCODED_BYTES} {TOKEN_ID}

Usage
-----
Local file:
    python hf_to_tiktoken.py tokenizer.json qwen3.5.tiktoken

Download from HuggingFace (requires ``huggingface_hub``):
    python hf_to_tiktoken.py Qwen/Qwen3.5-27B qwen3.5.tiktoken

The output is suitable for use with wordchipper's
``Tokenizer.from_tiktoken_file`` loader.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPT-2 / HuggingFace ByteLevel bytes-to-unicode mapping
# ---------------------------------------------------------------------------

def _build_bytes_to_unicode() -> dict[int, str]:
    """Return the canonical GPT-2 bytes-to-unicode table.

    This maps each of the 256 byte values to a printable Unicode character so
    that BPE can operate on "text" without null bytes or surrogates.  HuggingFace
    tokenizers uses the same table for all ByteLevel BPE models (GPT-2, Qwen, …).
    """
    # Bytes whose unicode code-point is already printable and unambiguous
    bs = (
        list(range(ord("!"), ord("~") + 1))   # 33-126  (ASCII printable)
        + list(range(ord("¡"), ord("¬") + 1)) # 161-172 (Latin-1 supplement)
        + list(range(ord("®"), ord("ÿ") + 1)) # 174-255
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# Build once at import time
_BYTE_TO_CHAR: dict[int, str] = _build_bytes_to_unicode()
_CHAR_TO_BYTE: dict[str, int] = {v: k for k, v in _BYTE_TO_CHAR.items()}


def decode_hf_token(token_str: str) -> bytes:
    """Decode a HuggingFace ByteLevel-BPE token string to raw bytes.

    Raises ``KeyError`` if the string contains characters outside the mapping
    (should not happen for well-formed ByteLevel vocabularies).
    """
    return bytes(_CHAR_TO_BYTE[ch] for ch in token_str)


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def load_tokenizer_json(source: str) -> dict:
    """Load tokenizers.json from a local path or a HuggingFace repo ID."""
    p = Path(source)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    # Treat as HuggingFace repo ID
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "error: 'huggingface_hub' is not installed.\n"
            "Install it with:  pip install huggingface_hub\n"
            "or download tokenizer.json manually and pass its local path.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading tokenizer.json from {source} …", file=sys.stderr)
    local = hf_hub_download(repo_id=source, filename="tokenizer.json")
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def convert(source: str, output_path: Path, *, include_added_tokens: bool = True) -> None:
    """Convert a HuggingFace tokenizers.json to tiktoken format.

    Parameters
    ----------
    source:
        Local path to tokenizer.json **or** a HuggingFace repo ID
        (e.g. ``"Qwen/Qwen3.5-27B"``).
    output_path:
        Destination ``.tiktoken`` file.
    include_added_tokens:
        Whether to include ``added_tokens`` (special tokens) in the output.
        Defaults to True so the file contains a complete ID mapping.
    """
    data = load_tokenizer_json(source)

    model = data.get("model", {})
    model_type = model.get("type", "")
    if model_type != "BPE":
        print(
            f"warning: model type is '{model_type}', expected 'BPE'. "
            "Conversion may produce incorrect output.",
            file=sys.stderr,
        )

    vocab: dict[str, int] = model.get("vocab", {})
    if not vocab:
        print("error: model.vocab is empty or missing.", file=sys.stderr)
        sys.exit(1)

    # id -> raw bytes
    all_tokens: dict[int, bytes] = {}
    decode_errors = 0

    for token_str, tok_id in vocab.items():
        try:
            raw = decode_hf_token(token_str)
        except KeyError:
            # Rare: token contains characters outside the bytes-to-unicode
            # table (can happen with some added special tokens stored in vocab).
            # Fall back to raw UTF-8 bytes.
            raw = token_str.encode("utf-8")
            decode_errors += 1
        existing = all_tokens.get(tok_id)
        if existing is not None and existing != raw:
            raise ValueError(
                f"duplicate token id {tok_id} maps to multiple byte sequences"
            )
        all_tokens[tok_id] = raw

    if decode_errors:
        print(
            f"warning: {decode_errors} token(s) fell back to raw UTF-8 "
            "(not in ByteLevel mapping — likely special tokens).",
            file=sys.stderr,
        )

    # added_tokens are special tokens stored separately; their "bytes" are
    # the raw UTF-8 of their content string (e.g. b"<|endoftext|>").
    added_count = 0
    if include_added_tokens:
        for entry in data.get("added_tokens", []):
            tok_id: int = entry["id"]
            content: str = entry["content"]
            raw = content.encode("utf-8")
            existing = all_tokens.get(tok_id)
            if existing is not None and existing != raw:
                raise ValueError(
                    f"duplicate token id {tok_id} maps to multiple byte sequences"
                )
            if tok_id not in all_tokens:
                all_tokens[tok_id] = raw
                added_count += 1

    # Write sorted by token ID
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        for tok_id in sorted(all_tokens):
            b64 = base64.standard_b64encode(all_tokens[tok_id]).decode("ascii")
            f.write(f"{b64} {tok_id}\n")

    print(
        f"Wrote {len(all_tokens)} tokens "
        f"({added_count} added/special) to {output_path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help=(
            "Path to tokenizer.json, or a HuggingFace repo ID "
            "(e.g. 'Qwen/Qwen3.5-27B')"
        ),
    )
    parser.add_argument(
        "output",
        help="Output path for the .tiktoken file",
    )
    parser.add_argument(
        "--no-added-tokens",
        action="store_true",
        help="Exclude added_tokens (special tokens) from the output",
    )
    args = parser.parse_args()

    convert(
        args.source,
        Path(args.output),
        include_added_tokens=not args.no_added_tokens,
    )


if __name__ == "__main__":
    main()

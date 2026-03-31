"""Qwen3.5 encode benchmarks: wordchipper vs tiktoken vs tokenizers.

Requires a converted .tiktoken vocabulary file — generate it with:
    python bindings/python/hf_to_tiktoken.py Qwen/Qwen3.5-27B qwen3.5.tiktoken

Then point to it via the environment variable:
    export QWEN_TIKTOKEN_PATH=/path/to/qwen3.5.tiktoken

Run with:
    pytest benchmarks/test_bench_qwen35.py -v
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

import wordchipper

# ---------------------------------------------------------------------------
# Qwen3.5 tokenizer constants
# Source: Qwen/Qwen3.5-27B tokenizer_config.json / tokenizer.json
# All Qwen3.5 sizes (0.6B–72B) share the same tokenizer.
# ---------------------------------------------------------------------------

QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Core special tokens (IDs sourced from tokenizer_config.json)
QWEN35_SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047,
    "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|quad_start|>": 248051,
    "<|quad_end|>": 248052,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|vision_pad|>": 248055,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
    "<tool_call>": 248058,
    "</tool_call>": 248059,
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|fim_suffix|>": 248062,
    "<|fim_pad|>": 248063,
    "<|repo_name|>": 248064,
    "<|file_sep|>": 248065,
    "<tool_response>": 248066,
    "</tool_response>": 248067,
    "<think>": 248068,
    "</think>": 248069,
    "<|audio_start|>": 248070,
    "<|audio_end|>": 248071,
    "<tts_pad|>": 248072,
    "<tts_text_bos|>": 248073,
    "<tts_text_eod|>": 248074,
    "<tts_text_bos_single|>": 248075,
    "<|audio_pad|>": 248076,
}

HF_REPO = "Qwen/Qwen3.5-27B"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiktoken_path() -> Path:
    val = os.environ.get("QWEN_TIKTOKEN_PATH")
    if not val:
        pytest.skip(
            "QWEN_TIKTOKEN_PATH not set — run hf_to_tiktoken.py first:\n"
            "  python bindings/python/hf_to_tiktoken.py Qwen/Qwen3.5-27B qwen3.5.tiktoken\n"
            "  export QWEN_TIKTOKEN_PATH=/path/to/qwen3.5.tiktoken"
        )
    p = Path(val)
    if not p.exists():
        pytest.skip(f"QWEN_TIKTOKEN_PATH={val!r} does not exist")
    return p


def _load_mergeable_ranks(path: Path) -> dict[bytes, int]:
    """Load a .tiktoken file as a bytes→id dict for use with tiktoken.Encoding."""
    ranks: dict[bytes, int] = {}
    with path.open(encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, id_str = line.split(" ", 1)
            ranks[base64.standard_b64decode(b64)] = int(id_str)
    return ranks


@pytest.fixture(scope="session")
def tiktoken_path() -> Path:
    return _tiktoken_path()


@pytest.fixture(scope="session")
def wc_tokenizer(tiktoken_path):
    """wordchipper Tokenizer loaded from the converted .tiktoken file."""
    options = wordchipper.TokenizerOptions.default()
    options.set_parallel(False)
    options.set_accelerated_lexers(False)
    return wordchipper.Tokenizer.from_tiktoken_file(
        str(tiktoken_path),
        QWEN35_PATTERN,
        QWEN35_SPECIAL_TOKENS,
        options,
    )


@pytest.fixture(scope="session")
def wc_tokenizer_accel(tiktoken_path):
    """wordchipper Tokenizer with accelerated lexers enabled."""
    options = wordchipper.TokenizerOptions.default()
    options.set_parallel(False)
    options.set_accelerated_lexers(True)
    return wordchipper.Tokenizer.from_tiktoken_file(
        str(tiktoken_path),
        QWEN35_PATTERN,
        QWEN35_SPECIAL_TOKENS,
        options,
    )


@pytest.fixture(scope="session")
def wc_tokenizer_parallel(tiktoken_path):
    """wordchipper Tokenizer with parallel batch encoding enabled."""
    options = wordchipper.TokenizerOptions.default()
    options.set_parallel(True)
    options.set_accelerated_lexers(False)
    return wordchipper.Tokenizer.from_tiktoken_file(
        str(tiktoken_path),
        QWEN35_PATTERN,
        QWEN35_SPECIAL_TOKENS,
        options,
    )


@pytest.fixture(scope="session")
def tiktoken_enc(tiktoken_path):
    """tiktoken Encoding constructed from the same .tiktoken file."""
    import tiktoken

    ranks = _load_mergeable_ranks(tiktoken_path)
    # Remove special tokens from mergeable_ranks — tiktoken requires them
    # to be passed separately.
    special_ids = set(QWEN35_SPECIAL_TOKENS.values())
    base_ranks = {k: v for k, v in ranks.items() if v not in special_ids}

    return tiktoken.Encoding(
        name="qwen3.5",
        pat_str=QWEN35_PATTERN,
        mergeable_ranks=base_ranks,
        special_tokens=QWEN35_SPECIAL_TOKENS,
    )


@pytest.fixture(scope="session")
def hf_tokenizer():
    """HuggingFace tokenizers.Tokenizer loaded from the HF hub."""
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(HF_REPO)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Single-string benchmarks
# ---------------------------------------------------------------------------

class TestSingleEncode:
    def test_wordchipper_english(self, benchmark, wc_tokenizer, english_text):
        wc_tokenizer.encode(english_text)  # warmup
        benchmark.group = "qwen3.5/single/english"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(wc_tokenizer.encode, english_text)

    def test_wordchipper_english_accel(self, benchmark, wc_tokenizer_accel, english_text):
        wc_tokenizer_accel.encode(english_text)
        benchmark.group = "qwen3.5/single/english"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(wc_tokenizer_accel.encode, english_text)

    def test_wordchipper_diverse(self, benchmark, wc_tokenizer, diverse_text):
        wc_tokenizer.encode(diverse_text)
        benchmark.group = "qwen3.5/single/diverse"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(wc_tokenizer.encode, diverse_text)

    def test_wordchipper_diverse_accel(self, benchmark, wc_tokenizer_accel, diverse_text):
        wc_tokenizer_accel.encode(diverse_text)
        benchmark.group = "qwen3.5/single/diverse"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(wc_tokenizer_accel.encode, diverse_text)

    def test_tiktoken_english(self, benchmark, tiktoken_enc, english_text):
        tiktoken_enc.encode(english_text, allowed_special="all")
        benchmark.group = "qwen3.5/single/english"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tiktoken_enc.encode, english_text, allowed_special="all")

    def test_tiktoken_diverse(self, benchmark, tiktoken_enc, diverse_text):
        tiktoken_enc.encode(diverse_text, allowed_special="all")
        benchmark.group = "qwen3.5/single/diverse"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tiktoken_enc.encode, diverse_text, allowed_special="all")

    def test_tokenizers_english(self, benchmark, hf_tokenizer, english_text):
        hf_tokenizer.encode(english_text)
        benchmark.group = "qwen3.5/single/english"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(hf_tokenizer.encode, english_text)

    def test_tokenizers_diverse(self, benchmark, hf_tokenizer, diverse_text):
        hf_tokenizer.encode(diverse_text)
        benchmark.group = "qwen3.5/single/diverse"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(hf_tokenizer.encode, diverse_text)


# ---------------------------------------------------------------------------
# Parallel batch benchmarks
# ---------------------------------------------------------------------------

class TestBatchEncode:
    def test_wordchipper_parallel(self, benchmark, wc_tokenizer_parallel, fineweb_batch):
        texts, total_bytes = fineweb_batch
        wc_tokenizer_parallel.encode_batch(texts)
        benchmark.group = "qwen3.5/batch"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(wc_tokenizer_parallel.encode_batch, texts)

    def test_wordchipper_threadpool(self, benchmark, tiktoken_path, fineweb_batch, max_threads):
        texts, total_bytes = fineweb_batch

        from concurrent.futures import ThreadPoolExecutor

        options = wordchipper.TokenizerOptions.default()
        options.set_concurrent(True)
        options.set_accelerated_lexers(False)
        tok = wordchipper.Tokenizer.from_tiktoken_file(
            str(tiktoken_path),
            QWEN35_PATTERN,
            QWEN35_SPECIAL_TOKENS,
            options,
        )

        benchmark.group = "qwen3.5/batch"
        benchmark.extra_info["input_bytes"] = total_bytes

        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            def _batch(texts):
                return list(pool.map(tok.encode, texts))

            _batch(texts)  # warmup
            benchmark(_batch, texts)

    def test_tiktoken(self, benchmark, tiktoken_enc, fineweb_batch, max_threads):
        texts, total_bytes = fineweb_batch
        tiktoken_enc.encode_batch(texts, allowed_special="all")
        benchmark.group = "qwen3.5/batch"
        benchmark.extra_info["input_bytes"] = total_bytes
        num_threads = max_threads or 8
        benchmark(tiktoken_enc.encode_batch, texts, num_threads=num_threads, allowed_special="all")

    def test_tokenizers(self, benchmark, hf_tokenizer, fineweb_batch):
        texts, total_bytes = fineweb_batch
        hf_tokenizer.encode_batch(texts)
        benchmark.group = "qwen3.5/batch"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(hf_tokenizer.encode_batch, texts)

"""Python encode/encode_batch benchmarks: wordchipper vs tiktoken vs tokenizers.

Build the extension in release mode first for meaningful numbers:
    maturin develop --release

Run with:
    pytest benchmarks/ --benchmark-group-by=group --benchmark-sort=mean
"""

import itertools

import pytest

MODELS = ["cl100k_base", "o200k_base"]
BATCH_SIZES = [1, 10, 100, 1000]

# HuggingFace model identifiers (matching the Rust benchmarks)
HF_MODELS = {
    "cl100k_base": "Xenova/text-embedding-ada-002",
    "o200k_base": "Xenova/gpt-4o",
}


def _make_batch(lines, size):
    return [s for _, s in zip(range(size), itertools.cycle(lines))]


def _utf8_len(text):
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Single encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
class TestSingleEncode:
    def test_wordchipper_english(self, benchmark, model, english_text):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text)

    def test_wordchipper_diverse(self, benchmark, model, diverse_text):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text)

    def test_tiktoken_english(self, benchmark, model, english_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode_ordinary, english_text)

    def test_tiktoken_diverse(self, benchmark, model, diverse_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode_ordinary, diverse_text)

    def test_tokenizers_english(self, benchmark, model, english_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text)

    def test_tokenizers_diverse(self, benchmark, model, diverse_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text)


# ---------------------------------------------------------------------------
# Batch encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
class TestBatchEncode:
    def test_wordchipper_english(self, benchmark, model, batch_size, english_lines):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        batch = _make_batch(english_lines, batch_size)
        benchmark.group = f"batch/english/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_batch, batch)

    def test_wordchipper_diverse(self, benchmark, model, batch_size, diverse_lines):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        batch = _make_batch(diverse_lines, batch_size)
        benchmark.group = f"batch/diverse/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_batch, batch)

    def test_tiktoken_english(self, benchmark, model, batch_size, english_lines):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        batch = _make_batch(english_lines, batch_size)
        benchmark.group = f"batch/english/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_ordinary_batch, batch)

    def test_tiktoken_diverse(self, benchmark, model, batch_size, diverse_lines):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        batch = _make_batch(diverse_lines, batch_size)
        benchmark.group = f"batch/diverse/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_ordinary_batch, batch)

    def test_tokenizers_english(self, benchmark, model, batch_size, english_lines):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        batch = _make_batch(english_lines, batch_size)
        benchmark.group = f"batch/english/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_batch, batch)

    def test_tokenizers_diverse(self, benchmark, model, batch_size, diverse_lines):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        batch = _make_batch(diverse_lines, batch_size)
        benchmark.group = f"batch/diverse/{model}/n={batch_size}"
        benchmark.extra_info["input_bytes"] = sum(_utf8_len(s) for s in batch)
        benchmark(tok.encode_batch, batch)

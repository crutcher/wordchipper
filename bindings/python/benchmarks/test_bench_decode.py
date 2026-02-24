"""Python decode benchmarks: wordchipper vs tiktoken vs tokenizers.

Matches decoding_single.rs: pre-encode the corpus, then benchmark decode.
Throughput is measured as original text bytes / decode time.

Build the extension in release mode first for meaningful numbers:
    maturin develop --release

Run with:
    pytest benchmarks/
"""

import pytest

MODELS = ["cl100k_base", "o200k_base"]

HF_MODELS = {
    "cl100k_base": "Xenova/text-embedding-ada-002",
    "o200k_base": "Xenova/gpt-4o",
}


def _utf8_len(text):
    return len(text.encode("utf-8"))


@pytest.mark.parametrize("model", MODELS)
class TestSingleDecode:
    def test_wordchipper_english(self, benchmark, model, english_text):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        tokens = tok.encode(english_text)
        benchmark.group = f"decode/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.decode, tokens)

    def test_wordchipper_diverse(self, benchmark, model, diverse_text):
        import wordchipper

        tok = wordchipper.Tokenizer.from_pretrained(model)
        tokens = tok.encode(diverse_text)
        benchmark.group = f"decode/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.decode, tokens)

    def test_tiktoken_english(self, benchmark, model, english_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        tokens = tok.encode(english_text, allowed_special="all")
        benchmark.group = f"decode/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.decode, tokens)

    def test_tiktoken_diverse(self, benchmark, model, diverse_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        tokens = tok.encode(diverse_text, allowed_special="all")
        benchmark.group = f"decode/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.decode, tokens)

    def test_tokenizers_english(self, benchmark, model, english_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        tokens = tok.encode(english_text).ids
        benchmark.group = f"decode/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.decode, tokens)

    def test_tokenizers_diverse(self, benchmark, model, diverse_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        tokens = tok.encode(diverse_text).ids
        benchmark.group = f"decode/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.decode, tokens)

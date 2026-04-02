from abc import ABC, abstractmethod
import unittest

import pytest
import tokenizers
from wordchipper.compat import tokenizers as wc_tokenizers

# Cache HF tokenizer instances to avoid repeated network requests.
_hf_cache: dict[str, tokenizers.Tokenizer] = {}
_wc_cache: dict[str, wc_tokenizers.Tokenizer] = {}


def _get_hf(model: str) -> tokenizers.Tokenizer:
    if model not in _hf_cache:
        _hf_cache[model] = tokenizers.Tokenizer.from_pretrained(model)
    return _hf_cache[model]


def _get_wc(model: str) -> wc_tokenizers.Tokenizer:
    if model not in _wc_cache:
        _wc_cache[model] = wc_tokenizers.Tokenizer.from_pretrained(model)
    return _wc_cache[model]


# Only models that exist on HuggingFace Hub (Xenova/cl100k_base etc. are
# wordchipper-only mappings, not real HF repos).
HF_COMPARABLE_MODELS = [
    "Xenova/gpt-4o",
    "Xenova/gpt-4",
]

HF_DIVERSE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    "",
    " ",
    "   multiple   spaces   ",
    "\t\ttabs\t\t",
    "\n\nnewlines\n\n",
    "CJK: \u4f60\u597d\u4e16\u754c",
    "Emoji: \U0001f600\U0001f680\U0001f30d",
    "Mixed: Hello \u4e16\u754c! \U0001f389",
    "Numbers: 12345 3.14159 -42",
    "Code: def foo(x): return x + 1",
    "Accented: caf\u00e9 na\u00efve r\u00e9sum\u00e9",
    "Korean: \uc548\ub155\ud558\uc138\uc694",
    "Single char: x",
    "Repeated: aaaaaaaaaa",
    "Long: " + "word " * 50,
]


class TestHFTokenizerMatchesWordchipper:
    """Side-by-side comparison: real tokenizers vs wordchipper compat.

    Compares encode/decode outputs (token IDs and decoded text). Does NOT
    compare token_to_id/id_to_token/vocab_size because HF tokenizers use
    GPT-2 byte-level encoding (space -> Ġ, etc.) while wordchipper uses raw
    bytes, so the token string representations differ by design.
    """

    @pytest.mark.parametrize("model", HF_COMPARABLE_MODELS)
    @pytest.mark.parametrize("text", HF_DIVERSE_TEXTS, ids=lambda t: repr(t)[:40])
    def test_encode_ids_match(self, model, text):
        a = _get_hf(model)
        b = _get_wc(model)
        assert a.encode(text).ids == b.encode(text).ids

    @pytest.mark.parametrize("model", HF_COMPARABLE_MODELS)
    @pytest.mark.parametrize("text", HF_DIVERSE_TEXTS, ids=lambda t: repr(t)[:40])
    def test_decode_roundtrip_matches(self, model, text):
        a = _get_hf(model)
        b = _get_wc(model)
        tokens = a.encode(text).ids
        assert a.decode(tokens) == b.decode(tokens)

    @pytest.mark.parametrize("model", HF_COMPARABLE_MODELS)
    def test_encode_batch_ids_match(self, model):
        a = _get_hf(model)
        b = _get_wc(model)
        texts = ["hello world", "foo bar", "\u4f60\u597d", ""]
        a_batch = a.encode_batch(texts)
        b_batch = b.encode_batch(texts)
        assert len(a_batch) == len(b_batch)
        for a_enc, b_enc in zip(a_batch, b_batch):
            assert a_enc.ids == b_enc.ids

    @pytest.mark.parametrize("model", HF_COMPARABLE_MODELS)
    def test_pair_encode_ids_match(self, model):
        a = _get_hf(model)
        b = _get_wc(model)
        a_enc = a.encode("hello", pair="world")
        b_enc = b.encode("hello", pair="world")
        assert a_enc.ids == b_enc.ids

    @pytest.mark.parametrize("model", HF_COMPARABLE_MODELS)
    def test_encode_add_special_tokens_false_ids_match(self, model):
        a = _get_hf(model)
        b = _get_wc(model)
        a_enc = a.encode("hello", add_special_tokens=False)
        b_enc = b.encode("hello", add_special_tokens=False)
        assert a_enc.ids == b_enc.ids

    # Note: get_vocab_size is NOT compared cross-library because HF and
    # wordchipper have different vocab composition (HF includes all tokens
    # in its base vocabulary; wordchipper separates core vs special).


class TestHFEncoding:
    def test_encoding_dataclass(self):
        enc = wc_tokenizers.Encoding(ids=[1, 2, 3], tokens=["a", "b", "c"])
        assert enc.ids == [1, 2, 3]
        assert enc.tokens == ["a", "b", "c"]


class TokenizersBaseTests(ABC, unittest.TestCase):
    ENCODING_NAME = "Xenova/gpt-4o"

    @abstractmethod
    def get_module(self): ...

    def test_from_pretrained_hf_id(self):
        tok = self.get_module().Tokenizer.from_pretrained(self.ENCODING_NAME)
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def get_tok(self):
        return self.get_module().Tokenizer.from_pretrained(self.ENCODING_NAME)

    def test_encode_decode_roundtrip(self):
        tok = self.get_tok()
        text = "hello world"
        enc = tok.encode(text)
        assert tok.decode(enc.ids) == text

    def test_encode_batch(self):
        tok = self.get_tok()
        texts = ["hello", "world"]
        results = tok.encode_batch(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, self.get_module().Encoding)
            assert len(r.ids) == len(r.tokens)

    def test_decode_batch(self):
        tok = self.get_tok()
        texts = ["hello", "world"]
        batch = tok.encode_batch(texts)
        decoded = tok.decode_batch([r.ids for r in batch])
        assert decoded == texts

    def test_get_vocab_size(self):
        tok = self.get_tok()
        size = tok.get_vocab_size()
        assert isinstance(size, int)
        assert size > 0

    def test_token_to_id(self):
        tok = self.get_tok()
        tid = tok.token_to_id("hello")
        assert isinstance(tid, int)

    def test_token_to_id_unknown(self):
        tok = self.get_tok()
        assert tok.token_to_id("xyzzy_not_real_99999") is None

    def test_id_to_token(self):
        tok = self.get_tok()
        tid = tok.token_to_id("hello")
        assert tid is not None
        assert tok.id_to_token(tid) == "hello"

    def test_id_to_token_out_of_range(self):
        tok = self.get_tok()
        assert tok.id_to_token(999_999_999) is None

    def test_encode_empty(self):
        tok = self.get_tok()
        result = tok.encode("")
        assert result.ids == []
        assert result.tokens == []

    def test_decode_empty(self):
        tok = self.get_tok()
        assert tok.decode([]) == ""

    def test_encode_returns_encoding(self):
        tok = self.get_tok()
        result = tok.encode("hello world")
        assert isinstance(result, self.get_module().Encoding)
        assert isinstance(result.ids, list)
        assert isinstance(result.tokens, list)
        assert len(result.ids) == len(result.tokens)

    def test_encode_default_kwargs_still_work(self):
        tok = self.get_tok()
        result = tok.encode("hello", pair=None, is_pretokenized=False)
        assert len(result.ids) > 0


class TokenizersTests(TokenizersBaseTests):
    def get_module(self):
        return tokenizers


class CompatTokenizersTests(TokenizersBaseTests):
    def get_module(self):
        return wc_tokenizers

    def test_from_pretrained_bare_name(self):
        tok = self.get_module().Tokenizer.from_pretrained("cl100k_base")
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_from_pretrained_unknown(self):
        with pytest.raises(ValueError):
            self.get_module().Tokenizer.from_pretrained("Xenova/totally-unknown")

    def test_from_pretrained_extra_kwargs_ignored(self):
        tok = self.get_module().Tokenizer.from_pretrained(
            "cl100k_base", revision="main"
        )
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_encode_pair(self):
        tok = self.get_tok()
        enc = tok.encode("hello", pair="world")
        hello_enc = tok.encode("hello")
        world_enc = tok.encode("world")
        assert enc.ids == hello_enc.ids + world_enc.ids
        assert enc.tokens == hello_enc.tokens + world_enc.tokens

    def test_encode_batch_with_pairs(self):
        tok = self.get_tok()
        batch = tok.encode_batch([("hello", "world"), "foo"])
        assert len(batch) == 2
        pair_enc = tok.encode("hello", pair="world")
        assert batch[0].ids == pair_enc.ids
        foo_enc = tok.encode("foo")
        assert batch[1].ids == foo_enc.ids

    def test_encode_add_special_tokens_false(self):
        tok = self.get_tok()
        enc_true = tok.encode("hello", add_special_tokens=True)
        enc_false = tok.encode("hello", add_special_tokens=False)
        assert enc_true.ids == enc_false.ids

    def test_encode_batch_add_special_tokens_false(self):
        tok = self.get_tok()
        batch_true = tok.encode_batch(["hello"], add_special_tokens=True)
        batch_false = tok.encode_batch(["hello"], add_special_tokens=False)
        assert batch_true[0].ids == batch_false[0].ids

    def test_encode_raises_on_is_pretokenized(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode("hello", is_pretokenized=True)

    def test_encode_batch_raises_on_is_pretokenized(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode_batch(["hello"], is_pretokenized=True)

    def test_decode_skip_special_tokens_false(self):
        tok = self.get_tok()
        text = "hello"
        ids = tok.encode(text).ids
        assert tok.decode(ids, skip_special_tokens=False) == text

    def test_decode_batch_skip_special_tokens_false(self):
        tok = self.get_tok()
        texts = ["hello", "world"]
        batch = tok.encode_batch(texts)
        decoded = tok.decode_batch(
            [r.ids for r in batch], skip_special_tokens=False
        )
        assert decoded == texts

    def test_decode_skip_special_tokens_filters(self):
        tok = self.get_tok()
        hello_ids = tok.encode("hello").ids
        special_ids = list(tok._tok.specials.values())
        mixed = hello_ids + special_ids
        assert tok.decode(mixed, skip_special_tokens=True) == "hello"
        # False includes the special token text
        decoded_with = tok.decode(mixed, skip_special_tokens=False)
        assert len(decoded_with) > len("hello")

    def test_get_vocab_size_without_added_tokens(self):
        tok = self.get_tok()
        full = tok.get_vocab_size(with_added_tokens=True)
        core = tok.get_vocab_size(with_added_tokens=False)
        assert core <= full
        assert core == tok._tok.vocab_size

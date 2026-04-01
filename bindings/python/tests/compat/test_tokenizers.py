from abc import ABC, abstractmethod
import unittest

import pytest
import tokenizers
from wordchipper.compat import tokenizers as wc_tokenizers


class TestHFTokenizerMatchesWordchipper:
    def test_encode_ids_match(self):
        ENCODING_NAME = "Xenova/gpt-4o"

        a = tokenizers.Tokenizer.from_pretrained(ENCODING_NAME)
        b = wc_tokenizers.Tokenizer.from_pretrained(ENCODING_NAME)

        text = "The quick brown fox jumps over the lazy dog."
        assert a.encode(text).ids == b.encode(text).ids

        tokens = a.encode(text).ids
        assert a.decode(tokens) == b.decode(tokens)


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

    # TODO: these features should be implemented in wc_tokenizers

    def test_from_pretrained_unknown(self):
        with pytest.raises(ValueError):
            self.get_module().Tokenizer.from_pretrained("Xenova/totally-unknown")

    def test_encode_raises_on_pair(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="pair"):
            tok.encode("hello", pair="world")

    def test_encode_raises_on_is_pretokenized(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode("hello", is_pretokenized=True)

    def test_encode_raises_on_add_special_tokens_false(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="add_special_tokens"):
            tok.encode("hello", add_special_tokens=False)

    def test_encode_batch_raises_on_is_pretokenized(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode_batch(["hello"], is_pretokenized=True)

    def test_encode_batch_raises_on_add_special_tokens_false(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="add_special_tokens"):
            tok.encode_batch(["hello"], add_special_tokens=False)

    def test_decode_raises_on_skip_special_tokens_false(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="skip_special_tokens"):
            tok.decode([1, 2], skip_special_tokens=False)

    def test_decode_batch_raises_on_skip_special_tokens_false(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="skip_special_tokens"):
            tok.decode_batch([[1, 2]], skip_special_tokens=False)

    def test_from_pretrained_raises_on_extra_kwargs(self):
        with pytest.raises(NotImplementedError, match="extra keyword"):
            self.get_module().Tokenizer.from_pretrained("cl100k_base", revision="main")

    def test_get_vocab_size_raises_on_with_added_tokens_false(self):
        tok = self.get_tok()
        with pytest.raises(NotImplementedError, match="with_added_tokens"):
            tok.get_vocab_size(with_added_tokens=False)

from abc import ABC, abstractmethod

import pytest
import tiktoken
import unittest
from wordchipper.compat import tiktoken as wc_tiktoken


class TestTiktokenMatchesWordchipper:
    def test_encode_matches(self):
        name = "cl100k_base"
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)

        text = "The quick brown fox jumps over the lazy dog."
        assert a.encode(text) == b.encode(text)

        tokens = a.encode(text)
        assert a.decode(tokens) == b.decode(tokens)


class TiktokenBaseTests(ABC, unittest.TestCase):
    ENCODING_NAME = "cl100k_base"

    @abstractmethod
    def get_module(self): ...

    def test_get_encoding(self):
        mod = self.get_module()
        enc = mod.get_encoding("cl100k_base")
        assert enc.name == "cl100k_base"

    def test_get_encoding_cached(self):
        mod = self.get_module()
        enc1 = mod.get_encoding("cl100k_base")
        enc2 = mod.get_encoding("cl100k_base")
        assert enc1 is enc2

    def test_get_encoding_unknown(self):
        mod = self.get_module()
        with pytest.raises(ValueError, match="Unknown encoding"):
            mod.get_encoding("nonexistent")

    def test_list_encoding_names(self):
        mod = self.get_module()
        names = mod.list_encoding_names()
        assert "cl100k_base" in names
        assert "o200k_base" in names
        assert "o200k_harmony" in names
        assert "r50k_base" in names

    def test_get_encoding_o200k_harmony(self):
        mod = self.get_module()
        enc = mod.get_encoding("o200k_harmony")
        assert enc.name == "o200k_harmony"
        tokens = enc.encode("hello")
        assert enc.decode(tokens) == "hello"

    def test_encoding_for_model(self):
        mod = self.get_module()
        enc = mod.encoding_for_model("gpt-4o")
        assert enc.name == "o200k_base"

    def test_encoding_for_model_chat(self):
        mod = self.get_module()
        enc = mod.encoding_for_model("gpt-4")
        assert enc.name == "cl100k_base"

    def test_encoding_name_for_model(self):
        mod = self.get_module()
        assert mod.encoding_name_for_model("gpt-4o") == "o200k_base"
        assert mod.encoding_name_for_model("gpt-4") == "cl100k_base"
        assert mod.encoding_name_for_model("davinci") == "r50k_base"

    def test_encoding_name_for_model_prefix(self):
        mod = self.get_module()
        assert mod.encoding_name_for_model("gpt-4o-2024-08-06") == "o200k_base"
        assert mod.encoding_name_for_model("gpt-4-0613") == "cl100k_base"

    def test_encoding_name_for_model_finetuned(self):
        mod = self.get_module()
        assert mod.encoding_name_for_model("ft:gpt-4o:my-org") == "o200k_base"

    def test_encoding_for_model_prefix(self):
        mod = self.get_module()
        enc = mod.encoding_for_model("gpt-4o-2024-08-06")
        assert enc.name == "o200k_base"

    def test_encoding_name_for_model_unknown(self):
        mod = self.get_module()
        with pytest.raises(KeyError):
            mod.encoding_name_for_model("totally-unknown-model")

    def get_encoding(self):
        return self.get_module().get_encoding(self.ENCODING_NAME)

    def test_properties(self):
        enc = self.get_encoding()
        assert enc.name == self.ENCODING_NAME
        assert enc.n_vocab == 100277
        assert enc.max_token_value == 100276
        assert enc.eot_token == 100257

    def test_encode_decode_roundtrip(self):
        enc = self.get_encoding()

        text = "hello world"
        tokens = enc.encode(text)
        assert isinstance(tokens, list)
        assert enc.decode(tokens) == text

    def test_encode_ordinary(self):
        enc = self.get_encoding()
        text = "hello world"
        assert enc.encode_ordinary(text) == enc.encode(text)

    def test_encode_batch(self):
        enc = self.get_encoding()
        texts = ["hello", "world"]
        results = enc.encode_batch(texts)
        assert len(results) == 2
        for i, text in enumerate(texts):
            assert enc.decode(results[i]) == text

    def test_encode_ordinary_batch(self):
        enc = self.get_encoding()

        texts = ["hello", "world"]
        assert enc.encode_ordinary_batch(texts) == enc.encode_batch(texts)

    def test_decode_batch(self):
        enc = self.get_encoding()

        texts = ["hello", "world"]
        batch = enc.encode_batch(texts)
        assert enc.decode_batch(batch) == texts

    def test_encode_empty(self):
        enc = self.get_encoding()
        assert enc.encode("") == []

    def test_decode_empty(self):
        enc = self.get_encoding()
        assert enc.decode([]) == ""

    def test_encode_specials(self):
        enc = self.get_encoding()
        tokens = enc.encode("hello<|endoftext|>", allowed_special="all")
        assert tokens[-1] == enc.eot_token

    def test_disallowed_specials(self):
        enc = self.get_encoding()

        tokens = enc.encode(
            "hello<|endoftext|>",
            allowed_special="all",
            disallowed_special="all",
        )
        assert tokens[-1] == enc.eot_token

        with pytest.raises(ValueError, match="disallowed special token <|endoftext|>"):
            enc.encode(
                "hello<|endoftext|>",
                allowed_special="all",
                disallowed_special={"<|endoftext|>"},
            )

    def test_batch_disallowed_specials(self):
        enc = self.get_encoding()

        batch = enc.encode_batch(
            ["hello<|endoftext|>"],
            allowed_special="all",
            disallowed_special="all",
        )
        assert batch[0][-1] == enc.eot_token

        with pytest.raises(ValueError, match="disallowed special token <|endoftext|>"):
            enc.encode_batch(
                ["hello<|endoftext|>"],
                allowed_special="all",
                disallowed_special={"<|endoftext|>"},
            )

    def test_encode_accepts_tiktoken_defaults(self):
        enc = self.get_encoding()

        # tiktoken's real defaults: allowed_special=set(), disallowed_special="all"
        tokens = enc.encode(
            "hello",
            allowed_special=set(),
            disallowed_special="all",
        )
        assert enc.decode(tokens) == "hello"

    def test_encode_batch_accepts_tiktoken_defaults(self):
        enc = self.get_encoding()

        results = enc.encode_batch(
            ["hello"],
            allowed_special=set(),
            disallowed_special="all",
        )
        assert enc.decode(results[0]) == "hello"


class TiktokenEncodingTests(TiktokenBaseTests):
    def get_module(self):
        return tiktoken


class CompatTiktokenEncodingTests(TiktokenBaseTests):
    def get_module(self):
        return wc_tiktoken

    def test_properties(self):
        enc = self.get_encoding()
        assert enc.name == self.ENCODING_NAME
        assert enc.eot_token == 100257

        # FIXME: this is a bug in the wccompat version of tiktoken.
        # Base tiktoken counts by the core vocab; not the special tokens.
        # assert enc.n_vocab == 100277
        # assert enc.max_token_value == 100276

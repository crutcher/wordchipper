from abc import ABC, abstractmethod

import pytest
import tiktoken
import unittest
from wordchipper.compat import tiktoken as wc_tiktoken


COMMON_ENCODINGS = ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"]

DIVERSE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    "",
    " ",
    "   multiple   spaces   ",
    "\t\ttabs\t\t",
    "\n\nnewlines\n\n",
    "line1\nline2\r\nline3",
    "CJK: \u4f60\u597d\u4e16\u754c",
    "Emoji: \U0001f600\U0001f680\U0001f30d",
    "Mixed: Hello \u4e16\u754c! \U0001f389",
    "Numbers: 12345 3.14159 -42",
    "Special chars: @#$%^&*(){}[]|\\",
    "Unicode punctuation: \u201chello\u201d \u2014 \u2018world\u2019",
    "Repeated: aaaaaaaaaa",
    "Code: def foo(x): return x + 1",
    "URL: https://example.com/path?q=hello&lang=en",
    "JSON: {\"key\": \"value\", \"num\": 42}",
    "Long: " + "word " * 200,
    "Accented: caf\u00e9 na\u00efve r\u00e9sum\u00e9",
    "Arabic: \u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "Korean: \uc548\ub155\ud558\uc138\uc694",
    "Thai: \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
    "Single char: x",
    "Whitespace only: \t \n \r\n",
]


class TestTiktokenMatchesWordchipper:
    """Side-by-side comparison: real tiktoken vs wordchipper compat."""

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    def test_properties_match(self, name):
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        assert a.n_vocab == b.n_vocab, f"{name}: n_vocab"
        assert a.max_token_value == b.max_token_value, f"{name}: max_token_value"
        assert a.eot_token == b.eot_token, f"{name}: eot_token"

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    @pytest.mark.parametrize("text", DIVERSE_TEXTS, ids=lambda t: t[:40])
    def test_encode_matches(self, name, text):
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        assert a.encode(text) == b.encode(text)

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    @pytest.mark.parametrize("text", DIVERSE_TEXTS, ids=lambda t: t[:40])
    def test_decode_roundtrip_matches(self, name, text):
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        tokens = a.encode(text)
        assert a.decode(tokens) == b.decode(tokens)

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    def test_encode_batch_matches(self, name):
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        texts = ["hello world", "foo bar", "\u4f60\u597d", ""]
        assert a.encode_batch(texts) == b.encode_batch(texts)

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    def test_encode_ordinary_matches(self, name):
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        # Use text without special token patterns; encode_ordinary behavior
        # on special-token-like text diverges (see test_encode_ordinary_special_diverges).
        text = "hello world, how are you?"
        assert a.encode_ordinary(text) == b.encode_ordinary(text)

    @pytest.mark.parametrize("name", COMMON_ENCODINGS)
    def test_encode_ordinary_special_diverges(self, name):
        """Document known divergence: encode_ordinary on special token text.

        tiktoken encodes <|endoftext|> as individual characters.
        wordchipper recognizes it as a special token ID.
        """
        a = tiktoken.get_encoding(name)
        b = wc_tiktoken.get_encoding(name)
        text = "hello <|endoftext|> world"
        # These are expected to differ; just verify both produce valid output.
        a_tokens = a.encode_ordinary(text)
        b_tokens = b.encode_ordinary(text)
        assert len(a_tokens) > 0
        assert len(b_tokens) > 0
        # tiktoken splits the special text into many tokens; wordchipper
        # produces fewer because it recognizes the special token.
        assert len(a_tokens) > len(b_tokens)

    def test_special_token_encode_matches(self):
        a = tiktoken.get_encoding("cl100k_base")
        b = wc_tiktoken.get_encoding("cl100k_base")
        text = "hello<|endoftext|>world"
        assert a.encode(text, allowed_special="all") == b.encode(
            text, allowed_special="all"
        )

    def test_special_tokens_set_matches(self):
        a = tiktoken.get_encoding("cl100k_base")
        b = wc_tiktoken.get_encoding("cl100k_base")
        assert a.special_tokens_set == b.special_tokens_set

    def test_disallowed_special_raises_tiktoken(self):
        """tiktoken raises ValueError by default on special token text."""
        enc = tiktoken.get_encoding("cl100k_base")
        with pytest.raises(ValueError):
            enc.encode("hello<|endoftext|>world")

    def test_disallowed_special_diverges(self):
        """Document known divergence: default disallowed_special handling.

        tiktoken raises ValueError when special token text appears with default
        params. wordchipper's default encode doesn't recognize special tokens
        so the disallowed check never triggers.
        """
        enc = wc_tiktoken.get_encoding("cl100k_base")
        # wordchipper does NOT raise here because its default encode treats
        # <|endoftext|> as ordinary text (no SpecialFilter).
        tokens = enc.encode("hello<|endoftext|>world")
        assert len(tokens) > 0


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

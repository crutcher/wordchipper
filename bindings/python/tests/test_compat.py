"""Tests for wordchipper.compat (tiktoken and tokenizers compatibility layers)."""

import pytest
from wordchipper import Tokenizer
from wordchipper.compat import tiktoken
from wordchipper.compat.tokenizers import Encoding as HFEncoding
from wordchipper.compat.tokenizers import Tokenizer as HFTokenizer

# ===================================================================
# tiktoken compat
# ===================================================================


class TestTiktokenGetEncoding:
    def test_get_encoding(self):
        enc = tiktoken.get_encoding("cl100k_base")
        assert enc.name == "cl100k_base"

    def test_get_encoding_cached(self):
        enc1 = tiktoken.get_encoding("cl100k_base")
        enc2 = tiktoken.get_encoding("cl100k_base")
        assert enc1 is enc2

    def test_get_encoding_unknown(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            tiktoken.get_encoding("nonexistent")

    def test_list_encoding_names(self):
        names = tiktoken.list_encoding_names()
        assert "cl100k_base" in names
        assert "o200k_base" in names
        assert "o200k_harmony" in names
        assert "r50k_base" in names

    def test_get_encoding_o200k_harmony(self):
        enc = tiktoken.get_encoding("o200k_harmony")
        assert enc.name == "o200k_harmony"
        tokens = enc.encode("hello")
        assert enc.decode(tokens) == "hello"


class TestTiktokenModelMapping:
    def test_encoding_for_model(self):
        enc = tiktoken.encoding_for_model("gpt-4o")
        assert enc.name == "o200k_base"

    def test_encoding_for_model_chat(self):
        enc = tiktoken.encoding_for_model("gpt-4")
        assert enc.name == "cl100k_base"

    def test_encoding_name_for_model(self):
        assert tiktoken.encoding_name_for_model("gpt-4o") == "o200k_base"
        assert tiktoken.encoding_name_for_model("gpt-4") == "cl100k_base"
        assert tiktoken.encoding_name_for_model("davinci") == "r50k_base"

    def test_encoding_name_for_model_prefix(self):
        assert tiktoken.encoding_name_for_model("gpt-4o-2024-08-06") == "o200k_base"
        assert tiktoken.encoding_name_for_model("gpt-4-0613") == "cl100k_base"

    def test_encoding_name_for_model_finetuned(self):
        assert tiktoken.encoding_name_for_model("ft:gpt-4o:my-org") == "o200k_base"

    def test_encoding_for_model_prefix(self):
        enc = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        assert enc.name == "o200k_base"

    def test_encoding_name_for_model_unknown(self):
        with pytest.raises(KeyError):
            tiktoken.encoding_name_for_model("totally-unknown-model")


class TestTiktokenEncoding:
    @pytest.fixture(scope="class")
    def enc(self):
        return tiktoken.get_encoding("cl100k_base")

    def test_encode_decode_roundtrip(self, enc):
        text = "hello world"
        tokens = enc.encode(text)
        assert isinstance(tokens, list)
        assert enc.decode(tokens) == text

    def test_encode_ordinary(self, enc):
        text = "hello world"
        assert enc.encode_ordinary(text) == enc.encode(text)

    def test_encode_batch(self, enc):
        texts = ["hello", "world"]
        results = enc.encode_batch(texts)
        assert len(results) == 2
        for i, text in enumerate(texts):
            assert enc.decode(results[i]) == text

    def test_encode_ordinary_batch(self, enc):
        texts = ["hello", "world"]
        assert enc.encode_ordinary_batch(texts) == enc.encode_batch(texts)

    def test_decode_batch(self, enc):
        texts = ["hello", "world"]
        batch = enc.encode_batch(texts)
        assert enc.decode_batch(batch) == texts

    def test_encode_empty(self, enc):
        assert enc.encode("") == []

    def test_decode_empty(self, enc):
        assert enc.decode([]) == ""

    def test_encode_specials(self, enc):
        tokens = enc.encode("hello<|endoftext|>", allowed_special="all")
        assert tokens[-1] == enc.eot_token

    def test_disallowed_specials(self, enc):
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

    def test_batch_disallowed_specials(self, enc):
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

    def test_encode_accepts_tiktoken_defaults(self, enc):
        # tiktoken's real defaults: allowed_special=set(), disallowed_special="all"
        tokens = enc.encode(
            "hello",
            allowed_special=set(),
            disallowed_special="all",
        )
        assert enc.decode(tokens) == "hello"

    def test_encode_batch_accepts_tiktoken_defaults(self, enc):
        results = enc.encode_batch(
            ["hello"],
            allowed_special=set(),
            disallowed_special="all",
        )
        assert enc.decode(results[0]) == "hello"


class TestTiktokenProperties:
    @pytest.fixture(scope="class")
    def enc(self):
        return tiktoken.get_encoding("cl100k_base")

    def test_n_vocab(self, enc):
        assert enc.n_vocab > 0

    def test_max_token_value(self, enc):
        assert enc.max_token_value > 0

    def test_eot_token(self, enc):
        assert isinstance(enc.eot_token, int)

    def test_special_tokens_set(self, enc):
        specials = enc.special_tokens_set
        assert isinstance(specials, set)
        assert "<|endoftext|>" in specials


class TestTiktokenMatchesWordchipper:
    def test_encode_matches(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tok = Tokenizer.from_pretrained("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        assert enc.encode(text) == tok.encode(text)

    def test_decode_matches(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tok = Tokenizer.from_pretrained("cl100k_base")
        tokens = tok.encode("hello world")
        assert enc.decode(tokens) == tok.decode(tokens)


# ===================================================================
# tokenizers compat
# ===================================================================


class TestHFEncoding:
    def test_encoding_dataclass(self):
        enc = HFEncoding(ids=[1, 2, 3], tokens=["a", "b", "c"])
        assert enc.ids == [1, 2, 3]
        assert enc.tokens == ["a", "b", "c"]


class TestHFTokenizer:
    @pytest.fixture(scope="class")
    def tok(self):
        return HFTokenizer.from_pretrained("Xenova/gpt-4o")

    def test_from_pretrained_hf_id(self):
        tok = HFTokenizer.from_pretrained("Xenova/gpt-4o")
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_from_pretrained_bare_name(self):
        tok = HFTokenizer.from_pretrained("cl100k_base")
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_from_pretrained_unknown(self):
        with pytest.raises(ValueError):
            HFTokenizer.from_pretrained("Xenova/totally-unknown")

    def test_encode_returns_encoding(self, tok):
        result = tok.encode("hello world")
        assert isinstance(result, HFEncoding)
        assert isinstance(result.ids, list)
        assert isinstance(result.tokens, list)
        assert len(result.ids) == len(result.tokens)

    def test_encode_decode_roundtrip(self, tok):
        text = "hello world"
        enc = tok.encode(text)
        assert tok.decode(enc.ids) == text

    def test_encode_batch(self, tok):
        texts = ["hello", "world"]
        results = tok.encode_batch(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, HFEncoding)
            assert len(r.ids) == len(r.tokens)

    def test_decode_batch(self, tok):
        texts = ["hello", "world"]
        batch = tok.encode_batch(texts)
        decoded = tok.decode_batch([r.ids for r in batch])
        assert decoded == texts

    def test_get_vocab_size(self, tok):
        size = tok.get_vocab_size()
        assert isinstance(size, int)
        assert size > 0

    def test_token_to_id(self, tok):
        tid = tok.token_to_id("hello")
        assert isinstance(tid, int)

    def test_token_to_id_unknown(self, tok):
        assert tok.token_to_id("xyzzy_not_real_99999") is None

    def test_id_to_token(self, tok):
        tid = tok.token_to_id("hello")
        assert tid is not None
        assert tok.id_to_token(tid) == "hello"

    def test_id_to_token_out_of_range(self, tok):
        assert tok.id_to_token(999_999_999) is None

    def test_encode_empty(self, tok):
        result = tok.encode("")
        assert result.ids == []
        assert result.tokens == []

    def test_decode_empty(self, tok):
        assert tok.decode([]) == ""

    def test_encode_raises_on_pair(self, tok):
        with pytest.raises(NotImplementedError, match="pair"):
            tok.encode("hello", pair="world")

    def test_encode_raises_on_is_pretokenized(self, tok):
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode("hello", is_pretokenized=True)

    def test_encode_raises_on_add_special_tokens_false(self, tok):
        with pytest.raises(NotImplementedError, match="add_special_tokens"):
            tok.encode("hello", add_special_tokens=False)

    def test_encode_batch_raises_on_is_pretokenized(self, tok):
        with pytest.raises(NotImplementedError, match="is_pretokenized"):
            tok.encode_batch(["hello"], is_pretokenized=True)

    def test_encode_batch_raises_on_add_special_tokens_false(self, tok):
        with pytest.raises(NotImplementedError, match="add_special_tokens"):
            tok.encode_batch(["hello"], add_special_tokens=False)

    def test_decode_raises_on_skip_special_tokens_false(self, tok):
        with pytest.raises(NotImplementedError, match="skip_special_tokens"):
            tok.decode([1, 2], skip_special_tokens=False)

    def test_decode_batch_raises_on_skip_special_tokens_false(self, tok):
        with pytest.raises(NotImplementedError, match="skip_special_tokens"):
            tok.decode_batch([[1, 2]], skip_special_tokens=False)

    def test_from_pretrained_raises_on_extra_kwargs(self):
        with pytest.raises(NotImplementedError, match="extra keyword"):
            HFTokenizer.from_pretrained("cl100k_base", revision="main")

    def test_get_vocab_size_raises_on_with_added_tokens_false(self, tok):
        with pytest.raises(NotImplementedError, match="with_added_tokens"):
            tok.get_vocab_size(with_added_tokens=False)

    def test_encode_default_kwargs_still_work(self, tok):
        result = tok.encode("hello", pair=None, is_pretokenized=False)
        assert len(result.ids) > 0


class TestHFTokenizerMatchesWordchipper:
    def test_encode_ids_match(self):
        hf = HFTokenizer.from_pretrained("cl100k_base")
        wc = Tokenizer.from_pretrained("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        assert hf.encode(text).ids == wc.encode(text)

    def test_vocab_size_matches(self):
        hf = HFTokenizer.from_pretrained("cl100k_base")
        wc = Tokenizer.from_pretrained("cl100k_base")
        assert hf.get_vocab_size() == wc.vocab_size


# ===================================================================
# Cross-validation against real tiktoken
# ===================================================================

_real_tiktoken = pytest.importorskip("tiktoken")

_CROSS_VALIDATION_TEXTS = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing spaces  ",
    "line\nbreaks\n\nand\ttabs",
    "Unicode: \u00e9\u00e0\u00fc \u4f60\u597d \U0001f680",
    "",
    "a" * 1000,
    "contractions: I'm can't won't they're",
    "code: def foo(x): return x + 1",
    "numbers 12345 and symbols !@#$%^&*()",
]


class TestTiktokenCrossValidation:
    """Compare our tiktoken compat layer against real tiktoken output."""

    @pytest.fixture(
        scope="class",
        params=["cl100k_base", "o200k_base"],
    )
    def encoding_pair(self, request):
        name = request.param
        ours = tiktoken.get_encoding(name)
        real = _real_tiktoken.get_encoding(name)
        return ours, real

    @pytest.mark.parametrize("text", _CROSS_VALIDATION_TEXTS)
    def test_encode_matches(self, encoding_pair, text):
        ours, real = encoding_pair
        assert ours.encode(text) == real.encode(text)

    @pytest.mark.parametrize("text", _CROSS_VALIDATION_TEXTS)
    def test_encode_ordinary_matches(self, encoding_pair, text):
        ours, real = encoding_pair
        assert ours.encode_ordinary(text) == real.encode_ordinary(text)

    @pytest.mark.parametrize("text", _CROSS_VALIDATION_TEXTS)
    def test_decode_matches(self, encoding_pair, text):
        ours, real = encoding_pair
        tokens = real.encode(text)
        assert ours.decode(tokens) == real.decode(tokens)

    def test_n_vocab_close(self, encoding_pair):
        ours, real = encoding_pair
        # wordchipper may not expose all special tokens that tiktoken includes,
        # so our n_vocab can be slightly smaller
        assert ours.n_vocab <= real.n_vocab
        assert ours.n_vocab >= real.n_vocab - 100

    @pytest.mark.parametrize(
        "model_name",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    )
    def test_encoding_for_model_matches(self, model_name):
        ours = tiktoken.encoding_for_model(model_name)
        real = _real_tiktoken.encoding_for_model(model_name)
        assert ours.name == real.name

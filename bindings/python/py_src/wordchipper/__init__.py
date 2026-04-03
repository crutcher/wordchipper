import functools
from typing import Optional

from wordchipper._wordchipper import SpecialFilter, _Tokenizer, _Vocab, TokenizerOptions

try:
    frozendict
except NameError:
    try:
        from frozendict import frozendict
    except ImportError:
        frozendict = dict


class Vocab:
    """Vocabulary for a BPE tokenizer.

    Supports dict-like access: ``vocab["hello"]`` returns the token ID.
    """

    _inner: _Vocab

    def __init__(self, inner: _Vocab) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, token: str) -> bool:
        return token in self._inner

    def __getitem__(self, token: str) -> int:
        return self._inner[token]

    @property
    def n_vocab(self) -> int:
        """Total number of token IDs in the vocabulary (core + special)."""
        return self._inner.n_vocab

    @property
    def max_token(self) -> int | None:
        """Highest token ID across core and special tokens."""
        return self._inner.max_token

    def token_to_id(self, token: str) -> int | None:
        """Look up the token ID for a token string. Returns None if not found."""
        return self._inner.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Look up the token string for a token ID. Returns None if not found."""
        return self._inner.id_to_token(id)

    def id_to_token_bytes(self, id: int) -> bytes | None:
        """Look up the raw bytes for a token ID. Returns None if not found."""
        return self._inner.id_to_token_bytes(id)

    def ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """Look up token strings for a list of token IDs in a single call."""
        return self._inner.ids_to_tokens(ids)

    @functools.cached_property
    def special_tokens(self) -> frozendict[str, int]:
        """Special tokens as a frozen mapping of name to ID."""
        return frozendict(self._inner.get_special_tokens())

    def to_dict(self) -> dict[str, int]:
        """Return the full vocabulary as a {token_string: id} dict."""
        return self._inner.to_dict()


class Tokenizer:
    """A BPE tokenizer."""

    _tok: _Tokenizer

    @staticmethod
    def available_models() -> list[str]:
        """List available pretrained model names."""
        return _Tokenizer.available_models()

    @staticmethod
    def from_pretrained(
        name: str, options: Optional[TokenizerOptions] = None
    ) -> "Tokenizer":
        """Load a pretrained OpenAI tokenizer by name.

        Names: "r50k_base", "p50k_base", "p50k_edit", "cl100k_base",
               "o200k_base", "o200k_harmony"
        """
        tok = _Tokenizer.from_pretrained(name, options)
        return Tokenizer(tok)

    def __init__(self, tok: _Tokenizer) -> "Tokenizer":
        self._tok = tok

    @functools.cached_property
    def vocab(self) -> Vocab:
        """The tokenizer's vocabulary."""
        return Vocab(self._tok.vocab)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the core vocabulary (excludes special tokens)."""
        return self._tok.vocab_size

    @property
    def max_token(self) -> int | None:
        """Highest token ID in the core vocabulary, or None if empty."""
        return self._tok.max_token

    def token_to_id(self, token: str) -> int | None:
        """Look up the token ID for a token string. Returns None if not found."""
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Look up the token string for a token ID. Returns None if not found."""
        return self._tok.id_to_token(id)

    @functools.cached_property
    def specials(self) -> frozendict[str, int]:
        """Special tokens as a frozen mapping of name to ID."""
        return frozendict(self._tok.get_special_tokens())

    def save_base64_vocab(self, path: str) -> None:
        """Save the vocabulary to a file in base64 tiktoken format (excludes special tokens)."""
        self._tok.save_base64_vocab(path)

    def encode(
        self,
        text: str,
        special_filter: Optional[SpecialFilter] = None,
    ) -> list[int]:
        """Encode a single string to a list of token IDs."""
        return self._tok.encode(text, special_filter=special_filter)

    def encode_batch(
        self,
        texts: list[str],
        special_filter: Optional[SpecialFilter] = None,
    ) -> list[list[int]]:
        """Encode a batch of strings to a list of lists of token IDs."""
        return self._tok.encode_batch(texts, special_filter=special_filter)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs to a string."""
        return self._tok.decode(tokens)

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode a list of token IDs to raw bytes."""
        return self._tok.decode_bytes(tokens)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists to a list of strings."""
        return self._tok.decode_batch(batch)

    def decode_bytes_batch(self, batch: list[list[int]]) -> list[bytes]:
        """Decode a batch of token ID lists to a list of byte strings."""
        return self._tok.decode_bytes_batch(batch)


__all__ = ["SpecialFilter", "Tokenizer", "TokenizerOptions", "Vocab"]

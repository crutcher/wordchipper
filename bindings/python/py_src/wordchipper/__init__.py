import functools
from typing import Optional

from wordchipper._wordchipper import SpecialFilter, _Tokenizer, TokenizerOptions

try:
    frozendict
except NameError:
    try:
        from frozendict import frozendict
    except ImportError:
        frozendict = dict


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

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self._tok.vocab_size

    @property
    def max_token(self) -> int | None:
        """Highest token ID in the vocabulary, or None if the vocabulary is empty."""
        return self._tok.max_token

    def token_to_id(self, token: str) -> int | None:
        """Look up the token ID for a token string. Returns None if not found."""
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Look up the token string for a token ID. Returns None if not found."""
        return self._tok.id_to_token(id)

    @functools.cached_property
    def specials(self) -> frozendict[str, int]:
        """Get special tokens as a list of (name, id) tuples.

        Note: The order of returned tuples is not guaranteed.
        """
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

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists to a list of strings."""
        return self._tok.decode_batch(batch)


__all__ = ["SpecialFilter", "Tokenizer", "TokenizerOptions"]

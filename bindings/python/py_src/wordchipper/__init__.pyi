from typing import Optional


class SpecialFilter:
    """A filter for special tokens."""

    @staticmethod
    def include_all() -> "SpecialFilter":
        """Include all special tokens."""
        ...

    @staticmethod
    def include_none() -> "SpecialFilter":
        """Exclude all special tokens."""
        ...

    def include(self, tokens: list[str]) -> "SpecialFilter":
        """Create a filter that includes only the specified special tokens."""
        ...

    def __contains__(self, item: str) -> bool:
        """Checks if the item is in the filter."""
        ...


class TokenizerOptions:
    """Options for building Tokenizer."""

    @staticmethod
    def default() -> "TokenizerOptions":
        """Get default options."""
        ...

    def parallel(self) -> bool:
        """Is a multithreaded tokenizer requested?"""
        ...

    def set_parallel(self, parallel: bool) -> None:
        """Enable/disable multi-threaded implementation."""
        ...

    def accelerated_lexers(self) -> bool:
        """Whether accelerated lexers should be enabled.

        When enabled, and an accelerated lexer can be
        found for a given regex pattern; the regex accelerator
        will be used for spanners.
        """
        ...

    def set_accelerated_lexers(self, accel: bool) -> None:
        """Enable/disable accelerated lexers."""
        ...

    def is_concurrent(self) -> bool:
        """Is concurrent support requested?

        Returns self.parallel() || self.concurrent()
        """
        ...

    def concurrent(self) -> bool:
        """Is the tokenizer going to be used in concurrent contexts?"""
        ...

    def set_concurrent(self, concurrent: bool) -> None:
        """Enable/disable concurrent support."""
        ...


class Tokenizer:
    """A BPE tokenizer."""

    @staticmethod
    def from_pretrained(name: str, options=Optional[TokenizerOptions]) -> "Tokenizer":
        """Load a pretrained OpenAI tokenizer by name.

        Names: "r50k_base", "p50k_base", "p50k_edit", "cl100k_base",
               "o200k_base", "o200k_harmony"
        """
        ...

    def encode(self, text: str, special_filter: Optional[SpecialFilter] = None) -> list[int]:
        """Encode a single string to a list of token IDs."""
        ...

    def encode_batch(self, texts: list[str], special_filter: Optional[SpecialFilter] = None) -> list[list[int]]:
        """Encode a batch of strings to a list of lists of token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs to a string."""
        ...

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists to a list of strings."""
        ...

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        ...

    @property
    def max_token(self) -> int | None:
        """Highest token ID in the vocabulary, or None if the vocabulary is empty."""
        ...

    def token_to_id(self, token: str) -> int | None:
        """Look up the token ID for a token string. Returns None if not found."""
        ...

    def id_to_token(self, id: int) -> str | None:
        """Look up the token string for a token ID. Returns None if not found."""
        ...

    def get_special_tokens(self) -> list[tuple[str, int]]:
        """Get special tokens as a list of (name, id) tuples.

        Note: The order of returned tuples is not guaranteed.
        """
        ...

    @staticmethod
    def available_models() -> list[str]:
        """List available pretrained model names."""
        ...

    def save_base64_vocab(self, path: str) -> None:
        """Save the vocabulary to a file in base64 tiktoken format (excludes special tokens)."""
        ...

class Tokenizer:
    """A BPE tokenizer."""

    @staticmethod
    def from_pretrained(name: str) -> "Tokenizer":
        """Load a pretrained OpenAI tokenizer by name.

        Names: "r50k_base", "p50k_base", "p50k_edit", "cl100k_base",
               "o200k_base", "o200k_harmony"
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Encode a single string to a list of token IDs."""
        ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
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

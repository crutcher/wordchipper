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

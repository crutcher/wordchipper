"""Drop-in replacement for the ``tiktoken`` library, backed by wordchipper.

Typical migration::

    # Before
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # After
    from wordchipper.compat import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
"""

from __future__ import annotations

import functools
from typing import Any, Literal, AbstractSet, Collection, NoReturn

from wordchipper import Tokenizer, SpecialFilter

try:
    frozendict
except NameError:
    try:
        from frozendict import frozendict
    except ImportError:
        frozendict = dict

_SENTINEL = object()

# ---------------------------------------------------------------------------
# Model-to-encoding mappings (based on tiktoken, excluding gpt-2 entries)
# ---------------------------------------------------------------------------

MODEL_TO_ENCODING: dict[str, str] = {
    # chat
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # reasoning
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    # base
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # DALL-E
    "dall-e-2": "cl100k_base",
    "dall-e-3": "cl100k_base",
    # code
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    # edit
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # old completions
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # old embeddings
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
}

MODEL_PREFIX_TO_ENCODING: dict[str, str] = {
    "gpt-4o-": "o200k_base",
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "ft:gpt-4o": "o200k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
    "ft:davinci-002": "cl100k_base",
    "ft:babbage-002": "cl100k_base",
}

_ENCODING_NAMES = [
    m.split(":", 1)[-1] for m in Tokenizer.available_models() if not m.endswith(":gpt2")
]

# Encoding cache (keyed by encoding name)
_cache: dict[str, Encoding] = {}


class Encoding:
    """Wrapper around :class:`wordchipper.Tokenizer` with tiktoken's API."""

    def __init__(self, name: str, tokenizer: Tokenizer) -> None:
        self._name = name
        self._tok = tokenizer

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @functools.cached_property
    def max_token_value(self) -> int:
        core_max = self._tok.max_token
        if core_max is None:
            raise ValueError(
                f"encoding {self._name!r} has an empty core vocabulary"
            )
        special_max = max(self._tok.specials.values()) if self._tok.specials else 0
        return max(core_max, special_max)

    @property
    def n_vocab(self) -> int:
        return self.max_token_value + 1

    @functools.cached_property
    def eot_token(self) -> int:
        try:
            return self._tok.specials["<|endoftext|>"]
        except KeyError:
            raise ValueError(f"encoding {self._name!r} has no <|endoftext|> token")

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        """The set of special tokens in the encoding."""
        return set(self._tok.specials.keys())

    # -- encode / decode -----------------------------------------------------

    def _allowed_filter(
        self,
        allowed_special: Literal["all"] | AbstractSet[str],
    ) -> SpecialFilter:
        if allowed_special == "all":
            return SpecialFilter.include_all()
        elif len(allowed_special):
            return SpecialFilter.include(allowed_special)
        else:
            return SpecialFilter.include_none()

    def _disallowed_specials(
        self,
        allowed_filter: SpecialFilter,
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> frozendict[str, int]:
        if disallowed_special == "all":
            if allowed_filter.is_all():
                return frozendict()

            # every special not in the special filter
            return frozendict(
                {k: v for k, v in self._tok.specials.items() if k not in allowed_filter}
            )

        return frozendict(
            {k: v for k, v in self._tok.specials.items() if k in disallowed_special}
        )

    def _check_disallowed(
        self, tokens: list[int], disallowed: frozendict[str, int]
    ) -> None:
        if disallowed:
            values = set(disallowed.values())
            for t in tokens:
                if t in values:
                    # invert the value-to-key mapping to find the token string for the disallowed token ID
                    span = next(k for (k, v) in disallowed.items() if v == t)
                    raise_disallowed_special_token(span)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        allowed_filter = self._allowed_filter(allowed_special)
        tokens = self._tok.encode(text, special_filter=allowed_filter)

        disallowed = self._disallowed_specials(
            allowed_filter=allowed_filter,
            disallowed_special=disallowed_special,
        )
        self._check_disallowed(tokens, disallowed)

        return tokens

    def encode_ordinary(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel.

        `num_threads` is passed to the underlying tokenizer.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        ```
        >>> enc.encode_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
        allowed_filter = self._allowed_filter(allowed_special)
        batch = self._tok.encode_batch(text, special_filter=allowed_filter)

        disallowed = self._disallowed_specials(
            allowed_filter=allowed_filter,
            disallowed_special=disallowed_special,
        )
        for tokens in batch:
            self._check_disallowed(tokens, disallowed)

        return batch

    def encode_ordinary_batch(self, text: list[str]) -> list[list[int]]:
        return self._tok.encode_batch(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        return self._tok.decode_batch(batch)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def get_encoding(encoding_name: str) -> Encoding:
    """Return an :class:`Encoding` for the given encoding name (cached)."""
    if encoding_name not in _ENCODING_NAMES:
        raise ValueError(
            f"Unknown encoding {encoding_name!r}. "
            f"Available: {', '.join(_ENCODING_NAMES)}"
        )
    if encoding_name not in _cache:
        tok = Tokenizer.from_pretrained(encoding_name)
        _cache[encoding_name] = Encoding(encoding_name, tok)
    return _cache[encoding_name]


def encoding_name_for_model(model_name: str) -> str:
    """Return the encoding name for a model (without loading the encoding)."""
    if model_name in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model_name]
    for prefix, enc_name in MODEL_PREFIX_TO_ENCODING.items():
        if model_name.startswith(prefix):
            return enc_name
    raise KeyError(f"No encoding for model {model_name!r}")


def encoding_for_model(model_name: str) -> Encoding:
    """Return an :class:`Encoding` for the given model name."""
    return get_encoding(encoding_name_for_model(model_name))


def list_encoding_names() -> list[str]:
    """Return the list of available encoding names."""
    return list(_ENCODING_NAMES)


def raise_disallowed_special_token(token: str) -> NoReturn:
    raise ValueError(
        f"Encountered text corresponding to disallowed special token {token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
        f"If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )

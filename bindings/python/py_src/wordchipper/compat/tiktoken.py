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
        val = self._tok.vocab.max_token
        if val is None:
            raise ValueError(
                f"encoding {self._name!r} has an empty vocabulary"
            )
        return val

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
        self, text: str, disallowed: frozendict[str, int]
    ) -> None:
        for token_str in disallowed:
            if token_str in text:
                raise_disallowed_special_token(token_str)

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
        disallowed = self._disallowed_specials(
            allowed_filter=allowed_filter,
            disallowed_special=disallowed_special,
        )
        self._check_disallowed(text, disallowed)

        return self._tok.encode(text, special_filter=allowed_filter)

    def encode_to_numpy(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ):
        import numpy as np

        return np.array(
            self.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            ),
            dtype=np.uint32,
        )

    def encode_ordinary(self, text: str) -> list[int]:
        return self._tok.encode(text, special_filter=SpecialFilter.include_none())

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
        disallowed = self._disallowed_specials(
            allowed_filter=allowed_filter,
            disallowed_special=disallowed_special,
        )
        for t in text:
            self._check_disallowed(t, disallowed)

        return self._tok.encode_batch(text, special_filter=allowed_filter)

    def encode_ordinary_batch(self, text: list[str]) -> list[list[int]]:
        return self._tok.encode_batch(
            text, special_filter=SpecialFilter.include_none()
        )

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        if isinstance(text_or_bytes, bytes):
            try:
                text_or_bytes = text_or_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise KeyError(text_or_bytes)
        token_id = self._tok.vocab.token_to_id(text_or_bytes)
        if token_id is None:
            raise KeyError(text_or_bytes)
        return token_id

    def decode_single_token_bytes(self, token: int) -> bytes:
        raw = self._tok.vocab.id_to_token_bytes(token)
        if raw is None:
            raise KeyError(token)
        return raw

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        return [self.decode_single_token_bytes(t) for t in tokens]

    def is_special_token(self, token: int) -> bool:
        return token in self._special_token_ids

    @functools.cached_property
    def _special_token_ids(self) -> frozenset[int]:
        return frozenset(self._tok.specials.values())

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        if errors == "replace":
            return self._tok.decode(tokens)
        raw = self._tok.decode_bytes(tokens)
        return raw.decode("utf-8", errors=errors)

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return self._tok.decode_bytes(tokens)

    def decode_batch(
        self,
        batch: list[list[int]],
        *,
        errors: str = "replace",
        num_threads: int = 8,
    ) -> list[str]:
        if errors == "replace":
            return self._tok.decode_batch(batch)
        raw_batch = self._tok.decode_bytes_batch(batch)
        return [raw.decode("utf-8", errors=errors) for raw in raw_batch]

    def decode_bytes_batch(
        self,
        batch: list[list[int]],
        *,
        num_threads: int = 8,
    ) -> list[bytes]:
        return self._tok.decode_bytes_batch(batch)

    def token_byte_values(self) -> list[bytes]:
        return [
            self._tok.vocab.id_to_token_bytes(i) or b""
            for i in range(self.max_token_value + 1)
        ]


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

"""Drop-in replacement for the HuggingFace ``tokenizers`` library.

Typical migration::

    # Before
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("Xenova/gpt-4o")

    # After
    from wordchipper.compat.tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("Xenova/gpt-4o")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wordchipper import Tokenizer as _WCTokenizer, SpecialFilter

# ---------------------------------------------------------------------------
# HuggingFace identifier -> wordchipper encoding name
# ---------------------------------------------------------------------------

HF_TO_WORDCHIPPER: dict[str, str] = {
    "Xenova/gpt-4o": "o200k_base",
    "Xenova/gpt-4": "cl100k_base",
    "Xenova/cl100k_base": "cl100k_base",
    "Xenova/o200k_base": "o200k_base",
    "Xenova/text-davinci-003": "p50k_base",
    "Xenova/text-embedding-ada-002": "cl100k_base",
}


@dataclass
class Encoding:
    """Result of a single encode call (mirrors ``tokenizers.Encoding``)."""

    ids: list[int]
    tokens: list[str]


class Tokenizer:
    """Wrapper around :class:`wordchipper.Tokenizer` with HuggingFace's API."""

    def __init__(self, inner: _WCTokenizer) -> None:
        self._tok = inner

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs: Any) -> Tokenizer:
        """Load a tokenizer by HuggingFace identifier or bare encoding name.

        Extra keyword arguments (e.g. ``revision``) are accepted for API
        compatibility but ignored.
        """
        name = HF_TO_WORDCHIPPER.get(identifier, identifier)
        return cls(_WCTokenizer.from_pretrained(name))

    # -- encode / decode -----------------------------------------------------

    def _encode_one(self, text: str) -> Encoding:
        ids = self._tok.encode(text)
        tokens = [t or "" for t in self._tok.vocab.ids_to_tokens(ids)]
        return Encoding(ids=ids, tokens=tokens)

    def encode(
        self,
        sequence: str,
        pair: str | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode a string, returning an :class:`Encoding` with ids and tokens.

        ``is_pretokenized`` is accepted for API compatibility but raises
        :class:`NotImplementedError` when set to ``True``.
        """
        if is_pretokenized:
            raise NotImplementedError("is_pretokenized is not supported")
        enc = self._encode_one(sequence)
        if pair is not None:
            pair_enc = self._encode_one(pair)
            enc = Encoding(
                ids=enc.ids + pair_enc.ids,
                tokens=enc.tokens + pair_enc.tokens,
            )
        return enc

    def encode_batch(
        self,
        input: list[str | tuple[str, str]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        if is_pretokenized:
            raise NotImplementedError("is_pretokenized is not supported")
        result = []
        for item in input:
            if isinstance(item, tuple):
                result.append(self.encode(item[0], pair=item[1]))
            else:
                result.append(self.encode(item))
        return result

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to a string.

        When ``skip_special_tokens`` is ``True`` (default), special token IDs
        are filtered out before decoding.
        """
        if skip_special_tokens:
            ids = self._filter_specials(ids)
        return self._tok.decode(ids)

    def decode_batch(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if skip_special_tokens:
            sequences = [self._filter_specials(ids) for ids in sequences]
        return self._tok.decode_batch(sequences)

    def _filter_specials(self, ids: list[int]) -> list[int]:
        special_ids = self._special_id_set
        return [i for i in ids if i not in special_ids]

    @property
    def _special_id_set(self) -> frozenset[int]:
        try:
            return self.__special_id_set
        except AttributeError:
            self.__special_id_set = frozenset(self._tok.specials.values())
            return self.__special_id_set

    # -- vocab inspection ----------------------------------------------------

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        if with_added_tokens:
            return self._tok.vocab.n_vocab
        return self._tok.vocab_size

    def token_to_id(self, token: str) -> int | None:
        return self._tok.vocab.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._tok.vocab.id_to_token(id)

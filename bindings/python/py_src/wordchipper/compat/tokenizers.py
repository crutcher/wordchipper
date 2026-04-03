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


class Encoding:
    """Result of a single encode call (mirrors ``tokenizers.Encoding``)."""

    __slots__ = (
        "ids", "tokens", "attention_mask", "type_ids",
        "special_tokens_mask", "offsets",
    )

    def __init__(
        self,
        ids: list[int],
        tokens: list[str],
        *,
        attention_mask: list[int] | None = None,
        type_ids: list[int] | None = None,
        special_tokens_mask: list[int] | None = None,
        offsets: list[tuple[int, int]] | None = None,
    ) -> None:
        n = len(ids)
        self.ids = ids
        self.tokens = tokens
        self.attention_mask = attention_mask if attention_mask is not None else [1] * n
        self.type_ids = type_ids if type_ids is not None else [0] * n
        self.special_tokens_mask = (
            special_tokens_mask if special_tokens_mask is not None else [0] * n
        )
        self.offsets = offsets if offsets is not None else [(0, 0)] * n

    def __len__(self) -> int:
        return len(self.ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Encoding):
            return NotImplemented
        return self.ids == other.ids and self.tokens == other.tokens

    def __repr__(self) -> str:
        return f"Encoding(num_tokens={len(self.ids)})"


class Tokenizer:
    """Wrapper around :class:`wordchipper.Tokenizer` with HuggingFace's API."""

    def __init__(self, inner: _WCTokenizer) -> None:
        self._tok = inner
        self._padding: dict | None = None
        self._truncation: dict | None = None

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs: Any) -> Tokenizer:
        """Load a tokenizer by HuggingFace identifier or bare encoding name.

        Extra keyword arguments (e.g. ``revision``) are accepted for API
        compatibility but ignored.
        """
        name = HF_TO_WORDCHIPPER.get(identifier, identifier)
        return cls(_WCTokenizer.from_pretrained(name))

    # -- encode / decode -----------------------------------------------------

    def _make_encoding(
        self,
        ids: list[int],
        type_id: int = 0,
    ) -> Encoding:
        tokens = [t or "" for t in self._tok.vocab.ids_to_tokens(ids)]
        return Encoding(
            ids=ids,
            tokens=tokens,
            type_ids=[type_id] * len(ids),
        )

    def _apply_truncation(self, enc: Encoding) -> Encoding:
        if self._truncation is None:
            return enc
        max_length = self._truncation["max_length"]
        if len(enc.ids) <= max_length:
            return enc
        direction = self._truncation.get("direction", "right")
        if direction == "left":
            s = len(enc.ids) - max_length
            return Encoding(
                ids=enc.ids[s:],
                tokens=enc.tokens[s:],
                attention_mask=enc.attention_mask[s:],
                type_ids=enc.type_ids[s:],
                special_tokens_mask=enc.special_tokens_mask[s:],
                offsets=enc.offsets[s:],
            )
        return Encoding(
            ids=enc.ids[:max_length],
            tokens=enc.tokens[:max_length],
            attention_mask=enc.attention_mask[:max_length],
            type_ids=enc.type_ids[:max_length],
            special_tokens_mask=enc.special_tokens_mask[:max_length],
            offsets=enc.offsets[:max_length],
        )

    def _apply_padding(self, enc: Encoding) -> Encoding:
        if self._padding is None:
            return enc
        length = self._padding.get("length")
        pad_to_multiple = self._padding.get("pad_to_multiple_of")
        target = length
        if target is None:
            target = len(enc.ids)
        if pad_to_multiple and target % pad_to_multiple:
            target += pad_to_multiple - (target % pad_to_multiple)
        if len(enc.ids) >= target:
            return enc
        pad_id = self._padding.get("pad_id", 0)
        pad_token = self._padding.get("pad_token", "[PAD]")
        pad_type_id = self._padding.get("pad_type_id", 0)
        pad_n = target - len(enc.ids)
        direction = self._padding.get("direction", "right")
        pad_ids = [pad_id] * pad_n
        pad_tokens = [pad_token] * pad_n
        pad_mask = [0] * pad_n
        pad_type = [pad_type_id] * pad_n
        pad_special = [1] * pad_n
        pad_offsets = [(0, 0)] * pad_n
        if direction == "left":
            return Encoding(
                ids=pad_ids + enc.ids,
                tokens=pad_tokens + enc.tokens,
                attention_mask=pad_mask + enc.attention_mask,
                type_ids=pad_type + enc.type_ids,
                special_tokens_mask=pad_special + enc.special_tokens_mask,
                offsets=pad_offsets + enc.offsets,
            )
        return Encoding(
            ids=enc.ids + pad_ids,
            tokens=enc.tokens + pad_tokens,
            attention_mask=enc.attention_mask + pad_mask,
            type_ids=enc.type_ids + pad_type,
            special_tokens_mask=enc.special_tokens_mask + pad_special,
            offsets=enc.offsets + pad_offsets,
        )

    def _postprocess(self, enc: Encoding) -> Encoding:
        enc = self._apply_truncation(enc)
        enc = self._apply_padding(enc)
        return enc

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
        ids = self._tok.encode(sequence)
        enc = self._make_encoding(ids, type_id=0)
        if pair is not None:
            pair_ids = self._tok.encode(pair)
            pair_enc = self._make_encoding(pair_ids, type_id=1)
            enc = Encoding(
                ids=enc.ids + pair_enc.ids,
                tokens=enc.tokens + pair_enc.tokens,
                type_ids=enc.type_ids + pair_enc.type_ids,
            )
        return self._postprocess(enc)

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

    # -- padding / truncation ------------------------------------------------

    def enable_padding(
        self,
        *,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self._padding = {
            "direction": direction,
            "pad_id": pad_id,
            "pad_type_id": pad_type_id,
            "pad_token": pad_token,
            "length": length,
            "pad_to_multiple_of": pad_to_multiple_of,
        }

    def no_padding(self) -> None:
        self._padding = None

    @property
    def padding(self) -> dict | None:
        return self._padding

    def enable_truncation(
        self,
        max_length: int,
        *,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None:
        self._truncation = {
            "max_length": max_length,
            "stride": stride,
            "strategy": strategy,
            "direction": direction,
        }

    def no_truncation(self) -> None:
        self._truncation = None

    @property
    def truncation(self) -> dict | None:
        return self._truncation

    # -- vocab inspection ----------------------------------------------------

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        return self._tok.vocab.to_dict()

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        if with_added_tokens:
            return self._tok.vocab.n_vocab
        return self._tok.vocab_size

    def token_to_id(self, token: str) -> int | None:
        return self._tok.vocab.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._tok.vocab.id_to_token(id)

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        return 0

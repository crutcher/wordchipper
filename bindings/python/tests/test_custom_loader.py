import base64
import importlib.util
import json
from pathlib import Path

from wordchipper import SpecialFilter, Tokenizer


def _load_converter_module():
    module_path = Path(__file__).resolve().parents[1] / "hf_to_tiktoken.py"
    spec = importlib.util.spec_from_file_location("hf_to_tiktoken", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf_to_tiktoken = _load_converter_module()


def _write_tokenizer_json(path: Path) -> None:
    payload = {
        "model": {
            "type": "BPE",
            "vocab": {
                "a": 97,
                "b": 98,
                "ab": 300,
            },
        },
        "added_tokens": [
            {"id": 301, "content": "<|end|>"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_tiktoken_file(path: Path) -> dict[int, bytes]:
    tokens: dict[int, bytes] = {}
    with path.open(encoding="ascii") as f:
        for line in f:
            b64, token_id = line.rstrip().split(" ", 1)
            tokens[int(token_id)] = base64.standard_b64decode(b64)
    return tokens


def test_convert_writes_expected_token_bytes(tmp_path):
    source = tmp_path / "tokenizer.json"
    output = tmp_path / "custom.tiktoken"
    _write_tokenizer_json(source)

    hf_to_tiktoken.convert(str(source), output)

    assert _read_tiktoken_file(output) == {
        97: b"a",
        98: b"b",
        300: b"ab",
        301: b"<|end|>",
    }


def test_from_tiktoken_file_loads_custom_vocab_and_specials(tmp_path):
    source = tmp_path / "tokenizer.json"
    output = tmp_path / "custom.tiktoken"
    _write_tokenizer_json(source)
    hf_to_tiktoken.convert(str(source), output)

    tok = Tokenizer.from_tiktoken_file(
        str(output),
        pattern=r"[a-z]+|\s+|.",
        special_tokens={"<|end|>": 301},
    )

    assert tok.encode("ab") == [300]
    assert tok.decode([300]) == "ab"
    assert tok.specials == {"<|end|>": 301}
    assert tok.vocab["<|end|>"] == 301
    assert tok.encode("<|end|>", special_filter=SpecialFilter.include_all()) == [301]
    assert tok.encode("<|end|>", special_filter=SpecialFilter.include_none()) != [301]
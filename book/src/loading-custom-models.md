# Loading Custom Models

wordchipper ships with OpenAI's public BPE vocabularies (GPT-2, GPT-3.5, GPT-4, etc.). But not all
BPE tokenizers follow the OpenAI standard. This chapter explains how to load non-OpenAI models like
Qwen3.5 that have different pre-tokenization patterns, special tokens, or merge table differences.

## When you need a custom loader

You'll need `from_tiktoken_file` when:

- **Using HuggingFace models** like Qwen, Llama, or Mistral that publish their tokenizer vocabulary
  but not as OpenAI-style BPE files.
- **Supporting multiple tokenizers** in the same application and you want a uniform loading path.
- **Distributing a tokenizer** alongside your application and want a simple format.
- **Experimenting with tokenizer design** and training custom vocabularies.

If you're using OpenAI models (GPT-2, GPT-3.5, GPT-4, etc.), stick with `from_pretrained` — it's
faster and handles special tokens automatically.

## Converting HuggingFace tokenizers

Most modern LLMs on HuggingFace Hub use the `tokenizers` library and publish a `tokenizer.json`
file in their model repository. The `hf_to_tiktoken.py` script converts these to tiktoken format,
which wordchipper can load.

### Step 1: Get hf_to_tiktoken.py

The script is in the wordchipper Python bindings:

```bash
cd bindings/python
ls hf_to_tiktoken.py  # It's at the top level of the bindings directory
```

Or for a fresh clone, find it in the repository at `bindings/python/hf_to_tiktoken.py`.

### Step 2: Convert the tokenizer

You can convert from a local `tokenizer.json` file or directly from a HuggingFace model ID.

**From a local file:**

```bash
python hf_to_tiktoken.py tokenizer.json output.tiktoken
```

**From HuggingFace Hub** (requires `huggingface_hub` package):

```bash
# Download and convert Qwen3.5
python hf_to_tiktoken.py Qwen/Qwen3.5-27B qwen3.5.tiktoken

# Or any other model with ByteLevel BPE
python hf_to_tiktoken.py meta-llama/Llama-2-7b llama2.tiktoken
```

The script requires `huggingface_hub` if downloading from the Hub. Install it with:

```bash
pip install huggingface_hub
```

### Step 3: Load in Python

Once you have a `.tiktoken` file, load it with `from_tiktoken_file`:

```python
from wordchipper import Tokenizer

tok = Tokenizer.from_tiktoken_file(
    "qwen3.5.tiktoken",
    pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    special_tokens={
        "<|endoftext|>": 248044,
        "<|im_start|>": 248045,
        "<|im_end|>": 248046,
    },
)

tokens = tok.encode("hello world")
```

## Understanding the parameters

`from_tiktoken_file` takes four arguments:

```python
Tokenizer.from_tiktoken_file(
    path: str,              # Path to .tiktoken file
    pattern: str,           # Pre-tokenization regex pattern
    special_tokens: dict,   # Optional: {token_name: token_id}
    options: TokenizerOptions,  # Optional: performance tuning
)
```

### The pattern parameter

The `pattern` parameter is a Unicode regex string used for pre-tokenization — the first step before
BPE merging. Different models use different patterns:

- **OpenAI (GPT-4, o200k)**: Recognizes CamelCase, Unicode categories (`\p{Lu}`, `\p{Ll}`), handles
  casing transitions.
- **Qwen3.5 and similar**: Simpler pattern, treats contractions differently, uses Unicode character
  classes (`\p{L}`, `\p{M}`) for letters and marks.

Always check the HuggingFace model's `tokenizer_config.json` or `tokenizer.json` for the correct
pattern. For Qwen, it's in the `"regex"` field of the tokenizer metadata.

### The special_tokens parameter

Special tokens are reserved IDs that encode model-specific control sequences like end-of-text,
instruction markers, or tool delimiters. They must be passed separately because:

- The `.tiktoken` file only contains the base vocabulary (individual bytes and their merges).
- Special tokens are defined in the model's tokenizer metadata or training configuration.
- They're often higher-numbered IDs that don't appear in the BPE training corpus.

For Qwen3.5, special tokens look like:

```python
special_tokens = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047,
    # ... more tokens ...
}
```

To extract them:

1. **From HuggingFace**: Look in `tokenizer.json` under `"added_tokens"` or in `tokenizer_config.json`
   under `"chat_template"` and related fields.
2. **From the model paper or documentation**: Models often list their special tokens.
3. **Using the conversion script**: `hf_to_tiktoken.py` with `--include-added-tokens` (default True)
   saves special tokens in the output file's metadata.

## Case study: Qwen3.5

Qwen3.5 is a family of models (0.6B to 72B parameters) that share a single tokenizer. All Qwen3.5
sizes use the same vocabulary and special tokens, so you only need to convert once.

### Loading Qwen3.5 in Python

Here's a complete example:

```python
from wordchipper import Tokenizer, TokenizerOptions, SpecialFilter

# Define the pre-tokenization pattern
QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Define special tokens
QWEN35_SPECIAL_TOKENS = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    # ... all other special ids ...
}

# Load the tokenizer
tok = Tokenizer.from_tiktoken_file(
    "qwen3.5.tiktoken",
    QWEN35_PATTERN,
    QWEN35_SPECIAL_TOKENS,
)

# Encode text
tokens = tok.encode("Hello, world!")
```

### Why Qwen needs BpeBacktrack

Qwen vocabularies have a special property: **pruned parent pairs**. Some merged tokens in the
vocabulary don't have their constituent sub-tokens preserved, breaking the standard BPE assumption.

**Solution: Use BpeBacktrack** (which is the wordchipper default) to ensure that every pair is
validated against the actual merge table.

## Filtering special tokens

Use `SpecialFilter` to control which special tokens are recognized during encoding:

```python
from wordchipper import SpecialFilter

# Include all special tokens (default)
tokens = tok.encode(text, special_filter=SpecialFilter.include_all())

# Exclude all special tokens (treat them as regular text)
tokens = tok.encode(text, special_filter=SpecialFilter.include_none())

# Include only specific special tokens
tokens = tok.encode(
    text,
    special_filter=SpecialFilter.include(["<|endoftext|>"]),
)
```

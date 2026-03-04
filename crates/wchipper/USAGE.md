# Command-Line Help for `wchipper`

This document contains the help content for the `wchipper` command-line program.

**Command Overview:**

* [`wchipper`↴](#wchipper)
* [`wchipper cat`↴](#wchipper-cat)
* [`wchipper lexers`↴](#wchipper-lexers)
* [`wchipper lexers list`↴](#wchipper-lexers-list)
* [`wchipper lexers stress`↴](#wchipper-lexers-stress)
* [`wchipper models`↴](#wchipper-models)
* [`wchipper models list`↴](#wchipper-models-list)
* [`wchipper train`↴](#wchipper-train)
* [`wchipper doc`↴](#wchipper-doc)

## `wchipper`

Text tokenizer multi-tool

**Usage:** `wchipper <COMMAND>`

###### **Subcommands:**

* `cat` — Act as a streaming tokenizer
* `lexers` — Lexers sub-menu
* `models` — Models sub-menu
* `train` — Train a new model
* `doc` — Generate markdown documentation



## `wchipper cat`

Act as a streaming tokenizer

**Usage:** `wchipper cat [OPTIONS] <--model <MODEL>> <--encode|--decode>`

###### **Options:**

* `--model <MODEL>` — Model to use for encoding

  Default value: `openai:r50k_base`
* `--encode` — Encode from text to tokens
* `--decode` — Decode from tokens to text
* `--input <INPUT>` — Optional input file; "-" may be used to indicate stdin
* `--output <OUTPUT>` — Optional output file; "-" may be used to indicate stdout
* `--cache-dir <CACHE_DIR>` — Cache directory



## `wchipper lexers`

Lexers sub-menu

**Usage:** `wchipper lexers <COMMAND>`

###### **Subcommands:**

* `list` — List available lexers
* `stress` — Stress test a regex accelerator



## `wchipper lexers list`

List available lexers

**Usage:** `wchipper lexers list [OPTIONS]`

**Command Alias:** `ls`

###### **Options:**

* `-p`, `--patterns` — Display the patterns



## `wchipper lexers stress`

Stress test a regex accelerator

**Usage:** `wchipper lexers stress [OPTIONS] --input-format <INPUT_FORMAT> <--lexer-model <LEXER_MODEL>|--pattern <PATTERN>> [FILES]...`

###### **Arguments:**

* `<FILES>` — Input files

###### **Options:**

* `--input-format <INPUT_FORMAT>` — The input shard file format

  Possible values:
  - `text`:
    Simple text files
  - `parquet`:
    Parquet files

* `--input-batch-size <INPUT_BATCH_SIZE>` — The input batch size

  Default value: `100`
* `-q`, `--quiet` — Silence log messages
* `-v`, `--verbose` — Turn debugging information on (-v, -vv, -vvv)
* `-t`, `--ts` — Enable timestamped logging
* `--lexer-model <LEXER_MODEL>` — Model name for selection
* `--pattern <PATTERN>` — Pattern for selection
* `--pre-context <PRE_CONTEXT>` — Span context before error

  Default value: `8`
* `--post-context <POST_CONTEXT>` — Span context after error

  Default value: `8`



## `wchipper models`

Models sub-menu

**Usage:** `wchipper models <COMMAND>`

###### **Subcommands:**

* `list` — List available models



## `wchipper models list`

List available models

**Usage:** `wchipper models list`

**Command Alias:** `ls`



## `wchipper train`

Train a new model

**Usage:** `wchipper train [OPTIONS] --input-format <INPUT_FORMAT> <--lexer-model <LEXER_MODEL>|--pattern <PATTERN>> [FILES]...`

###### **Arguments:**

* `<FILES>` — Input files

###### **Options:**

* `--input-format <INPUT_FORMAT>` — The input shard file format

  Possible values:
  - `text`:
    Simple text files
  - `parquet`:
    Parquet files

* `--input-batch-size <INPUT_BATCH_SIZE>` — The input batch size

  Default value: `100`
* `-q`, `--quiet` — Silence log messages
* `-v`, `--verbose` — Turn debugging information on (-v, -vv, -vvv)
* `-t`, `--ts` — Enable timestamped logging
* `--vocab-size <VOCAB_SIZE>` — Max vocab size

  Default value: `50281`
* `--lexer-model <LEXER_MODEL>` — Model name for selection
* `--pattern <PATTERN>` — Pattern for selection
* `--output <OUTPUT>` — Optional output file; "-" may be used to indicate stdout



## `wchipper doc`

Generate markdown documentation

**Usage:** `wchipper doc`



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>


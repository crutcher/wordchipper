# Interactive Tokenizer Demo

Try wordchipper's BPE tokenizer directly in the browser via WebAssembly. Select a model, type some
text, and click **Encode & Decode** to see the token IDs, roundtrip verification, and timing.

<div id="tokenizer-demo">
  <label for="demo-model">Model</label>
  <select id="demo-model">
    <option value="o200k_base" selected>o200k_base (GPT-4o)</option>
    <option value="cl100k_base">cl100k_base (GPT-3.5/4)</option>
    <option value="p50k_base">p50k_base (Codex)</option>
    <option value="r50k_base">r50k_base (GPT-2)</option>
  </select>

  <label for="demo-text">Text</label>
  <textarea id="demo-text">Hello, world! This is wordchipper running in the browser via WASM.</textarea>

  <button id="demo-run" disabled>Encode &amp; Decode</button>

  <div id="demo-status">Loading WASM...</div>
  <pre id="demo-output"></pre>
</div>

<style>
  #tokenizer-demo label {
    display: block;
    margin-top: 1rem;
    font-weight: 600;
  }
  #tokenizer-demo select,
  #tokenizer-demo textarea {
    width: 100%;
    padding: 0.5rem;
    margin-top: 0.25rem;
    font-family: inherit;
  }
  #tokenizer-demo textarea {
    min-height: 80px;
    resize: vertical;
  }
  #tokenizer-demo button {
    margin-top: 1rem;
    padding: 0.5rem 1.5rem;
    cursor: pointer;
  }
  #demo-status {
    margin-top: 0.5rem;
    color: #666;
  }
  #demo-output {
    margin-top: 1rem;
    white-space: pre-wrap;
    font-family: monospace;
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 4px;
    min-height: 60px;
  }

  /* Dark themes. */
  html.coal #demo-output,
  html.navy #demo-output,
  html.ayu #demo-output {
    background: var(--sidebar-bg);
  }
  html.coal #tokenizer-demo select,
  html.coal #tokenizer-demo textarea,
  html.navy #tokenizer-demo select,
  html.navy #tokenizer-demo textarea,
  html.ayu #tokenizer-demo select,
  html.ayu #tokenizer-demo textarea {
    background: var(--sidebar-bg);
    color: var(--fg);
    border-color: var(--sidebar-bg);
  }
</style>

<script type="module">
  const statusEl = document.getElementById("demo-status");
  const outputEl = document.getElementById("demo-output");
  const modelEl = document.getElementById("demo-model");
  const textEl = document.getElementById("demo-text");
  const runBtn = document.getElementById("demo-run");

  if (!statusEl || !runBtn) throw new Error("demo elements not found");

  const VOCAB_URLS = {
    r50k_base:   "wasm/vocab/r50k_base.tiktoken",
    p50k_base:   "wasm/vocab/p50k_base.tiktoken",
    cl100k_base: "wasm/vocab/cl100k_base.tiktoken",
    o200k_base:  "wasm/vocab/o200k_base.tiktoken",
  };

  let init, Tokenizer;
  try {
    const mod = await import("./wasm/wordchipper_wasm.js");
    init = mod.default;
    Tokenizer = mod.Tokenizer;
    await init();
  } catch (e) {
    statusEl.textContent = "WASM not loaded. Run: cargo x book-demo-setup";
    throw e;
  }

  const cache = {};

  statusEl.textContent = "WASM loaded. Select a model and click Encode & Decode.";
  runBtn.disabled = false;

  async function getTokenizer(model) {
    if (cache[model]) return cache[model];
    statusEl.textContent = `Fetching ${model} vocab...`;
    const resp = await fetch(VOCAB_URLS[model]);
    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
    const data = new Uint8Array(await resp.arrayBuffer());
    const tok = Tokenizer.fromVocabData(model, data);
    cache[model] = tok;
    return tok;
  }

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    outputEl.textContent = "";
    try {
      const model = modelEl.value;
      const tokenizer = await getTokenizer(model);
      const text = textEl.value;
      statusEl.textContent = "Encoding...";

      const t0 = performance.now();
      const tokens = tokenizer.encode(text);
      const encMs = (performance.now() - t0).toFixed(2);

      const t1 = performance.now();
      const decoded = tokenizer.decode(tokens);
      const decMs = (performance.now() - t1).toFixed(2);

      const lines = [
        `Model:      ${model}`,
        `Vocab size: ${tokenizer.vocabSize}`,
        `Tokens:     ${tokens.length}`,
        `Token IDs:  [${[...tokens].join(", ")}]`,
        `Decoded:    "${decoded}"`,
        `Roundtrip:  ${text === decoded ? "OK" : "MISMATCH"}`,
        `Encode:     ${encMs} ms`,
        `Decode:     ${decMs} ms`,
      ];
      outputEl.textContent = lines.join("\n");
      statusEl.textContent = "Done.";
    } catch (e) {
      outputEl.textContent = `Error: ${e.message}`;
      statusEl.textContent = "Error occurred.";
    } finally {
      runBtn.disabled = false;
    }
  });
</script>

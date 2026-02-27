// Node.js example for wordchipper WASM bindings.
// Run `cargo x wasm-node` first, then: node index.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load WASM glue and binary from this directory.
const { default: init, Tokenizer } = await import(
  resolve(__dirname, "wordchipper_wasm.js")
);
const wasmBytes = await readFile(resolve(__dirname, "wordchipper_wasm_bg.wasm"));
await init(wasmBytes);

// Load vocab from this directory.
const vocabData = await readFile(resolve(__dirname, "o200k_base.tiktoken"));
const tokenizer = Tokenizer.fromVocabData("o200k_base", vocabData);

console.log(`Available models: ${Tokenizer.availableModels()}`);
console.log(`Vocab size: ${tokenizer.vocabSize}`);
console.log(`Max token: ${tokenizer.maxToken}\n`);

// Encode and decode.
const text = "Hello, world! This is wordchipper running in Node.js via WASM.";
const tokens = tokenizer.encode(text);
console.log(`Text:    "${text}"`);
console.log(`Tokens:  [${tokens}]`);
console.log(`Count:   ${tokens.length}`);

const decoded = tokenizer.decode(tokens);
console.log(`Decoded: "${decoded}"`);
console.log(`Match:   ${text === decoded}\n`);

// Batch encode/decode.
const texts = ["Hello", "world", "foo bar baz"];
const batch = tokenizer.encodeBatch(texts);
console.log("Batch encode:");
for (let i = 0; i < texts.length; i++) {
  console.log(`  "${texts[i]}" -> [${batch[i]}]`);
}

// Special tokens.
const specials = tokenizer.getSpecialTokens();
console.log(`\nSpecial tokens: ${specials.length}`);
for (const [name, id] of specials) {
  console.log(`  ${name} -> ${id}`);
}

// Token lookup.
const helloId = tokenizer.tokenToId("Hello");
console.log(`\ntokenToId("Hello") = ${helloId}`);
if (helloId !== null) {
  console.log(`idToToken(${helloId}) = "${tokenizer.idToToken(helloId)}"`);
}

tokenizer.free();
console.log("\nDone.");

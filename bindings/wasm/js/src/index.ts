import init, {
  Tokenizer as WasmTokenizer,
  type InitInput,
} from "../../pkg/wordchipper_wasm.js";
import { MODEL_CONFIGS } from "./models.js";

let initPromise: Promise<void> | null = null;

/** Ensure WASM module is initialized. Call with custom input to override default loading. */
export async function ensureInit(input?: InitInput): Promise<void> {
  if (!initPromise) {
    initPromise = init(input).then(() => {});
  }
  await initPromise;
}

/** High-level tokenizer wrapping the WASM implementation. */
export class Tokenizer {
  private inner: WasmTokenizer;

  private constructor(inner: WasmTokenizer) {
    this.inner = inner;
  }

  /**
   * Create a tokenizer by fetching vocabulary data from the OpenAI CDN.
   *
   * @param name - Model name (e.g. "cl100k_base", "o200k_base")
   * @param wasmInput - Optional custom WASM module input for `init()`
   */
  static async fromPretrained(
    name: string,
    wasmInput?: InitInput,
  ): Promise<Tokenizer> {
    const config = MODEL_CONFIGS[name];
    if (!config) {
      const valid = Object.keys(MODEL_CONFIGS).join(", ");
      throw new Error(`Unknown model "${name}". Valid models: ${valid}`);
    }

    await ensureInit(wasmInput);

    const response = await fetch(config.url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch vocab for ${name}: ${response.status} ${response.statusText}`,
      );
    }
    const data = new Uint8Array(await response.arrayBuffer());
    return new Tokenizer(WasmTokenizer.fromVocabData(config.baseModel, data));
  }

  /**
   * Create a tokenizer from raw tiktoken vocabulary bytes.
   *
   * @param model - Base model name for pattern/special token resolution
   * @param data - Raw bytes of a .tiktoken file
   * @param wasmInput - Optional custom WASM module input for `init()`
   */
  static async fromVocabData(
    model: string,
    data: Uint8Array,
    wasmInput?: InitInput,
  ): Promise<Tokenizer> {
    await ensureInit(wasmInput);
    return new Tokenizer(WasmTokenizer.fromVocabData(model, data));
  }

  /** Encode text into token IDs. */
  encode(text: string): Uint32Array {
    return this.inner.encode(text);
  }

  /** Decode token IDs back into text. */
  decode(tokens: Uint32Array): string {
    return this.inner.decode(tokens);
  }

  /** Encode multiple texts into arrays of token IDs. */
  encodeBatch(texts: string[]): Uint32Array[] {
    return this.inner.encodeBatch(texts) as Uint32Array[];
  }

  /** Decode multiple token arrays back into strings. */
  decodeBatch(batch: Uint32Array[]): string[] {
    return this.inner.decodeBatch(batch) as string[];
  }

  /** Vocabulary size. */
  get vocabSize(): number {
    return this.inner.vocabSize;
  }

  /** Maximum token ID, or null if empty. */
  get maxToken(): number | null {
    return this.inner.maxToken as number | null;
  }

  /** Look up the token ID for a string. Returns null if not found. */
  tokenToId(token: string): number | null {
    return this.inner.tokenToId(token) as number | null;
  }

  /** Look up the string for a token ID. Returns null if not found. */
  idToToken(id: number): string | null {
    return this.inner.idToToken(id) as string | null;
  }

  /** Get all special tokens as [name, id] pairs. */
  getSpecialTokens(): [string, number][] {
    return this.inner.getSpecialTokens() as [string, number][];
  }

  /** List available model names. */
  static availableModels(): string[] {
    return WasmTokenizer.availableModels() as string[];
  }

  /** Free the underlying WASM memory. */
  free(): void {
    this.inner.free();
  }
}

export { MODEL_CONFIGS } from "./models.js";

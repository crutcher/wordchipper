/** Mapping from model name to tiktoken vocabulary URL and the base model name for pattern resolution. */
export const MODEL_CONFIGS: Record<string, { url: string; baseModel: string }> = {
  r50k_base: {
    url: "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    baseModel: "r50k_base",
  },
  p50k_base: {
    url: "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    baseModel: "p50k_base",
  },
  p50k_edit: {
    // p50k_edit shares p50k_base's vocabulary
    url: "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    baseModel: "p50k_edit",
  },
  cl100k_base: {
    url: "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    baseModel: "cl100k_base",
  },
  o200k_base: {
    url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    baseModel: "o200k_base",
  },
  o200k_harmony: {
    // o200k_harmony shares o200k_base's vocabulary
    url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    baseModel: "o200k_harmony",
  },
};

# Models

Hearth does not ship any model weights. Bring your own GGUF file.

## Which models work

| Family | Examples | Chat format |
|--------|----------|-------------|
| Llama 3 / 3.1 / 3.2 Instruct | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct | Llama 3 |
| Qwen2 / Qwen2.5 Instruct | Qwen2.5-0.5B-Instruct | ChatML |

| Number format | Notes |
|---------------|-------|
| **Q8_0**, **Q4_0** | Recommended. Fastest, and they also run on the GPU engine |
| **Q4_K_M** | Good quality per GB; CPU engine only |
| Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q5_K (e.g. Q5_K_M, Q3_K_M) | Converted to Q8_0 when loaded |
| F32, F16, BF16 | Full precision; works, but slow (reference path) |

As a rule of thumb, a model needs about its file size in free RAM, plus a little for the context.

## Where to get them

GGUF files are on Hugging Face, for example:
- [unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF)
- [unsloth/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF)
- [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)

## Using a model

- **App:** put `.gguf` files in a `models` folder next to `hearth.exe`, or click **Add model…** and pick any file. The app checks each file and marks it *ready*, *CPU only* or *not supported*, with the reason.
- **Command line:** `hearth chat --model path\to\model.gguf`

## Test models (for developers)

The model-tier tests and verification scripts use a fixed set of files.
`pwsh -File scripts\fetch-models.ps1` downloads them into this folder, derives the
quantized variants with llama.cpp's `llama-quantize`, and checks every file against the
SHA-256 hashes pinned in `scripts\pins.psd1`:

| Set | Command | Files |
|-----|---------|-------|
| 1B (default) | `fetch-models.ps1` | Llama-3.2-1B-Instruct: BF16 source; F32, Q8_0, Q4_0 (`--pure`) and Q4_K_M |
| 3B | `fetch-models.ps1 -Set 3b` | Llama-3.2-3B-Instruct: Q8_0, Q4_0 (`--pure`) |
| Qwen | `fetch-models.ps1 -Set qwen0.5b` | Qwen2.5-0.5B-Instruct: F32, Q8_0, Q4_0 (`--pure`) |

The perplexity checks also use WikiText-2 (raw), which the same script fetches.

## Licenses

Model weights are not part of Hearth and keep their own licenses: Llama 3.2 is under the
Llama 3.2 Community License (with an Acceptable Use Policy); Qwen2.5 is under Apache-2.0.
WikiText-2 is CC BY-SA 3.0. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

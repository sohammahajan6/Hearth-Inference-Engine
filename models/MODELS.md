# Models

Model files are not committed; this folder is gitignored except for this file.
- **Download and derive:** `pwsh -File scripts\fetch-models.ps1` (1B set) or `-Set 3b`.
- **Pins:** every file's provenance and sha256 below is also pinned in `scripts\pins.psd1`, and the scripts verify them.

**License:** Llama 3.2 Community License (free of charge, with an Acceptable Use Policy). See THIRD_PARTY_NOTICES.md.

## Llama-3.2-1B-Instruct
**Source file:** downloaded from the ungated mirror `unsloth/Llama-3.2-1B-Instruct-GGUF`, revision `b69aef112e9f895e6f98d7ae0949f72ff09aa401`.

| File | Bytes | sha256 |
|------|-------|--------|
| `Llama-3.2-1B-Instruct-BF16.gguf` | 2,479,595,264 | `f5d05e90d179fa27c01a049be1574a0201bdc09ba0b345eb85e1dbf3b87610f3` |

**Contents:** 147 tensors: 34 F32 (norms, `rope_freqs`) and 113 BF16. There is no `output.weight`, because the embeddings are tied.

**Derived files:** made locally by `llama-quantize --pure <src> <dst> <TYPE> 4` from llama.cpp b11960. `--pure` gives every 2-D weight the same type.

| File | Bytes | sha256 | Used for |
|------|-------|--------|----------|
| `Llama-3.2-1B-Instruct-F32-pure.gguf` | 4,951,088,896 | `61839d5dfb84b8f57b880ddcd12d2d14f4d5277af9d87e980e4c714b44218dc0` | FP32 gates only (BF16 → F32 is lossless) |
| `Llama-3.2-1B-Instruct-Q8_0-pure.gguf` | 1,321,082,624 | `5cadbba719694451b37bcf883aede2f6ed24f585cac15b7e28dd0e568bba0476` | Quantised gates, benchmarks |
| `Llama-3.2-1B-Instruct-Q4_0-pure.gguf` | 703,209,216 | `7268cd950902931966e24e6f4e493e45b11a7204746852a1b19e7cb4b752437e` | Quantised gates, benchmarks |

**K-quant file** (2026-11-13): made by `llama-quantize <src> <dst> Q4_K_M 4`, without `--pure`. It uses llama-quantize's own Q4_K_M recipe, because that is what K-quant files look like in practice: 96 matrices Q4_K, and 17 Q6_K (`token_embd`, the tied LM head, and `attn_v`/`ffn_down` of layers 0, 1, 4, 7, 10, 13, 14, 15). Two runs gave the same bytes.

| File | Bytes | sha256 | Used for |
|------|-------|--------|----------|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807,694,080 | `717810cca515d7b632bb2816129d15873f7d3b737f1dec4a9993a27d116b38c6` | K-quant gates ([ADR 0022](../docs/adr/0022-k-quant-gates.md)), `cpu` backend only |

## Llama-3.2-3B-Instruct (fetched 2026-11-06)
**Source file:** `unsloth/Llama-3.2-3B-Instruct-GGUF`, revision `e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31`.

| File | sha256 |
|------|--------|
| `Llama-3.2-3B-Instruct-BF16.gguf` (6,433,687,744 bytes; kept with `-KeepSource`, the quality reference of the 3B gates) | `9b8dce13b6cbcd8b20037bd6383ee8e747b5034ca32f40b5b8ee2efa9ebf56b8` |

**Derived with `llama-quantize --pure`** (pinned in `scripts\pins.psd1`):

| File | Bytes | sha256 | Use |
|------|-------|--------|-----|
| `Llama-3.2-3B-Instruct-Q8_0-pure.gguf` | 3,421,898,944 | `4378ff9c03a7adf707f500ad837e7214129218f2316ddb41c23d367d84e27a88` | Quantised gates, benchmarks |
| `Llama-3.2-3B-Instruct-Q4_0-pure.gguf` | 1,815,611,584 | `dc269781e84f7ce67150329d92aa94e0fa3005b11db9eaf305a178e411d68b14` | Quantised gates, benchmarks |

## Qwen2.5-0.5B-Instruct (fetched 2026-11-13, `fetch-models.ps1 -Set qwen0.5b`)
**Source file:** Qwen's own repository `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (ungated, Apache-2.0), revision `9217f5db79a29953eb74d5343926648285ec7e67`: `qwen2.5-0.5b-instruct-fp16.gguf` (1,266,425,696 bytes, sha256 `8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc`), saved as `Qwen2.5-0.5B-Instruct-F16.gguf` and deleted after deriving (kept with `-KeepSource`).

**Contents:** architecture `qwen2`, 291 tensors (Q/K/V biases, an untied `output.weight`), vocabulary 151,936.

**Derived with `llama-quantize --pure`** (pinned in `scripts\pins.psd1`):

| File | sha256 | Use |
|------|--------|-----|
| `Qwen2.5-0.5B-Instruct-F32-pure.gguf` | `5580787a2a366d26e4d7e468f601ff42917cb7e94da591f36cf30d22a73f6f5c` | FP32 gates, quality reference |
| `Qwen2.5-0.5B-Instruct-Q8_0-pure.gguf` | `51b344f6dbf12a9e0d7557453f1b115e72277727231b3198dc2f3e5331fb7616` | Quantised gates, benchmarks |
| `Qwen2.5-0.5B-Instruct-Q4_0-pure.gguf` | `fbf0f0348be09e5b163d25de24c8a8d499d708795eb6a64b8c3f5774428a0513` | Quantised gates |

There is no 3B F32 file: it would take about 12.9 GB (twice the BF16 file), and E: must keep 8 GB free. The 3B quality reference is llama.cpp on the BF16 file instead.

## Evaluation text
**Source:** WikiText-2 raw (CC BY-SA 3.0), `ggml-org/ci` dataset, revision `927b3642933080f1b0e811e2f916e14c292992f9`.

| File | sha256 |
|------|--------|
| `wikitext-2-raw-v1.zip` | `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11` |
| `data/wikitext-2-raw/wiki.test.raw` (extracted) | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |

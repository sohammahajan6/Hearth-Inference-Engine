# tools/oracle: hetero-oracle-dump

## Purpose
`hetero-oracle-dump.exe` runs llama.cpp's own forward pass, from the pinned source (b11960), over a
list of token ids and writes chosen intermediate tensors to disk. `hetero dump` writes the same
format from our engine, and `tools/layer_diff.py` compares two dumps tensor by tensor and layer by
layer, so a divergence points at the first operation that differs
.

The tensors are captured through llama.cpp's graph callback (`cb_eval`), under the names llama.cpp's
graph gives them.

## Build
```
pwsh -File scripts\setup.ps1 -Only deps     # once: the pinned llama.cpp source in third_party\llama.cpp
pwsh -File scripts\setup.ps1 -Only oracle   # -> third_party\install\oracle\bin\hetero-oracle-dump.exe
```
`-Only oracle` configures this folder's CMake project with MSVC 14.44 + Ninja (Release, `-Jobs`
parallel jobs, default 4) in `third_party\build\oracle`, then installs only the executable and checks
that it starts. It rebuilds on every run: a first build took 143 s, an up-to-date one 3 s. A build
folder configured from another source folder (a git worktree, say) is deleted and built afresh,
because CMake refuses to reuse it.

`CMakeLists.txt` fixes what is built:
- llama.cpp and ggml as static libraries, with the CPU backend only;
- no llama.cpp common library, tools, examples, server, tests or downloads;
- the instruction set of ggml's `icelake` CPU variant (AVX-512 with VBMI and VNNI, no AVX-VNNI). The
  official b11960 CPU release loads that variant (`ggml-cpu-icelake.dll`) on this machine;
- ggml's own thread pool (no OpenMP), `/fp:precise`.

## Command line
```
hetero-oracle-dump --model F (--tokens 1,2,3 | --tokens-file F) --out DIR
                   --names l_out,result_output,... --positions all|LIST --ctx N
                   [--f32-kv] [--threads N]
```
| Option | Meaning |
|--------|---------|
| `--model F` | GGUF model file |
| `--tokens LIST` | token ids, comma separated (`llama-tokenize -m F -p "text" --ids` prints them) |
| `--tokens-file F` | token ids, one per line (blank lines are skipped) |
| `--out DIR` | output directory, created if missing |
| `--names LIST` | tensor names, comma separated. A name matches the graph tensor with exactly that name and every per-layer tensor `NAME-<layer>`: `l_out` gives `l_out-0` … `l_out-15`, `l_out-3` only layer 3. |
| `--positions P` | `all`, or a comma list of positions and inclusive ranges, e.g. `0-63,100,2040-2047` |
| `--ctx N` | context size: at least the number of tokens, at most the model's trained context |
| `--f32-kv` | F32 KV cache (default F16, as in llama.cpp) |
| `--threads N` | CPU threads (default 4) |

What a run does:
- loads the model on the CPU only (no offload device), with flash attention off;
- evaluates all tokens teacher-forced in one batch and one ubatch (`n_batch = n_ubatch` = the number
  of tokens): positions 0, 1, … in sequence 0, with the logits of every token as outputs;
- writes, for each requested name, the tensor's values at the selected positions. When several graph
  nodes carry the same name, the file holds the last one in graph order and `count` says how many
  there were;
- fails if a requested name matches no graph tensor, and lists the names the graph has.
  `result_output` is the exception: if the graph did not show it, the logits returned by
  `llama_get_logits_ith` are written instead (`"source": "llama_get_logits_ith"`). When the graph did
  show it, it is compared bit for bit with `llama_get_logits_ith` at every written position, and the
  run fails if they differ; this checks the position mapping of the whole dump;
- prints the top-5 next tokens after the last position, as a quick sanity check.

Exit status: 0 on success, 1 on a run-time error, 2 on a usage error. Standard input is never read.

Example:
```
llama-tokenize -m M -p "The capital of France is" --ids --log-disable
  -> [128000, 791, 6864, 315, 9822, 374]
hetero-oracle-dump --model M --tokens 128000,791,6864,315,9822,374 --out DIR --names l_out,result_output --positions all --ctx 64
  -> top-5 after position 5: 12366 ' Paris' 19.152876, 264 ' a' 16.367407, ... (Q8_0 model)
```

## Output format
`DIR/index.json` is written last, so a directory with an `index.json` holds a complete dump. A run
deletes an existing `index.json` first, so a failed run never leaves an old one behind.
```json
{
  "producer": "hetero-oracle-dump (llama.cpp 0.8.0-dev, ggml 0.28.0 8642541)",
  "model": "E:\\Hetero-Engine\\models\\Llama-3.2-1B-Instruct-Q8_0-pure.gguf",
  "tokens": [128000, 791, 6864, 315, 9822, 374],
  "ctx": 64,
  "n_ctx_llama": 256,
  "f32_kv": false,
  "flash_attn": false,
  "threads": 4,
  "tensors": [
    {"name": "l_out-0", "ne": [2048, 6], "positions": [0, 1, 2, 3, 4, 5], "file": "l_out-0.bin", "count": 1, "ggml_ne": [2048, 6], "ggml_type": "f32", "source": "cb_eval"},
    {"name": "Qcur-0", "ne": [2048, 6], "positions": [0, 1, 2, 3, 4, 5], "file": "Qcur-0.bin", "count": 3, "ggml_ne": [64, 32, 6], "ggml_type": "f32", "source": "cb_eval"}
  ]
}
```
| Field | Meaning |
|-------|---------|
| `ctx` | `--ctx` |
| `n_ctx_llama` | the context llama.cpp actually allocated: it rounds `n_ctx` up to a multiple of 256 |
| `ne` | `[width, n_tokens]`. `width` is the number of floats per position, all axes below the token axis flattened: 2048 for `l_out`, 64 × 32 heads = 2048 for `Qcur`, 64 × 8 = 512 for `Kcur` and `Vcur`. `n_tokens` is the number of tokens in the graph, not the number of positions written. |
| `positions` | the positions in the file, ascending, without duplicates |
| `file` | raw little-endian float32, `len(positions) × width` values: one row per position, in `positions` order, and ggml order inside a row (`ne[0]` fastest). f16 and bf16 graph tensors are converted to f32. |
| `count` | how many graph nodes carry this name; the file holds the last |
| `ggml_ne` | ggml's shape of that node, up to and including the token axis |
| `ggml_type` | its ggml type before conversion |
| `source` | `cb_eval`, or `llama_get_logits_ith` (see above) |

Reading one tensor in Python:
```python
t = index["tensors"][0]
x = np.fromfile(d / t["file"], dtype="<f4").reshape(len(t["positions"]), t["ne"][0])
```

## llama.cpp b11960's Llama graph
Observed on 2026-11-04 with Llama-3.2-1B (Q8_0 and F32 files):
- Per-layer tensors are named `<name>-<layer>` and the others `<name>`. ggml names unnamed nodes
  `node_<n>`, and derived tensors after their source, as in `<source> (view)` or
  `<source> (reshaped)`; those match a `--names` entry only when it is given in full.
- The names llama.cpp gives (the error for an unknown name lists them, without the derived ones):
  `embd`, `norm`, `attn_norm`, `Qcur`, `Kcur`, `Vcur`, `kq`, `kq_soft_max`, `kqv`, `kqv_out`,
  `attn_out`, `ffn_inp`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_swiglu`, `ffn_out`, `l_out`,
  `result_norm`, `result_output`.
- `Qcur-<l>` and `Kcur-<l>` appear 3 times per layer: the matmul output (`[2048, n]` and `[512, n]`),
  its reshape to heads (`[64, 32, n]`, `[64, 8, n]`) and the RoPE output, which is what the dump
  keeps. `Vcur-<l>` appears twice (matmul, reshape). `norm-<l>` appears twice (the plain RMS norms
  before `attn_norm` and before `ffn_norm`); the dump keeps the second. Every other name appears
  once.
- `ffn_out-<l>` is the FFN output before the residual add; llama.cpp names the residual sum `l_out`.
- `result_output` is `[128256, n]` f32, and bit-identical to what `llama_get_logits_ith` returns.
- `kq-<l>` is `[256, n, 32]` at `--ctx 64`: attention runs over all of llama.cpp's 256 cells, the
  unused ones masked.
- No CPU weight repacking happens for these files: the matmuls read the memory-mapped GGUF data.

Measured on the same day: dumps are bit-identical across runs and across 1, 4 and 8 threads, and
the dumped values do not depend on which other tensors are requested (requesting `norm`, which
splits the CPU backend's fused RMS norm, leaves `l_out` and the logits bit-identical). 2048 tokens
in one ubatch (Q8_0 file, F32 KV cache) took 66 s, with a 1106 MiB compute buffer.

## Limits
- Only tensors whose outermost axis is the token axis can be dumped. That covers every name in
  docs/architecture/05-forward-pass.md. Attention-score tensors such as `kq-<l>`
  (`[n_kv, n_tokens, n_head]`) are rejected with an error; when `n_tokens` equals `n_head` they
  cannot be told apart from a per-token tensor, so do not request them.
- Memory grows with the number of tokens, because all of them are one ubatch: llama.cpp keeps the
  logits of every token (`n_vocab × n_tokens` floats, 1 GB at 2048 tokens), and without flash
  attention `kq` holds `n_kv × n_tokens × n_head` floats.

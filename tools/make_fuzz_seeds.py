"""Write the GGUF fuzzer's seed inputs and dictionary.

Seeds mirror the real Llama-3.2-1B file: its 32 metadata keys (the token,
token-type and merge arrays truncated to 64 entries, llama.vocab_size set to
match) and its full 147-tensor directory with real shapes (vocabulary
dimension 64), but no tensor data: each seed starts with an 8-byte logical
file size (see tests/fuzz/gguf_target.h) so the bounds checks pass. Plus small
seeds that reach other paths (no tensors; every tensor type; every value type).

Usage (repo root, after scripts\\fetch-models.ps1):
    .venv\\Scripts\\python -I tools\\make_fuzz_seeds.py
"""

import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader, GGUFValueType  # noqa: E402

OUT = ROOT / "tests" / "fuzz" / "seeds" / "gguf"
DICT = ROOT / "tests" / "fuzz" / "gguf.dict"
ALIGN = 32
VOCAB = 64
TYPE_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 30: (1, 2), 12: (256, 144), 39: (32, 17)}


def gstr(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def value(t: int, v) -> bytes:
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    if t == 8:
        return gstr(v)
    if t == 9:
        elem, items = v
        out = struct.pack("<IQ", elem, len(items))
        for x in items:
            out += value(elem, x)
        return out
    return struct.pack(fmt[t], v)


def gguf(kvs, tensors) -> bytes:
    """kvs: [(key, type, value)]; tensors: [(name, ne, type)]. Returns logical-size prefix + header."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for key, t, v in kvs:
        out += gstr(key.encode()) + struct.pack("<I", t) + value(t, v)
    offset = 0
    for name, ne, t in tensors:
        out += gstr(name.encode()) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
        out += struct.pack("<IQ", t, offset)
        bs, ts = TYPE_SIZES[t]
        n = 1
        for x in ne:
            n *= x
        nbytes = n // bs * ts
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    if tensors:
        out += b"\0" * ((ALIGN - len(out) % ALIGN) % ALIGN)
    return struct.pack("<Q", len(out) + offset) + out


def real_seed():
    r = GGUFReader(str(ROOT / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf"))
    kvs = []
    for name, f in r.fields.items():
        if name.startswith("GGUF."):
            continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY:
            elem = f.types[-1]
            items = [f.parts[i] for i in f.data[:VOCAB]]
            if elem == GGUFValueType.STRING:
                items = [bytes(x) for x in items]
            else:
                items = [x[0].item() for x in items]
            kvs.append((name, 9, (int(elem), items)))
        elif t == GGUFValueType.STRING:
            kvs.append((name, 8, bytes(f.parts[f.data[0]])))
        else:
            v = f.parts[f.data[0]][0].item()
            if name == "llama.vocab_size":
                v = VOCAB
            kvs.append((name, int(t), v))
    tensors = []
    for t in r.tensors:
        ne = [int(x) for x in t.shape]
        if t.name == "token_embd.weight":
            ne[1] = VOCAB
        tensors.append((t.name, ne, int(t.tensor_type)))
    return kvs, tensors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kvs, tensors = real_seed()
    seeds = {
        "llama-1b-shapes.bin": gguf(kvs, tensors),
        "vocab-only.bin": gguf([(k, t, v) for k, t, v in kvs if k.startswith(("tokenizer.", "general."))], []),
        "every-value-type.bin": gguf(
            [("u8", 0, 7), ("i8", 1, -7), ("u16", 2, 7), ("i16", 3, -7), ("u32", 4, 7), ("i32", 5, -7), ("f32", 6, 0.5),
             ("bool", 7, 1), ("str", 8, "h\xe9".encode()), ("u64", 10, 7), ("i64", 11, -7), ("f64", 12, 0.25),
             ("arr", 9, (4, [1, 2, 3])), ("strs", 9, (8, [b"a", b"bc"]))], []),
        "every-tensor-type.bin": gguf([], [("f32", [8], 0), ("f16", [8], 1), ("q4", [32, 2], 2), ("q8", [64], 8),
                                           ("bf", [4], 30), ("q4k", [256], 12), ("mx", [32], 39)]),
    }
    for name, data in seeds.items():
        (OUT / name).write_bytes(data)
        print(f"wrote {OUT / name} ({len(data)} bytes)")
    words = sorted({k for k, _, _ in kvs} | {t[0] for t in tensors} | {"GGUF", "general.alignment", "llama"})
    DICT.write_text("".join(f'"{w}"\n' for w in words), encoding="ascii", newline="\n")
    print(f"wrote {DICT} ({len(words)} entries)")


if __name__ == "__main__":
    main()

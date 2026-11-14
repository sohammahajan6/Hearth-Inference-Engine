"""Cross-check hearth's GGUF parser against gguf-py (an independent reader).

For each model file: run `hearth info --model F --json`, read the same file
with gguf-py from the pinned llama.cpp checkout, and compare the version,
alignment, data offset, every metadata key (type and value; string arrays by
an FNV-1a fingerprint of every string), and every tensor (name, type, shape,
offset, size). Exit status 0 only if everything agrees.

Usage (from the repo root):
    .venv\\Scripts\\python -I tools\\gguf_crosscheck.py build\\release\\cli\\hearth.exe models\\*.gguf
"""

import glob
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader, GGUFValueType  # noqa: E402  (path set above)

TYPE_NAMES = {
    GGUFValueType.UINT8: "u8", GGUFValueType.INT8: "i8", GGUFValueType.UINT16: "u16", GGUFValueType.INT16: "i16",
    GGUFValueType.UINT32: "u32", GGUFValueType.INT32: "i32", GGUFValueType.FLOAT32: "f32", GGUFValueType.BOOL: "bool",
    GGUFValueType.STRING: "string", GGUFValueType.ARRAY: "array", GGUFValueType.UINT64: "u64",
    GGUFValueType.INT64: "i64", GGUFValueType.FLOAT64: "f64",
}


def fnv1a(strings):
    h = 1469598103934665603
    for s in strings:
        for b in len(s).to_bytes(8, "little") + s:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def check(hearth_exe, path):
    ours = json.loads(subprocess.run([hearth_exe, "info", "--model", path, "--json"], capture_output=True,
                                     check=True, stdin=subprocess.DEVNULL, timeout=600).stdout)
    r = GGUFReader(path)
    problems = []

    def expect(what, got, want):
        if got != want:
            problems.append(f"{what}: hearth {got!r}, gguf-py {want!r}")

    fields = {name: f for name, f in r.fields.items() if not name.startswith("GGUF.")}
    expect("version", ours["version"], int(r.fields["GGUF.version"].parts[-1][0]))
    expect("alignment", ours["alignment"], r.alignment)
    expect("data offset", ours["data_offset"], r.data_offset)
    expect("metadata key count", len(ours["metadata"]), len(fields))
    for kv in ours["metadata"]:
        f = fields.get(kv["key"])
        if f is None:
            problems.append(f"key {kv['key']} missing in gguf-py")
            continue
        expect(f"{kv['key']} type", kv["type"], TYPE_NAMES[f.types[0]])
        if f.types[0] == GGUFValueType.ARRAY:
            expect(f"{kv['key']} element type", kv["elem"], TYPE_NAMES[f.types[-1]])
            expect(f"{kv['key']} count", kv["count"], len(f.data))
            if f.types[-1] == GGUFValueType.STRING:
                expect(f"{kv['key']} strings", kv["fnv1a"], fnv1a([bytes(f.parts[i]) for i in f.data]))
        else:
            want = f.contents()
            got = kv["value"]
            if isinstance(want, float):
                ok = abs(got - want) <= 1e-12 * max(1.0, abs(want))
                if not ok:
                    problems.append(f"{kv['key']}: hearth {got}, gguf-py {want}")
            else:
                expect(kv["key"], got, want)
    theirs = {t.name: t for t in r.tensors}
    expect("tensor count", len(ours["tensors"]), len(theirs))
    for t in ours["tensors"]:
        g = theirs.get(t["name"])
        if g is None:
            problems.append(f"tensor {t['name']} missing in gguf-py")
            continue
        expect(f"{t['name']} type", t["type"], g.tensor_type.name)
        expect(f"{t['name']} shape", t["ne"], [int(x) for x in g.shape])
        expect(f"{t['name']} offset", t["offset"] + ours["data_offset"], int(g.data_offset))
        expect(f"{t['name']} bytes", t["bytes"], int(g.n_bytes))
    return problems


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exe = sys.argv[1]
    files = [f for pattern in sys.argv[2:] for f in sorted(glob.glob(pattern))]
    failed = 0
    for path in files:
        problems = check(exe, path)
        name = pathlib.Path(path).name
        if problems:
            failed += 1
            print(f"FAIL {name}: {len(problems)} difference(s)")
            for p in problems[:20]:
                print(f"     {p}")
        else:
            print(f"ok   {name}: metadata, string arrays and tensor directory agree with gguf-py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

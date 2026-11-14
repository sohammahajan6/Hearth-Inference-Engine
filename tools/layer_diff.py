"""Compare two tensor dumps (ours vs llama.cpp's oracle) layer by layer.

Both directories hold index.json + one raw float32 file per tensor
(tools/oracle/README.md). For every tensor present in both, over the
positions both recorded:
    rel-L2 = ||ours - ref|| / ||ref||,  max |delta|
For result_output (the logits), per position, with FP64 softmax:
    KLD(ref || ours), top-1 agreement, top-2 probability gap of the reference.

The files are memory-mapped and processed in chunks of positions, so a
2048-position dump of the logits (1 GB per side) needs little RAM.

With --gate fp32, quant or quant-short the docs/ACCEPTANCE.md thresholds
(quantized ones: docs/adr/0007-quantized-gates.md) are checked and the exit
status is 1 if any is missed; quant-k and quant-k-short are the same checks
with the bars of docs/adr/0022-k-quant-gates.md, for files whose matrices are
K-quants (Q8_K activations: one scale per 256 values, a higher floor);
quant-qwen and quant-qwen-short those of docs/adr/0023-qwen-gates.md (Qwen2.5,
whose floor is about twice Llama's). --gate identical requires both dumps to hold
the same tensors at the same positions with the same bits (a backend that
must reproduce another exactly, such as hearth against cpu).

Usage:
    .venv\\Scripts\\python -I tools\\layer_diff.py OURS_DIR ORACLE_DIR [--gate fp32|quant|quant-k|identical]
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np

CHUNK = 64  # positions per step


def load(d: pathlib.Path):
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    out = {}
    for t in index["tensors"]:
        width, n = t["ne"][0], len(t["positions"])
        path = d / t["file"]
        if path.stat().st_size != 4 * width * n:
            raise SystemExit(f"{path}: {path.stat().st_size} bytes, expected 4 x {width} x {n}")
        out[t["name"]] = (t["positions"], np.memmap(path, dtype="<f4", mode="r", shape=(n, width)))
    return index, out


def shared_rows(a, b):
    """Row indices into a and b for the positions both recorded, ascending."""
    pa, pb = a[0], b[0]
    where_b = {p: i for i, p in enumerate(pb)}
    shared = [p for p in pa if p in where_b]
    ia = {p: i for i, p in enumerate(pa)}
    return shared, [ia[p] for p in shared], [where_b[p] for p in shared]


def bit_differences(a, b):
    """How many float32 values differ in their bits (NaN payloads and signed zeros count)."""
    if list(a[0]) != list(b[0]):
        return None
    xa, xb = a[1], b[1]
    count = 0
    for s in range(0, xa.shape[0], CHUNK):
        count += int(np.count_nonzero(xa[s:s + CHUNK].view("<u4") != xb[s:s + CHUNK].view("<u4")))
    return count


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def layer_key(name):
    m = re.match(r"(.*)-(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def compare(a, b):
    """rel-L2 and max |delta| over the shared positions, chunk by chunk in FP64."""
    pos, ia, ib = shared_rows(a, b)
    num = den = 0.0
    worst = 0.0
    for s in range(0, len(pos), CHUNK):
        x = a[1][ia[s:s + CHUNK]].astype(np.float64)
        y = b[1][ib[s:s + CHUNK]].astype(np.float64)
        d = x - y
        num += float((d * d).sum())
        den += float((y * y).sum())
        worst = max(worst, float(np.abs(d).max()))
    return len(pos), (num ** 0.5) / max(den ** 0.5, 1e-30), worst


def logits_stats(a, b):
    """Per-position KLD(ref || ours), top-1 of each, and the reference's top-2 gap."""
    pos, ia, ib = shared_rows(a, b)
    kld, top_ours, top_ref, gap = [], [], [], []
    for s in range(0, len(pos), CHUNK):
        q = a[1][ia[s:s + CHUNK]].astype(np.float64)  # ours
        p = b[1][ib[s:s + CHUNK]].astype(np.float64)  # reference
        lp, lq = log_softmax(p), log_softmax(q)
        pp = np.exp(lp)
        kld.append((pp * (lp - lq)).sum(axis=1))
        top_ours.append(q.argmax(axis=1))
        top_ref.append(p.argmax(axis=1))
        two = np.partition(pp, -2, axis=1)[:, -2:]
        gap.append(two[:, 1] - two[:, 0])
    return (len(pos), np.concatenate(kld), np.concatenate(top_ours), np.concatenate(top_ref), np.concatenate(gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("oracle")
    ap.add_argument("--gate", choices=["fp32", "quant", "quant-short", "quant-k", "quant-k-short", "quant-qwen",
                                       "quant-qwen-short", "identical"],
                    help="fp32: ACCEPTANCE's FP32 bars; quant: the quantized agreement bars of ADR 0007 "
                         "(>= 2048 positions); quant-short: only its mean-KLD and near-tie parts, for short runs; "
                         "quant-k, quant-k-short: the same for K-quant files, with ADR 0022's bars; "
                         "quant-qwen, quant-qwen-short: for Qwen2.5 files, with ADR 0023's bars; "
                         "identical: every tensor and position bit for bit")
    ap.add_argument("--save", help="write the per-position logit statistics (KLD, top-1 of each, the "
                                   "reference's top-2 gap) to this .npz, for tools/kld_aggregate.py")
    args = ap.parse_args()
    _, ours = load(pathlib.Path(args.ours))
    _, ref = load(pathlib.Path(args.oracle))
    names = sorted(set(ours) & set(ref), key=layer_key)
    if not names:
        raise SystemExit("no tensor appears in both dumps")
    for name in names:
        if ours[name][1].shape[1] != ref[name][1].shape[1]:
            raise SystemExit(f"{name}: width {ours[name][1].shape[1]} vs {ref[name][1].shape[1]}")
    failures = []
    print(f"{'tensor':22} {'positions':>9} {'rel-L2':>11} {'max|d|':>11}")
    rel = {}
    for name in names:
        n, r, worst = compare(ours[name], ref[name])
        rel[name] = r
        print(f"{name:22} {n:9} {r:11.3e} {worst:11.3e}")
    checks = []
    if "result_output" in names:
        n, kld, top_ours, top_ref, gap = logits_stats(ours["result_output"], ref["result_output"])
        if args.save:
            np.savez(args.save, kld=kld, top_ours=top_ours, top_ref=top_ref, gap=gap)
        agree = float(np.mean(top_ref == top_ours) * 100)
        print(f"logits: positions {n}, KLD mean {kld.mean():.3e}, p99 {np.percentile(kld, 99):.3e}, "
              f"max {kld.max():.3e}; top-1 agreement {agree:.2f}%")
        differ = top_ref != top_ours
        if differ.any():
            g = gap[differ]
            print(f"top-1 disagreements: {int(differ.sum())}; the reference's top-2 gap there: median {np.median(g):.3e}, "
                  f"max {g.max():.3e}; with gap > 0.01: {int((g > 0.01).sum())}, > 0.05: {int((g > 0.05).sum())}")
        if args.gate == "fp32":
            layers = sorted(layer_key(x)[1] for x in names if layer_key(x)[0] == "l_out")
            if not layers:
                raise SystemExit("--gate fp32 needs l_out in both dumps")
            checks = [("l_out first layer rel-L2", rel[f"l_out-{layers[0]}"], 1e-4),
                      ("l_out last layer rel-L2", rel[f"l_out-{layers[-1]}"], 2e-3),
                      ("logits rel-L2", rel["result_output"], 1e-3),
                      ("mean KLD", float(kld.mean()), 1e-5),
                      ("top-1 disagreements where gap > 0.01", int(np.sum((top_ref != top_ours) & (gap > 0.01))), 0)]
        elif args.gate in ("quant", "quant-short", "quant-k", "quant-k-short", "quant-qwen", "quant-qwen-short"):
            # docs/adr/0007-quantized-gates.md: just above the floor that two correct but not
            # bit-identical implementations reach (8-bit activation codes amplify FP32 rounding).
            # K-quant files: their Q8_K activations share one
            # scale per 256 values, so a flipped code is a larger step and the floor is ~3.5x higher.
            # Qwen2.5: Q8_0 activations too, but llama.cpp against
            # itself sits at 1.2-1.4e-3 on its files, about twice Llama's floor.
            k = args.gate.startswith("quant-k")
            qwen = args.gate.startswith("quant-qwen")
            long_run = args.gate in ("quant", "quant-k", "quant-qwen")
            mean_bar = (3.5e-3 if k else 2e-3 if qwen else 1e-3) * (1 if long_run else 2)
            near_tie_only = int(np.sum((top_ref != top_ours) & (gap > 0.2)))
            checks = [("mean KLD", float(kld.mean()), mean_bar),
                      ("top-1 disagreements where the reference's top-2 gap > 0.2", near_tie_only, 0)]
            if long_run:
                checks += [("p99 KLD", float(np.percentile(kld, 99)), 3.5e-2 if k else 1.5e-2 if qwen else 1e-2),
                           ("top-1 disagreement %", 100.0 - agree, 4.0 if k else 3.0)]
    elif args.gate and args.gate != "identical":
        raise SystemExit(f"--gate {args.gate} needs result_output in both dumps")
    if args.gate == "identical":
        only = sorted(set(ours) ^ set(ref), key=layer_key)
        checks = [("tensors in only one dump", len(only), 0)]
        differing, mismatched = 0, []
        for name in names:
            d = bit_differences(ours[name], ref[name])
            if d is None:
                mismatched.append(name)
            else:
                differing += d
        checks += [("tensors recorded at different positions", len(mismatched), 0),
                   ("values with different bits", differing, 0)]
        print(f"identical: {len(names)} tensors compared bit for bit, {sum(ours[n][1].size for n in names)} values")
    for what, value, bound in checks:
        ok = value <= bound
        shown = f"{bound:.0e}" if float(f"{bound:.0e}") == bound else f"{bound:.1e}"
        print(f"gate {'ok  ' if ok else 'FAIL'} {what}: {value:.3e} (bound {shown})")
        if not ok:
            failures.append(what)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

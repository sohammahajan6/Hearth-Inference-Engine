"""Combine per-chunk logit statistics (tools/layer_diff.py --save) and check
the quantized gates of docs/adr/0007-quantized-gates.md over all positions.

    .venv\\Scripts\\python -I tools\\kld_aggregate.py --agreement A1.npz A2.npz ...
        [--quality-ours Q1.npz ... --quality-llama L1.npz ...]

--agreement: hearth vs the baseline on the same quantized file. Bars: mean
KLD <= 1e-3, p99 <= 1e-2, top-1 agreement >= 97 %, every disagreement at a
reference top-2 gap <= 0.2. With --bars adr0022 (K-quant files, whose Q8_K
activations have a higher floor; docs/adr/0022-k-quant-gates.md): mean KLD
<= 3.5e-3, p99 <= 3.5e-2, top-1 agreement >= 96 %, the same near-tie rule.
With --bars adr0023 (Qwen2.5 files; docs/adr/0023-qwen-gates.md): mean KLD
<= 2e-3, p99 <= 1.5e-2, the other bars as ADR 0007's.
--quality-*: hearth and the baseline, each vs the reference model
(--reference-name; the F32 file for the 1B, the BF16 file for the 3B). Bar: mean KLD
ratio <= 1.1.
The baseline is llama.cpp; for the GPU backends it is hearth's own `cpu`
backend. --ours-name and
--baseline-name only change the labels.
Exit status 1 if any bar is missed.
"""

import argparse
import sys

import numpy as np


def load(paths):
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([x[k] for x in parts]) for k in ("kld", "top_ours", "top_ref", "gap")}


def describe(what, s):
    agree = float(np.mean(s["top_ours"] == s["top_ref"]) * 100)
    differ = s["top_ours"] != s["top_ref"]
    worst = float(s["gap"][differ].max()) if differ.any() else 0.0
    print(f"{what}: positions {len(s['kld'])}, KLD mean {s['kld'].mean():.3e}, p99 {np.percentile(s['kld'], 99):.3e}, "
          f"max {s['kld'].max():.3e}; top-1 agreement {agree:.2f}% ({int(differ.sum())} disagreements, largest "
          f"reference top-2 gap at one {worst:.3e})")
    return agree, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", nargs="+", required=True)
    ap.add_argument("--quality-ours", nargs="*", default=[])
    ap.add_argument("--quality-llama", nargs="*", default=[])
    ap.add_argument("--reference-name", default="the F32 model")
    ap.add_argument("--ours-name", default="hearth")
    ap.add_argument("--baseline-name", default="llama.cpp")
    ap.add_argument("--bars", choices=["adr0007", "adr0022", "adr0023"], default="adr0007")
    args = ap.parse_args()
    k = args.bars == "adr0022"
    qwen = args.bars == "adr0023"
    a = load(args.agreement)
    ours, base = args.ours_name, args.baseline_name
    agree, worst = describe(f"agreement, {ours} vs {base}", a)
    checks = [("mean KLD", float(a["kld"].mean()), 3.5e-3 if k else 2e-3 if qwen else 1e-3),
              ("p99 KLD", float(np.percentile(a["kld"], 99)), 3.5e-2 if k else 1.5e-2 if qwen else 1e-2),
              ("top-1 disagreement %", 100.0 - agree, 4.0 if k else 3.0),
              ("top-1 disagreements where the reference's top-2 gap > 0.2",
               int(np.sum((a["top_ours"] != a["top_ref"]) & (a["gap"] > 0.2))), 0)]
    if args.quality_ours or args.quality_llama:
        q_ours, q_llama = load(args.quality_ours), load(args.quality_llama)
        describe(f"quality, {ours} vs {args.reference_name}", q_ours)
        describe(f"quality, {base} vs {args.reference_name}", q_llama)
        ratio = float(q_ours["kld"].mean() / q_llama["kld"].mean())
        print(f"quality ratio: {q_ours['kld'].mean():.4e} / {q_llama['kld'].mean():.4e} = {ratio:.4f}")
        checks.append((f"quality ratio (mean KLD vs {args.reference_name})", ratio, 1.1))
    failed = []
    for what, value, bound in checks:
        ok = value <= bound
        print(f"gate {'ok  ' if ok else 'FAIL'} {what}: {value:.4g} (bound {bound:g})")
        if not ok:
            failed.append(what)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

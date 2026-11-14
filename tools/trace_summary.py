"""Summarise a `hearth ... --trace out.json` file: time per operation.

    .venv\\Scripts\\python -I tools\\trace_summary.py out.json [--tokens-min N]

Prints, for every event name, the number of events, the total and mean time,
and the share of the time spent inside "forward" events. Events nested in
other events (for example "quantize" inside a matmul) are listed under their
own name, so the shares of nested names overlap their parents.
Events on a named lane (the GPU's, from --backend gpu) are listed too, and
the lane's busy time is printed as a share of the forward time.
--tokens-min N / --tokens-max N keep only "forward" events (one per chunk of
at most 512 tokens of a forward pass) of that many tokens, and the events
inside them: to separate prefill from decode.
"""

import argparse
import bisect
import json
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--tokens-min", type=int, default=0)
    ap.add_argument("--tokens-max", type=int, default=1 << 62)
    a = ap.parse_args()
    with open(a.trace, "rb") as f:
        doc = json.load(f)
    # Complete events only; metadata events ("ph": "M") name the lanes.
    events = [e for e in doc["traceEvents"] if e["ph"] == "X"]
    lanes = {e["tid"]: e["args"]["name"] for e in doc["traceEvents"]
             if e["ph"] == "M" and e["name"] == "thread_name"}
    forwards = [e for e in events if e["name"] == "forward"
                and a.tokens_min <= e["args"].get("tokens", 0) <= a.tokens_max]
    if not forwards:
        print("no forward events match", file=sys.stderr)
        return 1
    # Times in whole nanoseconds (the trace holds microseconds with 0.1 us
    # resolution), so that containment is not decided by float rounding.
    def span(e):
        t0 = round(e["ts"] * 1000)
        return t0, t0 + round(e["dur"] * 1000)

    # Forward events run one after another on one thread, so they do not overlap.
    spans = sorted(span(e) for e in forwards)
    starts = [s0 for s0, _ in spans]

    def inside(e) -> bool:
        t0, t1 = span(e)
        i = bisect.bisect_right(starts, t0) - 1
        return i >= 0 and t1 <= spans[i][1]

    total_forward = sum(s1 - s0 for s0, s1 in spans)
    by_name = defaultdict(lambda: [0, 0.0])
    lane_busy = defaultdict(float)
    for e in events:
        if inside(e):  # a selected forward event contains itself
            by_name[e["name"]][0] += 1
            by_name[e["name"]][1] += e["dur"]
            if e["tid"] in lanes:
                lane_busy[e["tid"]] += e["dur"]
    print(f"{len(forwards)} forward events, {total_forward / 1e6:.1f} ms in total"
          f" (dropped events: {doc.get('otherData', {}).get('dropped_events', 0)})")
    for tid, busy in sorted(lane_busy.items()):
        print(f"lane '{lanes[tid]}': busy {busy / 1e3:.1f} ms, {100 * busy * 1000 / total_forward:.1f}% of the forward time")
    print(f"{'name':<16} {'count':>8} {'total ms':>10} {'mean us':>10} {'share':>7}")
    for name, (n, dur) in sorted(by_name.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:<16} {n:>8} {dur / 1e3:>10.1f} {dur / n:>10.1f} {100 * dur * 1000 / total_forward:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

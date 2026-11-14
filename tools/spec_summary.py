"""Summarize `hearth bench --draft D --prompts FILE` reports into one table.

Usage: python -I tools/spec_summary.py REPORT.txt [REPORT.txt ...]

For each report: the target and draft files, the draft settings, the plain and
speculative decode rates (all tokens / all time), the speedup (all tokens /
all time, and the geometric mean of the per-prompt ratios), the share of draft
tokens kept, and how many runs gave identical output. Reports that did not
finish are listed as such.
"""

import re
import sys
from pathlib import Path


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {"file": path.name}
    m = re.search(r"^bench: models\\(\S+?)\.gguf", text, re.M)
    row["target"] = m.group(1).replace("Llama-3.2-", "").replace("-Instruct", "").replace("-pure", "") if m else "?"
    m = re.search(r"^speculative: draft (\S+?)\.gguf .*?, up to (\d+) tokens, ([^;]+);", text, re.M)
    if m:
        row["draft"] = m.group(1).replace("Llama-3.2-", "").replace("-Instruct", "").replace("-pure", "")
        row["settings"] = f"max {m.group(2)}, {m.group(3)}"
    m = re.search(r"^plain decode tok/s: .*all tokens / all time ([\d.]+)", text, re.M)
    row["plain"] = m.group(1) if m else None
    m = re.search(r"^speculative decode tok/s: .*all tokens / all time ([\d.]+)", text, re.M)
    row["spec"] = m.group(1) if m else None
    m = re.search(r"geometric mean ([\d.]+)x; all tokens / all time ([\d.]+)x", text)
    row["geo"], row["speedup"] = (m.group(1), m.group(2)) if m else (None, None)
    m = re.search(r"draft tokens kept: \d+ of \d+ \(([\d.]+) %\); identical output in (\d+) of (\d+)", text)
    row["kept"], row["identical"] = (m.group(1), f"{m.group(2)}/{m.group(3)}") if m else (None, None)
    return row


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cols = ["file", "target", "draft", "settings", "plain", "spec", "speedup", "geo", "kept", "identical"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for arg in sys.argv[1:]:
        r = parse(Path(arg))
        if r.get("speedup") is None:
            print(f"| {r['file']} | (did not finish) |" + " |" * (len(cols) - 2))
            continue
        print("| " + " | ".join(str(r.get(c, "?")) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

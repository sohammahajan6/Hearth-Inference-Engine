"""Fail if a float multiply, add or subtract in a shader lacks the
NoContraction decoration: the
canonical FP32 chain of ADR 0009 must not be fused into FMAs by the driver.

    .venv\\Scripts\\python -I tools\\check_nocontraction.py SHADER.spvasm [--allow-in FUNCTION ...]

SHADER.spvasm is glslang's human-readable SPIR-V (`glslang -V -H`). Operations
inside a function named with --allow-in (for example the exact fp16 decoder,
whose only multiply is by a power of two) are exempt.
"""

import argparse
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spvasm")
    ap.add_argument("--allow-in", nargs="*", default=[])
    a = ap.parse_args()
    text = open(a.spvasm, encoding="utf-8", errors="replace").read()
    decorated = set(re.findall(r"Decorate (\d+) NoContraction", text))
    bad = []
    function = None
    for line in text.splitlines():
        # A definition: "13(half_bits_to_float(u1;):   10(float) Function None 11".
        m = re.match(r"\s*(\d+)\(([^():;]+)[^:]*:.*\bFunction (None|Inline|DontInline|Pure|Const)\b", line)
        if m:
            function = m.group(2)
        m = re.match(r"\s*(\d+):\s+\d+\(float\)\s+F(Mul|Add|Sub)\b", line)
        if m and m.group(1) not in decorated and function not in a.allow_in:
            bad.append(f"{function}: {line.strip()}")
    for b in bad:
        print("missing NoContraction:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
